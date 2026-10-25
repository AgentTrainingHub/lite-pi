"""Bounded commands in a disposable container, including detached descendants."""

import csv
import io
import json
import os
import re
import selectors
import shutil
import subprocess
import tarfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from lite_pi.outcomes import tool_outcome
from lite_pi.runtime import PATHS
from lite_pi.tools.contracts import JsonObject
from lite_pi.tools.shell_workspace import (
    MAX_WORKSPACE_BYTES,
    MAX_WORKSPACE_ENTRIES,
    ScopeRejected,
    ShellWorkspace,
)

MODEL_VISIBLE_BYTES = 8000
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
IMAGE_FILE = PATHS.state / "runtime" / "image.json"
DOCKER_ENV = {"PATH": "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin"}
if "HOME" in os.environ:
    DOCKER_ENV["HOME"] = os.environ["HOME"]


def _docker(args: list[str], *, timeout: float = 15) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, env=DOCKER_ENV)


def _mount(source: Path, target: str, *, readonly: bool = False) -> list[str]:
    value = io.StringIO()
    fields = ["type=bind", f"src={source.resolve()}", f"dst={target}"]
    if readonly:
        fields.append("readonly")
    csv.writer(value, lineterminator="").writerow(fields)
    return ["--mount", value.getvalue()]


def _capture(
    process: subprocess.Popen[bytes],
    name: str,
    path: Path,
    timeout: float,
    cancel_after: float | None,
    cancel_event: threading.Event | None = None,
) -> tuple[bytearray, int, str | None, str | None]:
    if process.stdout is None:
        raise RuntimeError("command process has no stdout pipe")
    visible = bytearray()
    total = 0
    reason = None
    deadline = time.monotonic() + min(timeout, cancel_after if cancel_after is not None else timeout)
    stop_time = None
    stopped_at = None
    with selectors.DefaultSelector() as selector, path.open("xb") as output:
        selector.register(process.stdout, selectors.EVENT_READ)
        while selector.get_map() or process.poll() is None:
            if reason is None and cancel_event is not None and cancel_event.is_set():
                reason = "cancelled"
            elif reason is None and time.monotonic() >= deadline:
                reason = "cancelled" if cancel_after is not None and cancel_after <= timeout else "timeout"
            for key, _ in selector.select(0.05):
                chunk = os.read(key.fd, 8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                visible.extend(chunk[: max(0, MODEL_VISIBLE_BYTES - len(visible))])
                output.write(chunk[: max(0, MAX_OUTPUT_BYTES - total)])
                total += len(chunk)
                if total > MAX_OUTPUT_BYTES and reason is None:
                    reason = "output_limit"
            if reason is not None and stop_time is None:
                _docker(["kill", name], timeout=5)
                stop_time = time.monotonic()
                stopped_at = datetime.now(UTC).isoformat()
            if stop_time is not None and time.monotonic() >= stop_time + 2:
                break
        output.flush()
        os.fsync(output.fileno())
    return visible, total, reason, stopped_at


def _guest_modes(name: str, evidence: Path) -> dict[str, int]:
    modes: dict[str, int] = {}
    total = 0
    # Docker Desktop can retain guest permission bits separately from host bind-mount metadata.
    with (evidence / (name + "-archive.stderr")).open("wb") as errors:
        process = subprocess.Popen(
            ["docker", "cp", name + ":/workspace/.", "-"],
            stdout=subprocess.PIPE,
            stderr=errors,
            env=DOCKER_ENV,
        )
        if process.stdout is None:
            raise RuntimeError("command process has no stdout pipe")
        deadline = threading.Timer(15, process.kill)
        deadline.start()
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                for member in archive:
                    total += member.size
                    if total > MAX_WORKSPACE_BYTES or len(modes) > MAX_WORKSPACE_ENTRIES:
                        raise ValueError("command output workspace exceeds publication limit")
                    path = Path(member.name)
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError("invalid command archive path")
                    if str(path) != ".":
                        modes[str(path)] = member.mode & 0o777
            if process.wait(timeout=10):
                raise ValueError("cannot inspect guest workspace metadata")
        except tarfile.TarError as exc:
            raise ValueError("invalid guest workspace archive") from exc
        finally:
            deadline.cancel()
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            process.stdout.close()
    (evidence / (name + "-modes.json")).write_text(json.dumps(modes) + "\n")
    return modes


def run_shell(
    root: Path,
    command: str,
    *,
    evidence: Path,
    timeout: float,
    cancel_after: float | None = None,
    readonly: bool = False,
    cancel_event: threading.Event | None = None,
    allowed_paths: tuple[str, ...] | None = None,
) -> JsonObject:
    if shutil.which("docker", path=DOCKER_ENV["PATH"]) is None or not IMAGE_FILE.is_file():
        return tool_outcome(
            {"text": "bash blocked: run uv run python scripts/prepare_runtime.py with Docker running"},
            is_error=True,
            executed=False,
            reason="isolation_unavailable",
            blocks_run=True,
        )
    image = json.loads(IMAGE_FILE.read_text())["image"]
    if not isinstance(image, str) or re.fullmatch(r"python@sha256:[0-9a-f]{64}", image) is None:
        raise ValueError("container image must be the prepared Python digest")
    root = root.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    name = "lite-pi-" + uuid4().hex
    workspace = ShellWorkspace(root, evidence / (name + "-workspace"))
    mounts = _mount(workspace.path, "/workspace", readonly=readonly)
    path = evidence / (name + ".log")
    state: JsonObject = {}
    process = None
    visible = bytearray()
    total = 0
    reason = None
    stopped_at = None
    error = None
    cleanup_error = None
    created = False
    guest_modes = None
    try:
        create = _docker(
            [
                "create",
                "--name",
                name,
                "--init",
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "64",
                "--memory",
                "256m",
                "--cpus",
                "1",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--workdir",
                "/workspace",
                "--tmpfs",
                "/tmp:rw,nosuid,noexec,size=16m",
                "--env",
                "PYTHONDONTWRITEBYTECODE=1",
                "--env",
                "PYTHONNOUSERSITE=1",
                *mounts,
                image,
                "/bin/bash",
                "-c",
                command,
            ]
        )
        if create.returncode:
            reason, error = "isolation_unavailable", create.stderr.strip()
        else:
            created = True
            process = subprocess.Popen(
                ["docker", "start", "--attach", name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=DOCKER_ENV,
            )
            visible, total, reason, stopped_at = _capture(
                process, name, path, timeout, cancel_after, cancel_event
            )
            inspect = _docker(["inspect", "--format", "{{json .State}}", name])
            if inspect.returncode:
                reason, error = "runtime_error", inspect.stderr.strip()
            else:
                state = json.loads(inspect.stdout)
                if not readonly and not state.get("Running", True):
                    guest_modes = _guest_modes(name, evidence)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        reason = "runtime_error" if process is not None else "isolation_unavailable"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        # Removal is attempted even if create timed out after the daemon accepted it.
        try:
            cleanup = _docker(["rm", "--force", name])
            if cleanup.returncode and (created or "no such container" not in cleanup.stderr.lower()):
                cleanup_error = cleanup.stderr.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            if process.stdout is not None:
                process.stdout.close()
    if cleanup_error and process is not None:
        reason = "cleanup_failed"
    if process is not None and cleanup_error is None and guest_modes is not None and not readonly:
        try:
            workspace.publish(guest_modes, allowed_paths=allowed_paths)
        except ScopeRejected as exc:
            reason, error = "path_scope", ", ".join(exc.paths)
        except (OSError, ValueError) as exc:
            reason, error = "workspace_sync_failed", f"{type(exc).__name__}: {exc}"
    code = state.get("ExitCode")
    text = f"exit_code: {code}\n" + visible.decode(errors="replace")
    if total > len(visible):
        text += f"\n[truncated; captured output: {path}]"
    if reason:
        text += f"\n[{reason}] {error or ''} {cleanup_error or ''}".rstrip()
    return tool_outcome(
        {
            "text": text,
            "exit_code": code,
            "timed_out": reason == "timeout",
            "cancelled": reason == "cancelled",
            "stopped_at": stopped_at,
            "command": command,
            "output_path": str(path) if path.exists() else None,
            "output_bytes": total,
            "output_complete": reason
            not in {"output_limit", "runtime_error", "cleanup_failed", "workspace_sync_failed"},
            "container": name,
            "process_reaped": cleanup_error is None,
            "cleanup_error": cleanup_error,
            "isolation": "docker-pid-namespace",
            "image": image,
            "container_state": state,
            "workspace": str(workspace.path),
            "published": workspace.published,
        },
        executed=process is not None,
        is_error=code != 0 or reason is not None,
        reason=reason,
        effect="unknown"
        if reason in {"cleanup_failed", "runtime_error", "workspace_sync_failed"}
        else "none",
        blocks_run=reason
        in {"isolation_unavailable", "cleanup_failed", "runtime_error", "workspace_sync_failed"},
    )
