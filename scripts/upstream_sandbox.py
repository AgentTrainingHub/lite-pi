"""Sandbox for the frozen upstream baseline only; product Shell uses containers."""

import json
import os
import selectors
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from lite_pi.tools.contracts import JsonObject

MODEL_VISIBLE_BYTES = 8000
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
RUNTIME_PATH = "/opt/homebrew/opt/python@3.12/libexec/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def run_shell(
    root: Path, command: str, *, evidence: Path, timeout: float = 10, cancel_after: float | None = None
) -> JsonObject:
    sandbox = shutil.which("sandbox-exec")
    if sandbox is None:
        return {"is_error": True, "executed": False, "text": "bash refused: macOS sandbox unavailable"}
    work = root.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    output_path = evidence / ("shell-" + uuid4().hex + ".log")
    # Only these values enter the subprocess. The parent may contain arbitrary credentials.
    env = {
        "PATH": RUNTIME_PATH,
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    visible = bytearray()
    total = 0
    reason = None
    descendants: set[int] = set()
    with tempfile.NamedTemporaryFile("w", suffix=".sb") as profile:
        profile.write(_profile(work))
        profile.flush()
        process = subprocess.Popen(
            [sandbox, "-f", profile.name, "/bin/sh", "-c", command],
            cwd=work,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        assert process.stdout is not None
        deadline = time.monotonic() + min(timeout, cancel_after if cancel_after is not None else timeout)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            with output_path.open("xb") as output:
                while selector.get_map() or process.poll() is None:
                    descendants.update(_descendants(process.pid, descendants))
                    if time.monotonic() >= deadline and reason is None:
                        reason = (
                            "cancelled" if cancel_after is not None and cancel_after <= timeout else "timeout"
                        )
                        _kill(process.pid, descendants)
                    if process.poll() is not None:
                        _kill(process.pid, descendants)
                    for key, _ in selector.select(0.02):
                        chunk = os.read(key.fd, 8192)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        if len(visible) < MODEL_VISIBLE_BYTES:
                            visible.extend(chunk[: MODEL_VISIBLE_BYTES - len(visible)])
                        remaining = MAX_OUTPUT_BYTES - total
                        output.write(chunk[: max(0, remaining)])
                        total += len(chunk)
                        if total > MAX_OUTPUT_BYTES and reason is None:
                            reason = "output_limit"
                            _kill(process.pid, descendants)
                    if reason is not None and time.monotonic() > deadline + 2:
                        break
                output.flush()
                os.fsync(output.fileno())
        finally:
            _kill(process.pid, descendants)
            process.wait(timeout=5)
            selector.close()
            process.stdout.close()
    remaining_pids = [pid for pid in descendants if _alive(pid)]
    if remaining_pids:
        reason = "cleanup_failed"
    text = visible.decode(errors="replace")
    if total > len(visible):
        text += f"\n[truncated; full captured output: {output_path}]"
    if reason:
        text += f"\n[{reason}]"
    text = f"exit_code: {process.returncode}\n" + text
    return {
        "is_error": reason is not None or process.returncode != 0,
        "executed": True,
        "text": text,
        "output_path": str(output_path),
        "output_bytes": total,
        "output_complete": reason != "output_limit",
        "exit_code": process.returncode,
        "timed_out": reason == "timeout",
        "cancelled": reason == "cancelled",
        "reason": reason,
        "command": command,
        "pid": process.pid,
        "descendants": sorted(descendants),
        "remaining_pids": remaining_pids,
        "process_reaped": not remaining_pids,
        "isolation": "macos-seatbelt",
    }


def _profile(work: Path) -> str:
    read_paths = ["/System", "/usr/lib", "/usr/share", "/usr/bin", "/usr/sbin", "/bin", "/sbin"]
    # Resolve only runtime packages, never grant all Homebrew or all temporary files.
    for package in ("python@3.12", "openssl@3", "gettext", "libffi", "mpdecimal", "xz"):
        path = Path("/opt/homebrew/opt") / package
        if path.exists():
            read_paths.extend([str(path), str(path.resolve())])
    reads = "\n".join(f"(allow file-read* (subpath {json.dumps(path)}))" for path in read_paths)
    escaped = json.dumps(str(work))
    protected = r'"(^|/)([.][eE][nN][vV]([.][^/]*)?|[.][gG][iI][tT]|[.][sS][sS][hH]|[.][aA][wW][sS])(/|$)"'
    artifacts = json.dumps(str(work / ".artifacts"))
    return f"""(version 1)
(deny default)
(allow process-exec process-fork)
(allow signal (target self))
(allow sysctl-read)
(allow mach-lookup (global-name "com.apple.system.logger"))
(allow file-read-metadata)
(allow file-read* (literal "/") (literal "/dev/null") (literal "/dev/urandom") (literal "/dev/random"))
{reads}
(allow file-read* file-write* (subpath {escaped}))
(deny file-read* file-write* (regex #{protected}) (subpath {artifacts}))
(deny network*)
"""


def _descendants(parent: int, known: set[int]) -> set[int]:
    listing = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid="],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
        check=True,
    )
    pairs = [tuple(map(int, line.split())) for line in listing.stdout.splitlines() if line.strip()]
    found = {parent} | known
    previous = -1
    while previous != len(found):
        previous = len(found)
        found.update(pid for pid, ppid in pairs if ppid in found)
    return found - {parent}


def _kill(group: int, descendants: set[int]) -> None:
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for pid in descendants:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
