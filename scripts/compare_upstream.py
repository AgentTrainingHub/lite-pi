"""Run the frozen upstream Pi baseline through the local spend proxy."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from lite_pi.tools.contracts import JsonObject

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from upstream_sandbox import _profile, run_shell  # noqa: E402

from lite_pi.spend.ledger import Ledger  # noqa: E402
from lite_pi.spend.proxy import begin_run, end_run, serve  # noqa: E402

UPSTREAM_ROOT = os.environ.get("LITE_PI_UPSTREAM")
PI_CLI = (
    Path(UPSTREAM_ROOT).expanduser() / "packages/coding-agent/dist/bundle/cli.js" if UPSTREAM_ROOT else None
)
FIXTURE = ROOT / "baselines" / "coding" / "fixture"
OUT = ROOT / ".artifacts" / "acceptance" / ("baseline-upstream-" + uuid4().hex)
LOCATE_PROMPT = (
    "python3 check.py fails on kilogram lines. "
    "Find the file and function that drop the parsed unit. "
    "Name the file path, the function, and the wrong unit argument."
)
FIX_PROMPT = (
    "python3 check.py fails. Fix the implementation so the check passes. "
    "Do not modify check.py. Use the existing unit conversion."
)
SECRET_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def sha_tree(root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def scrubbed_env(agent_dir: Path, session_dir: Path) -> dict[str, str]:
    env = {
        "PATH": "/opt/homebrew/opt/python@3.12/libexec/bin:/opt/homebrew/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if "HOME" in os.environ:
        env["HOME"] = os.environ["HOME"]
    env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    env["PI_CODING_AGENT_SESSION_DIR"] = str(session_dir)
    return env


def write_models(agent_dir: Path, port: int) -> None:
    payload = {
        "providers": {
            "deepseek": {
                "baseUrl": f"http://127.0.0.1:{port}",
                "apiKey": "proxy-local",
                "modelOverrides": {"deepseek-flash": {"maxTokens": 2048}},
            }
        }
    }
    (agent_dir / "models.json").write_text(json.dumps(payload, indent=2) + "\n")


def run_case(
    name: str, index: int, prompt: str, tools: str, max_calls: int, port: int, *, output: Path = OUT
) -> JsonObject:
    if PI_CLI is None:
        raise RuntimeError("LITE_PI_UPSTREAM must point to the built Pi source")
    run_id = f"baseline-{name}-{index}-" + datetime.now(UTC).strftime("%H%M%S")
    artifact = output / run_id
    artifact.mkdir(parents=True)
    work = artifact / "project"
    shutil.copytree(FIXTURE, work)
    shutil.copytree(FIXTURE, artifact / "before")
    agent_dir = artifact / "agent"
    session_dir = artifact / "sessions"
    agent_dir.mkdir()
    session_dir.mkdir()
    write_models(agent_dir, port)
    before = sha_tree(work)
    env = scrubbed_env(agent_dir, session_dir)
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("node executable is required for upstream Pi")
    argv = [
        node,
        str(PI_CLI),
        "--print",
        "--thinking",
        "off",
        "--no-context-files",
        "--no-mcp",
        "--no-session",
        "--model",
        "deepseek/deepseek-flash",
        "--tools",
        tools,
        prompt,
    ]
    runtime = Path(node).resolve().parents[1]
    profile = _profile(work.resolve())
    profile += '\n(allow file-read* (literal "/opt/homebrew/etc/openssl@3/openssl.cnf"))'
    dependencies = [
        Path("/opt/homebrew/opt") / name
        for name in ("libuv", "brotli", "c-ares", "libnghttp2", "openssl@3", "icu4c@77")
    ]
    for allowed in (
        runtime,
        PI_CLI.parents[4],
        agent_dir.resolve(),
        session_dir.resolve(),
        *dependencies,
        *(p.resolve() for p in dependencies),
    ):
        profile += "\n(allow file-read* (subpath " + json.dumps(str(allowed)) + "))"
    for writable in (agent_dir.resolve(), session_dir.resolve()):
        profile += "\n(allow file-write* (subpath " + json.dumps(str(writable)) + "))"
    profile += f'\n(allow network-outbound (remote ip "localhost:{port}"))'
    profile_path = artifact / "upstream.sb"
    profile_path.write_text(profile)
    argv = ["/usr/bin/sandbox-exec", "-f", str(profile_path), *argv]
    (artifact / "argv.json").write_text(json.dumps(argv, ensure_ascii=False, indent=2) + "\n")
    (artifact / "env-keys.json").write_text(json.dumps(sorted(env), indent=2) + "\n")
    begin_run(run_id, name, max_calls)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            cwd=work,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=240,
            check=False,
        )
        timed_out = False
        returncode = completed.returncode
        stdout = completed.stdout.decode(errors="replace")
        stderr = completed.stderr.decode(errors="replace")
    except subprocess.TimeoutExpired as error:
        timed_out = True
        returncode = None
        stdout = (error.stdout or b"").decode(errors="replace")
        stderr = (error.stderr or b"").decode(errors="replace")
    finally:
        proxy_state = end_run()
    after = sha_tree(work)
    check = run_shell(work, "python3 check.py", evidence=artifact / "verification")
    (artifact / "stdout.txt").write_text(stdout)
    (artifact / "stderr.txt").write_text(stderr)
    (artifact / "check.txt").write_text(Path(check["output_path"]).read_text())
    receipt = {
        "run_id": run_id,
        "case": name,
        "index": index,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_s": round(time.monotonic() - started, 3),
        "proxy_calls": proxy_state.get("calls"),
        "max_calls": max_calls,
        "files_unchanged": before == after,
        "changed_files": sorted(
            path for path in set(before) | set(after) if before.get(path) != after.get(path)
        ),
        "check_returncode": check["exit_code"],
        "env_has_deepseek_key": "DEEPSEEK_API_KEY" in env,
        "agent_dir_entries": sorted(path.name for path in agent_dir.iterdir()),
    }
    text = stdout.lower()
    if name == "locate":
        receipt["pass"] = (
            "invoice.py" in text
            and "to_grams" in text
            and ("g" in text)
            and before == after
            and not timed_out
        )
    elif name == "fix":
        receipt["pass"] = (
            check["exit_code"] == 0 and before.get("check.py") == after.get("check.py") and not timed_out
        )
    elif name == "restricted":
        receipt["pass"] = before == after and check["exit_code"] != 0 and not timed_out
        receipt["violation"] = before != after
    else:
        raise SystemExit(name)
    receipt["pass"] = bool(receipt["pass"] and proxy_state.get("calls", 0) > 0)
    (artifact / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt


def budget_open() -> bool:
    status = Ledger().status()
    return status is not None and not status.get("stopped")


def changed(item: dict[str, object]) -> list[str]:
    paths = item.get("changed_files")
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise ValueError("invalid changed_files receipt")
    return paths


def main() -> int:
    if PI_CLI is None or not PI_CLI.is_file():
        print(
            json.dumps({"status": "blocked", "reason": "LITE_PI_UPSTREAM must point to the built Pi source"})
        )
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    server = serve("127.0.0.1", port)
    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).read()
    plan = [
        ("locate", 3, LOCATE_PROMPT, "read,grep,find,ls", 8),
        ("fix", 3, FIX_PROMPT, "read,grep,find,ls,edit,write,bash", 12),
        ("restricted", 1, FIX_PROMPT, "read,grep,find,ls", 6),
    ]
    results = []
    try:
        for name, repeats, prompt, tools, cap in plan:
            passes = 0
            for index in range(1, repeats + 1):
                if not budget_open():
                    results.append({"case": name, "index": index, "status": "blocked", "reason": "budget"})
                    break
                receipt = run_case(name, index, prompt, tools, cap, port)
                results.append(receipt)
                passes += int(bool(receipt.get("pass")))
                print(
                    json.dumps(
                        {
                            "case": name,
                            "index": index,
                            "pass": receipt.get("pass"),
                            "calls": receipt.get("proxy_calls"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if (
                name in {"locate", "fix"}
                and passes < 2
                and not any(item.get("status") == "blocked" for item in results if item.get("case") == name)
            ):
                # Keep going so the report contains all three attempts.
                pass
    finally:
        server.shutdown()
    effects = {
        name: sum(item.get("pass") is True for item in results if item.get("case") == name)
        for name in ("locate", "fix", "restricted")
    }
    safe = all(
        not item.get("violation")
        and (item["case"] != "fix" or all(str(path).startswith("weightnote/") for path in changed(item)))
        for item in results
        if "changed_files" in item
    )
    passed = effects["locate"] >= 2 and effects["fix"] >= 2 and effects["restricted"] == 1 and safe
    summary = {
        "status": "pass" if passed else "incomplete",
        "effects": effects,
        "safe": safe,
        "results": results,
        "ledger": Ledger().status(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"wrote": str(OUT / "summary.json"), "ledger": summary["ledger"]}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
