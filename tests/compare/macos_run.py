"""One disposable macOS install: real code task, reopen, shell cancel, MCP cleanup."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from support import ROOT, check

from compare.combo import marker_closed, marker_count
from compare.judge import FIX_TASK, check_unchanged, implementation_changed, sha_tree
from compare.platform import fresh_import, host_facts
from lite_pi.spend.ledger import LEDGER_PATH
from lite_pi.tools.contracts import JsonObject

FIXTURE = ROOT / "baselines" / "coding" / "fixture"
MARKER = ROOT / "tests" / "fixtures" / "mcp_server.py"


def run(base: Path) -> JsonObject:
    home = base / "home"
    venv = base / "venv"
    project = base / "project"
    session = base / "session.jsonl"
    base.mkdir(parents=True)
    installed = fresh_import(venv, ROOT)
    shutil.copytree(FIXTURE, project)
    before = sha_tree(project)
    env = _env(home, venv)
    code = _python(
        venv,
        env,
        [
            "-m",
            "lite_pi",
            "code",
            "--root",
            str(project),
            "--session",
            str(session),
            "--max-turns",
            "8",
            "--max-task-calls",
            "12",
            FIX_TASK,
        ],
    )
    checked = check(project)
    after = sha_tree(project)
    code_ok = (
        code.get("returncode") == 0
        and code.get("status") == "stop"
        and code.get("model") == "deepseek-flash"
        and checked["returncode"] == 0
        and check_unchanged(before, after)
        and implementation_changed(before, after)
    )
    reopened: JsonObject = _reopen(venv, env, project, session) if code_ok else {"status": "not-run"}
    if reopened.get("returncode") == 0:
        again = check(project)
        reopened["check"] = again["returncode"]
        reopened["check_unchanged"] = check_unchanged(after, sha_tree(project))
    shell = _shell_cancel(venv, env, base / "shell-project")
    mcp = _mcp(venv, env, base / "mcp-project")
    ledger = _python(venv, env, ["-c", "from lite_pi.spend.ledger import Ledger; print(Ledger().path)"])
    py_files = [path for path in project.rglob("*.py") if "__pycache__" not in path.parts]
    report = {
        "host": host_facts(),
        "install": installed,
        "env_has_api_key": "DEEPSEEK_API_KEY" in env,
        "ledger_path": (ledger.get("stdout") or "").strip(),
        "main_ledger": str(LEDGER_PATH),
        "code": {
            "returncode": code.get("returncode"),
            "run_id": code.get("run_id"),
            "model": code.get("model"),
            "status": code.get("status"),
            "check": checked["returncode"],
            "passed": code_ok,
        },
        "reopen": reopened,
        "shell_cancel": shell,
        "mcp": mcp,
        "newline_crlf": any(b"\r\n" in path.read_bytes() for path in py_files),
        "paths_inside_project": all(not path.is_symlink() for path in project.rglob("*")),
    }

    report["passed"] = (
        code_ok
        and reopened.get("returncode") == 0
        and reopened.get("status") == "stop"
        and reopened.get("request_count", 0) > 0
        and reopened.get("model") == "deepseek-flash"
        and reopened.get("check") == 0
        and reopened.get("check_unchanged") is True
        and shell.get("model") == "deepseek-flash"
        and shell.get("returncode") == 0
        and shell.get("request_count", 0) > 0
        and shell.get("cancelled") is True
        and shell.get("process_reaped") is True
        and mcp.get("model") == "deepseek-flash"
        and mcp.get("returncode") == 0
        and mcp.get("status") == "stop"
        and mcp.get("mark_recorded") is True
        and mcp.get("pid_alive") is False
        and mcp.get("pid_observed") is True
        and report["ledger_path"] == report["main_ledger"]
    )
    (base / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _env(home: Path, venv: Path) -> dict[str, str]:
    home.mkdir(parents=True, exist_ok=True)
    return {
        "PATH": f"{venv / 'bin'}:/usr/bin:/bin",
        "HOME": str(home),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "C.UTF-8",
    }


def _python(venv: Path, env: dict[str, str], args: list[str]) -> JsonObject:
    completed = subprocess.run(
        [str(venv / "bin" / "python"), *args],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    payload: JsonObject = {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    text = completed.stdout.strip()
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            payload["parse_error"] = str(error)
            parsed = {}
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                if key in {"text", "events", "messages", "transcript"}:
                    continue
                if isinstance(value, (str, int, float, bool)) or value is None:
                    payload[key] = value
    else:
        payload["stdout"] = text
    if completed.returncode != 0:
        payload["error"] = completed.stderr[-300:]
    return payload


def _reopen(venv: Path, env: dict[str, str], project: Path, session: Path) -> JsonObject:
    script = (
        "import json\n"
        "from lite_pi.session import AgentSession\n"
        f"result = AgentSession.open({str(session)!r}).submit("
        "'Run python3 check.py once and stop. Do not modify files.',"
        " max_turns=3, max_tokens=400, max_task_calls=16)\n"
        "print(json.dumps({'run_id': result.get('run_id'), 'status': result.get('status'),"
        " 'model': result.get('model'), 'request_count': result.get('request_count')}))\n"
    )
    observed = _python(venv, env, ["-c", script])
    observed["new_process"] = True
    return observed


def _shell_cancel(venv: Path, env: dict[str, str], work: Path) -> JsonObject:
    work.mkdir()
    script = (
        "import json\n"
        "from pathlib import Path\n"
        "from lite_pi.session import AgentSession\n"
        f"result = AgentSession(Path({str(work)!r}), tools=('bash',)).submit("
        "'Use bash to run sleep 30 once; do not run any other command.',"
        " max_turns=2, max_tokens=400, max_task_calls=3, shell_timeout=10, cancel_bash_after=0.4)\n"
        "outcomes = [o for t in result.get('transcript', []) for o in t.get('results', [])"
        " if o.get('name') == 'bash' and o.get('executed')]\n"
        "matched = [o for o in outcomes if 'sleep 30' in o.get('command', '')]\n"
        "print(json.dumps({'run_id': result.get('run_id'), 'model': result.get('model'),"
        " 'request_count': result.get('request_count'),"
        " 'cancelled': len(matched) == 1 and matched[0].get('reason') == 'cancelled',"
        " 'process_reaped': bool(matched) and matched[0].get('process_reaped') is True}))\n"
    )
    return _python(venv, env, ["-c", script])


def _mcp(venv: Path, env: dict[str, str], work: Path) -> JsonObject:
    work.mkdir()
    state = work.parent / "mcp-observer"
    config = {
        "servers": [
            {
                "name": "marker",
                "command": str(venv / "bin" / "python"),
                "args": [str(MARKER), "--name", "marker", "--mode", "serve", "--state", str(state)],
                "tools": ["mark"],
                "timeout_seconds": 20,
            }
        ]
    }
    lite = work / ".lite-pi"
    lite.mkdir()
    (lite / "mcp.json").write_text(json.dumps(config), encoding="utf-8")
    session = work.parent / "mcp-session.jsonl"
    script = (
        "import json\n"
        "from pathlib import Path\n"
        "from lite_pi.journal import Journal\n"
        "from lite_pi.session import AgentSession\n"
        f"work = Path({str(work)!r})\n"
        f"journal = Journal.create(Path({str(session)!r}), work, tools=())\n"
        "result = AgentSession(work, tools=(), journal=journal).submit("
        "'Call the mark tool once with label comparison and then stop.', max_turns=3, max_tokens=400,"
        " max_task_calls=4)\n"
        "marks = [o for t in result.get('transcript', []) for o in t.get('results', [])"
        " if o.get('name') == 'mcp__marker__mark' and o.get('executed') and not o.get('is_error')]\n"
        "print(json.dumps({'run_id': result.get('run_id'), 'status': result.get('status'),"
        " 'model': result.get('model'), 'mark_calls': len(marks)}))\n"
    )
    observed = _python(venv, env, ["-c", script])
    calls = state / "calls.jsonl"
    count = marker_count(state) if calls.is_file() else 0
    observed["mark_count"] = count
    observed["mark_recorded"] = count == 1 and observed.get("mark_calls") == 1
    observed["pid_observed"] = (state / "pid").is_file()
    observed["pid_alive"] = not marker_closed(state) if observed["pid_observed"] else None
    return observed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = run(args.output.resolve())
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
