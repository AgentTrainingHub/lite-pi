"""Real DeepSeek recovery. Each crash is a separate process, judged from disk."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from support import ROOT, Suite

READ_TASK = (
    "Call the read tool once with path token.txt. "
    "After its result, reply with the exact file contents and no other text."
)
WRITE_TASK = (
    "Call the write tool once. path is mark.txt and content is exactly {marker}. Do not call another tool."
)


def run(suite: Suite) -> None:
    for index in range(1, 4):
        _read_recovery(suite, index)
    _write_reconcile(suite)
    _result_backfill(suite)
    _approval(suite)
    _shared_calls(suite)
    _time_budget(suite)


def _read_recovery(suite: Suite, index: int) -> None:
    project = suite.project(f"recovery-read-{index}")
    token = f"recovery-read-{index}-{project.name[-8:]}"
    target = project / "token.txt"
    target.write_text(token, encoding="utf-8")
    before = _fingerprint(target)
    journal = project / "session.jsonl"
    code, out, err, pid = _drive(
        ["code", "--root", str(project), "--session", str(journal), "--max-turns", "2", READ_TASK],
        crash="intent",
    )
    intent = _intent(journal)
    hit = (
        code == 86
        and _dead(pid)
        and intent is not None
        and intent["name"] == "read"
        and intent["status"] == "intent"
        and intent["replay"] == "safe"
    )
    observed: dict[str, object] = {}
    resumed = (0, "", "", 0)
    if hit and intent is not None:
        resumed = _drive(["resume", "--session", str(journal), "--max-turns", "2"])
        observed = _observed(resumed[1])
    tools = [] if intent is None else _tools_for(journal, str(intent["call_id"]))
    after = _fingerprint(target)
    text = str(observed.get("text") or "")
    passed = (
        hit
        and observed.get("status") in {"stop", "turn_limit"}
        and len(tools) == 1
        and token in _content(tools[0])
        and after == before
        and _owner(journal) >= 2
    )
    suite.record(
        f"recovery-read-recovery-{index}",
        passed,
        effect="read-recovery",
        safe=after == before,
        hit=hit,
        crash_code=code,
        pid=pid,
        process_dead=_dead(pid),
        run_status=observed.get("status"),
        owner_version=_owner(journal),
        tool_count=len(tools),
        text_has_token=token in text,
    )
    _save(project, out, err, resumed)


def _write_reconcile(suite: Suite) -> None:
    project = suite.project("recovery-write")
    marker = f"MARKER-{project.name[-8:]}"
    journal = project / "session.jsonl"
    target = project / "mark.txt"
    code, out, err, pid = _drive(
        [
            "code",
            "--root",
            str(project),
            "--session",
            str(journal),
            "--max-turns",
            "2",
            WRITE_TASK.format(marker=marker),
        ],
        crash="effect",
    )
    intent = _intent(journal)
    digest = _fingerprint(target)
    hit = (
        code == 86
        and _dead(pid)
        and intent is not None
        and intent["name"] == "write"
        and intent["status"] == "intent"
        and _matches_intent(target, intent)
    )
    stamp = target.stat().st_mtime_ns if target.is_file() else None
    observed: dict[str, object] = {}
    resumed = (0, "", "", 0)
    if hit and intent is not None:
        resumed = _resume_readonly(journal, max_turns=2)
        observed = _observed(resumed[1])
    tools = [] if intent is None else _tools_for(journal, str(intent["call_id"]))
    current = _intent(journal)
    passed = (
        hit
        and stamp is not None
        and target.is_file()
        and target.stat().st_mtime_ns == stamp
        and _fingerprint(target) == digest
        and current is not None
        and current["status"] == "settled"
        and len(tools) == 1
        and _outcome(tools[0]).get("effect") == "reconciled"
    )
    suite.record(
        "recovery-write-reconcile",
        passed,
        safe=bool(hit and _fingerprint(target) == digest),
        hit=hit,
        crash_code=code,
        process_dead=_dead(pid),
        run_status=observed.get("status"),
        intent_status=None if current is None else current["status"],
    )
    _save(project, out, err, resumed)


def _result_backfill(suite: Suite) -> None:
    project = suite.project("recovery-result")
    marker = f"RESULT-{project.name[-8:]}"
    journal = project / "session.jsonl"
    target = project / "mark.txt"
    code, out, err, pid = _drive(
        [
            "code",
            "--root",
            str(project),
            "--session",
            str(journal),
            "--max-turns",
            "2",
            WRITE_TASK.format(marker=marker),
        ],
        crash="result",
    )
    intent = _intent(journal)
    hit = (
        code == 86
        and _dead(pid)
        and intent is not None
        and intent["status"] == "result_saved"
        and _matches_intent(target, intent)
        and _tools(journal) == []
    )
    stamp = target.stat().st_mtime_ns if target.is_file() else None
    first = (0, "", "", 0)
    second = (0, "", "", 0)
    if hit and intent is not None:
        first = _resume_readonly(journal, max_turns=2)
        second = _resume_readonly(journal, max_turns=1, max_task_calls=1)
    tools = [] if intent is None else _tools_for(journal, str(intent["call_id"]))
    current = _intent(journal)
    second_observed = _observed(second[1])
    passed = (
        hit
        and intent is not None
        and stamp is not None
        and len(tools) == 1
        and _body(tools[0]).get("intent_id") == intent["intent_id"]
        and current is not None
        and current["status"] == "settled"
        and target.stat().st_mtime_ns == stamp
        and second_observed.get("request_count") == 0
    )
    suite.record(
        "recovery-result-backfill",
        passed,
        safe=True,
        hit=hit,
        crash_code=code,
        process_dead=_dead(pid),
        tool_count=len(tools),
        second_requests=second_observed.get("request_count"),
    )
    _save(project, out, err, first)
    (project / "second.stdout").write_text(second[1], encoding="utf-8")


def _approval(suite: Suite) -> None:
    project = suite.project("recovery-approval")
    marker = f"APPROVED-{project.name[-8:]}"
    journal = project / "session.jsonl"
    target = project / "mark.txt"
    code, out, err, pid = _drive(
        [
            "code",
            "--root",
            str(project),
            "--session",
            str(journal),
            "--max-turns",
            "2",
            "--approvals",
            "--max-task-calls",
            "1",
            WRITE_TASK.format(marker=marker),
        ]
    )
    proposed = _observed(out)
    intent = _intent(journal)
    approval = _approval_row(journal)
    arguments = intent["arguments"] if intent is not None and isinstance(intent["arguments"], dict) else {}
    hit = (
        code == 0
        and _dead(pid)
        and proposed.get("status") == "awaiting_approval"
        and intent is not None
        and intent["name"] == "write"
        and intent["status"] == "awaiting_approval"
        and arguments.get("path") == "mark.txt"
        and approval is not None
        and "mark.txt" in str(approval["preview"])
        and not target.exists()
    )
    waiting_observed: dict[str, object] = {}
    receipt: dict[str, object] = {}
    duplicate: dict[str, object] = {}
    executed_observed: dict[str, object] = {}
    again_observed: dict[str, object] = {}
    stamp = None
    if hit and approval is not None:
        waiting = _drive(["resume", "--session", str(journal), "--max-turns", "2"])
        (project / "waiting.stdout").write_text(waiting[1], encoding="utf-8")
        (project / "waiting.stderr").write_text(waiting[2], encoding="utf-8")
        waiting_observed = _observed(waiting[1])
        still_waiting = (
            waiting_observed.get("status") == "awaiting_approval"
            and waiting_observed.get("request_count") == 0
            and not target.exists()
        )
        if still_waiting:
            receipt = _observed(
                _drive(
                    [
                        "approve",
                        "--session",
                        str(journal),
                        "--id",
                        str(approval["approval_id"]),
                        "--decision",
                        "approve",
                    ]
                )[1]
            )
            duplicate = _observed(
                _drive(
                    [
                        "approve",
                        "--session",
                        str(journal),
                        "--id",
                        str(approval["approval_id"]),
                        "--decision",
                        "approve",
                    ]
                )[1]
            )
            if (
                receipt.get("decision") == "approved"
                and receipt.get("executed") is False
                and receipt == duplicate
            ):
                executed = _drive(["resume", "--session", str(journal), "--max-turns", "2"])
                executed_observed = _observed(executed[1])
                if target.is_file():
                    stamp = target.stat().st_mtime_ns
                again = _drive(["resume", "--session", str(journal), "--max-turns", "1"])
                again_observed = _observed(again[1])
    tools = [] if intent is None else _tools_for(journal, str(intent["call_id"]))
    content = ""
    if intent is not None and isinstance(intent["arguments"], dict):
        raw = intent["arguments"].get("content")
        content = raw if isinstance(raw, str) else ""
    passed = (
        hit
        and waiting_observed.get("status") == "awaiting_approval"
        and waiting_observed.get("request_count") == 0
        and receipt.get("decision") == "approved"
        and receipt.get("executed") is False
        and receipt == duplicate
        and executed_observed.get("status") == "budget"
        and executed_observed.get("reason") == "task_calls"
        and executed_observed.get("request_count") == 0
        and target.is_file()
        and target.read_text(encoding="utf-8") == content
        and marker in content
        and len(tools) == 1
        and again_observed.get("request_count") == 0
        and stamp is not None
        and target.stat().st_mtime_ns == stamp
    )
    suite.record(
        "recovery-approval-restart",
        passed,
        safe=not target.exists() or target.read_bytes() == content.encode(),
        hit=hit,
        waiting_status=waiting_observed.get("status"),
        executed_status=executed_observed.get("status"),
        executed_reason=executed_observed.get("reason"),
        again_requests=again_observed.get("request_count"),
    )
    _save(project, out, err, (code, out, err, pid))


def _shared_calls(suite: Suite) -> None:
    project = suite.project("recovery-calls")
    token = f"recovery-calls-{project.name[-8:]}"
    target = project / "token.txt"
    target.write_text(token, encoding="utf-8")
    before = _fingerprint(target)
    journal = project / "session.jsonl"
    code, out, err, pid = _drive(
        [
            "code",
            "--root",
            str(project),
            "--session",
            str(journal),
            "--max-turns",
            "2",
            "--max-task-calls",
            "1",
            READ_TASK,
        ],
        crash="intent",
    )
    intent = _intent(journal)
    hit = (
        code == 86
        and _dead(pid)
        and intent is not None
        and intent["name"] == "read"
        and intent["status"] == "intent"
    )
    observed: dict[str, object] = {}
    resumed = (0, "", "", 0)
    if hit and intent is not None:
        resumed = _drive(["resume", "--session", str(journal), "--max-turns", "2"])
        observed = _observed(resumed[1])
    tools = [] if intent is None else _tools_for(journal, str(intent["call_id"]))
    passed = (
        hit
        and observed.get("status") == "budget"
        and observed.get("reason") == "task_calls"
        and observed.get("request_count") == 0
        and len(tools) == 1
        and token in _content(tools[0])
        and _fingerprint(target) == before
    )
    suite.record(
        "recovery-shared-calls",
        passed,
        safe=_fingerprint(target) == before,
        hit=hit,
        crash_code=code,
        process_dead=_dead(pid),
        run_status=observed.get("status"),
        run_reason=observed.get("reason"),
        request_count=observed.get("request_count"),
    )
    _save(project, out, err, resumed)


def _time_budget(suite: Suite) -> None:
    project = suite.project("recovery-time")
    token = f"recovery-time-{project.name[-8:]}"
    (project / "token.txt").write_text(token, encoding="utf-8")
    journal = project / "session.jsonl"
    code, out, err, pid = _drive(
        [
            "code",
            "--root",
            str(project),
            "--session",
            str(journal),
            "--max-turns",
            "3",
            "--max-task-seconds",
            "1",
            READ_TASK,
        ]
    )
    observed = _observed(out)
    tools = _tools(journal)
    content = _content(tools[0]) if tools else ""
    hit = observed.get("status") == "budget" and observed.get("reason") == "time_budget"
    passed = (
        hit
        and _dead(pid)
        and 1 <= observed.get("request_count", 0) <= 3
        and len(tools) <= 1
        and (not tools or token in content)
    )
    suite.record(
        "recovery-time-budget",
        passed,
        safe=True,
        hit=bool(hit),
        return_code=code,
        process_dead=_dead(pid),
        run_status=observed.get("status"),
        run_reason=observed.get("reason"),
        request_count=observed.get("request_count"),
        tool_count=len(tools),
    )
    _save(project, out, err, (code, out, err, pid))


def _drive(args: list[str], crash: str | None = None) -> tuple[int, str, str, int]:
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": os.environ.get("HOME", ""),
    }
    if crash is not None:
        env["LITE_PI_CRASH_AFTER"] = crash
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "tests/crash_injection.py"), *args],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=180)
        code = proc.returncode if proc.returncode is not None else 1
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        code = proc.returncode if proc.returncode is not None else -9
    return code, stdout, stderr, proc.pid


def _resume_readonly(
    journal: Path, *, max_turns: int, max_task_calls: int | None = None
) -> tuple[int, str, str, int]:
    limit = "None" if max_task_calls is None else str(max_task_calls)
    script = f"""
import json
from pathlib import Path
from lite_pi.session import AgentSession
session = AgentSession.open(Path({str(journal)!r}), tools=("read",))
limits = {{"max_turns": {max_turns}, "max_tokens": 640}}
if {limit} is not None:
    limits["max_task_calls"] = {limit}
result = session.resume(**limits)
print(json.dumps({{
    "status": result["status"],
    "reason": result.get("reason"),
    "text": result.get("text"),
    "request_count": result.get("request_count"),
    "run_id": result["run_id"],
}}, ensure_ascii=False))
"""
    return _python(script)


def _python(script: str) -> tuple[int, str, str, int]:
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": os.environ.get("HOME", ""),
    }
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=180)
        code = proc.returncode if proc.returncode is not None else 1
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        code = proc.returncode if proc.returncode is not None else -9
    return code, stdout, stderr, proc.pid


def _db(journal: Path) -> Path:
    return journal.with_name(journal.name + ".tasks.sqlite")


def _intent(journal: Path) -> dict[str, object] | None:
    path = _db(journal)
    if not path.is_file():
        return None
    with sqlite3.connect(path) as db:
        row = db.execute(
            """
            SELECT intent_id, call_id, name, status, replay, arguments, after_sha256
            FROM intents ORDER BY rowid LIMIT 1
            """
        ).fetchone()
    if row is None:
        return None
    arguments = json.loads(row[5])
    return {
        "intent_id": row[0],
        "call_id": row[1],
        "name": row[2],
        "status": row[3],
        "replay": row[4],
        "arguments": arguments if isinstance(arguments, dict) else {},
        "after_sha256": row[6],
    }


def _approval_row(journal: Path) -> dict[str, object] | None:
    path = _db(journal)
    if not path.is_file():
        return None
    with sqlite3.connect(path) as db:
        row = db.execute(
            "SELECT approval_id, decision, preview FROM approvals ORDER BY rowid LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    return {"approval_id": row[0], "decision": row[1], "preview": row[2]}


def _owner(journal: Path) -> int:
    path = _db(journal)
    if not path.is_file():
        return 0
    with sqlite3.connect(path) as db:
        row = db.execute("SELECT version FROM ownership WHERE session_id IS NOT NULL LIMIT 1").fetchone()
    return 0 if row is None else int(row[0])


def _tools(journal: Path) -> list[dict[str, object]]:
    if not journal.is_file():
        return []
    found = []
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if isinstance(record, dict) and record.get("kind") == "tool":
            found.append(record)
    return found


def _tools_for(journal: Path, call_id: str) -> list[dict[str, object]]:
    return [record for record in _tools(journal) if _body(record).get("tool_call_id") == call_id]


def _body(record: dict[str, object]) -> dict[str, object]:
    body = record.get("body")
    return body if isinstance(body, dict) else {}


def _content(record: dict[str, object]) -> str:
    content = _body(record).get("content")
    return content if isinstance(content, str) else ""


def _outcome(record: dict[str, object]) -> dict[str, object]:
    outcome = _body(record).get("outcome")
    return outcome if isinstance(outcome, dict) else {}


def _matches_intent(path: Path, intent: dict[str, object]) -> bool:
    arguments = intent["arguments"]
    content = arguments.get("content") if isinstance(arguments, dict) else None
    if not isinstance(content, str) or not path.is_file():
        return False
    data = path.read_bytes()
    return data == content.encode() and hashlib.sha256(data).hexdigest() == intent["after_sha256"]


def _fingerprint(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dead(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _load(stdout: str) -> dict[str, object]:
    line = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return {"status": "unparsed", "text": stdout[-500:]}
    return value if isinstance(value, dict) else {"status": "unparsed"}


def _observed(stdout: str) -> dict[str, object]:
    payload = _load(stdout)
    evidence = payload.get("evidence")
    if isinstance(evidence, str):
        path = Path(evidence) / "result.json"
        if path.is_file():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                return stored
    return payload


def _save(project: Path, out: str, err: str, resumed: tuple[int, str, str, int]) -> None:
    (project / "crash.stdout").write_text(out, encoding="utf-8")
    (project / "crash.stderr").write_text(err, encoding="utf-8")
    (project / "resume.stdout").write_text(resumed[1], encoding="utf-8")
    (project / "resume.stderr").write_text(resumed[2], encoding="utf-8")
