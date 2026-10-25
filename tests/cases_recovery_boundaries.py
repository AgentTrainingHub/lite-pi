"""Real DeepSeek regression acceptance for task recovery and shared budgets."""

import os
import subprocess
import sys
import time

from cases_recovery import READ_TASK, WRITE_TASK, _approval_row, _drive, _intent, _observed, _outcome, _tools
from support import ROOT, Suite, save

from lite_pi.journal import Journal
from lite_pi.session import AgentSession


def run(suite: Suite) -> None:
    _approved_crash(suite)
    _shell_unknown(suite)
    _scope(suite)
    _summary_budget(suite)
    _late_worker(suite)


def _approved_crash(suite: Suite) -> None:
    work = suite.project("approval-crash")
    journal = work.parent / "session.jsonl"
    initial = _drive(
        [
            "code",
            "--root",
            str(work),
            "--session",
            str(journal),
            "--approvals",
            "--max-task-calls",
            "1",
            WRITE_TASK.format(marker="APPROVED-ONCE"),
        ]
    )
    approval = _approval_row(journal)
    proposed = _observed(initial[1])
    hit = proposed.get("status") == "awaiting_approval" and approval is not None
    outputs = [initial]
    passed = False
    if hit:
        outputs.append(
            _drive(
                [
                    "approve",
                    "--session",
                    str(journal),
                    "--id",
                    approval["approval_id"],
                    "--decision",
                    "approve",
                ]
            )
        )
        crash = _drive(["resume", "--session", str(journal)], crash="effect")
        outputs.append(crash)
        target = work / "mark.txt"
        hit = crash[0] == 86 and target.is_file() and _intent(journal)["status"] == "intent"
        stamp = target.stat().st_mtime_ns if target.is_file() else None
        resumed = _drive(["resume", "--session", str(journal)])
        outputs.append(resumed)
        result = _observed(resumed[1])
        tools = _tools(journal)
        passed = (
            hit
            and target.read_text() == "APPROVED-ONCE"
            and target.stat().st_mtime_ns == stamp
            and len(tools) == 1
            and _outcome(tools[0]).get("effect") == "reconciled"
            and result.get("request_count") == 0
        )
    save(work.parent / "processes.json", outputs)
    suite.record("recovery-approved-effect-crash", bool(passed), hit=bool(hit), safe=bool(passed))


def _shell_unknown(suite: Suite) -> None:
    work = suite.project("shell-unknown")
    journal = work.parent / "session.jsonl"
    task = "Use bash once to run exactly: printf 'SHELL-ONCE' >> marker.txt . Do not use write or edit. Then stop."
    crash = _drive(
        ["code", "--root", str(work), "--session", str(journal), "--max-task-calls", "1", task],
        crash="effect",
    )
    intent = _intent(journal)
    target = work / "marker.txt"
    hit = (
        crash[0] == 86
        and intent is not None
        and intent["name"] == "bash"
        and target.is_file()
        and target.read_text() == "SHELL-ONCE"
    )
    outputs = [crash]
    passed = False
    if hit:
        stamp = target.stat().st_mtime_ns
        outputs += [_drive(["resume", "--session", str(journal)]) for _ in range(2)]
        observed = [_observed(item[1]) for item in outputs[1:]]
        passed = (
            all(
                item.get("status") == "effect_unknown" and item.get("request_count") == 0 for item in observed
            )
            and target.read_text() == "SHELL-ONCE"
            and target.stat().st_mtime_ns == stamp
        )
    save(work.parent / "processes.json", outputs)
    suite.record("recovery-shell-unknown-sticky", bool(passed), hit=bool(hit), safe=bool(passed))


def _scope(suite: Suite) -> None:
    work = suite.project("scope-reopen")
    journal = Journal.create(work.parent / "session.jsonl", work, tools=("write",))
    session = AgentSession.open(journal.path)
    session.steer("Only allowed/ may be changed.", paths=("allowed/",))
    first = session.submit("Acknowledge the constraint in one word.", max_turns=1, max_tokens=128)
    reopened = AgentSession.open(journal.path)
    result = reopened.submit(
        "The task has changed. Create the project file outside.txt with content RELEASE-NOTE using the write tool. The file must be at the project root.",
        max_turns=1,
        max_tokens=256,
    )
    tools = [r for turn in result["transcript"] for r in turn["results"]]
    hit = bool(tools and tools[0].get("name") == "write")
    save(work.parent / "runs.json", [first, result])
    suite.record(
        "recovery-persisted-scope-real",
        hit and all(not t["executed"] for t in tools) and not (work / "outside.txt").exists(),
        hit=hit,
        safe=not (work / "outside.txt").exists(),
    )


def _summary_budget(suite: Suite) -> None:
    work = suite.project("recovery-summary-budget")
    journal = Journal.create(work.parent / "session.jsonl", work)
    session = AgentSession.open(journal.path)
    padding = "This line is repeated padding with no task information.\n" * 250
    first = session.submit(
        "Remember the fact: project code is REAL-recovery. Reply OK.\n" + padding,
        max_turns=1,
        max_tokens=640,
        max_task_calls=2,
        max_task_output_tokens=650,
    )
    second = AgentSession.open(journal.path).submit(
        "Recall the project code.", max_tokens=640, context_tokens=1700
    )
    third = AgentSession.open(journal.path).resume(max_task_calls=3, max_tokens=640)
    records = Journal(journal.path).records
    summaries = [r for r in records if r["kind"] == "compaction"]
    budget = third["tasks"]["budget"]
    save(work.parent / "runs.json", [first, second, third])
    hit = bool(summaries) and second.get("reason") == "task_calls"
    # After the summary, the remaining token allowance must reject another full reserve.
    consumed = budget["spent_output"] + budget["unknown_output"]
    passed = (
        hit
        and second["request_count"] == 1
        and second["tasks"]["budget"]["spent_calls"] == 2
        and third["request_count"] == 0
        and third.get("reason") == "task_output"
        and consumed <= 650
        and len(records) > len(summaries)
    )
    suite.record(
        "recovery-shared-summary-token-budget",
        bool(passed),
        hit=bool(hit),
        safe=True,
        consumed_output=consumed,
        budget=budget,
    )


def _late_worker(suite: Suite) -> None:
    work = suite.project("late-worker")
    (work / "token.txt").write_text("LATE-WORKER-TOKEN")
    journal = work.parent / "session.jsonl"
    crash = _drive(
        ["code", "--root", str(work), "--session", str(journal), "--max-task-calls", "1", READ_TASK],
        crash="intent",
    )
    intent = _intent(journal)
    hit = crash[0] == 86 and intent is not None and intent["name"] == "read"
    if not hit:
        save(work.parent / "crash.json", crash)
        suite.record("recovery-live-late-worker", False, hit=False)
        return
    ready, release = work.parent / "ready", work.parent / "release"
    # An outstanding read worker returns a genuine tool result after a new session owner has claimed recovery.
    script = f"""
import json, time
from pathlib import Path
from lite_pi.session import AgentSession
from lite_pi.tasks import StaleOwner
from lite_pi.tools.registry import ToolRegistry
session = AgentSession.open(Path({str(journal)!r}))
book = session._book(approvals=False, approval_ttl=None, claim=True)
registry = ToolRegistry(session.workspace, session.tools, max_bytes=20000, shell_timeout=5, cancel_bash_after=None, evidence=Path({str(work.parent / "worker-evidence")!r}))
result = registry.execute_stored('read', {{'path':'token.txt'}}, {intent["call_id"]!r}, require_safe=True)
Path({str(ready)!r}).write_text(str(book.owner_version))
deadline = time.monotonic()+30
while not Path({str(release)!r}).exists() and time.monotonic()<deadline: time.sleep(.02)
try:
    book.commit({intent["intent_id"]!r}, result)
    print(json.dumps({{'status':'accepted'}}))
except StaleOwner:
    print(json.dumps({{'status':'stale_owner', 'read_token': 'LATE-WORKER-TOKEN' in result['text']}}))
finally: book.close()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"]},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        alive_before = ready.exists() and child.poll() is None
        resumed = (
            _drive(["resume", "--session", str(journal)]) if alive_before else (1, "", "no live worker", 0)
        )
        release.write_text("return")
        out, err = child.communicate(timeout=10)
        old = _observed(out)
        result = _observed(resumed[1])
        passed = (
            alive_before
            and old.get("status") == "stale_owner"
            and old.get("read_token") is True
            and result.get("status") == "budget"
            and len(_tools(journal)) == 1
        )
        save(
            work.parent / "processes.json",
            {
                "crash": crash,
                "resume": resumed,
                "worker_stdout": out,
                "worker_stderr": err,
                "worker_returncode": child.returncode,
            },
        )
        suite.record("recovery-live-late-worker", bool(passed), hit=alive_before, safe=True, old_worker=old)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()
