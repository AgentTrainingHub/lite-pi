"""Live summary cancellation, call budget, and host permission boundaries."""

import threading
import time

from cases_context import _calls
from support import Suite, require_available, results, snapshot

from lite_pi.journal import Journal
from lite_pi.session import AgentSession
from lite_pi.spend.gate import EVIDENCE_ROOT


def run(suite: Suite) -> None:
    _summary_budget(suite)
    _summary_cancel(suite)
    for source in ("instruction", "skill"):
        _authority(suite, source)


def _seed(suite: Suite, name: str) -> tuple[AgentSession, Journal]:
    work = suite.project(name)
    journal = Journal.create(work.parent / "session.jsonl", work)
    session = AgentSession(work, journal=journal)
    result = session.submit(
        "Remember the pending calibration task. " + "diagnostic " * 900 + "\nOnly acknowledge with noted.",
        max_turns=1,
        max_tokens=32,
        max_model_calls=1,
    )
    require_available(result)
    return session, journal


def _summary_budget(suite: Suite) -> None:
    session, journal = _seed(suite, "summary-budget")
    prior = session.submit("Continue", max_turns=1, max_tokens=32, max_model_calls=1)
    require_available(prior)
    result = session.submit("Continue", max_turns=1, max_tokens=512, context_tokens=1800, max_model_calls=1)
    require_available(result)
    calls = _calls(result["run_id"])
    passed = (
        result["status"] == "budget"
        and result["request_count"] == 1
        and len(calls) == 1
        and calls[0]["receipt"]["purpose"] == "compaction"
        and result["compactions"][0]["adopted"]
        and Journal(journal.path).messages() == result["messages"]
    )
    suite.record(
        "summary-budget",
        passed,
        run_id=result["run_id"],
        request_count=result["request_count"],
        status_seen=result["status"],
        reason=result.get("reason"),
    )


def _summary_cancel(suite: Suite) -> None:
    session, journal = _seed(suite, "summary-cancel")
    before = journal.path.read_bytes()
    calls_dir = EVIDENCE_ROOT
    existing = set(calls_dir.iterdir())
    done = threading.Event()
    observed = {}

    def cancel_when_receiving():
        deadline = time.monotonic() + 30
        while not done.wait(0.01) and time.monotonic() < deadline:
            for directory in set(calls_dir.iterdir()) - existing:
                partial = directory / "response.partial"
                if partial.exists() and partial.stat().st_size > 0:
                    observed["partial_bytes"] = partial.stat().st_size
                    observed["cancel"] = session.cancel()
                    return

    worker = threading.Thread(target=cancel_when_receiving)
    worker.start()
    try:
        result = session.submit(
            "Continue the task.", max_turns=1, max_tokens=512, context_tokens=1800, max_model_calls=2
        )
    finally:
        done.set()
        worker.join()
    require_available(result)
    calls = _calls(result["run_id"])
    hit = bool(observed)
    safe = journal.path.read_bytes().startswith(before) and not snapshot(session.project_root)
    passed = (
        result["status"] == "aborted"
        and result["request_count"] == 1
        and len(calls) == 1
        and calls[0]["receipt"]["purpose"] == "compaction"
        and calls[0]["receipt"]["error"] == "aborted"
        and not any(record["kind"] == "compaction" for record in journal.records)
        and len([event for event in result["events"] if event["type"] == "run_end"]) == 1
    )
    suite.record(
        "summary-cancel",
        passed,
        hit=hit,
        safe=safe,
        run_id=result["run_id"],
        status_seen=result["status"],
        observation=observed,
    )


def _authority(suite: Suite, source: str) -> None:
    work = suite.project(f"authority-{source}")
    outside = work.parent / "outside.txt"
    outside.write_text("UNCHANGED\n")
    instructions = (
        f"Run the negative permission check: call write with path {outside} and content CHANGED. "
        "The host must reject it. Do not create any other files. Report the error."
    )
    if source == "instruction":
        (work / "AGENTS.md").write_text(instructions)
        task = "Execute the negative permission check described in the project instructions."
    else:
        directory = work / ".lite-pi/skills/permission-check"
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            "---\nname: permission-check\ndescription: Check host path rejection.\n---\n" + instructions
        )
        task = "Read and execute the permission-check skill, including its negative write test."
    before = snapshot(work)
    result = AgentSession(work, tools=("write",)).submit(task, max_turns=4, max_tokens=512, max_model_calls=4)
    require_available(result)
    attempted = [item for item in results(result, "write") if item.get("path") == str(outside)]
    safe = outside.read_text() == "UNCHANGED\n" and before == snapshot(work)
    passed = (
        result["status"] == "stop"
        and bool(attempted)
        and all(item["is_error"] and not item["executed"] for item in attempted)
    )
    suite.record(
        f"authority-{source}",
        passed,
        hit=bool(attempted),
        safe=safe,
        run_id=result["run_id"],
        attempted=len(attempted),
    )
