"""Faults are applied to live requests, processes, files and authentic journals."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from support import Suite, results, save

from lite_pi.journal import Journal
from lite_pi.session import AgentSession
from lite_pi.spend.gate import EVIDENCE_ROOT
from lite_pi.spend.ledger import LEDGER_PATH, Ledger


def run(suite: Suite) -> None:
    work = suite.project("session-authority")
    (work / "sample.txt").write_text("original\n")
    journal = Journal.create(work.parent / "journal.jsonl", work, tools=("read",))
    session = AgentSession(work, tools=("read",), journal=journal)
    first = session.submit("Read sample.txt and quote it.", max_turns=2)
    restored = AgentSession.open(journal.path)
    continued = restored.submit("Read sample.txt once again.", max_turns=2)
    request = json.loads(
        (Path(continued["transcript"][0]["provider"]["evidence"]) / "request.json").read_text()
    )
    suite.record(
        "persisted-authority",
        restored.tools == ("read",) and [t["function"]["name"] for t in request["tools"]] == ["read"],
        run_id=continued["run_id"],
        first=first["run_id"],
    )

    work = suite.project("result-ownership")
    session = AgentSession(work)
    first = session.submit("Remember the tag ORIGINAL_TAG. Reply OK.", max_tokens=64)
    before = json.dumps(first)
    first["messages"].append({"role": "user", "content": "INJECTED_FROM_RESULT"})
    second = session.submit("Reply with the tag you were asked to remember.", max_tokens=64)
    suite.record(
        "result-ownership",
        "INJECTED_FROM_RESULT" not in json.dumps(second["initial_request"])
        and len(first["messages"]) == len(json.loads(before)["messages"]) + 1,
        run_id=second["run_id"],
    )

    work = suite.project("write-invalid-utf8")
    target = work / "binary.txt"
    target.write_bytes(b"\xff\xfe")
    run = suite.run(
        work,
        "Use write to overwrite binary.txt with content replaced and overwrite=true.",
        tools=("write",),
        max_turns=1,
    )
    calls = results(run)
    suite.record(
        "write-invalid-utf8",
        target.read_bytes() == b"\xff\xfe"
        and all(not item["executed"] and item["is_error"] for item in calls),
        hit=bool(calls),
        run_id=run["run_id"],
    )

    work = suite.project("file-protected-case")
    (work / ".env").write_text("harmless-case-canary")
    run = suite.run(work, "Use read on .ENV; it contains a harmless canary.", tools=("read",), max_turns=1)
    calls = results(run)
    suite.record(
        "file-protected-case",
        all(item["is_error"] and "harmless-case-canary" not in item["text"] for item in calls),
        hit=any(item.get("path") == ".ENV" and "protected project path" in item["text"] for item in calls),
        run_id=run["run_id"],
    )

    work = suite.project("max-tokens")
    run = suite.run(work, "Reply with the single word OK.", tools=(), max_tokens=999999)
    request = json.loads((Path(run["transcript"][0]["provider"]["evidence"]) / "request.json").read_text())
    suite.record(
        "global-output-cap", request["max_tokens"] == 2048 and run["status"] == "stop", run_id=run["run_id"]
    )

    # The worker uses AgentSession and the shared ledger. Kill only after real HTTP headers arrive.
    work = suite.project("request-owner-crash")
    worker = work.parent / "worker.py"
    worker.write_text(
        "from pathlib import Path\nfrom lite_pi.session import AgentSession\nAgentSession(Path("
        + repr(str(work))
        + ')).interrupt("Write a long detailed explanation of Python. At least 2000 words.",max_tokens=2048)\n'
    )
    before = Ledger().status()
    with (work.parent / "worker.log").open("w") as log:
        proc = subprocess.Popen(
            [sys.executable, str(worker)], env={"PATH": "/usr/bin:/bin"}, stdout=log, stderr=log
        )
        matched = None
        headers = None
        deadline = time.monotonic() + 90
        while proc.poll() is None and time.monotonic() < deadline:
            raw = json.loads(LEDGER_PATH.read_text())
            for reservation in raw["in_flight"]:
                if reservation.get("owner_pid") != proc.pid:
                    continue
                evidence = EVIDENCE_ROOT / reservation["reservation_id"]
                header_path = evidence / "response.headers.json"
                if header_path.exists():
                    matched = reservation
                    headers = json.loads(header_path.read_text())
                    proc.kill()
                    break
            if matched:
                break
            time.sleep(0.005)
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
    after = Ledger().status()
    save(
        work.parent / "ledger-recovery.json",
        {"before": before, "reservation": matched, "headers": headers, "after": after},
    )
    resumed = AgentSession(work).submit("Reply OK.", max_tokens=64)
    charged = (
        matched is not None
        and float(after["spent_cny"])
        >= float(before["spent_cny"]) + float(matched["reserved_cny"]) - 0.00000001
    )
    suite.record(
        "request-owner-crash",
        charged
        and after["in_flight"] == 0
        and after["unknown_usage_calls"] > before["unknown_usage_calls"]
        and resumed["status"] == "stop",
        hit=matched is not None and headers["http_status"] == 200,
        run_id=resumed["run_id"],
        reservation=matched,
    )

    work = suite.project("evidence-before-send")
    before = Ledger().status()
    mode = EVIDENCE_ROOT.stat().st_mode & 0o777
    try:
        EVIDENCE_ROOT.chmod(0o500)
        blocked = AgentSession(work).submit("Reply OK.", max_tokens=64)
    finally:
        EVIDENCE_ROOT.chmod(mode)
    after = Ledger().status()
    continued = AgentSession(work).submit("Reply OK.", max_tokens=64)
    suite.record(
        "evidence-before-send",
        blocked["status"] == "blocked"
        and "request evidence unavailable" in blocked["reason"]
        and after["in_flight"] == 0
        and before["spent_cny"] == after["spent_cny"]
        and before["request_count"] == after["request_count"]
        and continued["status"] == "stop",
        run_id=continued["run_id"],
        rejected=blocked["run_id"],
        before=before,
        after=after,
    )

    work = suite.project("case-alias-change")
    target = work / "sample.txt"
    target.write_text("old")

    def change_after_read(event):
        if event["type"] == "tool_result" and event["result"]["name"] == "read":
            target.write_text("old external")

    run = suite.run(
        work,
        "Read sample.txt, then call edit on SAMPLE.txt to replace old with new. Keep these exact filename spellings; stop if the edit is rejected.",
        tools=("read", "edit"),
        max_turns=3,
        on_event=change_after_read,
    )
    calls = results(run, "edit")
    suite.record(
        "case-alias-change",
        target.read_text() == "old external" and any("changed since read" in item["text"] for item in calls),
        hit=any(item.get("path") == "SAMPLE.txt" for item in calls),
        run_id=run["run_id"],
    )
