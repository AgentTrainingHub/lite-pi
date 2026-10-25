"""Real DeepSeek coverage for compaction, project instructions, and skills."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from support import Suite, require_available, snapshot

from lite_pi.compaction import estimate_request
from lite_pi.journal import Journal
from lite_pi.session import AgentSession
from lite_pi.spend.gate import EVIDENCE_ROOT


def run(suite: Suite) -> None:
    for index in range(3):
        _compact(suite, index)
    for index in range(3):
        _instructions(suite, index)
    _override(suite)
    for index in range(3):
        _skill(suite, index)


def _compact(suite: Suite, index: int) -> None:
    work = suite.project(f"compact-{index}")
    drop = f"diagnostic-padding-{index}-7f3a"
    contract = (
        "Pending export contract: create notes/decision.json with exactly three JSON fields: "
        'unit="grams", pending="calibrate", limit=3. Do not create it yet. '
        "Wait until asked to complete the export.\n"
    )
    first = contract + (drop + " ") * 300 + "\nReply exactly: noted"
    journal = Journal.create(work.parent / "session.jsonl", work, tools=("write",))
    session = AgentSession(work, tools=("write",), journal=journal)
    opened = session.submit(first, max_turns=1, max_tokens=32, context_tokens=100000)
    require_available(opened)
    before = snapshot(work)
    projected = journal.messages()
    early = any(record["kind"] == "compaction" for record in journal.records)
    task = "Complete the pending export now."
    steer = "Latest decision: change the export limit to 7. Only create notes/decision.json."
    history = projected + [{"role": "user", "content": task}, {"role": "user", "content": steer}]
    definitions = _calls(opened["run_id"])[0]["request"].get("tools", [])
    reserve = 768
    context_tokens = estimate_request(history[-2:], definitions, reserve) + reserve + 1600
    triggered = context_tokens < estimate_request(history, definitions, reserve) + reserve
    session.steer(steer)
    result = session.submit(
        task, max_turns=4, max_tokens=reserve, context_tokens=context_tokens, max_model_calls=6
    )
    require_available(result)
    calls = _calls(result["run_id"])
    continuation = [
        item for item in calls if str(item["receipt"].get("purpose", "")).startswith("session-turn")
    ]
    summary_calls = [item for item in calls if item["receipt"].get("purpose") == "compaction"]
    request_text = json.dumps(continuation[0]["request"]) if continuation else ""
    adopted = [item for item in result["compactions"] if item.get("adopted")]
    latest = adopted[0] if adopted else {}
    summary = latest.get("summary", "")
    target = work / "notes/decision.json"
    try:
        contents = json.loads(target.read_text()) if target.is_file() else None
    except json.JSONDecodeError:
        contents = None
    expected = {"unit": "grams", "pending": "calibrate", "limit": 7}
    changed = _changed(before, work)
    before_resume = snapshot(work)
    reopened = AgentSession.open(journal.path)
    resumed = reopened.submit(
        "Confirm the completed export in one short sentence without modifying files.",
        max_turns=1,
        max_tokens=512,
        max_model_calls=1,
    )
    require_available(resumed)
    raw_pad = (drop + " ") * 20
    facts_kept = all(value in summary for value in ("grams", "calibrate", "notes/decision.json"))
    safe = (
        not before
        and set(changed) <= {"notes/decision.json"}
        and before_resume == snapshot(work)
        and raw_pad in journal.path.read_text()
        and (not adopted or facts_kept)
    )
    passed = (
        opened["status"] == "stop"
        and result["status"] == "stop"
        and resumed["status"] == "stop"
        and triggered
        and not early
        and bool(adopted)
        and bool(summary_calls)
        and facts_kept
        and contents == expected
        and steer in request_text
        and raw_pad not in request_text
        and raw_pad not in summary
        and raw_pad not in _request_text(resumed["run_id"])
        and isinstance(latest.get("deviation"), int)
        and latest.get("estimate_method") == "spend_input_estimate"
    )
    suite.record(
        f"compact-{index}",
        passed,
        effect="compaction",
        safe=safe,
        run_id=result["run_id"],
        resumed_run_id=resumed["run_id"],
        status_seen=result["status"],
        reason=result.get("reason"),
        context_tokens=context_tokens,
        summary_calls=len(summary_calls),
        deviation=latest.get("deviation"),
        contents=contents,
        changed=changed,
    )


def _instructions(suite: Suite, index: int) -> None:
    work = suite.project(f"instructions-{index}")
    rule = "PROJECT-RULE-9c2e: create only notes/convention.txt whose entire contents are convention."
    (work / "AGENTS.md").write_text(rule + "\n")
    (work.parent / "AGENTS.md").write_text("PARENT-RULE-9c2e do not apply this file.\n")
    (work / "CLAUDE.md").write_text("CLAUDE-RULE-9c2e do not apply this file.\n")
    digest = hashlib.sha256((work / "AGENTS.md").read_bytes()).hexdigest()
    before = snapshot(work)
    journal = Journal.create(work.parent / "session.jsonl", work, tools=("read", "write"))
    session = AgentSession(work, tools=("read", "write"), journal=journal)
    result = session.submit(
        "Follow the project instructions and finish the one file they name.",
        max_turns=4,
        max_tokens=256,
    )
    require_available(result)
    changed = _changed(before, work)
    text = _request_text(result["run_id"])
    target = work / "notes/convention.txt"
    contents = target.read_text() if target.is_file() else ""
    users = [record["body"].get("content", "") for record in journal.records if record["kind"] == "user"]
    leaked = "PARENT-RULE-9c2e" in text or "CLAUDE-RULE-9c2e" in text
    passed = (
        result["status"] == "stop"
        and rule in text
        and digest in text
        and result["context"]["instructions"]["adopted"]["path"] == "AGENTS.md"
        and contents.strip() == "convention"
        and changed == ["notes/convention.txt"]
        and all("PROJECT-RULE-9c2e" not in item for item in users)
        and "bash" not in result["tools"]
    )
    suite.record(
        f"instructions-{index}",
        passed,
        effect="instructions",
        safe=not leaked and "PARENT-RULE-9c2e" not in text,
        run_id=result["run_id"],
        changed=changed,
        contents=contents,
    )


def _override(suite: Suite) -> None:
    work = suite.project("override")
    (work / "AGENTS.override.md").write_text("OVERRIDE-RULE-9c2e adopt this instruction.\n")
    (work / "AGENTS.md").write_text("SHADOW-RULE-9c2e do not adopt this sentence.\n")
    result = AgentSession(work).submit("Reply with the single word ok.", max_turns=1, max_tokens=16)
    require_available(result)
    text = _request_text(result["run_id"])
    adopted = result["context"]["instructions"]["adopted"]["path"]
    shadowed = [item["path"] for item in result["context"]["instructions"]["shadowed"]]
    calls = _calls(result["run_id"])
    passed = (
        result["status"] == "stop"
        and adopted == "AGENTS.override.md"
        and shadowed == ["AGENTS.md"]
        and "OVERRIDE-RULE-9c2e" in text
        and "SHADOW-RULE-9c2e" not in text
        and calls
        and calls[0]["receipt"].get("http_status") == 200
    )
    suite.record("override", passed, run_id=result["run_id"], adopted=adopted, shadowed=shadowed)


def _skill(suite: Suite, index: int) -> None:
    work = suite.project(f"skill-{index}")
    token = f"SKILL-BODY-{index}-9c2e"
    directory = work / ".lite-pi/skills/marker"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\n"
        "name: marker\n"
        "description: Load the marker procedure. The marker text is only in a resource.\n"
        "---\n"
        "Call the skill tool again with resource token.txt. "
        "Then create only notes/marker.txt containing exactly the resource text.\n"
    )
    (directory / "token.txt").write_text(token + "\n")
    digest = hashlib.sha256((directory / "SKILL.md").read_bytes()).hexdigest()
    before = snapshot(work)
    result = AgentSession(work, tools=("read", "write")).submit(
        "Use the skill named marker. Read its body and its token.txt resource before writing the file.",
        max_turns=6,
        max_tokens=320,
    )
    require_available(result)
    changed = _changed(before, work)
    calls = _calls(result["run_id"])
    turns = [item for item in calls if str(item["receipt"].get("purpose", "")).startswith("session-turn")]
    first = json.dumps(turns[0]["request"]) if turns else ""
    later = json.dumps([item["request"] for item in turns[1:]])
    target = work / "notes/marker.txt"
    contents = target.read_text() if target.is_file() else ""
    skill_results = [
        item
        for turn in result["transcript"]
        for item in turn["results"]
        if item.get("name") == "skill" and item.get("executed") and not item.get("is_error")
    ]
    passed = (
        result["status"] == "stop"
        and "Load the marker procedure" in first
        and "Then create only notes/marker.txt" not in first
        and token not in first
        and digest in first
        and token in later
        and any(item.get("resource") is None and item.get("sha256") == digest for item in skill_results)
        and any(item.get("resource") == "token.txt" and item.get("sha256") for item in skill_results)
        and contents.strip() == token
        and changed == ["notes/marker.txt"]
        and "bash" not in result["tools"]
        and "skill" in result["tools"]
    )
    suite.record(
        f"skill-{index}",
        passed,
        effect="skill",
        safe="bash" not in result["tools"],
        run_id=result["run_id"],
        changed=changed,
        contents=contents,
        skill_calls=len(skill_results),
    )


def _changed(before: dict, work: Path) -> list[str]:
    after = snapshot(work)
    return sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))


def _calls(run_id: str) -> list[dict]:
    found = []
    root = EVIDENCE_ROOT
    for path in root.glob("*/receipt.json"):
        receipt = json.loads(path.read_text())
        if receipt.get("run_id") != run_id:
            continue
        found.append(
            {
                "receipt": receipt,
                "request": json.loads((path.parent / "request.json").read_text()),
                "mtime": path.stat().st_mtime,
            }
        )
    return sorted(found, key=lambda item: item["mtime"])


def _request_text(run_id: str) -> str:
    return json.dumps([item["request"] for item in _calls(run_id)])
