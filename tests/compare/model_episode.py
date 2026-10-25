"""Capture one tool-free public Session, including failure-side spend and partial events."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from support import save

from compare.evidence_failures import record_failure
from lite_pi.journal import Journal
from lite_pi.session import AgentSession
from lite_pi.spend.ledger import Ledger
from lite_pi.tools.contracts import JsonObject


def run_episode(directory: Path, prompt: str) -> JsonObject:
    """Caller owns the directory; this function owns its project, session and receipts."""
    work = directory / "project"
    work.mkdir()
    journal = Journal.create(directory / "session.jsonl", work, tools=())
    ledger = Ledger()
    before = ledger.status()
    offset = len(ledger.calls_path.read_text().splitlines()) if ledger.calls_path.exists() else 0
    events: list[JsonObject] = []
    save(
        directory / "request-input.json",
        {"prompt": prompt, "tools": [], "max_calls": 1, "max_output": 512, "context_tokens": 4096},
    )
    started = time.monotonic()
    try:
        result = AgentSession(work, tools=(), journal=journal).submit(
            prompt,
            on_event=events.append,
            max_turns=1,
            max_model_calls=1,
            max_task_calls=1,
            max_tokens=512,
            max_task_output_tokens=512,
            context_tokens=4096,
        )
        save(directory / "result.json", result)
        return result
    except BaseException as error:
        record_failure(directory / "failure.json", error)
        raise
    finally:
        original = sys.exception()
        try:
            save(directory / "events.json", events)
            rows = (
                [json.loads(line) for line in ledger.calls_path.read_text().splitlines()[offset:]]
                if ledger.calls_path.exists()
                else []
            )
            save(directory / "attempts.json", rows)
            save(
                directory / "spend.json",
                {"before": before, "after": ledger.status(), "elapsed": time.monotonic() - started},
            )
            save(directory / "budget.json", AgentSession.open(journal.path).inspect()["budget"])
            ids = {e["run_id"] for e in events if e.get("run_id")}
            if any(row["run_id"] not in ids for row in rows):
                raise RuntimeError("unattributed_episode_spend")
        except Exception as capture_error:
            record_failure(directory / "capture-failure.json", capture_error, primary=original)
            if original is None:
                raise
