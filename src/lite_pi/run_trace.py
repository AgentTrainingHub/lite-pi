"""One owner for event delivery and final run evidence, after resource cleanup."""

import json
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import NotRequired, TypedDict

from lite_pi.control import ControlInput
from lite_pi.outcomes import RunStatus, ToolControl
from lite_pi.storage import atomic_write
from lite_pi.tools.contracts import JsonObject


class RunOutcome(TypedDict):
    """Loop state before session cleanup and final event publication."""

    status: RunStatus
    text: str
    reason: str | None
    request_count: NotRequired[int]
    total_request_count: NotRequired[int]
    raw_stop_reason: NotRequired[str | None]
    model: NotRequired[str | None]
    usage: NotRequired[JsonObject | None]
    usage_unknown: NotRequired[bool]
    transcript: NotRequired[list[JsonObject]]
    controls: NotRequired[list[ControlInput]]
    initial_request: NotRequired[list[JsonObject]]
    source_ids: NotRequired[list[str]]
    messages: NotRequired[list[JsonObject]]
    tools: NotRequired[list[str]]
    context: NotRequired[JsonObject | None]
    compactions: NotRequired[list[JsonObject]]
    extensions: NotRequired[list[JsonObject]]
    hook_log: NotRequired[list[JsonObject]]
    checked_files: NotRequired[bool]
    tasks: NotRequired[JsonObject]
    cleanup_errors: NotRequired[list[str]]


class RunTrace:
    def __init__(
        self, run_id: str, evidence: Path, on_event: Callable[[JsonObject], None] | None = None
    ) -> None:
        self.run_id = run_id
        self.evidence = evidence
        self._on_event = on_event
        self._events: list[JsonObject] = []

    def _append(self, event: JsonObject) -> JsonObject:
        snapshot = {
            **deepcopy(event),
            "run_id": self.run_id,
            "seq": len(self._events) + 1,
            "at": datetime.now(UTC).isoformat(),
        }
        self._events.append(snapshot)
        return snapshot

    def emit(self, event: JsonObject) -> None:
        snapshot = self._append(event)
        if self._on_event:
            self._on_event(deepcopy(snapshot))

    def unresolved_effect(self) -> bool:
        """Retain observed effects even if an event consumer interrupts the loop."""
        pending: set[str] = set()
        unqueryable = False
        for event in self._events:
            if event.get("type") != "tool_end":
                continue
            outcome = ToolControl.read(event["result"])
            resolved = outcome.resolves_effect
            if isinstance(resolved, str):
                pending.discard(resolved)
            if outcome.effect == "unknown":
                key = outcome.effect_key
                if isinstance(key, str):
                    pending.add(key)
                else:
                    unqueryable = True
        return unqueryable or bool(pending)

    def finish(self, outcome: RunOutcome, mcp: JsonObject) -> JsonObject:
        receipt = {
            **outcome,
            "run_id": self.run_id,
            "evidence": str(self.evidence),
            "mcp": mcp,
        }
        end = self._append(
            {"type": "run_end", "status": outcome["status"], "text": outcome["text"], "result": receipt}
        )
        result = {**receipt, "events": deepcopy(self._events)}
        # result.json is the completion marker; no provisional success is written.
        atomic_write(
            self.evidence / "events.jsonl",
            "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in self._events).encode(),
        )
        atomic_write(
            self.evidence / "result.json", (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()
        )
        if self._on_event:
            self._on_event(deepcopy(end))
        return result
