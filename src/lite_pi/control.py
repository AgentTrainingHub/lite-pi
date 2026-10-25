"""Steering, follow-up, and cancellation for one public session."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from threading import Event, Lock
from typing import Literal, TypedDict
from uuid import uuid4

from lite_pi.tools.contracts import JsonObject


class ControlInput(TypedDict):
    id: str
    kind: str
    text: str
    paths: tuple[str, ...] | None
    submitted_at: str
    accepted_at: str | None
    entered_at: str | None
    disposition: Literal["queued", "rejected_after_cancel", "entered", "not_sent", "dropped_on_cancel"]


def _now() -> str:
    return datetime.now(UTC).isoformat()


class RunControl:
    """Queues belong to the session. A run drains them; cancellation does not rewind files."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.cancel = Event()
        self._shared_cancel = False
        self._steering: list[ControlInput] = []
        self._follow_ups: list[ControlInput] = []
        self.inputs: dict[str, ControlInput] = {}
        self.scope: tuple[str, ...] | None = None
        self.cancel_count = 0
        self.cancelled_at: str | None = None

    def adopt(self, cancel: Event) -> None:
        with self._lock:
            self.cancel = cancel
            self._shared_cancel = True

    def begin_run(self) -> None:
        with self._lock:
            if not self._shared_cancel:
                self.cancel.clear()
            self.cancel_count = 0
            self.cancelled_at = None
            self.scope = None

    def submit(self, kind: str, text: str, paths: tuple[str, ...] | None = None) -> ControlInput:
        if kind not in {"steer", "follow_up"}:
            raise ValueError("control kind must be steer or follow_up")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("control text must not be empty")
        if paths is not None and (not paths or not all(isinstance(path, str) and path for path in paths)):
            raise ValueError("path scope must be a non-empty tuple of relative paths")
        now = _now()
        item: ControlInput = {
            "id": uuid4().hex,
            "kind": kind,
            "text": text,
            "paths": paths,
            "submitted_at": now,
            "accepted_at": None,
            "entered_at": None,
            "disposition": "queued",
        }
        with self._lock:
            if self.cancel.is_set():
                item["disposition"] = "rejected_after_cancel"
                self.inputs[item["id"]] = item
                return deepcopy(item)
            item["accepted_at"] = now
            self.inputs[item["id"]] = item
            (self._steering if kind == "steer" else self._follow_ups).append(item)
            return deepcopy(item)

    def restore_scope(self, scope: tuple[str, ...] | None) -> None:
        with self._lock:
            self.scope = scope

    def queued_steer(self) -> ControlInput | None:
        with self._lock:
            if not self._steering:
                return None
            return deepcopy(self._steering[0])

    def return_pending(self) -> list[ControlInput]:
        """Return inputs that missed the final drain, without sending them next run."""
        with self._lock:
            pending = [*self._steering, *self._follow_ups]
            self._steering.clear()
            self._follow_ups.clear()
            for item in pending:
                item["disposition"] = "not_sent"
            return deepcopy(pending)

    def drain(self, kind: str) -> ControlInput | None:
        with self._lock:
            if self.cancel.is_set():
                return None
            queue = self._steering if kind == "steer" else self._follow_ups
            if not queue:
                return None
            return deepcopy(queue.pop(0))

    def mark_entered(self, item_id: str) -> ControlInput:
        with self._lock:
            item = self.inputs[item_id]
            item["entered_at"] = _now()
            item["disposition"] = "entered"
            if item["kind"] == "follow_up":
                self.scope = None
            elif item["paths"] is not None:
                self.scope = tuple(item["paths"])
            return deepcopy(item)

    def cancel_run(self) -> JsonObject:
        with self._lock:
            self.cancel_count += 1
            first = self.cancel_count == 1
            if first:
                self.cancelled_at = _now()
                self.cancel.set()
                for item in [*self._steering, *self._follow_ups]:
                    item["disposition"] = "dropped_on_cancel"
                self._steering.clear()
                self._follow_ups.clear()
            return {"first": first, "at": self.cancelled_at, "count": self.cancel_count}

    def snapshot(self) -> list[ControlInput]:
        with self._lock:
            return [deepcopy(item) for item in self.inputs.values()]
