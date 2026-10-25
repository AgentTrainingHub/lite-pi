"""Persisted per-attempt task budget; shares the task database transaction boundary."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Protocol, cast
from uuid import uuid4

from lite_pi.task_storage import assert_owner, optional_row, transaction
from lite_pi.tools.contracts import JsonObject


@dataclass(frozen=True)
class BudgetLimits:
    max_calls: int | None
    max_output: int | None
    max_seconds: float | None


class BudgetExceeded(RuntimeError):
    def __init__(self, reason: str, attempts: int = 0):
        super().__init__(reason)
        self.reason, self.attempts = reason, attempts


class RequestBudget(Protocol):
    def reserve(self, purpose: str, output_tokens: int) -> str | None: ...
    def settle(self, reservation_id: str, usage: JsonObject | None, attempts: int) -> None: ...
    def release(self, reservation_id: str) -> None: ...
    def stop_reason(self) -> str | None: ...
    def snapshot(self) -> JsonObject: ...


class RunRequestBudget:
    """One run's allowance shared by loop, summaries, reranking and children."""

    def __init__(self, parent: RequestBudget, max_calls: int | None) -> None:
        self.parent, self.max_calls = parent, max_calls
        self._pending: dict[str, str] = {}
        self._settled = 0
        self._lock = threading.RLock()

    @property
    def consumed_calls(self) -> int:
        with self._lock:
            return self._settled + len(self._pending)

    def stop_reason(self) -> str | None:
        with self._lock:
            parent_reason = self.parent.stop_reason()
            if parent_reason:
                return parent_reason
            if self.max_calls is not None and self.consumed_calls >= self.max_calls:
                return "run_calls"
            return None

    def reserve(self, purpose: str, output_tokens: int) -> str | None:
        with self._lock:
            reason = self.stop_reason()
            if reason:
                return None
            parent_id = self.parent.reserve(purpose, output_tokens)
            if parent_id is None:
                return None
            ticket = uuid4().hex
            self._pending[ticket] = parent_id
            return ticket

    def settle(self, reservation_id: str, usage: JsonObject | None, attempts: int) -> None:
        with self._lock:
            if reservation_id not in self._pending:
                return
            parent_id = self._pending[reservation_id]
            self.parent.settle(parent_id, usage, attempts)
            del self._pending[reservation_id]
            self._settled += 1

    def release(self, reservation_id: str) -> None:
        with self._lock:
            if reservation_id not in self._pending:
                return
            parent_id = self._pending[reservation_id]
            self.parent.release(parent_id)
            del self._pending[reservation_id]

    def snapshot(self) -> JsonObject:
        return {**self.parent.snapshot(), "run_calls": self.consumed_calls}


class TaskBudget:
    def __init__(self, db: sqlite3.Connection, session_id: str):
        self._db, self.session_id = db, session_id
        self.owner_version = 0
        self._lock = threading.RLock()

    def _tx(self) -> AbstractContextManager[None]:
        return _locked(self._lock, self._db)

    def configure(self, max_calls: int | None, max_output: int | None, max_seconds: float | None) -> None:
        with self._tx():
            self._db.execute(
                """INSERT OR IGNORE INTO budget(session_id,spent_calls,reserved_calls,
                unknown_calls,spent_output,reserved_output,unknown_output,max_calls,max_output,max_seconds,started_at)
                VALUES(?,0,0,0,0,0,0,?,?,?,?)""",
                (self.session_id, max_calls, max_output, max_seconds, time.time()),
            )
            self._db.execute(
                """UPDATE budget SET max_calls=COALESCE(?,max_calls),
                max_output=COALESCE(?,max_output),max_seconds=COALESCE(?,max_seconds) WHERE session_id=?""",
                (max_calls, max_output, max_seconds, self.session_id),
            )

    def limits(self) -> BudgetLimits:
        row = self._budget()
        if row is None:
            raise ValueError("task budget is missing")
        return BudgetLimits(
            _optional_int(row["max_calls"]),
            _optional_int(row["max_output"]),
            _optional_float(row["max_seconds"]),
        )

    def snapshot(self) -> JsonObject:
        row = self._budget()
        if row is None:
            raise ValueError("task budget is missing")
        return _public_budget(row)

    def stop_reason(self) -> str | None:
        row = self._budget()
        if row is None:
            return None
        if row["max_seconds"] is not None and time.time() - float(row["started_at"]) >= float(
            row["max_seconds"]
        ):
            return "time_budget"
        if row["max_calls"] is not None and _consumed_calls(row) >= int(row["max_calls"]):
            return "task_calls"
        if row["max_output"] is not None and _consumed_output(row) >= int(row["max_output"]):
            return "task_output"
        return None

    def reserve(self, purpose: str, output_tokens: int) -> str | None:
        if output_tokens < 1:
            raise ValueError("output reserve must be positive")
        reservation_id = uuid4().hex
        with self._tx():
            assert_owner(self._db, self.session_id, self.owner_version)
            row = self._budget()
            if row is None:
                raise ValueError("task budget is missing")
            if self._over_cap(row, output_tokens):
                return None
            self._db.execute(
                """
                INSERT INTO reservations(
                    reservation_id, session_id, owner_version, purpose, output_tokens, state
                ) VALUES(?,?,?,?,?,'reserved')
                """,
                (reservation_id, self.session_id, self.owner_version, purpose, output_tokens),
            )
            self._db.execute(
                """
                UPDATE budget SET reserved_calls=reserved_calls+1, reserved_output=reserved_output+?
                WHERE session_id=?
                """,
                (output_tokens, self.session_id),
            )
        return reservation_id

    def settle(self, reservation_id: str, usage: JsonObject | None, attempts: int) -> None:
        if attempts != 1:
            raise ValueError("each reservation represents exactly one HTTP attempt")
        with self._tx():
            assert_owner(self._db, self.session_id, self.owner_version)
            row = self._reservation(reservation_id)
            if row is None or row["state"] != "reserved" or int(row["owner_version"]) != self.owner_version:
                return
            output = _completion_tokens(usage)
            unknown = output is None
            self._consume_reservation(row, attempts=attempts, output=output, unknown=unknown, usage=usage)

    def release(self, reservation_id: str) -> None:
        with self._tx():
            assert_owner(self._db, self.session_id, self.owner_version)
            row = self._reservation(reservation_id)
            if row is None or row["state"] != "reserved" or int(row["owner_version"]) != self.owner_version:
                return
            self._consume_reservation(row, attempts=0, output=0, unknown=False, usage=None, release=True)

    def reap(self, version: int) -> None:
        rows = list(
            self._db.execute(
                """
                SELECT * FROM reservations
                WHERE session_id=? AND state='reserved' AND owner_version!=?
                """,
                (self.session_id, version),
            )
        )
        for row in rows:
            self._consume_reservation(row, attempts=1, output=None, unknown=True, usage=None, reap=True)

    def _consume_reservation(
        self,
        row: sqlite3.Row,
        *,
        attempts: int,
        output: int | None,
        unknown: bool,
        usage: JsonObject | None,
        release: bool = False,
        reap: bool = False,
    ) -> None:
        reserved_output = int(row["output_tokens"])
        if release:
            state, call_spent, call_unknown, output_spent, output_unknown = "released", 0, 0, 0, 0
        elif unknown or reap:
            state = "unknown"
            call_spent, call_unknown = 0, max(1, attempts)
            output_spent, output_unknown = 0, reserved_output
        else:
            state = "settled"
            call_spent, call_unknown = attempts, 0
            output_spent, output_unknown = output or 0, 0
        cursor = self._db.execute(
            "UPDATE reservations SET state=?, attempts=?, usage_json=? WHERE reservation_id=? AND state='reserved'",
            (
                state,
                attempts,
                None if usage is None else json.dumps(usage, ensure_ascii=False),
                row["reservation_id"],
            ),
        )
        if cursor.rowcount != 1:
            return
        self._db.execute(
            """
            UPDATE budget SET
                reserved_calls=reserved_calls-1,
                spent_calls=spent_calls+?,
                unknown_calls=unknown_calls+?,
                reserved_output=reserved_output-?,
                spent_output=spent_output+?,
                unknown_output=unknown_output+?
            WHERE session_id=?
            """,
            (
                call_spent,
                call_unknown,
                reserved_output,
                output_spent,
                output_unknown,
                self.session_id,
            ),
        )

    def _budget(self) -> sqlite3.Row | None:
        with self._lock:
            return optional_row(
                self._db.execute("SELECT * FROM budget WHERE session_id=?", (self.session_id,)).fetchone()
            )

    def _reservation(self, reservation_id: str) -> sqlite3.Row | None:
        return optional_row(
            self._db.execute(
                "SELECT * FROM reservations WHERE reservation_id=?", (reservation_id,)
            ).fetchone()
        )

    def _over_cap(self, row: sqlite3.Row, output_tokens: int) -> bool:
        if row["max_calls"] is not None and _consumed_calls(row) + 1 > int(row["max_calls"]):
            return True
        if row["max_seconds"] is not None and time.time() - float(row["started_at"]) >= float(
            row["max_seconds"]
        ):
            return True
        return bool(
            row["max_output"] is not None and _consumed_output(row) + output_tokens > int(row["max_output"])
        )


@contextmanager
def _locked(lock: threading.RLock, db: sqlite3.Connection) -> Iterator[None]:
    with lock, transaction(db):
        yield


def _completion_tokens(usage: JsonObject | None) -> int | None:
    if usage is None:
        return None
    value = usage.get("completion_tokens")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _consumed_calls(row: sqlite3.Row) -> int:
    return int(row["spent_calls"]) + int(row["reserved_calls"]) + int(row["unknown_calls"])


def _consumed_output(row: sqlite3.Row) -> int:
    return int(row["spent_output"]) + int(row["reserved_output"]) + int(row["unknown_output"])


def _optional_int(value: object) -> int | None:
    return None if value is None else int(cast(int, value))


def _optional_float(value: object) -> float | None:
    return None if value is None else float(cast(float, value))


def _public_budget(row: sqlite3.Row) -> JsonObject:
    return {
        "spent_calls": int(row["spent_calls"]),
        "reserved_calls": int(row["reserved_calls"]),
        "unknown_calls": int(row["unknown_calls"]),
        "spent_output": int(row["spent_output"]),
        "reserved_output": int(row["reserved_output"]),
        "unknown_output": int(row["unknown_output"]),
        "max_calls": _optional_int(row["max_calls"]),
        "max_output": _optional_int(row["max_output"]),
        "max_seconds": _optional_float(row["max_seconds"]),
        "started_at": float(row["started_at"]),
        "output_unknown": int(row["unknown_output"]) > 0,
    }
