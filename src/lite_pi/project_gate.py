"""Single-host project admission; read slots and an exclusive writer."""

from __future__ import annotations

import hashlib
import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from lite_pi.process_identity import process_alive, process_identity
from lite_pi.runtime import PATHS
from lite_pi.storage import locked
from lite_pi.task_storage import connect, transaction
from lite_pi.tools.contracts import JsonObject

READ_LIMIT = 2


def gate_db_path(project: Path) -> Path:
    root = PATHS.state / "task-gates"
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(project.resolve()).encode()).hexdigest()[:24]
    return root / f"{digest}.sqlite"


class GateAdmission(TypedDict):
    worker_id: str
    state: str
    reason: str | None
    mode: str


class ProjectGate:
    def __init__(self, project: Path, session_id: str):
        self.session_id = session_id
        self.worker_id = ""
        self._held = False
        path = gate_db_path(project)
        # SQLite cannot always wait while two new connections switch journal mode.
        with locked(path.with_suffix(".init.lock")):
            self._gate = connect(path)
            try:
                self._gate.execute("""CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, mode TEXT NOT NULL,
                    state TEXT NOT NULL, pid INTEGER NOT NULL, reason TEXT, process_identity TEXT)""")
                if "process_identity" not in {
                    row[1] for row in self._gate.execute("PRAGMA table_info(workers)")
                }:
                    self._gate.execute("ALTER TABLE workers ADD COLUMN process_identity TEXT")
            except BaseException:
                self._gate.close()
                raise

    def _gate_tx(self) -> AbstractContextManager[None]:
        return transaction(self._gate)

    def close(self) -> None:
        try:
            self.release()
        finally:
            self._gate.close()

    def release(self) -> None:
        if self._held:
            self._release_slot()

    def snapshot(self) -> list[JsonObject]:
        return [
            dict(row)
            for row in self._gate.execute(
                "SELECT worker_id, session_id, mode, state, pid, reason FROM workers WHERE session_id=?",
                (self.session_id,),
            )
        ]

    def acquire(self, mode: str) -> GateAdmission:
        if mode not in {"read", "write"}:
            raise ValueError("concurrency mode must be read or write")
        worker_id = uuid4().hex
        with self._gate_tx():
            self._reap_workers()
            reads, writes = self._running_counts()
            if (mode == "write" and (reads or writes)) or (mode == "read" and writes):
                state, reason = "queued", "write_busy"
            elif mode == "read" and reads >= READ_LIMIT:
                state, reason = "queued", "read_limit"
            else:
                state, reason = "running", None
            self._gate.execute(
                "INSERT INTO workers(worker_id, session_id, mode, state, pid, reason, process_identity) VALUES(?,?,?,?,?,?,?)",
                (worker_id, self.session_id, mode, state, os.getpid(), reason, process_identity(os.getpid())),
            )
        self.worker_id = worker_id
        self._held = state == "running"
        return {"worker_id": worker_id, "state": state, "reason": reason, "mode": mode}

    def dismiss(self, worker_id: str) -> JsonObject:
        with self._gate_tx():
            cursor = self._gate.execute(
                "UPDATE workers SET state='cancelled' WHERE worker_id=? AND state='queued'", (worker_id,)
            )
            if cursor.rowcount == 1:
                return {"worker_id": worker_id, "state": "cancelled"}
            row = self._gate.execute("SELECT state FROM workers WHERE worker_id=?", (worker_id,)).fetchone()
        return {"worker_id": worker_id, "state": str(row["state"]) if row else "missing"}

    def _release_slot(self) -> None:
        with self._gate_tx():
            self._gate.execute(
                "UPDATE workers SET state='closed' WHERE worker_id=? AND state='running'", (self.worker_id,)
            )
        self._held = False

    def _reap_workers(self) -> None:
        rows = list(
            self._gate.execute("SELECT worker_id, pid, process_identity FROM workers WHERE state='running'")
        )
        for row in rows:
            if not process_alive(int(row["pid"]), row["process_identity"]):
                self._gate.execute(
                    "UPDATE workers SET state='closed', reason='owner_exited' WHERE worker_id=?",
                    (row["worker_id"],),
                )

    def _running_counts(self) -> tuple[int, int]:
        reads = self._gate.execute(
            "SELECT COUNT(*) AS n FROM workers WHERE state='running' AND mode='read'"
        ).fetchone()
        writes = self._gate.execute(
            "SELECT COUNT(*) AS n FROM workers WHERE state='running' AND mode='write'"
        ).fetchone()
        return int(reads["n"]), int(writes["n"])
