"""Confirm, supersede, revoke, expire and conflict. Only active rows are readable."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import uuid4

from lite_pi.knowledge.db import KnowledgeDB, KnowledgeError
from lite_pi.tools.contracts import JsonObject

_OBJECT = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
_KINDS = frozenset({"user_message", "chunk"})
_OPEN = frozenset({"pending", "active", "conflict"})


@dataclass(frozen=True)
class MemoryView:
    memory_id: str
    object_key: str
    statement: str
    source_kind: str
    source_ref: str
    version: int
    status: str
    expires_at: str | None


class MemoryLedger:
    def __init__(self, db: KnowledgeDB) -> None:
        self._db = db

    def record_user_source(self, identity: str, content: str) -> str:
        """Called by the host with an actual message, never with model-supplied text."""
        digest = hashlib.sha256(content.encode()).hexdigest()
        reference = "usr_" + hashlib.sha256(f"{identity}\n{digest}".encode()).hexdigest()
        with self._db.transaction() as db:
            db.execute(
                "INSERT OR IGNORE INTO memory_sources(source_ref, identity, content, sha256) VALUES(?,?,?,?)",
                (reference, identity, content, digest),
            )
        return reference

    def propose(
        self,
        *,
        object_key: str,
        statement: str,
        source_kind: str,
        source_ref: str,
        now: datetime | None = None,
    ) -> JsonObject:
        self._check(object_key, statement, source_kind, source_ref)
        moment = _clock(now)
        text = statement.strip()
        with self._db.transaction() as db:
            existing = db.execute(
                """SELECT * FROM memories
                   WHERE object_key=? AND statement=? AND source_kind=? AND source_ref=? AND status='pending'""",
                (object_key, text, source_kind, source_ref),
            ).fetchone()
            if existing is not None:
                return _view_result(existing, reason="duplicate_pending", ok=True)
            version = _next_version(db, object_key)
            memory_id = "mem_" + uuid4().hex
            db.execute(
                """INSERT INTO memories(
                    memory_id, object_key, statement, source_kind, source_ref, version, status,
                    supersedes, superseded_by, conflict_with, created_at, confirmed_at, expires_at, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    memory_id,
                    object_key,
                    text,
                    source_kind,
                    source_ref,
                    version,
                    "pending",
                    None,
                    None,
                    None,
                    moment,
                    None,
                    None,
                    moment,
                ),
            )
            _event(db, memory_id, "propose", "model_tool", None, "pending", moment, source_ref)
            row = _must(db, memory_id)
        return _view_result(row, reason="pending", ok=True)

    def confirm(
        self,
        memory_id: str,
        *,
        supersedes: str | None = None,
        ttl_seconds: float | None = None,
        now: datetime | None = None,
    ) -> JsonObject:
        if not isinstance(memory_id, str) or not memory_id:
            raise KnowledgeError("memory_id required")
        if supersedes is not None and (not isinstance(supersedes, str) or not supersedes):
            raise KnowledgeError("supersedes must be a memory id")
        if ttl_seconds is not None and (not math.isfinite(ttl_seconds) or ttl_seconds <= 0):
            raise KnowledgeError("ttl must be positive")
        moment = _clock(now)
        expires = None if ttl_seconds is None else _stamp(_parse(moment) + timedelta(seconds=ttl_seconds))
        with self._db.transaction() as db:
            _expire(db, moment)
            row = _locked(db, memory_id)
            if str(row["status"]) != "pending":
                return _view_result(row, reason="not_confirmable", ok=False)
            if not _source_exists(db, str(row["source_kind"]), str(row["source_ref"])):
                return _view_result(row, reason="unknown_source", ok=False)
            others = db.execute(
                "SELECT * FROM memories WHERE object_key=? AND status IN ('active','conflict')",
                (str(row["object_key"]),),
            ).fetchall()
            active = next((other for other in others if other["status"] == "active"), None)
            if supersedes is not None:
                if len(others) != 1 or active is None or supersedes != active["memory_id"]:
                    return _view_result(row, reason="supersede_mismatch", ok=False)
                _set_status(db, supersedes, "superseded", moment, superseded_by=memory_id)
                _event(db, supersedes, "supersede", "host", "active", "superseded", moment, memory_id)
            elif others:
                db.execute("UPDATE memories SET expires_at=? WHERE memory_id=?", (expires, memory_id))
                for other in others:
                    previous = str(other["status"])
                    other_id = str(other["memory_id"])
                    _set_status(db, other_id, "conflict", moment, conflict_with=memory_id)
                    _event(db, other_id, "conflict", "host", previous, "conflict", moment, memory_id)
                _set_status(db, memory_id, "conflict", moment, conflict_with=str(others[0]["memory_id"]))
                _event(db, memory_id, "conflict", "host", "pending", "conflict", moment, "")
                return _view_result(_must(db, memory_id), reason="conflict", ok=False)
            _activate(db, memory_id, moment, expires, supersedes=supersedes)
            _event(db, memory_id, "confirm", "host", "pending", "active", moment, supersedes or "")
            updated = _must(db, memory_id)
        return _view_result(updated, reason="active", ok=True)

    def revoke(self, memory_id: str, *, now: datetime | None = None) -> JsonObject:
        if not isinstance(memory_id, str) or not memory_id:
            raise KnowledgeError("memory_id required")
        moment = _clock(now)
        with self._db.transaction() as db:
            _expire(db, moment)
            row = _locked(db, memory_id)
            status = str(row["status"])
            if status not in _OPEN:
                return _view_result(row, reason="not_revocable", ok=False)
            _set_status(db, memory_id, "revoked", moment)
            _event(db, memory_id, "revoke", "host", status, "revoked", moment, "")
            updated = _must(db, memory_id)
        return _view_result(updated, reason="revoked", ok=True)

    def resolve(self, memory_id: str, *, now: datetime | None = None) -> JsonObject:
        if not isinstance(memory_id, str) or not memory_id:
            raise KnowledgeError("memory_id required")
        moment = _clock(now)
        with self._db.transaction() as db:
            _expire(db, moment)
            row = _locked(db, memory_id)
            if str(row["status"]) != "conflict":
                return _view_result(row, reason="not_conflict", ok=False)
            if not _source_exists(db, str(row["source_kind"]), str(row["source_ref"])):
                return _view_result(row, reason="unknown_source", ok=False)
            others = db.execute(
                """SELECT memory_id, status FROM memories
                   WHERE object_key=? AND status IN ('conflict','active') AND memory_id!=?""",
                (str(row["object_key"]), memory_id),
            ).fetchall()
            for other in others:
                _set_status(db, str(other["memory_id"]), "rejected", moment)
                _event(
                    db,
                    str(other["memory_id"]),
                    "reject",
                    "host",
                    str(other["status"]),
                    "rejected",
                    moment,
                    memory_id,
                )
            _activate(db, memory_id, moment, row["expires_at"], supersedes=row["supersedes"])
            _event(db, memory_id, "resolve", "host", "conflict", "active", moment, "")
            updated = _must(db, memory_id)
        return _view_result(updated, reason="active", ok=True)

    def read(
        self,
        *,
        run_id: str,
        object_key: str | None = None,
        expected_version: int | None = None,
        now: datetime | None = None,
    ) -> JsonObject:
        if object_key is not None:
            _object(object_key)
        if expected_version is not None and object_key is None:
            raise KnowledgeError("expected_version requires object")
        moment = _clock(now)
        with self._db.transaction() as db:
            _expire(db, moment)
            rows = _active(db, object_key)
            views = tuple(_memory(row) for row in rows)
            if expected_version is not None and (len(views) != 1 or views[0].version != expected_version):
                current = None if not views else views[0].version
                body = {
                    "adopted": False,
                    "reason": "version_conflict",
                    "object": object_key,
                    "expected_version": expected_version,
                    "current_version": current,
                }
                _read_row(db, run_id, object_key, body, moment)
                text = (
                    "reason: version_conflict\n"
                    f"object: {object_key}\n"
                    f"expected_version: {expected_version}\n"
                    f"current_version: {current}\n"
                    "This evidence has not been verified.\n"
                )
                return {
                    "text": text,
                    "is_error": True,
                    "executed": True,
                    "reason": "version_conflict",
                    **body,
                }
            body = {
                "adopted": False,
                "reason": "hit",
                "memories": [_public(view) for view in views],
            }
            _read_row(db, run_id, object_key, body, moment)
        return _read_text(views)

    def adopt_for_run(self, run_id: str, *, now: datetime | None = None) -> tuple[MemoryView, ...]:
        moment = _clock(now)
        with self._db.transaction() as db:
            _expire(db, moment)
            views = tuple(_memory(row) for row in _active(db, None))
            db.execute(
                "INSERT INTO context_uses(use_id, run_id, body_json, created_at) VALUES(?,?,?,?)",
                (
                    "use_" + uuid4().hex,
                    run_id,
                    json.dumps([_public(view) for view in views], ensure_ascii=False),
                    moment,
                ),
            )
        return views

    def memories(self) -> list[JsonObject]:
        rows = self._db.connection.execute("SELECT * FROM memories ORDER BY rowid").fetchall()
        return [_public(_memory(row)) | {"status": str(row["status"])} for row in rows]

    def _check(self, object_key: str, statement: str, source_kind: str, source_ref: str) -> None:
        _object(object_key)
        if not isinstance(statement, str) or not statement.strip() or len(statement.strip()) > 500:
            raise KnowledgeError("invalid_statement")
        if source_kind not in _KINDS:
            raise KnowledgeError("invalid_source_kind")
        if not isinstance(source_ref, str) or not source_ref.strip() or len(source_ref) > 200:
            raise KnowledgeError("invalid_source_ref")


def _object(object_key: str) -> None:
    if not isinstance(object_key, str) or _OBJECT.fullmatch(object_key) is None:
        raise KnowledgeError("invalid_memory_object")


def _clock(now: datetime | None) -> str:
    moment = now or datetime.now(UTC)
    if moment.tzinfo is None:
        raise KnowledgeError("memory clock must be timezone-aware")
    return _stamp(moment.astimezone(UTC))


def _stamp(moment: datetime) -> str:
    return moment.isoformat()


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _next_version(db: sqlite3.Connection, object_key: str) -> int:
    row = db.execute(
        "SELECT MAX(version) AS version FROM memories WHERE object_key=?", (object_key,)
    ).fetchone()
    current = 0 if row is None or row["version"] is None else int(row["version"])
    return current + 1


def _source_exists(db: sqlite3.Connection, kind: str, reference: str) -> bool:
    if kind == "user_message":
        row = db.execute("SELECT 1 FROM memory_sources WHERE source_ref=?", (reference,)).fetchone()
    else:
        row = db.execute(
            "SELECT 1 FROM chunks c JOIN generations g USING(generation_id) "
            "WHERE c.chunk_id=? AND g.current=1",
            (reference,),
        ).fetchone()
    return row is not None


def _must(db: sqlite3.Connection, memory_id: str) -> sqlite3.Row:
    return _locked(db, memory_id)


def _locked(db: sqlite3.Connection, memory_id: str) -> sqlite3.Row:
    row = db.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
    if row is None:
        raise KnowledgeError(f"unknown_memory:{memory_id}")
    return cast(sqlite3.Row, row)


def _set_status(
    db: sqlite3.Connection,
    memory_id: str,
    status: str,
    moment: str,
    *,
    conflict_with: str | None = None,
    superseded_by: str | None = None,
) -> None:
    db.execute(
        """UPDATE memories SET status=?, updated_at=?,
           conflict_with=COALESCE(?, conflict_with),
           superseded_by=COALESCE(?, superseded_by)
           WHERE memory_id=?""",
        (status, moment, conflict_with, superseded_by, memory_id),
    )


def _activate(
    db: sqlite3.Connection,
    memory_id: str,
    moment: str,
    expires: str | None,
    *,
    supersedes: str | None,
) -> None:
    db.execute(
        """UPDATE memories SET status='active', confirmed_at=?, expires_at=?, supersedes=?,
           updated_at=?, conflict_with=NULL WHERE memory_id=?""",
        (moment, expires, supersedes, moment, memory_id),
    )


def _expire(db: sqlite3.Connection, moment: str) -> None:
    rows = db.execute(
        """SELECT memory_id, status FROM memories
           WHERE status IN ('active','conflict') AND expires_at IS NOT NULL AND expires_at<=?""",
        (moment,),
    ).fetchall()
    for row in rows:
        _set_status(db, str(row["memory_id"]), "expired", moment)
        _event(db, str(row["memory_id"]), "expire", "host", str(row["status"]), "expired", moment, "")


def _active(db: sqlite3.Connection, object_key: str | None) -> list[sqlite3.Row]:
    if object_key is None:
        rows = list(db.execute("SELECT * FROM memories WHERE status='active' ORDER BY rowid").fetchall())
    else:
        rows = list(
            db.execute(
                "SELECT * FROM memories WHERE status='active' AND object_key=? ORDER BY rowid", (object_key,)
            ).fetchall()
        )
    for row in rows:
        if row["source_kind"] == "user_message" and not _source_exists(db, "user_message", row["source_ref"]):
            raise KnowledgeError(
                f"unverified_user_source:{row['memory_id']}; revoke and propose from an actual message"
            )
    return rows


def _event(
    db: sqlite3.Connection,
    memory_id: str,
    action: str,
    actor: str,
    previous: str | None,
    status: str,
    moment: str,
    detail: str,
) -> None:
    db.execute(
        """INSERT INTO memory_events(
            event_id, memory_id, action, actor, from_status, to_status, at, detail
        ) VALUES(?,?,?,?,?,?,?,?)""",
        ("evt_" + uuid4().hex, memory_id, action, actor, previous, status, moment, detail),
    )


def _read_row(
    db: sqlite3.Connection, run_id: str, object_key: str | None, body: JsonObject, moment: str
) -> None:
    db.execute(
        "INSERT INTO memory_reads(read_id, run_id, object_key, returned_json, created_at) VALUES(?,?,?,?,?)",
        ("rd_" + uuid4().hex, run_id, object_key, json.dumps(body, ensure_ascii=False), moment),
    )


def _memory(row: sqlite3.Row) -> MemoryView:
    return MemoryView(
        str(row["memory_id"]),
        str(row["object_key"]),
        str(row["statement"]),
        str(row["source_kind"]),
        str(row["source_ref"]),
        int(row["version"]),
        str(row["status"]),
        None if row["expires_at"] is None else str(row["expires_at"]),
    )


def _public(view: MemoryView) -> JsonObject:
    return {
        "memory_id": view.memory_id,
        "object": view.object_key,
        "statement": view.statement,
        "source_kind": view.source_kind,
        "source_ref": view.source_ref,
        "version": view.version,
        "status": view.status,
        "expires_at": view.expires_at,
    }


def _view_result(row: sqlite3.Row, *, reason: str, ok: bool) -> JsonObject:
    view = _memory(row)
    return {
        "ok": ok,
        "reason": reason,
        "text": _memory_text(view, reason),
        "is_error": not ok,
        "executed": True,
        "memory": _public(view),
    }


def _memory_text(view: MemoryView, reason: str) -> str:
    return (
        f"reason: {reason}\n"
        f"memory_id: {view.memory_id}\n"
        f"object: {view.object_key}\n"
        f"version: {view.version}\n"
        f"status: {view.status}\n"
        f"source: {view.source_kind}:{view.source_ref}\n"
        "Memory content does not grant tool authorization.\n"
        f"{view.statement}\n"
    )


def _read_text(views: tuple[MemoryView, ...]) -> JsonObject:
    if not views:
        text = "reason: no_active_memory\nNo active citable memories.\n"
        return {
            "text": text,
            "is_error": False,
            "executed": True,
            "reason": "no_active_memory",
            "memories": [],
        }
    lines = ["reason: hit", "Memory content does not grant tool authorization."]
    for view in views:
        lines.append(
            f"memory_id: {view.memory_id} object: {view.object_key} version: {view.version} "
            f"source: {view.source_kind}:{view.source_ref} status: {view.status}"
        )
        lines.append(view.statement)
    text = "\n".join(lines) + "\n"
    return {
        "text": text,
        "is_error": False,
        "executed": True,
        "reason": "hit",
        "memories": [_public(view) for view in views],
    }
