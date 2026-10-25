"""Open and close the project knowledge database. Domain rules live elsewhere."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from lite_pi.runtime import PATHS
from lite_pi.storage import locked
from lite_pi.task_storage import connect, transaction
from lite_pi.tools.workspace import Workspace

SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    generation_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    algorithm_json TEXT NOT NULL,
    current INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    generation_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_version TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    text TEXT NOT NULL,
    ordinal INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS queries (
    query_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    generation_id TEXT,
    hit_json TEXT NOT NULL,
    stale_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    repeat_of TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    object_key TEXT NOT NULL,
    statement TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL,
    supersedes TEXT,
    superseded_by TEXT,
    conflict_with TEXT,
    created_at TEXT NOT NULL,
    confirmed_at TEXT,
    expires_at TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_events (
    event_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    from_status TEXT,
    to_status TEXT NOT NULL,
    at TEXT NOT NULL,
    detail TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_reads (
    read_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    object_key TEXT,
    returned_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS context_uses (
    use_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    body_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class KnowledgeError(ValueError):
    pass


class KnowledgeDB:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self._closed = False

    @classmethod
    def open(cls, root: Path) -> KnowledgeDB:
        workspace = Workspace(root)
        path = workspace.root / ".lite-pi" / "knowledge.sqlite"
        key = hashlib.sha256(str(workspace.root).encode()).hexdigest()
        locks = PATHS.state / "knowledge-locks"
        with (
            locked(locks / f"{key}.lock"),
            workspace.parent(".lite-pi/knowledge.sqlite", create=True) as (directory, name),
        ):
            for suffix in ("", "-wal", "-shm", "-journal"):
                _check_file(directory, name + suffix)
            try:
                descriptor = os.open(
                    name,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory,
                )
            except FileExistsError:
                pass
            else:
                os.close(descriptor)
            # sqlite3 needs a pathname. The host owns this directory for the connection
            # lifetime; arbitrary external directory replacement is outside that contract.
            if not os.path.samestat(os.fstat(directory), path.parent.stat(follow_symlinks=False)):
                raise KnowledgeError("knowledge directory changed")
            connection = connect(path)
            try:
                _ensure_schema(connection)
                return cls(connection)
            except BaseException:
                connection.close()
                raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        if self._closed:
            raise KnowledgeError("knowledge database is closed")
        with transaction(self.connection):
            yield self.connection


def _check_file(directory: int, name: str) -> None:
    try:
        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise KnowledgeError(f"unsafe knowledge file: {name}")


def _ensure_schema(connection: sqlite3.Connection) -> None:
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version not in {0, 1, 2, 3}:
        raise KnowledgeError(f"unsupported_knowledge_schema:{version}")
    if version == 0:
        connection.executescript(SCHEMA)
    if version < 2:
        with transaction(connection):
            # Do not silently discard existing conflicting active rows during migration.
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS one_active_memory "
                "ON memories(object_key) WHERE status='active'"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS memory_sources ("
                "source_ref TEXT PRIMARY KEY, identity TEXT NOT NULL, "
                "content TEXT NOT NULL, sha256 TEXT NOT NULL)"
            )
            connection.execute("ALTER TABLE generations ADD COLUMN manifest_json TEXT")
            connection.execute("PRAGMA user_version=2")
    if version < 3:
        with transaction(connection):
            connection.execute("ALTER TABLE generations ADD COLUMN representation_json TEXT")
            connection.execute("ALTER TABLE queries ADD COLUMN trace_json TEXT NOT NULL DEFAULT '{}'")
            connection.execute(
                "CREATE TABLE vectors (chunk_id TEXT PRIMARY KEY, generation_id TEXT NOT NULL, vector_json TEXT NOT NULL)"
            )
            connection.execute("PRAGMA user_version=3")
