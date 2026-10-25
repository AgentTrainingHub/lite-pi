"""SQLite connection setup and explicit transactions."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5, isolation_level=None, check_same_thread=False)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection
    except BaseException:
        connection.close()
        raise


def optional_row(row: object) -> sqlite3.Row | None:
    if row is None:
        return None
    if not isinstance(row, sqlite3.Row):
        raise TypeError("expected a database row")
    return row


@contextmanager
def transaction(db: sqlite3.Connection) -> Iterator[None]:
    db.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        db.execute("ROLLBACK")
        raise
    else:
        db.execute("COMMIT")


class StaleOwner(ValueError):
    def __init__(self, identity: str):
        super().__init__(f"stale_owner:{identity}")


def assert_owner(db: sqlite3.Connection, session_id: str, version: int) -> None:
    row = db.execute("SELECT version FROM ownership WHERE session_id=?", (session_id,)).fetchone()
    if row is None or int(row["version"]) != version:
        raise StaleOwner(session_id)
