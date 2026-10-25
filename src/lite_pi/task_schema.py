"""Task database schema and transactional migration of legacy call identities."""

import sqlite3

from lite_pi.journal import Journal
from lite_pi.task_storage import transaction


def migrate(db: sqlite3.Connection, journal: Journal) -> None:
    if int(db.execute("PRAGMA user_version").fetchone()[0]) > 4:
        raise ValueError("unknown_task_schema")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS ownership (
            session_id TEXT PRIMARY KEY, version INTEGER NOT NULL, worker_id TEXT NOT NULL, pid INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS intents (
            intent_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            assistant_id TEXT NOT NULL,
            call_id TEXT NOT NULL,
            name TEXT NOT NULL,
            arguments TEXT NOT NULL,
            replay TEXT NOT NULL,
            status TEXT NOT NULL,
            candidate_version TEXT NOT NULL,
            object_key TEXT NOT NULL,
            before_sha256 TEXT,
            after_sha256 TEXT,
            preview TEXT NOT NULL,
            owner_version INTEGER NOT NULL,
            requirement_version INTEGER NOT NULL DEFAULT 0,
            result_json TEXT,
            reason TEXT,
            UNIQUE(session_id, assistant_id, call_id)
        );
        CREATE TABLE IF NOT EXISTS reconciliations (
            intent_id TEXT PRIMARY KEY, query_intent_id TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS approvals (
            approval_id TEXT PRIMARY KEY,
            intent_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            candidate_version TEXT NOT NULL,
            decision TEXT NOT NULL,
            preview TEXT NOT NULL,
            scope TEXT NOT NULL,
            constraint_token TEXT NOT NULL,
            expires_at REAL,
            receipt_json TEXT
        );
        CREATE TABLE IF NOT EXISTS budget (
            session_id TEXT PRIMARY KEY,
            spent_calls INTEGER NOT NULL,
            reserved_calls INTEGER NOT NULL,
            unknown_calls INTEGER NOT NULL,
            spent_output INTEGER NOT NULL,
            reserved_output INTEGER NOT NULL,
            unknown_output INTEGER NOT NULL,
            max_calls INTEGER,
            max_output INTEGER,
            max_seconds REAL,
            started_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reservations (
            reservation_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            owner_version INTEGER NOT NULL,
            purpose TEXT NOT NULL,
            output_tokens INTEGER NOT NULL,
            state TEXT NOT NULL,
            attempts INTEGER,
            usage_json TEXT
        );
        CREATE TABLE IF NOT EXISTS workflows (
            event_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            anchor_id TEXT NOT NULL,
            request_json TEXT NOT NULL,
            state_json TEXT NOT NULL,
            UNIQUE(session_id, anchor_id, request_json)
        );
        """
    )
    columns = {row["name"] for row in db.execute("PRAGMA table_info(intents)")}
    if "assistant_id" not in columns:
        with transaction(db):
            rows = list(db.execute("SELECT * FROM intents"))
            mapping = {}
            for row in rows:
                candidates = [
                    record["id"]
                    for record in journal.records
                    if record["kind"] == "assistant"
                    and any(call["id"] == row["call_id"] for call in record["body"].get("tool_calls", []))
                ]
                if len(candidates) != 1:
                    raise ValueError("ambiguous legacy task identity; database left unchanged")
                mapping[row["intent_id"]] = candidates[0]
            db.execute("""CREATE TABLE intents_v2 (
            intent_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            assistant_id TEXT NOT NULL,
            call_id TEXT NOT NULL,
            name TEXT NOT NULL,
            arguments TEXT NOT NULL,
            replay TEXT NOT NULL,
            status TEXT NOT NULL,
            candidate_version TEXT NOT NULL,
            object_key TEXT NOT NULL,
            before_sha256 TEXT,
            after_sha256 TEXT,
            preview TEXT NOT NULL,
            owner_version INTEGER NOT NULL,
            requirement_version INTEGER NOT NULL DEFAULT 0,
            result_json TEXT,
            reason TEXT,
            UNIQUE(session_id, assistant_id, call_id)
        );""")
            for row in rows:
                values = {**dict(row), "assistant_id": mapping[row["intent_id"]]}
                permitted = {
                    "intent_id",
                    "session_id",
                    "assistant_id",
                    "call_id",
                    "name",
                    "arguments",
                    "replay",
                    "status",
                    "candidate_version",
                    "object_key",
                    "before_sha256",
                    "after_sha256",
                    "preview",
                    "owner_version",
                    "requirement_version",
                    "result_json",
                    "reason",
                }
                if values.keys() - permitted:
                    raise ValueError("unsupported legacy intent columns")
                db.execute(
                    f"INSERT INTO intents_v2 ({','.join(values)}) VALUES({','.join('?' for _ in values)})",
                    tuple(values.values()),
                )
            db.execute("DROP TABLE intents")
            db.execute("ALTER TABLE intents_v2 RENAME TO intents")
    if "requirement_version" not in {row[1] for row in db.execute("PRAGMA table_info(intents)")}:
        db.execute("ALTER TABLE intents ADD COLUMN requirement_version INTEGER NOT NULL DEFAULT 0")
    db.execute("PRAGMA user_version=4")
    db.execute("""UPDATE intents SET status='effect_unknown'
        WHERE status='settled' AND intent_id NOT IN (SELECT intent_id FROM reconciliations) AND (json_extract(result_json,'$.effect')='unknown'
        OR json_extract(result_json,'$.reason') IN ('effect_unknown','cleanup_failed','workspace_sync_failed'))""")
