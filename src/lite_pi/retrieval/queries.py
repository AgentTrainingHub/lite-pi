"""Persist query receipts and load generations; source freshness is checked at commit."""

import json
import sqlite3
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from lite_pi.knowledge.db import KnowledgeDB, KnowledgeError
from lite_pi.outcomes import tool_outcome
from lite_pi.retrieval.hits import SearchHit
from lite_pi.retrieval.records import StaleSource, StoredChunk
from lite_pi.retrieval.sources import NOTICE, CorpusSources, snapshot_token
from lite_pi.tools.contracts import JsonObject


class QueryStore:
    def __init__(self, db: KnowledgeDB, sources: CorpusSources) -> None:
        self._db, self.sources = db, sources

    def current_generation(self) -> str | None:
        row = self._db.connection.execute("SELECT generation_id FROM generations WHERE current=1").fetchone()
        if row is None:
            return None
        return str(row["generation_id"])

    def queries(self) -> list[JsonObject]:
        rows = self._db.connection.execute("SELECT * FROM queries ORDER BY rowid").fetchall()
        found = []
        for row in rows:
            found.append(
                {
                    "query_id": str(row["query_id"]),
                    "run_id": str(row["run_id"]),
                    "query_text": str(row["query_text"]),
                    "generation_id": row["generation_id"],
                    "hits": json.loads(str(row["hit_json"])),
                    "stale": json.loads(str(row["stale_json"])),
                    "reason": str(row["reason"]),
                    "repeat_of": row["repeat_of"],
                    "trace": json.loads(str(row["trace_json"])),
                }
            )
        return found

    def load(self) -> tuple[str | None, tuple[StoredChunk, ...]]:
        generation = self.current_generation()
        if generation is None:
            return None, ()
        rows = self._db.connection.execute(
            "SELECT * FROM chunks WHERE generation_id=? ORDER BY ordinal", (generation,)
        ).fetchall()
        return generation, tuple(_stored(row) for row in rows)

    def stored_query(self, query_id: str) -> tuple[list[SearchHit], JsonObject]:
        row = self._db.connection.execute(
            "SELECT hit_json,trace_json FROM queries WHERE query_id=?", (query_id,)
        ).fetchone()
        if row is None:
            return [], {}
        loaded = json.loads(str(row["hit_json"]))
        if not isinstance(loaded, list) or any(not isinstance(item, dict) for item in loaded):
            raise KnowledgeError("invalid stored query hits")
        return cast(list[SearchHit], loaded), cast(JsonObject, json.loads(str(row["trace_json"])))

    def save(
        self,
        *,
        run_id: str,
        query: str,
        generation: str | None,
        reason: str,
        hits: list[SearchHit],
        stale: list[StaleSource],
        repeat_of: str | None,
        is_error: bool,
        executed: bool,
        expect_generation: str | None = None,
        recheck: tuple[StoredChunk, ...] | None = None,
        snapshot: str | None = None,
        trace: JsonObject | None = None,
    ) -> JsonObject:
        query_id = "qry_" + uuid4().hex
        try:
            with self._db.transaction() as db:
                if expect_generation is not None:
                    current = db.execute("SELECT generation_id FROM generations WHERE current=1").fetchone()
                    live = None if current is None else str(current["generation_id"])
                    if live != expect_generation:
                        reason = "generation_changed"
                        hits = []
                        is_error = True
                        generation = live
                        recheck = None
                if (
                    recheck is not None
                    and generation is not None
                    and not self.sources.manifest_current(generation)
                ):
                    reason, hits, recheck = "index_stale", [], None
                if recheck is not None and generation is not None:
                    fresh, live_stale = self.sources.freshness(recheck)
                    live_token = snapshot_token(generation, fresh, live_stale)
                    if snapshot is not None and live_token != snapshot:
                        reason = "source_changed"
                        hits = []
                        stale = live_stale
                        is_error = False
                    else:
                        stale_paths = {str(item["source"]) for item in live_stale}
                        hits = [hit for hit in hits if str(hit.get("source")) not in stale_paths]
                        stale = live_stale
                        if reason == "empty_query":
                            hits = []
                        elif not hits and not fresh and live_stale:
                            reason = "index_stale"
                        elif hits and live_stale:
                            reason = "partial_stale"
                db.execute(
                    """INSERT INTO queries(
                        query_id, run_id, query_text, generation_id, hit_json, stale_json,
                        reason, repeat_of, created_at, trace_json
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        query_id,
                        run_id,
                        query,
                        generation,
                        json.dumps(hits, ensure_ascii=False),
                        json.dumps(stale, ensure_ascii=False),
                        reason,
                        repeat_of,
                        datetime.now(UTC).isoformat(),
                        json.dumps(trace or {}, ensure_ascii=False),
                    ),
                )
        except sqlite3.Error as error:
            return tool_outcome(
                {"text": f"knowledge_db:{error}\n"},
                is_error=True,
                executed=True,
                effect="unknown",
                reason="knowledge_db",
            )
        return tool_outcome(
            {
                "text": _render(query_id, generation, reason, hits, stale, repeat_of),
                "query_id": query_id,
                "generation_id": generation,
                "hits": hits,
                "stale": stale,
                "repeat_of": repeat_of,
                "trace": trace or {},
            },
            is_error=is_error,
            executed=executed,
            reason=reason,
        )


def _stored(row: sqlite3.Row) -> StoredChunk:
    return StoredChunk(
        str(row["chunk_id"]),
        str(row["generation_id"]),
        str(row["source_path"]),
        str(row["source_version"]),
        str(row["source_sha256"]),
        int(row["start_line"]),
        int(row["end_line"]),
        str(row["text"]),
        int(row["ordinal"]),
    )


def _render(
    query_id: str,
    generation: str | None,
    reason: str,
    hits: list[SearchHit],
    stale: list[StaleSource],
    repeat_of: str | None,
) -> str:
    lines = [
        f"query_id: {query_id}",
        f"generation: {generation or '-'}",
        f"reason: {reason}",
        NOTICE,
    ]
    if repeat_of:
        lines.append(f"repeat_of: {repeat_of}")
    if reason == "retrieval_exhausted":
        lines.append("Search quota exhausted. No new results were retrieved; do not invent citations.")
    if not hits and reason != "retrieval_exhausted":
        lines.append("No citable passages. Do not invent citations.")
    for hit in hits:
        lines.append(
            "hit chunk_id={chunk_id} source={source} version={version} sha256={sha256} "
            "lines={start_line}-{end_line} score={score:.4f}".format(**hit)
        )
        lines.append(str(hit["text"]).rstrip("\n"))
        terms = hit["terms"]
        if isinstance(terms, list):
            rendered = ", ".join(
                f"{part['term']}:{part['score']:.4f}" for part in terms if isinstance(part, dict)
            )
            lines.append(f"terms: {rendered}")
    for item in stale:
        lines.append(
            f"stale source={item['source']} indexed_sha256={item['indexed_sha256']} "
            f"current_sha256={item['current_sha256']}"
        )
    return "\n".join(lines) + "\n"
