"""Index versioned project sources and record every query."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from lite_pi.knowledge.db import KnowledgeDB, KnowledgeError
from lite_pi.retrieval.budget import RetrievalMeter
from lite_pi.retrieval.builder import IndexBuilder, retrieval_error
from lite_pi.retrieval.embedding import (
    DEFAULT_MODEL_DIR,
    Encoder,
    Encoding,
    LocalEncoder,
    Mode,
    validate_vector,
)
from lite_pi.retrieval.fusion import Candidate, reciprocal_rank, semantic_rank
from lite_pi.retrieval.hits import SearchHit
from lite_pi.retrieval.queries import QueryStore
from lite_pi.retrieval.rank import RankConfig, ScoredDocument, rank, tokenize
from lite_pi.retrieval.records import RankedQuery, RankingTrace, StoredChunk
from lite_pi.retrieval.rerank import rerank
from lite_pi.retrieval.sources import CorpusSources, snapshot_token
from lite_pi.task_budget import RequestBudget
from lite_pi.tools.contracts import JsonObject
from lite_pi.tools.workspace import Workspace


class CorpusIndex:
    def __init__(
        self,
        db: KnowledgeDB,
        workspace: Workspace,
        config: RankConfig | None = None,
        *,
        embedding_dir: Path | None = None,
        stopped: Callable[[], bool] = lambda: False,
        budget: RequestBudget | None = None,
    ) -> None:
        self._db = db
        self._workspace = workspace
        self._config = config or RankConfig()
        self._embedding_dir = embedding_dir or DEFAULT_MODEL_DIR
        self._stopped = stopped
        self._encoder: Encoder | None = None
        self._budget = budget
        self.sources = CorpusSources(workspace, db)
        self.records = QueryStore(db, self.sources)

    def close(self) -> None:
        if self._encoder is not None:
            self._encoder.close()
            self._encoder = None

    def _embedding(self) -> Encoder:
        if self._encoder is None:
            self._encoder = LocalEncoder(self._embedding_dir)
        return self._encoder

    def rebuild(self, *, mode: Mode = "bm25") -> JsonObject:
        builder = IndexBuilder(
            self._db, self._workspace, self.sources, self._config, self._embedding, self._stopped
        )
        return builder.rebuild(mode=mode)

    def current_generation(self) -> str | None:
        return self.records.current_generation()

    def queries(self) -> list[JsonObject]:
        return self.records.queries()

    def search(
        self, query: str, *, run_id: str, limit: int, meter: RetrievalMeter, mode: Mode = "bm25"
    ) -> JsonObject:
        if mode not in {"bm25", "semantic", "hybrid"}:
            return retrieval_error("invalid_retrieval_mode")
        if type(limit) is not int or not 1 <= limit <= 4:
            return retrieval_error("invalid_limit")
        if mode != "bm25":
            try:
                self._embedding()
            except (KnowledgeError, OSError, ValueError) as error:
                return retrieval_error(str(error))
        normalized = " ".join(query.split()).casefold()
        trace = RankingTrace(
            mode,
            self._embedding().representation if mode != "bm25" else None,
            f"bm25-k1={self._config.k1}-b={self._config.b}+e5-exact+rrf60+rerank-v1",
        )
        if not meter.allow_search():
            return self.records.save(
                run_id=run_id,
                query=query,
                generation=self.current_generation(),
                reason="retrieval_exhausted",
                hits=[],
                stale=[],
                repeat_of=None,
                is_error=True,
                executed=False,
            )
        generation, chunks = self.records.load()
        if generation is None:
            meter.mark_search()
            return self.records.save(
                run_id=run_id,
                query=query,
                generation=None,
                reason="no_index",
                hits=[],
                stale=[],
                repeat_of=None,
                is_error=True,
                executed=True,
            )
        if not self.sources.manifest_current(generation):
            meter.mark_search()
            return self.records.save(
                run_id=run_id,
                query=query,
                generation=generation,
                reason="index_stale",
                hits=[],
                stale=[],
                repeat_of=None,
                is_error=False,
                executed=True,
            )
        fresh, stale = self.sources.freshness(chunks)
        if mode != "bm25":
            try:
                self._check_representation(generation)
            except KnowledgeError as error:
                meter.mark_search()
                return self.records.save(
                    run_id=run_id,
                    query=query,
                    generation=generation,
                    reason=str(error),
                    hits=[],
                    stale=stale,
                    repeat_of=None,
                    is_error=True,
                    executed=True,
                    trace=trace.payload(),
                )
        token = snapshot_token(generation, fresh, stale)
        normalized = json.dumps(trace.payload(), sort_keys=True) + ":" + normalized
        previous = meter.previous(normalized, limit)
        if previous is not None and previous.fingerprint == token:
            prior_id = previous.query_id
            prior_hits, prior_trace = self.records.stored_query(prior_id)
            meter.mark_search()
            return self.records.save(
                run_id=run_id,
                query=query,
                generation=generation,
                reason="repeat",
                hits=prior_hits,
                stale=stale,
                repeat_of=prior_id,
                is_error=False,
                executed=True,
                expect_generation=generation,
                recheck=chunks,
                snapshot=token,
                trace=prior_trace,
            )
        stale_paths = {item["source"] for item in stale}
        usable = tuple(chunk for chunk in chunks if chunk.source_path not in stale_paths)
        ranked = self._rank_candidates(query, generation, usable, limit, trace, run_id)
        hits, failure_reason = ranked.hits, ranked.error
        if failure_reason:
            meter.mark_search()
            return self.records.save(
                run_id=run_id,
                query=query,
                generation=generation,
                reason=failure_reason,
                hits=[],
                stale=stale,
                repeat_of=None,
                is_error=True,
                executed=True,
                trace=trace.payload(),
            )
        if not tokenize(query):
            reason = "empty_query"
            hits = []
            is_error = True
        elif not usable and stale:
            reason = "index_stale"
            is_error = False
        elif not hits:
            reason = "no_hit"
            is_error = False
        elif stale:
            reason = "partial_stale"
            is_error = False
        else:
            reason = "ok"
            is_error = False
        saved = self.records.save(
            run_id=run_id,
            query=query,
            generation=generation,
            reason=reason,
            hits=hits,
            stale=stale,
            repeat_of=None,
            is_error=is_error,
            executed=True,
            expect_generation=generation,
            recheck=chunks,
            snapshot=token,
            trace=trace.payload(),
        )
        if saved.get("query_id") and saved.get("reason") not in {
            "source_changed",
            "generation_changed",
            "knowledge_db",
            "empty_query",
            "retrieval_exhausted",
        }:
            meter.remember(normalized, token, str(saved["query_id"]), limit)
        if saved.get("reason") != "knowledge_db":
            meter.mark_search()
        return saved

    def _rank_candidates(
        self,
        query: str,
        generation: str,
        chunks: tuple[StoredChunk, ...],
        limit: int,
        trace: RankingTrace,
        run_id: str,
    ) -> RankedQuery:
        if not chunks or not tokenize(query):
            return RankedQuery([], trace)
        by_id = {c.chunk_id: c for c in chunks}
        try:
            ranked = (
                rank(
                    query,
                    tuple((c.chunk_id, c.text) for c in chunks),
                    limit=limit if trace.mode == "bm25" else 8,
                    config=self._config,
                )
                if trace.mode != "semantic"
                else ()
            )
            trace.bm25 = tuple(Candidate(r.doc_id, r.score) for r in ranked)
            if trace.mode == "bm25":
                return RankedQuery([_hit(by_id[r.doc_id], r) for r in ranked], trace)
            trace.semantic, trace.encoding = self._semantic(query, generation, chunks)
            trace.fused = (
                reciprocal_rank(trace.bm25, trace.semantic) if trace.mode == "hybrid" else trace.semantic
            )
            candidates = [
                _hit(by_id[c.chunk_id], ScoredDocument(c.chunk_id, c.score, ())) for c in trace.fused
            ]
            if trace.mode == "semantic":
                return RankedQuery(candidates[:limit], trace)
            ordered = rerank(query, candidates, run_id=run_id, budget=self._budget, stopped=self._stopped)
            trace.rerank, trace.reranked = ordered.receipt, ordered.order
            if ordered.error:
                return RankedQuery([], trace, ordered.error)
            by_hit = {c["chunk_id"]: c for c in candidates}
            return RankedQuery([by_hit[i] for i in ordered.order[:limit]], trace)
        except (KnowledgeError, OSError, ValueError) as error:
            return RankedQuery([], trace, str(error))

    def _semantic(
        self, query: str, generation: str, chunks: tuple[StoredChunk, ...]
    ) -> tuple[tuple[Candidate, ...], Encoding]:
        expected = self._embedding().representation
        rows = self._db.connection.execute(
            "SELECT chunk_id,vector_json FROM vectors WHERE generation_id=?", (generation,)
        ).fetchall()
        usable = {c.chunk_id for c in chunks}
        vectors = {
            str(r["chunk_id"]): validate_vector(json.loads(r["vector_json"]), expected.dimension)
            for r in rows
            if r["chunk_id"] in usable
        }
        if set(vectors) != usable:
            raise KnowledgeError("embedding_index_incomplete")
        encoded = self._embedding().encode([query], query=True, stopped=self._stopped)
        return semantic_rank(validate_vector(encoded.vectors[0], expected.dimension), vectors), encoded

    def _check_representation(self, generation: str) -> None:
        row = self._db.connection.execute(
            "SELECT representation_json FROM generations WHERE generation_id=?", (generation,)
        ).fetchone()
        expected = self._embedding().representation
        if row is None or not row["representation_json"]:
            raise KnowledgeError("no_semantic_index")
        if json.loads(row["representation_json"]) != expected.payload():
            raise KnowledgeError("embedding_representation_mismatch")


def _hit(chunk: StoredChunk, scored: ScoredDocument) -> SearchHit:
    return {
        "chunk_id": chunk.chunk_id,
        "source": chunk.source_path,
        "version": chunk.source_version,
        "sha256": chunk.source_sha256,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "score": scored.score,
        "text": chunk.text,
        "terms": [
            {"term": part.term, "tf": part.tf, "idf": part.idf, "score": part.score} for part in scored.terms
        ],
    }
