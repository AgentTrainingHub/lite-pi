"""Build and atomically publish a complete index generation."""

import hashlib
import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

from lite_pi.knowledge.db import KnowledgeDB, KnowledgeError
from lite_pi.outcomes import tool_outcome
from lite_pi.retrieval.embedding import Encoder, Mode, validate_vector
from lite_pi.retrieval.rank import RankConfig, algorithm_record
from lite_pi.retrieval.sources import NOTICE, SOURCE_MAX_BYTES, CorpusSources, load_manifest, manifest_json
from lite_pi.tools.contracts import JsonObject
from lite_pi.tools.workspace import Workspace


class IndexBuilder:
    def __init__(
        self,
        db: KnowledgeDB,
        workspace: Workspace,
        sources: CorpusSources,
        config: RankConfig,
        embedding: Callable[[], Encoder],
        stopped: Callable[[], bool],
    ) -> None:
        self._db, self._workspace, self._config = db, workspace, config
        self._embedding, self._stopped = embedding, stopped
        self.sources = sources

    def rebuild(self, *, mode: Mode = "bm25") -> JsonObject:
        if mode not in {"bm25", "semantic", "hybrid"}:
            return retrieval_error("invalid_retrieval_mode")
        representation: JsonObject | None = None
        metrics: JsonObject | None = None
        vectors: tuple[tuple[float, ...], ...] = ()
        try:
            specs = load_manifest(self._workspace)
            prepared = self.sources.read(specs)
            if mode != "bm25":
                if any(len(c.text) > 800 for s in prepared for c in s.chunks):
                    raise KnowledgeError("embedding_chunk_too_long:split_long_lines")
                encoder = self._embedding()
                encoded = encoder.encode(
                    [c.text for s in prepared for c in s.chunks], query=False, stopped=self._stopped
                )
                vectors = tuple(validate_vector(v, encoder.representation.dimension) for v in encoded.vectors)
                if len(vectors) != sum(len(s.chunks) for s in prepared):
                    raise KnowledgeError("embedding_batch_size_mismatch")
                representation, metrics = encoder.representation.payload(), encoded.metrics()
            if self._stopped():
                raise KnowledgeError("cancelled")
        except KnowledgeError as error:
            return retrieval_error(str(error))
        except (OSError, ValueError, UnicodeError) as error:
            return retrieval_error(f"{type(error).__name__}: {error}")
        generation = "gen_" + uuid4().hex
        created = datetime.now(UTC).isoformat()
        algorithm = algorithm_record(self._config)
        count = 0
        try:
            with self._db.transaction() as db:
                if load_manifest(self._workspace) != specs:
                    raise KnowledgeError("manifest_changed")
                db.execute("UPDATE generations SET current=0 WHERE current=1")
                db.execute(
                    "INSERT INTO generations(generation_id, created_at, algorithm_json, current, manifest_json, representation_json) VALUES(?,?,?,1,?,?)",
                    (
                        generation,
                        created,
                        json.dumps(algorithm, ensure_ascii=False),
                        manifest_json(specs),
                        json.dumps(representation) if representation else None,
                    ),
                )
                for source in prepared:
                    path, version, digest = source.spec.path, source.spec.version, source.digest
                    for chunk in source.chunks:
                        count += 1
                        identity = hashlib.sha256(
                            f"{generation}\n{path}\n{chunk.start_line}\n{chunk.text}".encode()
                        ).hexdigest()[:16]
                        db.execute(
                            """INSERT INTO chunks(
                                chunk_id, generation_id, source_path, source_version, source_sha256,
                                start_line, end_line, text, ordinal
                            ) VALUES(?,?,?,?,?,?,?,?,?)""",
                            (
                                "chk_" + identity,
                                generation,
                                path,
                                version,
                                digest,
                                chunk.start_line,
                                chunk.end_line,
                                chunk.text,
                                count,
                            ),
                        )
                        if vectors:
                            db.execute(
                                "INSERT INTO vectors(chunk_id,generation_id,vector_json) VALUES(?,?,?)",
                                ("chk_" + identity, generation, json.dumps(vectors[count - 1])),
                            )
                        if self._stopped():
                            raise KnowledgeError("cancelled")
                if load_manifest(self._workspace) != specs:
                    raise KnowledgeError("manifest_changed")
                for source in prepared:
                    current = self._workspace.read(
                        source.spec.path, remember=False, max_bytes=SOURCE_MAX_BYTES
                    )
                    if hashlib.sha256(current).hexdigest() != source.digest:
                        raise KnowledgeError("source_changed")
        except KnowledgeError as error:
            return retrieval_error(str(error))
        except sqlite3.Error as error:
            return retrieval_error(f"knowledge_db:{error}")
        text = f"generation: {generation}\nsources: {len(prepared)}\nchunks: {count}\n{NOTICE}\n"
        return tool_outcome(
            {
                "text": text,
                "generation_id": generation,
                "chunks": count,
                "representation": representation,
                "embedding_metrics": metrics,
            },
            is_error=False,
            executed=True,
            reason="rebuilt",
        )


def retrieval_error(reason: str) -> JsonObject:
    return tool_outcome({"text": reason + "\n"}, is_error=True, executed=False, reason=reason)
