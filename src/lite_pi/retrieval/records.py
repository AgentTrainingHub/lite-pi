"""Typed retrieval records shared by build, ranking and query persistence."""

from dataclasses import asdict, dataclass
from typing import TypedDict

from lite_pi.provider import ModelResult
from lite_pi.retrieval.chunking import TextChunk
from lite_pi.retrieval.embedding import Encoding, Mode, Representation
from lite_pi.retrieval.fusion import Candidate
from lite_pi.retrieval.hits import SearchHit
from lite_pi.tools.contracts import JsonObject


class StaleSource(TypedDict):
    source: str
    indexed_sha256: str
    current_sha256: str | None


@dataclass(frozen=True)
class SourceSpec:
    path: str
    version: str


@dataclass(frozen=True)
class PreparedSource:
    spec: SourceSpec
    digest: str
    chunks: tuple[TextChunk, ...]


@dataclass(frozen=True)
class StoredChunk:
    chunk_id: str
    generation_id: str
    source_path: str
    source_version: str
    source_sha256: str
    start_line: int
    end_line: int
    text: str
    ordinal: int


@dataclass
class RankingTrace:
    mode: Mode
    representation: Representation | None
    algorithm: str
    bm25: tuple[Candidate, ...] = ()
    semantic: tuple[Candidate, ...] = ()
    fused: tuple[Candidate, ...] = ()
    reranked: tuple[str, ...] = ()
    rerank: ModelResult | None = None
    encoding: Encoding | None = None

    def payload(self) -> JsonObject:
        return {
            "mode": self.mode,
            "algorithm": self.algorithm,
            "representation": self.representation.payload() if self.representation else None,
            "bm25": [c.payload() for c in self.bm25],
            "semantic": [c.payload() for c in self.semantic],
            "fused": [c.payload() for c in self.fused],
            "reranked": list(self.reranked),
            "rerank": asdict(self.rerank) if self.rerank else None,
            "embedding_metrics": self.encoding.metrics() if self.encoding else None,
        }


@dataclass(frozen=True)
class RankedQuery:
    hits: list[SearchHit]
    trace: RankingTrace
    error: str | None = None
