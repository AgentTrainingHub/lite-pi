"""Exact-vector ranking and reciprocal-rank fusion over shared chunk identities."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from lite_pi.tools.contracts import JsonObject


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    score: float

    def payload(self) -> JsonObject:
        return asdict(self)


def semantic_rank(query: tuple[float, ...], vectors: dict[str, tuple[float, ...]]) -> tuple[Candidate, ...]:
    return tuple(
        sorted(
            (
                Candidate(identity, sum(a * b for a, b in zip(query, vector, strict=True)))
                for identity, vector in vectors.items()
            ),
            key=lambda item: (-item.score, item.chunk_id),
        )[:8]
    )


def reciprocal_rank(*rankings: tuple[Candidate, ...]) -> tuple[Candidate, ...]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, 1):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0) + 1 / (60 + position)
    return tuple(
        sorted((Candidate(k, v) for k, v in scores.items()), key=lambda item: (-item.score, item.chunk_id))[
            :8
        ]
    )
