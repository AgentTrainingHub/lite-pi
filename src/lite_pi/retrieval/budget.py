"""Per-run caps for search and rebuild. Model-call budget stays in TaskBudget."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class QuerySnapshot:
    fingerprint: str
    query_id: str


@dataclass
class RetrievalMeter:
    search_limit: int
    rebuild_limit: int = 2
    searches: int = 0
    rebuilds: int = 0
    snapshots: dict[tuple[str, int], QuerySnapshot] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.search_limit < 1 or self.rebuild_limit < 1:
            raise ValueError("retrieval limits must be positive")

    def allow_search(self) -> bool:
        return self.searches < self.search_limit

    def mark_search(self) -> None:
        self.searches += 1

    def allow_rebuild(self) -> bool:
        return self.rebuilds < self.rebuild_limit

    def mark_rebuild(self) -> None:
        self.rebuilds += 1

    def previous(self, normalized: str, limit: int = 4) -> QuerySnapshot | None:
        return self.snapshots.get((normalized, limit))

    def remember(self, normalized: str, snapshot: str, query_id: str, limit: int = 4) -> None:
        self.snapshots[(normalized, limit)] = QuerySnapshot(snapshot, query_id)
