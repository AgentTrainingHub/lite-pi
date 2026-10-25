"""Stable, locatable candidates shared by ranking and relevance requests."""

from typing import TypedDict


class TermScore(TypedDict):
    term: str
    tf: int
    idf: float
    score: float


class SearchHit(TypedDict):
    chunk_id: str
    source: str
    version: str
    sha256: str
    start_line: int
    end_line: int
    score: float
    text: str
    terms: list[TermScore]
