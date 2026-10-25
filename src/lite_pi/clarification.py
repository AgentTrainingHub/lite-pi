"""A clarification carries information; it never authorizes a project effect."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class Question:
    id: str
    text: str
    options: list[str]
    paths: list[str]
    step_ids: list[str]
    requirement_version: int
    branch_id: str
    asked_record_id: str
    status: Literal["pending", "answered", "cancelled", "superseded"] = "pending"
    answer: str | None = None
    answer_record_id: str | None = None
