"""Classify retrieval outcomes. This is a test judge, not a product module."""

from __future__ import annotations

import json
import re


def classify_retrieval(
    *,
    corpus_has_fact: bool,
    hit_count: int,
    cited_ids: list[str],
    supported_ids: set[str],
) -> str:
    """Return missing, recall_failure, evidence_misuse, or supported."""
    cited = set(cited_ids)
    if cited and not cited <= supported_ids:
        return "evidence_misuse"
    if not corpus_has_fact and hit_count == 0 and not cited:
        return "missing"
    if not corpus_has_fact and (hit_count or cited):
        return "evidence_misuse"
    if corpus_has_fact and hit_count == 0:
        return "recall_failure"
    if corpus_has_fact and hit_count and cited and cited <= supported_ids:
        return "supported"
    return "evidence_misuse"


def contract_answer(text: str, supported: set[str], returned: set[str]) -> bool:
    """A closed factual report prevents an otherwise valid citation hiding extra claims."""
    report = _report(text)
    if not isinstance(report, dict) or set(report) != {
        "parameter",
        "parameter_type",
        "return_type",
        "citations",
    }:
        return False
    citations = report["citations"]
    if not isinstance(citations, list) or not citations or any(not isinstance(c, str) for c in citations):
        return False
    cited = {c.removeprefix("cite:") for c in citations}
    return (
        report["parameter"] == "user_id"
        and report["parameter_type"] == "str"
        and report["return_type"] == "str"
        and bool(cited & supported)
        and cited <= returned
    )


def gap_answer(text: str) -> bool:
    """Check the factual outcome without requiring a particular natural-language sentence."""
    report = _report(text)
    return isinstance(report, dict) and report == {"status": "insufficient_evidence", "citations": []}


def _report(text: str) -> object:
    fenced = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text, object_pairs_hook=_unique_fields)
    except (ValueError, TypeError):
        return None


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate report field")
        result[key] = value
    return result
