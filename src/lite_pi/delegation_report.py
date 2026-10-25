"""Typed investigation receipts and their tool-visible projection."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from lite_pi.task_budget import RequestBudget
from lite_pi.tools.contracts import JsonObject


@dataclass(frozen=True)
class ChildBinding:
    budget: RequestBudget
    scope: tuple[str, ...]


@dataclass(frozen=True)
class Survey:
    task: str
    paths: tuple[str, ...]
    hashes: dict[str, str]


@dataclass(frozen=True, kw_only=True)
class ChildReport:
    survey_id: str
    generation: int
    status: str
    reason: str | None
    narrative: str
    run_id: str | None
    parent_run_id: str
    paths: tuple[str, ...]
    tools: tuple[str, ...]
    refusals: tuple[str, ...]
    unsupported: tuple[str, ...]
    completion_claim: str
    model_calls: int
    started: float
    ended: float
    started_at: str
    ended_at: str
    kept: bool


@dataclass(frozen=True)
class FileProof:
    path: str
    exists: bool
    sha256: str | None
    before_sha256: str | None
    excerpt: str
    error: str | None


def child_status(value: object) -> str:
    if value == "stop":
        return "complete"
    if isinstance(value, str) and value:
        return value
    return "error"


def call_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def failure_result(text: str, reason: str) -> JsonObject:
    return {
        "text": text,
        "is_error": True,
        "executed": True,
        "reason": reason,
        "narrative_is_not_evidence": True,
        "surveys": [],
        "proofs": [],
    }


def cancelled_report(generation: int, parent_run_id: str) -> JsonObject:
    return {
        "text": "父任务已取消，没有启动子调查。子任务叙述不能作为证据。",
        "is_error": True,
        "executed": True,
        "reason": "cancelled",
        "narrative_is_not_evidence": True,
        "generation": generation,
        "parent_run_id": parent_run_id,
        "failure_policy": "continue_siblings",
        "surveys": [],
        "proofs": [],
        "dropped": [],
        "child_write_detected": False,
        "peak_concurrency": 0,
        "overlapped": False,
        "joined": True,
    }


def rejected_report(
    survey_id: str,
    generation: int,
    parent_run_id: str,
    paths: tuple[str, ...],
    reason: str,
) -> ChildReport:
    started = time.monotonic()
    return make_report(
        survey_id=survey_id,
        generation=generation,
        parent_run_id=parent_run_id,
        paths=paths,
        status="rejected",
        reason=reason,
        narrative="",
        run_id=None,
        tools=(),
        refusals=(),
        started=started,
        started_at=now_iso(),
    )


def make_report(
    *,
    survey_id: str,
    generation: int,
    parent_run_id: str,
    paths: tuple[str, ...],
    status: str,
    reason: str | None,
    narrative: str,
    run_id: str | None,
    tools: tuple[str, ...],
    refusals: tuple[str, ...],
    started: float,
    started_at: str,
    model_calls: int = 0,
    unsupported: tuple[str, ...] = (),
    completion_claim: str = "none",
) -> ChildReport:
    return ChildReport(
        survey_id=survey_id,
        generation=generation,
        status=status,
        reason=reason,
        narrative=narrative[:2000],
        run_id=run_id,
        parent_run_id=parent_run_id,
        paths=paths,
        tools=tools,
        refusals=refusals,
        unsupported=unsupported,
        completion_claim=completion_claim,
        model_calls=model_calls,
        started=started,
        ended=time.monotonic(),
        started_at=started_at,
        ended_at=now_iso(),
        kept=False,
    )


def tool_refusals(outcome: JsonObject) -> tuple[str, ...]:
    found: list[str] = []
    transcript = outcome.get("transcript")
    if not isinstance(transcript, list):
        return ()
    for turn in transcript:
        if not isinstance(turn, dict):
            continue
        results = turn.get("results")
        if not isinstance(results, list):
            continue
        for item in results:
            if not isinstance(item, dict):
                continue
            reason = item.get("reason")
            name = item.get("name")
            if reason in {"tool_not_authorized", "unknown_tool", "path_scope"} and isinstance(reason, str):
                tool_name = name if isinstance(name, str) else ""
                found.append(f"{tool_name}:{reason}")
    return tuple(found)


def _overlapped(reports: list[ChildReport]) -> bool:
    spans = [(item.started, item.ended) for item in reports if item.status != "rejected"]
    for index, left in enumerate(spans):
        for right in spans[index + 1 :]:
            if left[0] < right[1] and right[0] < left[1]:
                return True
    return False


def _survey_json(report: ChildReport) -> JsonObject:
    value = asdict(report)
    for key in ("paths", "tools", "refusals", "unsupported"):
        value[key] = list(value[key])
    return value


def _proof_json(proof: FileProof) -> JsonObject:
    return asdict(proof)


def render_report(
    generation: int,
    reports: list[ChildReport],
    proofs: tuple[FileProof, ...],
    *,
    child_write_detected: bool,
    joined: bool,
    parent_run_id: str,
    dropped: tuple[str, ...],
    peak: int,
    budget: JsonObject,
) -> JsonObject:
    lines = [
        "narrative_is_not_evidence: true",
        "failure_policy: continue_siblings",
        f"child_write_detected: {str(child_write_detected).lower()}",
        "以下摘录由宿主在子调查返回后回读。子任务的完成声明未核验。",
    ]
    for proof in proofs:
        lines.append(f"HOST {proof.path} sha256={proof.sha256 or ''} before={proof.before_sha256 or ''}")
        if proof.excerpt:
            lines.append(proof.excerpt)
        if proof.error:
            lines.append(f"HOST error: {proof.error}")
    for report in reports:
        lines.append(
            f"SURVEY {report.survey_id} status={report.status} claim={report.completion_claim} "
            f"reason={report.reason or ''}"
        )
        if report.narrative:
            lines.append("子任务叙述（未核验）：")
            lines.append(report.narrative)
        if report.unsupported:
            lines.append("unsupported: " + " | ".join(report.unsupported))
        if report.refusals:
            lines.append("refusals: " + " | ".join(report.refusals))
    reason = (
        "child_write_detected"
        if child_write_detected
        else "evidence_unavailable"
        if any(proof.error for proof in proofs)
        else None
    )
    return {
        "text": "\n".join(lines),
        "is_error": reason is not None,
        "executed": True,
        "reason": reason,
        "narrative_is_not_evidence": True,
        "generation": generation,
        "parent_run_id": parent_run_id,
        "failure_policy": "continue_siblings",
        "surveys": [_survey_json(report) for report in reports],
        "proofs": [_proof_json(proof) for proof in proofs],
        "dropped": list(dropped),
        "child_write_detected": child_write_detected,
        "peak_concurrency": peak,
        "overlapped": _overlapped(reports),
        "joined": joined,
        "budget": budget,
    }
