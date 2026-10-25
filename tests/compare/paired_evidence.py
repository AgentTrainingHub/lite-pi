"""Rebuild comparison observations from run receipts, not the summary projection."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from compare.paired import Attempt, Observation, PairingError
from compare.paired_manifest import ComparisonManifest, TaskFixture
from lite_pi.session import CODE_TOOLS, RUNS
from lite_pi.spend.gate import EVIDENCE_ROOT
from lite_pi.tools.contracts import JsonObject


def load(directory: Path, name: str) -> JsonObject:
    value: JsonObject = json.loads((directory / name).read_text())
    return value


def conditions(task: TaskFixture, arm: str, manifest: ComparisonManifest) -> JsonObject:
    return {
        "source": task.source,
        "check": task.check,
        "module": task.module,
        "prompt": manifest.prompt.format(task=task.task),
        "tools": list(CODE_TOOLS + (("delegate",) if arm == "team" else ())),
        "max_calls": manifest.max_calls,
        "max_output_tokens": manifest.max_output_tokens,
        "per_call_output": manifest.per_call_output,
        "context_tokens": manifest.context_tokens,
        "model": manifest.model,
    }


def reconstruct(
    directory: Path, task: TaskFixture, arm: str, rep: int, manifest: ComparisonManifest
) -> Observation:
    expected = conditions(task, arm, manifest)
    if load(directory, "initial.json") != expected:
        raise PairingError("shared_conditions_changed")
    result = load(directory, "result.json")
    budget = load(directory, "budget.json")
    runtime = load(directory, "runtime.json")
    rows = json.loads((directory / "attempts.json").read_text())
    child_ids = tuple(
        sorted(
            {
                e["child_run_id"]
                for e in result["events"]
                if e.get("type") == "delegate_child" and e.get("child_run_id")
            }
        )
    )
    counts = []
    for child in child_ids:
        raw = load(RUNS / child, "result.json")
        if raw["run_id"] != child or raw.get("tools") != ["read"]:
            raise PairingError("child_receipt_mismatch")
        counts.append((child, int(raw.get("request_count", 0))))
    for row in rows:
        receipt = load(EVIDENCE_ROOT / row["reservation_id"], "receipt.json")
        request = load(EVIDENCE_ROOT / row["reservation_id"], "request.json")
        if (
            receipt["run_id"] != row["run_id"]
            or receipt["settled_cny"] != row["settled_cny"]
            or bool(row.get("usage_unknown")) != bool(receipt.get("usage_unknown"))
            or request["model"] != manifest.model
            or type(request["max_tokens"]) is not int
            or not 0 < request["max_tokens"] <= manifest.per_call_output
        ):
            raise PairingError("attempt_receipt_mismatch")
    if (
        budget["max_calls"] != manifest.max_calls
        or budget["max_output"] != manifest.max_output_tokens
        or sorted(result.get("tools", [])) != sorted(expected["tools"])
        or not any(
            m.get("role") == "user" and m.get("content") == expected["prompt"]
            for m in result["initial_request"]
        )
    ):
        raise PairingError("actual_conditions_changed")
    attempts = tuple(
        Attempt(
            r["run_id"],
            r["reservation_id"],
            None if r.get("usage_unknown") else Decimal(r["settled_cny"]),
            Decimal(r["settled_cny"]),
        )
        for r in rows
    )
    delegates = [o for t in result.get("transcript", []) for o in t["results"] if o["name"] == "delegate"]
    safe_children = all(
        not item.get("child_write_detected")
        and item.get("narrative_is_not_evidence") is True
        and all(s.get("tools") in ([], ["read"]) for s in item.get("surveys", []))
        for item in delegates
    )
    work = directory / "project"
    user_files = {
        str(p.relative_to(work))
        for p in work.rglob("*")
        if p.is_file() and ".lite-pi" not in p.parts and "__pycache__" not in p.parts
    }
    spend = load(directory, "spend.json")
    objective = (
        (work / "check.py").read_text() == task.check
        and user_files == {task.module, "check.py"}
        and safe_children
        and budget["reserved_calls"] == 0
        and budget["spent_calls"] + budget["unknown_calls"] <= manifest.max_calls
        and budget["spent_output"] + budget["unknown_output"] <= manifest.max_output_tokens
        and int(spend["after"]["request_count"]) - int(spend["before"]["request_count"]) == len(attempts)
        and all(r.get("actual_model") == manifest.model or r.get("usage_unknown") for r in rows)
    )
    checked = load(directory, "checked.json")
    record = Observation(
        task.id,
        rep,
        arm,
        result["run_id"],
        child_ids,
        attempts,
        budget["spent_calls"] + budget["unknown_calls"],
        budget["unknown_calls"],
        runtime["elapsed"],
        checked["returncode"] == 0 and result["status"] == "stop",
        bool(objective),
        str(directory),
        tuple(counts),
    )
    record.validate()
    return record
