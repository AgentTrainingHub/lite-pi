"""Task-level paired summaries. Unknown fees and missing observations stay explicit."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from decimal import Decimal
from statistics import mean

from lite_pi.tools.contracts import JsonObject


class PairingError(ValueError):
    pass


@dataclass(frozen=True)
class Attempt:
    run_id: str
    reservation_id: str
    known_cost: Decimal | None
    charged_cost: Decimal


@dataclass(frozen=True)
class Observation:
    task: str
    repetition: int
    arm: str
    parent_run_id: str
    child_run_ids: tuple[str, ...]
    attempts: tuple[Attempt, ...]
    budget_calls: int
    unknown_calls: int
    elapsed: float
    success: bool
    objective_valid: bool
    evidence: str
    child_call_counts: tuple[tuple[str, int], ...] = ()

    def validate(self) -> None:
        ids = (self.parent_run_id, *self.child_run_ids)
        if not all(ids) or len(set(ids)) != len(ids):
            raise PairingError("duplicate_run_identity")
        if len(self.attempts) != self.budget_calls:
            raise PairingError("omitted_attempts_or_child_calls")
        if sum(a.known_cost is None for a in self.attempts) != self.unknown_calls:
            raise PairingError("unknown_usage_mismatch")
        if any(a.run_id not in ids for a in self.attempts):
            raise PairingError("unattributed_attempt")
        child_counts = dict(self.child_call_counts)
        if set(child_counts) != set(self.child_run_ids) or len(child_counts) != len(self.child_call_counts):
            raise PairingError("child_receipt_missing")
        if any(
            count != sum(a.run_id == child for a in self.attempts) for child, count in child_counts.items()
        ):
            raise PairingError("child_attempt_count_mismatch")
        if not math.isfinite(self.elapsed) or self.elapsed < 0:
            raise PairingError("invalid_elapsed")
        if any(a.charged_cost < 0 or (a.known_cost is not None and a.known_cost < 0) for a in self.attempts):
            raise PairingError("invalid_cost")


def interval(values: list[float], *, seed: int, draws: int) -> JsonObject:
    rng = random.Random(seed)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return {
        "mean": mean(values),
        "interval95": [samples[int((draws - 1) * 0.025)], samples[int((draws - 1) * 0.975)]],
        "task_differences": values,
    }


def summarize(
    records: list[Observation],
    tasks: tuple[str, ...],
    repetitions: int,
    *,
    seed: int = 1804,
    draws: int = 5000,
) -> JsonObject:
    if len(set(tasks)) != len(tasks) or len(tasks) < 3 or repetitions < 3 or draws < 100:
        raise PairingError("invalid_manifest")
    expected = {(task, rep, arm) for task in tasks for rep in range(repetitions) for arm in ("solo", "team")}
    indexed: dict[tuple[str, int, str], Observation] = {}
    runs: set[str] = set()
    reservations: set[str] = set()
    for record in records:
        record.validate()
        key = record.task, record.repetition, record.arm
        ids = {record.parent_run_id, *record.child_run_ids}
        receipts = {a.reservation_id for a in record.attempts}
        if key in indexed or runs & ids or reservations & receipts or len(receipts) != len(record.attempts):
            raise PairingError("duplicate_identity")
        if key not in expected:
            raise PairingError("unexpected_identity")
        indexed[key] = record
        runs.update(ids)
        reservations.update(receipts)
    if indexed.keys() != expected:
        raise PairingError(f"missing_pairs:{len(expected - indexed.keys())}")
    unknown = sum(r.unknown_calls for r in records)
    metrics: JsonObject = {}
    for metric in ("success", "calls", "elapsed", "known_cost"):
        if metric == "known_cost" and unknown:
            metrics[metric] = None
            continue
        differences = []
        for task in tasks:
            task_pairs = []
            for rep in range(repetitions):
                values = []
                for arm in ("solo", "team"):
                    r = indexed[(task, rep, arm)]
                    value = {
                        "success": float(r.success and r.objective_valid),
                        "calls": float(len(r.attempts)),
                        "elapsed": r.elapsed,
                        "known_cost": float(
                            sum((a.known_cost or Decimal(0) for a in r.attempts), Decimal(0))
                        ),
                    }[metric]
                    values.append(value)
                task_pairs.append(values[1] - values[0])
            differences.append(mean(task_pairs))
        metrics[metric] = interval(differences, seed=seed, draws=draws)
    return {
        "status": "complete",
        "pairs": len(tasks) * repetitions,
        "tasks": list(tasks),
        "repetitions": repetitions,
        "unit": "task mean of paired repetitions",
        "direction": "team minus solo",
        "seed": seed,
        "bootstrap_draws": draws,
        "metrics": metrics,
        "unknown_usage_calls": unknown,
        "charged_total_cny": str(sum((a.charged_cost for r in records for a in r.attempts), Decimal(0))),
        "limitation": "three selected tasks; repeats are not independent tasks; no population superiority claim",
    }
