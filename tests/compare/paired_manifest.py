"""Validate the frozen experiment boundary before any model request."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from lite_pi.tools.contracts import JsonObject


@dataclass(frozen=True)
class TaskFixture:
    id: str
    module: str
    task: str
    source: str
    check: str


@dataclass(frozen=True)
class ComparisonManifest:
    payload: JsonObject
    tasks: tuple[TaskFixture, ...]
    arms: tuple[str, ...]
    repetitions: int
    model: str
    max_calls: int
    max_output_tokens: int
    per_call_output: int
    context_tokens: int
    max_turns: int
    prompt: str
    seed: int
    bootstrap_draws: int
    worst_cost: Decimal


def decode_manifest(raw: JsonObject) -> ComparisonManifest:
    def integer(name: str, minimum: int = 1) -> int:
        value = raw.get(name)
        if type(value) is not int or value < minimum:
            raise ValueError("invalid_manifest_" + name)
        return value

    tasks = []
    for item in raw.get("tasks", []):
        if not isinstance(item, dict) or any(
            not isinstance(item.get(k), str) or not item[k]
            for k in ("id", "module", "task", "source", "check")
        ):
            raise ValueError("invalid_manifest_task")
        if not re.fullmatch(r"[a-z][a-z0-9-]*", item["id"]) or not re.fullmatch(
            r"[a-z_]+\.py", item["module"]
        ):
            raise ValueError("invalid_task_identity_or_module")
        tasks.append(TaskFixture(item["id"], item["module"], item["task"], item["source"], item["check"]))
    if len(tasks) < 3 or len({t.id for t in tasks}) != len(tasks):
        raise ValueError("duplicate_or_missing_tasks")
    if raw.get("arms") != ["solo", "team"] or raw.get("model") != "deepseek-flash":
        raise ValueError("invalid_arms_or_model")
    reps = integer("repetitions", 3)
    calls, output, tokens, context = (
        integer("max_calls"),
        integer("max_output_tokens"),
        integer("per_call_output"),
        integer("context_tokens"),
    )
    if tokens != 512 or output != calls * tokens or context > 8192 or raw.get("pairs") != len(tasks) * reps:
        raise ValueError("invalid_shared_budget")
    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or prompt.count("{task}") != 1:
        raise ValueError("invalid_prompt_template")
    prompt.format(task="check")
    metrics = raw.get("metrics")
    if (
        not isinstance(metrics, dict)
        or type(metrics.get("seed")) is not int
        or type(metrics.get("bootstrap_draws")) is not int
        or metrics["bootstrap_draws"] < 100
    ):
        raise ValueError("invalid_statistics_manifest")
    cost = Decimal(2 * len(tasks) * reps * calls) * (
        Decimal(context) * Decimal(".000002") + Decimal(tokens) * Decimal(".000008")
    )
    if Decimal(raw["worst_pair_cost_cny"]) != cost:
        raise ValueError("invalid_cost_bound")
    return ComparisonManifest(
        raw,
        tuple(tasks),
        ("solo", "team"),
        reps,
        "deepseek-flash",
        calls,
        output,
        tokens,
        context,
        integer("max_turns"),
        prompt,
        metrics["seed"],
        metrics["bootstrap_draws"],
        cost,
    )
