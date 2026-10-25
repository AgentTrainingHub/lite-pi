"""Execute the frozen public-Session comparison; recompute summaries without new calls."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

from support import check, save

from compare.evidence_failures import record_failure
from compare.paired import Observation, PairingError, summarize
from compare.paired_evidence import conditions, reconstruct
from compare.paired_manifest import ComparisonManifest, TaskFixture, decode_manifest
from lite_pi.journal import Journal
from lite_pi.session import CODE_TOOLS, AgentSession
from lite_pi.spend.ledger import Ledger
from lite_pi.tools.contracts import JsonObject

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "fixtures/evaluation/paired-manifest.json"
PROMPT = (
    "修复目标模块，让 check.py 通过；禁止修改 check.py 或其他文件。先读取现行文件。"
    "如果工具清单提供 delegate，修改前安排一次只读调查；调查后父任务还要亲自读取并核对实际文件。"
    "没有 delegate 时直接调查。禁止用 shell 读源码；允许运行 python3 check.py。"
    "不要提交或联网。需求：{task}"
)


def encode_observation(record: Observation) -> JsonObject:
    value = asdict(record)
    value["attempts"] = [
        {
            **asdict(a),
            "known_cost": str(a.known_cost) if a.known_cost is not None else None,
            "charged_cost": str(a.charged_cost),
        }
        for a in record.attempts
    ]
    return value


def run_arm(
    directory: Path, task: TaskFixture, arm: str, repetition: int, manifest: ComparisonManifest
) -> Observation:
    directory.mkdir()
    work = directory / "project"
    work.mkdir()
    module = task.module
    (work / module).write_text(task.source)
    (work / "check.py").write_text(task.check)
    tools = CODE_TOOLS + (("delegate",) if arm == "team" else ())
    journal = Journal.create(directory / "session.jsonl", work, tools=tools)
    events: list[JsonObject] = []
    ledger = Ledger()
    before = ledger.require_status()
    offset = len(ledger.calls_path.read_text().splitlines()) if ledger.calls_path.exists() else 0
    save(directory / "initial.json", conditions(task, arm, manifest))
    start = time.monotonic()
    try:
        result = AgentSession(work, tools=tools, journal=journal).submit(
            manifest.prompt.format(task=task.task),
            on_event=events.append,
            max_turns=manifest.max_turns,
            max_model_calls=manifest.max_calls,
            max_task_calls=manifest.max_calls,
            max_task_output_tokens=manifest.max_output_tokens,
            max_tokens=manifest.per_call_output,
            context_tokens=manifest.context_tokens,
        )
        save(directory / "result.json", result)
        save(directory / "checked.json", check(work))
    except BaseException as error:
        record_failure(directory / "failure.json", error)
        raise
    finally:
        original = sys.exception()
        try:
            save(directory / "runtime.json", {"elapsed": time.monotonic() - start})
            save(directory / "observed-events.json", events)
            rows = (
                [json.loads(line) for line in ledger.calls_path.read_text().splitlines()[offset:]]
                if ledger.calls_path.exists()
                else []
            )
            save(directory / "attempts.json", rows)
            save(directory / "spend.json", {"before": before, "after": ledger.require_status()})
            save(directory / "budget.json", AgentSession.open(journal.path).inspect()["budget"])
            ids = {e["run_id"] for e in events if e.get("run_id")}
            ids.update(e["child_run_id"] for e in events if e.get("child_run_id"))
            if any(row["run_id"] not in ids for row in rows):
                raise PairingError("unattributed_ledger_delta")
        except Exception as capture_error:
            record_failure(directory / "capture-failure.json", capture_error, primary=original)
            if original is None:
                raise
    return reconstruct(directory, task, arm, repetition, manifest)


def evaluate(directory: Path, manifest: ComparisonManifest) -> JsonObject:
    projections = json.loads((directory / "observations.json").read_text())
    records = []
    try:
        tasks = {task.id: task for task in manifest.tasks}
        for raw in projections:
            if (
                raw.get("task") not in tasks
                or raw.get("arm") not in manifest.arms
                or type(raw.get("repetition")) is not int
                or raw["repetition"] not in range(manifest.repetitions)
            ):
                raise PairingError("unexpected_identity")
            expected_path = directory / f"{raw['task']}-{raw['repetition']}-{raw['arm']}"
            if raw["evidence"] != str(expected_path):
                raise PairingError("evidence_identity_mismatch")
            record = reconstruct(expected_path, tasks[raw["task"]], raw["arm"], raw["repetition"], manifest)
            if json.loads(json.dumps(encode_observation(record))) != raw:
                raise PairingError("observation_projection_mismatch")
            records.append(record)
        report = summarize(
            records,
            tuple(t.id for t in manifest.tasks),
            manifest.repetitions,
            seed=manifest.seed,
            draws=manifest.bootstrap_draws,
        )
    except PairingError as error:
        report = {"status": "incomplete", "error": str(error), "headline": None}
    report["observations"] = [encode_observation(r) for r in records]
    report["ablation"] = manifest.payload["ablation"]
    report["delegate_used"] = {r.evidence: bool(r.child_run_ids) for r in records}
    report["ablation_hit"] = all(bool(r.child_run_ids) == (r.arm == "team") for r in records)
    save(directory / "summary.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    if args.recompute:
        manifest = decode_manifest(json.loads((args.directory / "manifest.json").read_text()))
    else:
        manifest = decode_manifest(json.loads(args.manifest.read_text()))
        ledger = Ledger().require_status()
        available = (
            Decimal(ledger["allowance_cny"]) - Decimal(ledger["spent_cny"]) - Decimal(ledger["reserved_cny"])
        )
        if available < manifest.worst_cost or ledger["in_flight"]:
            raise RuntimeError("budget preflight blocked")
        args.directory.mkdir(parents=True)
        save(args.directory / "manifest.json", manifest.payload)
        records: list[JsonObject] = []
        save(args.directory / "observations.json", [])
        for task in manifest.tasks:
            for rep in range(manifest.repetitions):
                for arm in manifest.arms:
                    try:
                        record = run_arm(args.directory / f"{task.id}-{rep}-{arm}", task, arm, rep, manifest)
                    except Exception as error:
                        record_failure(
                            args.directory / "summary.json",
                            error,
                            details={
                                "status": "incomplete",
                                "failed_identity": [task.id, rep, arm],
                                "headline": None,
                                "completed_arms": len(records),
                                "expected_arms": len(manifest.tasks) * manifest.repetitions * 2,
                            },
                        )
                        raise
                    records.append(encode_observation(record))
                    save(args.directory / "observations.json", records)
                    print(
                        json.dumps(
                            {
                                "task": task.id,
                                "rep": rep,
                                "arm": arm,
                                "success": record.success,
                                "safe": record.objective_valid,
                                "calls": len(record.attempts),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    report = evaluate(args.directory, manifest)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("observations", "delegate_used")},
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "complete" and report["ablation_hit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
