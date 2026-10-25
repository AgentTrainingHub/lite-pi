from __future__ import annotations

import json
from pathlib import Path

from support import FIX, Suite, check, cli, results, save, snapshot

from lite_pi.journal import Journal


def run(suite: Suite) -> None:
    for index in range(3):
        work = suite.project(f"resume-{index}", fixture=True)
        baseline = snapshot(work)
        journal_path = work.parent / "journal.jsonl"
        first = cli("code", "--root", str(work), "--session", str(journal_path), "--max-turns", "1", FIX)
        initial = journal_path.read_bytes()
        run = cli("resume", "--session", str(journal_path), "--max-turns", "8")
        second = json.loads((Path(run["evidence"]) / "result.json").read_text())
        initial_records = [json.loads(line) for line in initial.splitlines()]
        source_ids = {item["id"] for item in initial_records if item["kind"] != "header"}
        resumed_snapshot = snapshot(work)
        save(work.parent / "resume-snapshots.json", {"before": baseline, "after": resumed_snapshot})
        changed = [
            name
            for name in baseline.keys() | resumed_snapshot.keys()
            if baseline.get(name) != resumed_snapshot.get(name)
        ]
        old_call_ids = {
            call["id"] for record in initial_records for call in record["body"].get("tool_calls", [])
        }
        new_results = results(second)
        duplicate_execution = any(item["tool_call_id"] in old_call_ids for item in new_results)
        suite.record(
            f"resume-{index}",
            check(work)["returncode"] == 0
            and journal_path.read_bytes().startswith(initial)
            and source_ids == set(second["source_ids"]),
            run_id=run["run_id"],
            first=first,
            effect="resume",
            safe=all(name.startswith("weightnote/") for name in changed) and not duplicate_execution,
            changed=changed,
            duplicate_execution=duplicate_execution,
        )
        journal = Journal(journal_path)
        leaf = journal.leaf_id()
        header = journal.records[0]["id"]
        disk = snapshot(work)
        branch = cli(
            "branch",
            "--session",
            str(journal_path),
            "--from",
            header,
            "Use write to create beta.txt containing BRANCH_BETA_ONLY.",
        )
        switched = cli("resume", "--session", str(journal_path), "--max-turns", "3")
        beta = json.loads((Path(switched["evidence"]) / "result.json").read_text())
        before_switch = snapshot(work)
        branch_safe = {
            name for name in disk.keys() | before_switch.keys() if disk.get(name) != before_switch.get(name)
        } == {"beta.txt"} and (work / "beta.txt").read_text().strip() == "BRANCH_BETA_ONLY"
        alpha_run = cli("resume", "--session", str(journal_path), "--leaf", leaf, "--max-turns", "1")
        alpha = json.loads((Path(alpha_run["evidence"]) / "result.json").read_text())
        suite.record(
            f"branch-{index}",
            branch["status"] == "branched"
            and branch_safe
            and FIX not in json.dumps(beta["initial_request"])
            and "BRANCH_BETA_ONLY" not in json.dumps(alpha["initial_request"])
            and before_switch == snapshot(work)
            and all(snapshot(work).get(p) == v for p, v in disk.items()),
            run_id=switched["run_id"],
            alpha_run_id=alpha_run["run_id"],
            effect="branch",
            safe=branch_safe
            and FIX not in json.dumps(beta["initial_request"])
            and "BRANCH_BETA_ONLY" not in json.dumps(alpha["initial_request"])
            and before_switch == snapshot(work)
            and all(snapshot(work).get(p) == v for p, v in disk.items()),
        )
        journal = Journal(journal_path)
        invalid_parent = next(
            (
                item["id"]
                for item in journal.records
                if item["kind"] == "assistant" and item["body"].get("tool_calls")
            ),
            None,
        )
        if invalid_parent:
            before = (journal_path.read_bytes(), journal.leaf_path.read_bytes())
            rejected = cli(
                "branch", "--session", str(journal_path), "--from", invalid_parent, "unsafe branch"
            )
            suite.record(
                f"partial-branch-{index}",
                rejected.get("reason") == "unpaired_tool_call"
                and before == (journal_path.read_bytes(), journal.leaf_path.read_bytes()),
                run_id=run["run_id"],
                rejection=rejected,
            )
        for kind in ("tail", "version", "parent", "duplicate", "kind"):
            bad = work.parent / (kind + ".jsonl")
            records = [json.loads(line) for line in journal_path.read_text().splitlines()]
            if kind == "kind":
                records[-1]["kind"] = []
            if kind == "version":
                records[-1]["version"] = 999
            if kind == "parent":
                records[-1]["parent_id"] = "absent"
            if kind == "duplicate":
                records[-1]["id"] = records[0]["id"]
            bad.write_text(
                "".join(json.dumps(item) + "\n" for item in records) + ("{" if kind == "tail" else "")
            )
            bad.with_suffix(".jsonl.leaf").write_bytes(journal.leaf_path.read_bytes())
            before = bad.read_bytes()
            rejected = cli("resume", "--session", str(bad))
            suite.record(
                f"{kind}-{index}",
                rejected["status"] == "rejected" and bad.read_bytes() == before,
                run_id=run["run_id"],
                rejection=rejected,
            )
