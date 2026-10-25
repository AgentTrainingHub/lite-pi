from __future__ import annotations

from support import FEATURE, FIX, Suite, check, output, results


def run(suite: Suite) -> None:
    for index in range(3):
        work = suite.project(f"fix-{index}", fixture=True)
        run = suite.run(work, FIX, max_turns=9, max_tokens=768)
        verification = check(work)
        items = results(run)
        fail_at = next(
            (i for i, item in enumerate(items) if item["name"] == "bash" and "FAIL" in output(item)), -1
        )
        edit_at = next(
            (
                i
                for i, item in enumerate(items)
                if i > fail_at >= 0 and item["name"] in {"edit", "write"} and not item["is_error"]
            ),
            -1,
        )
        recheck = any(
            i > edit_at >= 0
            and item["name"] == "bash"
            and "OK" in output(item)
            and "FAIL" not in output(item)
            for i, item in enumerate(items)
        )
        diff_ok = run["diffs_match"]
        suite.record(
            f"fix-{index}",
            verification["returncode"] == 0
            and bool(run["changed"])
            and all(p.startswith("weightnote/") for p in run["changed"])
            and recheck
            and diff_ok,
            run_id=run["run_id"],
            effect="fix",
            safe=all(p.startswith("weightnote/") for p in run["changed"]),
            changed=run["changed"],
            check=verification,
            failure_then_edit=recheck,
            diff_matches=diff_ok,
        )
        work = suite.project(f"feature-{index}", fixture=True)
        target = work / "weightnote/invoice.py"
        target.write_text(
            target.read_text()
            .replace("amount, _unit", "amount, unit")
            .replace('to_grams(amount, "g")', "to_grams(amount, unit)")
        )
        run = suite.run(work, FEATURE, max_turns=8, max_tokens=768)
        verification = check(work)
        functionality = check(
            work,
            "from weightnote.report import line_count; assert line_count(['200 g','1 kg']) == 2; assert line_count([]) == 0",
        )
        suite.record(
            f"feature-{index}",
            verification["returncode"] == 0
            and functionality["returncode"] == 0
            and run["changed"] == ["weightnote/report.py"],
            run_id=run["run_id"],
            effect="feature",
            safe=all(p == "weightnote/report.py" for p in run["changed"]),
            changed=run["changed"],
            check=verification,
            functionality=functionality,
        )
