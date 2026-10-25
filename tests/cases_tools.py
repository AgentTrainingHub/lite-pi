from __future__ import annotations

from support import Suite, results


def run(suite: Suite) -> None:
    for index in range(3):
        work = suite.project(f"read-{index}", fixture=True)
        run = suite.run(
            work,
            "Read the code to explain why python3 check.py fails on kilogram lines. Name the file and function that drop the unit.",
            tools=("read",),
            max_turns=6,
        )
        suite.record(
            f"read-{index}",
            not run["changed"]
            and "invoice.py" in run["text"]
            and "to_grams" in run["text"]
            and bool(results(run, "read"))
            and len(run["transcript"]) > 1,
            run_id=run["run_id"],
            effect="read",
            safe=not run["changed"],
        )
    work = suite.project("read-boundaries")
    data = "甲" * 35 + "END_SENTINEL\nlast\n"
    (work / "long.txt").write_text(data)
    run = suite.run(
        work,
        "Read long.txt completely, continuing at every returned cursor, then quote the final line.",
        tools=("read",),
        max_bytes=51,
        max_turns=7,
    )
    reads = results(run, "read")
    reconstructed = "".join(item.get("content", "") for item in reads)
    suite.record(
        "long-line",
        reconstructed == data and not run["changed"],
        hit=any(item.get("truncation") for item in reads),
        run_id=run["run_id"],
    )
    run = suite.run(work, "Read missing.txt and quote the file error.", tools=("read",), max_turns=2)
    calls = results(run, "read")
    suite.record(
        "missing-file",
        bool(calls) and calls[0]["is_error"] and not run["changed"],
        hit=bool(calls),
        run_id=run["run_id"],
    )
    run = suite.run(
        work,
        "Call read with path=long.txt, offset=-4 and limit=-1. This is an invalid-parameter boundary test; keep both negative values.",
        tools=("read",),
        max_turns=1,
    )
    calls = results(run, "read")
    hit = any("Validation failed" in item["text"] for item in calls)
    suite.record("invalid-parameter", hit and not run["changed"], hit=hit, run_id=run["run_id"])
    run = suite.run(
        work,
        "Call write to create huge.txt containing 3000 different words. Put all content in one tool call.",
        tools=("write",),
        max_turns=1,
        max_tokens=64,
    )
    calls = results(run)
    hit = run["raw_stop_reason"] == "length" and bool(calls)
    suite.record(
        "truncated-call",
        hit and all(not item["executed"] for item in calls) and not run["changed"],
        hit=hit,
        run_id=run["run_id"],
    )
    run = suite.run(work, "Read long.txt before answering.", tools=("read",), max_turns=1)
    suite.record(
        "turn-limit",
        run["status"] == "turn_limit" and len(run["transcript"]) == 1 and not run["changed"],
        hit=bool(results(run)),
        run_id=run["run_id"],
    )
