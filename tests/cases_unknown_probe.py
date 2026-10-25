"""Optional live-model unknown-tool probe; never counted as deterministic coverage."""

from support import Suite, results


def run(suite: Suite) -> None:
    work = suite.project("unknown-probe")
    run = suite.run(
        work,
        "Call the unregistered tool absent_tool with empty arguments; this protocol test requires that exact tool name.",
        tools=("read",),
        max_turns=1,
    )
    calls = results(run)
    hit = any(item["name"] == "absent_tool" for item in calls)
    suite.record(
        "unknown-tool",
        hit and all(not item["executed"] for item in calls) and not run["changed"],
        hit=hit,
        run_id=run["run_id"],
    )
