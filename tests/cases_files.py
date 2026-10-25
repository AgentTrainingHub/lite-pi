from __future__ import annotations

from support import Suite, results


def run(suite: Suite) -> None:
    for name, content, instruction, expected, error_text in [
        (
            "crlf",
            b"\xef\xbb\xbfhead\r\nold\r\ntail\n",
            "Replace the word old with new in sample.txt.",
            b"\xef\xbb\xbfhead\r\nnew\r\ntail\n",
            None,
        ),
        (
            "ambiguous",
            b"old old\n",
            "Replace old with new in sample.txt in one exact edit.",
            b"old old\n",
            "matched 2",
        ),
        (
            "overlap",
            b"abcdef\n",
            'Edit sample.txt with edits=[{"old":"abc","new":"Z"},{"old":"cde","new":"Q"}] in one call.',
            b"abcdef\n",
            "overlap",
        ),
        (
            "nested-invalid",
            b"old\n",
            "Call edit on sample.txt with edits=[null]. Keep this invalid payload for a schema test.",
            b"old\n",
            "Validation failed",
        ),
    ]:
        work = suite.project(name)
        target = work / "sample.txt"
        target.write_bytes(content)
        target.chmod(0o750)
        run = suite.run(work, instruction, tools=("edit",), max_turns=1)
        calls = results(run, "edit")
        hit = bool(calls) and (error_text is None or any(error_text in item["text"] for item in calls))
        suite.record(
            name,
            target.read_bytes() == expected and target.stat().st_mode & 0o777 == 0o750,
            hit=hit,
            run_id=run["run_id"],
        )
    work = suite.project("file-boundaries")
    outside = work.parent / "outside.txt"
    outside.write_text("keep\n")
    (work / "link.txt").symlink_to(outside)
    for name, path in (("path-escape", str(outside)), ("symlink", "link.txt")):
        run = suite.run(
            work, f"Use edit to replace keep with changed in {path}.", tools=("edit",), max_turns=1
        )
        calls = results(run, "edit")
        suite.record(
            name,
            outside.read_text() == "keep\n" and all(item["is_error"] for item in calls),
            hit=any(
                item.get("path") == path
                and ("escapes the project" in item["text"] or "Too many levels" in item["text"])
                for item in calls
            ),
            run_id=run["run_id"],
        )
    run = suite.run(work, "Use write to create new.txt containing first.", tools=("write",), max_turns=1)
    target = work / "new.txt"
    created = target.read_bytes() if target.exists() else None
    again = suite.run(
        work,
        "Use write to write second to new.txt without the overwrite argument.",
        tools=("write",),
        max_turns=1,
    )
    suite.record(
        "write-no-overwrite",
        created is not None
        and target.read_bytes() == created
        and any(item["is_error"] for item in results(again)),
        hit=bool(results(again)),
        run_id=run["run_id"],
        again=again["run_id"],
    )
    work = suite.project("object-change")
    target = work / "sample.txt"
    target.write_text("old\n")
    changed = []

    def mutate(event):
        if event["type"] == "tool_result" and event["result"]["name"] == "read" and not changed:
            target.write_text("old externally_changed\n")
            changed.append(True)

    run = suite.run(
        work,
        "Read sample.txt, then use edit to replace old with new exactly once. Stop if edit fails.",
        tools=("read", "edit"),
        max_turns=3,
        on_event=mutate,
    )
    edits = results(run, "edit")
    hit = any("changed since read" in item["text"] for item in edits)
    suite.record(
        "object-change", target.read_text() == "old externally_changed\n", hit=hit, run_id=run["run_id"]
    )
    run = suite.run(
        work, "Read sample.txt before answering.", tools=("read",), max_model_calls=1, max_turns=3
    )
    suite.record(
        "model-call-budget",
        run["status"] == "budget" and len(run["transcript"]) == 1,
        hit=bool(results(run)),
        run_id=run["run_id"],
    )
