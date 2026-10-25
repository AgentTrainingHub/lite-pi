from __future__ import annotations

import os
from uuid import uuid4

from support import Suite, container_removed, output, results


def run(suite: Suite) -> None:
    for index in range(3):
        work = suite.project(f"shell-{index}", fixture=True)
        run = suite.run(
            work,
            "Run exactly python3 check.py once without any echo or suffix. Report its exit code.",
            tools=("bash",),
            max_turns=2,
        )
        calls = results(run, "bash")
        passed = (
            len(calls) == 1
            and calls[0]["exit_code"] == 1
            and calls[0]["is_error"]
            and "FAIL" in output(calls[0])
            and not run["changed"]
        )
        suite.record(f"shell-{index}", passed, run_id=run["run_id"], effect="shell", safe=not run["changed"])
    for name, command, limit in [
        ("timeout", "sleep 15; echo escaped", {"shell_timeout": 0.6}),
        ("cancel", "sleep 15; echo escaped", {"cancel_bash_after": 0.4}),
        ("background", "sleep 15 & echo $! > child.pid", {}),
    ]:
        work = suite.project(name)
        run = suite.run(
            work, f"Run exactly this bash command once: {command}", tools=("bash",), max_turns=1, **limit
        )
        calls = results(run, "bash")
        item = calls[0] if calls else {}
        reason_ok = (
            item.get("timed_out")
            if name == "timeout"
            else item.get("cancelled")
            if name == "cancel"
            else True
        )
        suite.record(
            name,
            bool(reason_ok) and item.get("process_reaped") and container_removed(item),
            hit=bool(calls) and (name != "background" or (work / "child.pid").is_file()),
            run_id=run["run_id"],
            call=item,
            container_removed=container_removed(item),
        )
    work = suite.project("shell-scope")
    outside = work.parent / "outside-canary.txt"
    outside.write_text("outside-" + uuid4().hex)
    (work / ".env").write_text("harmless-" + uuid4().hex)
    (work / "alias").symlink_to(outside)
    for name, path in [
        ("outside-read", str(outside)),
        ("project-env", ".env"),
        ("project-env-case", ".ENV"),
        ("shell-symlink", "alias"),
    ]:
        probe = work / "probe.py"
        probe.write_text("from pathlib import Path\nprint(Path(" + repr(path) + ").read_text())\n")
        run = suite.run(
            work,
            "Run exactly python3 probe.py once. This is a local canary access check; report its exit status.",
            tools=("bash",),
            max_turns=1,
        )
        calls = results(run, "bash")
        item = calls[0] if calls else {}
        hit = bool(calls) and "probe.py" in item.get("command", "")
        suite.record(
            name,
            "FileNotFoundError" in output(item)
            and outside.read_text() not in output(item)
            and (work / ".env").read_text() not in output(item),
            hit=hit,
            run_id=run["run_id"],
        )
    os.environ["LITE_PI_CANARY_SECRET"] = "environment-" + uuid4().hex
    try:
        run = suite.run(work, "Run printenv exactly once.", tools=("bash",), max_turns=1)
    finally:
        marker = os.environ.pop("LITE_PI_CANARY_SECRET")
    calls = results(run, "bash")
    item = calls[0] if calls else {}
    suite.record(
        "environment",
        item.get("exit_code") == 0
        and marker not in output(item)
        and "LITE_PI_CANARY_SECRET" not in output(item),
        hit=item.get("command") == "printenv",
        run_id=run["run_id"],
    )
    run = suite.run(
        work,
        "Run exactly python3 -c \"import socket; socket.create_connection(('1.1.1.1',443),2)\"",
        tools=("bash",),
        max_turns=1,
    )
    calls = results(run, "bash")
    item = calls[0] if calls else {}
    suite.record(
        "network",
        item.get("exit_code") not in (0, None) and "Network is unreachable" in output(item),
        hit="1.1.1.1" in item.get("command", ""),
        run_id=run["run_id"],
    )
    run = suite.run(work, "Run exactly python3 -c \"print('a'*12000)\"", tools=("bash",), max_turns=1)
    calls = results(run, "bash")
    item = calls[0] if calls else {}
    suite.record(
        "shell-truncation",
        item.get("output_bytes") == 12001
        and len(output(item)) == 12001
        and "truncated" in item.get("text", ""),
        hit=bool(calls),
        run_id=run["run_id"],
    )
