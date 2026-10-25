"""Observe real tool-error feedback and a subsequent successful model action."""

import json
from pathlib import Path
from uuid import uuid4

from support import Suite, results


def run(suite: Suite) -> None:
    for index in range(3):
        work = suite.project(f"tool-recovery-{index}")
        marker = "RECOVERED_" + uuid4().hex
        (work / "available.txt").write_text(marker + "\n")
        run = suite.run(
            work,
            "First read missing.txt. If that fails, read available.txt and report its exact content. "
            "Make the second read only after receiving the first result.",
            tools=("read",),
            max_turns=4,
            max_model_calls=4,
        )
        calls = results(run)
        errors = [item for item in calls if item["is_error"] and "FileNotFoundError" in item["text"]]
        feedback_sent = False
        for turn in run["transcript"][1:]:
            request_path = Path(turn["provider"]["evidence"]) / "request.json"
            payload = json.loads(request_path.read_text())
            feedback_sent |= any(
                message.get("role") == "tool"
                and message.get("tool_call_id") == error["tool_call_id"]
                and message.get("content") == error["text"]
                for message in payload["messages"]
                for error in errors
            )
        recovered = any(not item["is_error"] and marker in item["text"] for item in calls)
        suite.record(
            f"tool-error-recovery-{index}",
            bool(errors)
            and feedback_sent
            and recovered
            and run["status"] == "stop"
            and marker in run["text"]
            and not run["changed"],
            hit=bool(errors),
            run_id=run["run_id"],
            feedback_sent=feedback_sent,
            safe=not run["changed"],
        )
