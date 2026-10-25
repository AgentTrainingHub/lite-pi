from __future__ import annotations

import json
from pathlib import Path

from support import Suite, cli, require_available

from lite_pi.session import AgentSession


def run(suite: Suite) -> None:
    task = "Given Python a=[1,2,3], what is a[0]-a[1]? Answer with the integer only."
    for index in range(3):
        work = suite.project(f"entry-{index}")
        run = suite.run(work, task, tools=(), max_tokens=128)
        api_pass = run["text"].strip() == "-1" and run["status"] == "stop" and not run["checked_files"]
        command = cli("submit", "--root", str(work), task)
        suite.record(
            f"entry-{index}",
            api_pass and command["text"].strip() == "-1" and command["returncode"] == 0,
            run_id=run["run_id"],
            cli=command,
            effect="entry",
        )
    work = suite.project("interrupt")
    run = AgentSession(work).interrupt(
        "Write a detailed 1000-word explanation of Python generators.", max_tokens=512
    )
    require_available(run)
    call = run["transcript"][0]["provider"]
    receipt = json.loads((Path(call["evidence"]) / "receipt.json").read_text())
    partial = (Path(call["evidence"]) / "response.bin").stat().st_size
    suite.record(
        "interrupt",
        run["status"] == "aborted" and receipt["error"] == "aborted" and run["usage_unknown"] and partial > 0,
        run_id=run["run_id"],
        partial_bytes=partial,
        receipt=receipt,
    )
