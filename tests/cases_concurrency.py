"""Two live session processes hold read slots before making real DeepSeek requests."""

import os
import subprocess
import sys
import time

from cases_recovery import _observed
from support import ROOT, Suite, save

from lite_pi.journal import Journal
from lite_pi.session import AgentSession


def run(suite: Suite) -> None:
    work = suite.project("parallel-readers")
    children = []
    release = work.parent / "release"
    try:
        for index in range(2):
            ready = work.parent / f"ready-{index}"
            journal = work.parent / f"reader-{index}.jsonl"
            script = f"""
import json, time
from pathlib import Path
from lite_pi.journal import Journal
from lite_pi.session import AgentSession
journal = Journal.create(Path({str(journal)!r}), Path({str(work)!r}), tools=('read',))
def observe(event):
    if event['type']=='run_start':
        Path({str(ready)!r}).write_text('holding')
        deadline=time.monotonic()+45
        while not Path({str(release)!r}).exists():
            if time.monotonic()>deadline: raise TimeoutError('test synchronization deadline')
            time.sleep(.02)
result=AgentSession.open(journal.path).submit('Reply exactly LIVE-READER. Do not call tools.', max_turns=1, max_tokens=64, on_event=observe)
print(json.dumps(result))
"""
            child = subprocess.Popen(
                [sys.executable, "-c", script],
                cwd=ROOT,
                env={"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"]},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            children.append(child)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not all(
            (work.parent / f"ready-{i}").exists() for i in range(2)
        ):
            time.sleep(0.02)
        hit = all(child.poll() is None for child in children) and all(
            (work.parent / f"ready-{i}").exists() for i in range(2)
        )
        reader = AgentSession.open(Journal.create(work.parent / "third.jsonl", work, tools=("read",)).path)
        third = reader.submit("Reply THIRD.", max_turns=1, max_tokens=64)
        writer = AgentSession.open(Journal.create(work.parent / "writer.jsonl", work, tools=("write",)).path)
        rejected_writer = writer.submit("Reply WRITER.", max_turns=1, max_tokens=64)
        cancelled = reader.dismiss(third["tasks"]["workers"][-1]["worker_id"])
        release.write_text("continue")
        processes = []
        for child in children:
            out, err = child.communicate(timeout=90)
            processes.append({"code": child.returncode, "result": _observed(out), "stderr": err})
        passed = (
            hit
            and third.get("reason") == "read_limit"
            and rejected_writer.get("reason") == "write_busy"
            and third["request_count"] == rejected_writer["request_count"] == 0
            and cancelled["state"] == "cancelled"
            and all(
                p["code"] == 0 and p["result"]["status"] == "stop" and p["result"]["request_count"] >= 1
                for p in processes
            )
        )
        save(
            work.parent / "results.json",
            {"readers": processes, "third": third, "writer": rejected_writer, "cancelled": cancelled},
        )
        suite.record(
            "recovery-public-process-concurrency",
            bool(passed),
            hit=hit or any(p["code"] != 0 for p in processes),
            safe=True,
        )
    finally:
        release.write_text("continue")
        for child in children:
            if child.poll() is None:
                child.kill()
                child.communicate()
