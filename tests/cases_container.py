"""Real-model regressions for container isolation and host effect publication."""

from __future__ import annotations

import os
import threading
import time

from support import Suite, container_removed, results

from lite_pi.session import RUNS


def run(suite: Suite) -> None:
    work = suite.project("closed-stdio")
    run = suite.run(
        work,
        "Run exactly: exec 1>&- 2>&-; sleep 0.2; printf finished > finished.txt",
        tools=("bash",),
        max_turns=1,
    )
    calls = results(run, "bash")
    suite.record(
        "closed-stdio",
        (work / "finished.txt").is_file()
        and (work / "finished.txt").read_text() == "finished"
        and calls[0]["exit_code"] == 0,
        hit=bool(calls),
        run_id=run["run_id"],
    )

    work = suite.project("detached-descendant")
    probe = work / "detach.py"
    probe.write_text(
        'import os,time\npid=os.fork()\nif pid:\n time.sleep(0.2)\n os._exit(0)\nos.setsid()\npid=os.fork()\nif pid:\n os._exit(0)\nopen("detached.pid","w").write(str(os.getpid()))\nos.close(1)\nos.close(2)\ntime.sleep(15)\n'
    )
    run = suite.run(
        work,
        "Run python3 detach.py once to verify cleanup of a short-lived worker. The bash tool uses a disposable Docker container. The script forks twice, writes detached.pid, closes output and sleeps briefly. Complete the execution after inspecting it.",
        tools=("bash",),
        max_turns=6,
    )
    calls = results(run, "bash")
    item = next((item for item in calls if "python3 detach.py" in item.get("command", "")), {})
    suite.record(
        "detached-descendant",
        item.get("process_reaped") and container_removed(item),
        hit=(work / "detached.pid").exists(),
        run_id=run["run_id"],
        call=item,
        container_removed=container_removed(item),
    )

    work = suite.project("shell-hardlink")
    (work / ".env").write_text("hardlink-canary-not-a-secret")
    os.link(work / ".env", work / "alias.txt")
    run = suite.run(work, "Run exactly cat alias.txt once.", tools=("bash",), max_turns=1)
    calls = results(run, "bash")
    suite.record(
        "shell-hardlink",
        bool(calls) and all(not item["executed"] and "single-link" in item["text"] for item in calls),
        hit=bool(calls),
        run_id=run["run_id"],
    )

    work = suite.project("shell-publish")
    (work / "old.txt").write_text("old")
    run = suite.run(
        work,
        "Run exactly: rm old.txt; mkdir newdir; printf done > newdir/result.txt; chmod 750 newdir/result.txt; chmod 555 newdir",
        tools=("bash",),
        max_turns=1,
    )
    calls = results(run, "bash")
    target = work / "newdir/result.txt"
    suite.record(
        "shell-publish",
        not (work / "old.txt").exists()
        and target.is_file()
        and target.read_text() == "done"
        and target.stat().st_mode & 0o777 == 0o750
        and target.parent.stat().st_mode & 0o777 == 0o555
        and all(not item["is_error"] for item in calls),
        hit=bool(calls),
        run_id=run["run_id"],
    )

    work = suite.project("shell-host-conflict")
    target = work / "sample.txt"
    target.write_text("initial")
    workers = []
    hit = []

    def start_writer(event):
        if event["type"] != "model_response" or workers:
            return
        evidence = RUNS / event["run_id"]

        def change_host():
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if any(evidence.glob("*-workspace/started")):
                    target.write_text("external")
                    hit.append(True)
                    return
                time.sleep(0.005)

        worker = threading.Thread(target=change_host)
        workers.append(worker)
        worker.start()

    run = suite.run(
        work,
        "Run exactly: touch started; sleep 0.5; printf from_command > sample.txt",
        tools=("bash",),
        max_turns=1,
        on_event=start_writer,
    )
    for worker in workers:
        worker.join()
    suite.record(
        "shell-host-conflict",
        run["status"] == "effect_unknown"
        and run["reason"] == "workspace_sync_failed"
        and target.read_text() == "external"
        and not (work / "started").exists(),
        hit=bool(hit),
        run_id=run["run_id"],
    )
