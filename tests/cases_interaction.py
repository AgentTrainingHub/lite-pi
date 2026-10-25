"""Real DeepSeek coverage for streaming, steering, follow-up, and cancellation."""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from support import ROOT, Suite, check, container_removed, require_available, snapshot

from lite_pi.journal import Journal
from lite_pi.session import CODE_TOOLS, RUNS, AgentSession
from lite_pi.spend.gate import EVIDENCE_ROOT
from lite_pi.tools.shell import DOCKER_ENV

STEER = "SCOPE-INVOICE-ONLY: change only weightnote/invoice.py."
FOLLOW = "Create notes/follow.txt containing exactly follow-up."
STREAM_TASK = (
    "List 12 distinct short facts about the number 17, at most 8 words each, "
    "one fact per line, with no introduction or tools."
)
CANCEL_TASK = "Write 80 numbered lines about rivers, one line each, and do not call any tools."
BASH_TASK = (
    "We are testing user cancellation in a disposable offline project. Run this command with bash: "
    "\n```sh\nprintf started > notes/started.txt; sleep 20 && printf done > notes/race.txt\n```\n"
    "An external test controller cancels "
    "the run after started.txt is written, during the sleep. Preserve the sleep so cancellation "
    "can be tested, and do not repeat the command."
)


def run(suite: Suite) -> None:
    for index in range(3):
        _stream(suite, index)
    _resume(suite)
    _cancel_stream(suite)
    _cli_cancel(suite)
    if _docker_ready():
        _cancel_command(suite)
        for index in range(3):
            _steer(suite, index)
    else:
        suite.record(
            "cancel-command",
            False,
            blocked=True,
            reason="docker CLI did not answer within 5s",
        )
        for index in range(3):
            suite.record(
                f"steer-{index}",
                False,
                blocked=True,
                effect="steer",
                reason="docker CLI did not answer within 5s",
            )


def _docker_ready() -> bool:
    try:
        observed = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=5,
            env=DOCKER_ENV,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return observed.returncode == 0 and bool(observed.stdout.strip())


def _stream(suite: Suite, index: int) -> None:
    work = suite.project(f"stream-{index}")
    observed = _cli(work, "submit", "--root", str(work), STREAM_TASK)
    data = observed["json"]
    require_available(data)
    events = _events(data["evidence"])
    log = _log_at(_spend(data["run_id"]))
    deltas = [event for event in events if event["type"] == "message_delta"]
    joined = "".join(event["text"] for event in deltas)
    chunks = [item for item in log if item["kind"] == "chunk"]
    terminal = log[-1] if log else {}
    incremental = (
        len(chunks) >= 2
        and len(deltas) >= 2
        and chunks[0]["at"] < terminal.get("at", "")
        and terminal.get("kind") == "finished"
        and all(event["at"] <= terminal["at"] for event in deltas)
    )
    displayed_once = observed["stderr"] == data["text"] and data["text"] == joined and bool(joined.strip())
    structured = [event["type"] for event in events]
    passed = (
        data["status"] == "stop"
        and data.get("model") == "deepseek-flash"
        and displayed_once
        and structured[0] == "run_start"
        and structured[-1] == "run_end"
        and "turn_start" in structured
        and "message_end" in structured
        and incremental
    )
    suite.record(
        f"stream-{index}",
        passed,
        hit=incremental or data["status"] != "stop",
        run_id=data.get("run_id"),
        effect="stream",
        chunks=len(chunks),
        deltas=len(deltas),
        terminal=terminal.get("kind"),
        stderr_matches_text=observed["stderr"] == data.get("text"),
        returncode=observed["returncode"],
    )


def _resume(suite: Suite) -> None:
    work = suite.project("resume")
    journal_path = work.parent / "session.jsonl"
    journal = Journal.create(journal_path, work)
    first = AgentSession(work, journal=journal).submit(STREAM_TASK, max_tokens=24)
    require_available(first)
    second = AgentSession.open(journal_path).resume(max_turns=1, max_tokens=800)
    require_available(second)
    request = _request(second)
    deltas = [event for event in second["events"] if event["type"] == "message_delta"]
    log = _log_at(Path(second["transcript"][0]["provider"]["evidence"]))
    chunks = [item for item in log if item["kind"] == "chunk"]
    terminal = log[-1] if log else {}
    hit = len(chunks) >= 2 and len(deltas) >= 2 and terminal.get("kind") == "finished"
    suite.record(
        "resume",
        hit
        and second["run_id"] != first["run_id"]
        and any(isinstance(item, dict) and "number 17" in json.dumps(item) for item in request)
        and second["events"][0]["type"] == "run_start"
        and "".join(event["text"] for event in deltas) == second["text"]
        and all(event["at"] <= terminal.get("at", "") for event in deltas),
        hit=hit,
        run_id=second["run_id"],
        previous_run_id=first["run_id"],
        chunks=len(chunks),
        first_status=first["status"],
    )


def _cancel_stream(suite: Suite) -> None:
    work = suite.project("cancel-stream")
    session = AgentSession(work)

    def observe(event):
        if event["type"] == "message_delta":
            session.cancel()
            session.cancel()

    run = session.submit(CANCEL_TASK, max_tokens=800, on_event=observe)
    require_available(run)
    body = _body(run)
    log = _log_at(Path(run["transcript"][0]["provider"]["evidence"]))
    chunks = [item for item in log if item["kind"] == "chunk"]
    done = b"[DONE]" in body
    hit = bool(chunks) and bool(run["text"]) and not done and log[-1]["kind"] == "aborted"
    endings = [event for event in run["events"] if event["type"] == "run_end"]
    suite.record(
        "cancel-stream",
        hit
        and run["status"] == "aborted"
        and len(endings) == 1
        and not any(event["type"] == "tool_start" for event in run["events"])
        and session.control.cancel_count == 2
        and all(event["at"] <= log[-1]["at"] for event in run["events"] if event["type"] == "message_delta"),
        hit=hit,
        safe=not any(event["type"] == "tool_start" for event in run["events"]),
        run_id=run["run_id"],
        chunks=len(chunks),
        completed_marker=done,
        text_bytes=len(run["text"].encode()),
    )


def _cli_cancel(suite: Suite) -> None:
    work = suite.project("cli-cancel")
    proc = subprocess.Popen(
        [sys.executable, "-m", "lite_pi", "submit", "--root", str(work), CANCEL_TASK],
        cwd=ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            **({"HOME": os.environ["HOME"]} if "HOME" in os.environ else {}),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    seen = ""
    deadline = time.monotonic() + 20
    assert proc.stderr is not None
    os.set_blocking(proc.stderr.fileno(), False)
    with selectors.DefaultSelector() as selector:
        selector.register(proc.stderr, selectors.EVENT_READ)
        while time.monotonic() < deadline and not seen and proc.poll() is None:
            if not selector.select(0.2):
                continue
            chunk = os.read(proc.stderr.fileno(), 256)
            if chunk:
                seen += chunk.decode(errors="replace")
    if seen and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
    stdout, stderr = proc.communicate(timeout=30)
    data = json.loads(stdout)
    require_available(data)
    displayed = seen + stderr
    hit = bool(seen) and data["status"] == "aborted"
    suite.record(
        "cli-cancel",
        hit and proc.returncode == 0 and displayed == data["text"],
        hit=hit,
        safe=True,
        run_id=data.get("run_id"),
        returncode=proc.returncode,
        run_status=data.get("status"),
    )


def _cancel_command(suite: Suite) -> None:
    work = suite.project("cancel-command")
    (work / "notes").mkdir()
    before = snapshot(work)
    session = AgentSession(work, tools=("bash",))
    finished = threading.Event()
    observed = {}
    worker = None

    def interrupt(evidence: Path):
        while not finished.wait(0.01):
            for marker in evidence.glob("*-workspace/notes/started.txt"):
                if marker.read_text() == "started":
                    observed["started_at"] = time.monotonic()
                    session.cancel()
                    session.cancel()
                    return

    def observe(event):
        nonlocal worker
        if event["type"] == "tool_start" and event["name"] == "bash" and worker is None:
            worker = threading.Thread(target=interrupt, args=(RUNS / event["run_id"],))
            worker.start()

    try:
        run = session.submit(BASH_TASK, max_turns=2, shell_timeout=30, on_event=observe)
    finally:
        finished.set()
        if worker is not None:
            worker.join(2)
    require_available(run)
    bash = [
        item
        for turn in run["transcript"]
        for item in turn["results"]
        if item.get("name") == "bash" and item.get("executed")
    ]
    matched = [item for item in bash if "sleep 20" in item.get("command", "")]
    hit = bool(observed) and bool(matched) and matched[0].get("reason") == "cancelled"
    later = (
        [
            item
            for turn in run["transcript"]
            for item in turn["results"]
            if item.get("executed") and item is not matched[0]
        ]
        if matched
        else []
    )
    elapsed = time.monotonic() - observed["started_at"] if observed else None
    started = work / "notes/started.txt"
    race = work / "notes" / "race.txt"
    published = matched[0].get("published", []) if matched else []
    consistent = (race.exists()) == ("notes/race.txt" in published)
    removed = bool(matched) and container_removed(matched[0]) and matched[0].get("process_reaped") is True
    suite.record(
        "cancel-command",
        hit
        and run["status"] == "aborted"
        and not later
        and removed
        and consistent
        and not race.exists()
        and started.is_file()
        and started.read_text() == "started"
        and elapsed is not None
        and elapsed < 7
        and len([event for event in run["events"] if event["type"] == "run_end"]) == 1
        and session.control.cancel_count == 2,
        hit=hit,
        safe=not later and (not matched or removed),
        run_id=run["run_id"],
        command=matched[0].get("command") if matched else None,
        reason=matched[0].get("reason") if matched else None,
        stopped_at=matched[0].get("stopped_at") if matched else None,
        started_observed=bool(observed),
        return_delay_seconds=elapsed,
        published=published,
        race_exists=race.exists(),
        changed=sorted(
            path
            for path in before.keys() | snapshot(work).keys()
            if before.get(path) != snapshot(work).get(path)
        ),
    )


def _steer(suite: Suite, index: int) -> None:
    work = suite.project(f"steer-{index}", fixture=True)
    journal = Journal.create(work.parent / "session.jsonl", work, tools=CODE_TOOLS)
    session = AgentSession(work, tools=CODE_TOOLS, journal=journal)
    original = snapshot(work)
    armed = {"value": False}

    def observe(event):
        if event["type"] == "message_end" and event["turn"] == 1 and not armed["value"]:
            armed["value"] = True
            session.steer(STEER, paths=("weightnote/invoice.py",))
            session.follow_up(FOLLOW)

    run = session.submit(
        "First run python3 check.py, read the failure, then fix weightnote/invoice.py. "
        "Do not change check.py. A later instruction may limit the file scope and add a follow-up.",
        max_turns=14,
        max_tokens=640,
        shell_timeout=20,
        on_event=observe,
    )
    require_available(run)
    requests = [_request(turn) for turn in run["transcript"]]
    steer_at = next((index for index, messages in enumerate(requests) if STEER in json.dumps(messages)), None)
    follow_at = next(
        (index for index, messages in enumerate(requests) if FOLLOW in json.dumps(messages)), None
    )
    entered = [item for item in run["controls"] if item["text"] in {STEER, FOLLOW}]
    timed = bool(entered) and all(
        item["submitted_at"]
        and item["accepted_at"]
        and item["entered_at"]
        and item["submitted_at"] <= item["accepted_at"] <= item["entered_at"]
        for item in entered
        if item["disposition"] == "entered"
    )
    cutoff = follow_at if follow_at is not None else len(run["transcript"])
    later_results = [item for turn in run["transcript"][1:cutoff] for item in turn["results"]]
    outside = [
        item
        for item in later_results
        if item.get("executed")
        and item.get("name") in {"edit", "write"}
        and item.get("path") not in {None, "weightnote/invoice.py"}
    ]
    verified = check(work)
    follow_file = work / "notes" / "follow.txt"
    changed = sorted(
        path
        for path in original.keys() | snapshot(work).keys()
        if original.get(path) != snapshot(work).get(path)
    )
    scope_held = set(changed) <= {"weightnote/invoice.py", "notes/follow.txt"} and not outside
    passed = (
        steer_at not in {None, 0}
        and follow_at is not None
        and follow_at > steer_at
        and timed
        and scope_held
        and verified["returncode"] == 0
        and "FAIL" not in verified["stdout"]
        and follow_file.is_file()
        and follow_file.read_text().strip() == "follow-up"
    )
    suite.record(
        f"steer-{index}",
        passed,
        hit=steer_at is not None,
        safe=scope_held,
        effect="steer",
        run_id=run["run_id"],
        steer_request=steer_at,
        follow_request=follow_at,
        check_returncode=verified["returncode"],
        follow_exists=follow_file.is_file(),
        changed=changed,
    )


def _cli(work: Path, *args: str) -> dict:
    del work
    proc = subprocess.run(
        [sys.executable, "-m", "lite_pi", *args],
        cwd=ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            **({"HOME": os.environ["HOME"]} if "HOME" in os.environ else {}),
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    return {"json": json.loads(proc.stdout), "stderr": proc.stderr, "returncode": proc.returncode}


def _events(evidence: str) -> list[dict]:
    path = Path(evidence) / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _request(run_or_turn: dict) -> list:
    if "provider" in run_or_turn:
        evidence = Path(run_or_turn["provider"]["evidence"])
    else:
        evidence = Path(run_or_turn["transcript"][0]["provider"]["evidence"])
    return json.loads((evidence / "request.json").read_text())["messages"]


def _log_at(evidence: Path) -> list[dict]:
    path = evidence / "stream.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _spend(run_id: str) -> Path:
    found = []
    for path in EVIDENCE_ROOT.iterdir():
        receipt = path / "receipt.json"
        if not receipt.is_file():
            continue
        data = json.loads(receipt.read_text())
        if data.get("run_id") == run_id and data.get("purpose") == "session-turn-1":
            found.append(path)
    if len(found) != 1:
        raise FileNotFoundError(f"{run_id}: found {len(found)} turn-1 calls")
    return found[0]


def _body(run: dict) -> bytes:
    evidence = Path(run["transcript"][0]["provider"]["evidence"])
    return (evidence / "response.bin").read_bytes()
