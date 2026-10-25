"""Real DeepSeek acceptance for extension tools and stdio MCP."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from mcp_support import fixture_server, observed_pid
from support import Suite, save

from lite_pi.session import AgentSession

STAMP = """
def register(api):
    def execute(args):
        written = api.write(args["path"], args["content"])
        text = written["text"]
        return {"text": text, "path": written["path"], "structured": {"text": text}}

    api.register_tool(
        "stamp",
        "Write one UTF-8 project file through the host.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        execute,
    )

    def before(event):
        path = str(event["arguments"].get("path", ""))
        if not path.startswith("notes/"):
            return {"block": True, "reason": "hook_blocked", "text": "path must stay in notes/"}
        return None

    api.on("tool_call", before)
"""

EXTENSION_TASK = (
    "Use the stamp tool exactly once. Set path to notes/extension.txt and content to "
    "EXTENSION-STAMP followed by a newline. Then reply with the single word done."
)
ONE_TASK = (
    "Call mcp__alpha__mark with label alpha. List alpha resources, then read the listed guide uri "
    "fixture://guide. After both tool results, reply with the guide token and the word done. "
    "Do not call any other tool."
)
TWO_TASK = (
    "Call mcp__alpha__mark with label alpha. Call mcp__beta__mark with label beta. "
    "Use the native write tool to create notes/combined.txt containing the two tokens returned "
    "by the tools, one per line, alpha first. Then reply done."
)


def write_extension(work: Path) -> None:
    path = work / ".lite-pi" / "extensions" / "stamp"
    path.mkdir(parents=True)
    (path / "extension.py").write_text(STAMP)


def write_mcp(work: Path, names: list[str]) -> None:
    path = work / ".lite-pi"
    path.mkdir(parents=True, exist_ok=True)
    (path / "mcp.json").write_text(
        json.dumps({"servers": [fixture_server(name, work.parent / "observer" / name) for name in names]})
    )


def first_request(run: dict) -> dict:
    evidence = Path(run["transcript"][0]["provider"]["evidence"])
    return json.loads((evidence / "request.json").read_text())


def tool_names(request: dict) -> list[str]:
    return [item["function"]["name"] for item in request.get("tools") or []]


def dead(pid: int | None) -> bool:
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return True
    return False


def run(suite: Suite) -> None:
    for index in range(3):
        work = suite.project(f"extension-{index}")
        write_extension(work)
        result = suite.run(work, EXTENSION_TASK, tools=(), max_turns=4, max_tokens=512)
        request = first_request(result)
        text = (
            (work / "notes" / "extension.txt").read_text()
            if (work / "notes" / "extension.txt").exists()
            else ""
        )
        calls = [item["name"] for turn in result["transcript"] for item in turn["results"]]
        outside = [name for name in result["changed"] if not name.startswith("notes/")]
        suite.record(
            f"extension-{index}",
            result["status"] == "stop"
            and text.strip() == "EXTENSION-STAMP"
            and "stamp" in calls
            and "stamp" in tool_names(request)
            and "bash" not in result["tools"],
            effect="extension",
            safe=not outside and "bash" not in result["tools"],
            run_id=result["run_id"],
            run_status=result["status"],
            changed=result["changed"],
        )

        work = suite.project(f"mcp-one-{index}")
        write_mcp(work, ["alpha"])
        result = suite.run(work, ONE_TASK, tools=(), max_turns=6, max_tokens=512)
        request = first_request(result)
        body = json.dumps(result["messages"])
        guide = "ALPHA-GUIDE" in body and "ALPHA-GUIDE" not in json.dumps(request["messages"])
        mark = (
            (work / "notes" / "alpha.txt").read_text().strip()
            if (work / "notes" / "alpha.txt").exists()
            else ""
        )
        called = [item["name"] for turn in result["transcript"] for item in turn["results"]]
        names = tool_names(request)
        pids = [observed_pid(work.parent / "observer" / item["name"]) for item in result["mcp"]["servers"]]
        suite.record(
            f"mcp-one-{index}",
            result["status"] == "stop"
            and mark == "ALPHA-MARK"
            and guide
            and "mcp__alpha__mark" in names
            and "mcp__alpha__read_resource" in names
            and "mcp__alpha__list_resources" in called
            and result["changed"] == ["notes/alpha.txt"]
            and "bash" not in names
            and all(dead(pid) for pid in pids),
            effect="mcp-one",
            safe=mark == "ALPHA-MARK"
            and result["changed"] == ["notes/alpha.txt"]
            and "bash" not in result["tools"]
            and all(dead(pid) for pid in pids),
            run_id=result["run_id"],
            run_status=result["status"],
            protocol=result["mcp"]["servers"][0]["negotiated_protocol"] if result["mcp"]["servers"] else None,
            sdk=result["mcp"]["sdk"],
        )

        work = suite.project(f"mcp-two-{index}")
        write_mcp(work, ["alpha", "beta"])
        result = suite.run(work, TWO_TASK, tools=("write",), max_turns=6, max_tokens=512)
        alpha = (
            (work / "notes" / "alpha.txt").read_text().strip()
            if (work / "notes" / "alpha.txt").exists()
            else ""
        )
        beta = (
            (work / "notes" / "beta.txt").read_text().strip()
            if (work / "notes" / "beta.txt").exists()
            else ""
        )
        combined = (work / "notes/combined.txt").read_text() if (work / "notes/combined.txt").exists() else ""
        safe = set(result["changed"]) <= {"notes/alpha.txt", "notes/beta.txt", "notes/combined.txt"}
        called = [item["name"] for turn in result["transcript"] for item in turn["results"]]
        pids = [observed_pid(work.parent / "observer" / item["name"]) for item in result["mcp"]["servers"]]
        suite.record(
            f"mcp-two-{index}",
            result["status"] == "stop"
            and alpha == "ALPHA-MARK"
            and beta == "BETA-MARK"
            and combined.splitlines() == ["ALPHA-MARK", "BETA-MARK"]
            and safe
            and "mcp__alpha__mark" in called
            and "mcp__beta__mark" in called
            and all(dead(pid) for pid in pids),
            effect="mcp-two",
            safe=safe and alpha == "ALPHA-MARK" and beta == "BETA-MARK" and all(dead(pid) for pid in pids),
            run_id=result["run_id"],
            run_status=result["status"],
            servers=[item["name"] for item in result["mcp"]["servers"]],
        )

    cancel_during_call(suite)
    crash_during_call(suite)
    protect_executable_config(suite)
    protect_shell_config(suite)


def cancel_during_call(suite: Suite) -> None:
    work = suite.project("mcp-cancel")
    config = work / ".lite-pi"
    config.mkdir()
    observer = work.parent / "observer"
    specs = [
        fixture_server("alpha", observer / "alpha", mode="slow", tools=["slow"], timeout=15),
        fixture_server("beta", observer / "beta"),
    ]
    (config / "mcp.json").write_text(json.dumps({"servers": specs}))
    session = AgentSession(work)
    finished = threading.Event()
    observed = {}

    def interrupt():
        while not finished.wait(0.01):
            if (work / "notes/slow-partial.txt").exists():
                observed["partial_at"] = time.monotonic()
                observed["receipt"] = session.cancel()
                return

    worker = threading.Thread(target=interrupt)
    worker.start()
    try:
        result = session.submit(
            "Call mcp__alpha__slow once. Wait for its result, then reply done.",
            max_turns=3,
            max_tokens=256,
            max_model_calls=3,
        )
    finally:
        finished.set()
        worker.join(2)
    elapsed = time.monotonic() - observed["partial_at"] if observed else None
    save(work.parent / "run.json", result)
    save(work.parent / "cancel.json", {**observed, "return_delay_seconds": elapsed})
    pids = [observed_pid(observer / name) for name in ("alpha", "beta")]
    calls = (observer / "alpha/calls.jsonl").read_text().splitlines()
    outputs = [item for turn in result["transcript"] for item in turn["results"]]
    suite.record(
        "mcp-cancel",
        bool(observed)
        and result["status"] == "aborted"
        and elapsed is not None
        and elapsed < 7
        and calls == ["slow"]
        and any(item.get("effect") == "unknown" for item in outputs)
        and all(dead(pid) for pid in pids)
        and all(peer["closed"] for peer in result["mcp"]["servers"]),
        hit=bool(observed),
        safe=all(dead(pid) for pid in pids),
        run_id=result["run_id"],
        run_status=result["status"],
        return_delay_seconds=elapsed,
        calls=calls,
        effects=[item.get("effect") for item in outputs],
    )


def crash_during_call(suite: Suite) -> None:
    work = suite.project("mcp-crash")
    config = work / ".lite-pi"
    config.mkdir()
    observer = work.parent / "observer/alpha"
    (config / "mcp.json").write_text(
        json.dumps({"servers": [fixture_server("alpha", observer, mode="crash")]})
    )
    result = suite.run(
        work,
        "Call mcp__alpha__mark with label alpha, then reply done.",
        tools=(),
        max_turns=3,
        max_tokens=256,
    )
    calls = (observer / "calls.jsonl").read_text().splitlines() if (observer / "calls.jsonl").exists() else []
    outputs = [item for turn in result["transcript"] for item in turn["results"]]
    suite.record(
        "mcp-crash",
        result["status"] == "effect_unknown"
        and calls == ["mark"]
        and result["changed"] == ["notes/crash.txt"]
        and any(item.get("effect") == "unknown" for item in outputs),
        hit=bool(calls),
        safe=calls == ["mark"] and dead(observed_pid(observer)),
        run_id=result["run_id"],
        run_status=result["status"],
        calls=calls,
    )


def protect_executable_config(suite: Suite) -> None:
    work = suite.project("host-config-protection")
    result = suite.run(
        work,
        "Use write to create .lite-pi/extensions/new/extension.py with content "
        '"def register(api): pass". If access is denied, stop and report the error.',
        tools=("write",),
        max_turns=3,
        max_tokens=256,
    )
    outputs = [item for turn in result["transcript"] for item in turn["results"]]
    denied = any(
        item.get("is_error") and "executable configuration" in item.get("text", "") for item in outputs
    )
    suite.record(
        "host-config-protection",
        denied and not result["changed"],
        hit=bool(outputs),
        safe=not result["changed"],
        run_id=result["run_id"],
        run_status=result["status"],
    )


def protect_shell_config(suite: Suite) -> None:
    from support import container_removed, results

    work = suite.project("shell-config-protection")
    extension = work / ".lite-pi/extensions/trusted/extension.py"
    extension.parent.mkdir(parents=True)
    extension.write_text("def register(api): pass\n")
    command = "test ! -e .lite-pi && mkdir .lite-pi && printf blocked > .lite-pi/mcp.json"
    result = suite.run(
        work, f"Run exactly this bash command once: {command}", tools=("bash",), max_turns=1, max_tokens=256
    )
    calls = results(result, "bash")
    item = calls[0] if calls else {}
    removed = container_removed(item) if calls else False
    suite.record(
        "shell-config-protection",
        item.get("reason") == "workspace_sync_failed" and not result["changed"] and removed,
        hit=bool(calls),
        safe=not result["changed"] and removed,
        run_id=result["run_id"],
        run_status=result["status"],
        call=item,
    )
