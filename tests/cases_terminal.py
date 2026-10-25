"""Actual terminal keys and real DeepSeek; fixture sources never replace the API."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from support import FIX, ROOT, Suite, check, container_removed
from tui_pty import Terminal

from lite_pi.journal import Journal
from lite_pi.session import RUNS, AgentSession
from lite_pi.spend.ledger import Ledger
from lite_pi.tools.contracts import JsonObject

PHASES = (
    "stop",
    "aborted",
    "budget",
    "effect_unknown",
    "awaiting_approval",
    "turn_limit",
    "blocked",
    "error",
)


def ready(terminal: Terminal) -> None:
    terminal.wait(lambda: "Enter 提交" in terminal.text)


def send(terminal: Terminal, text: str) -> None:
    terminal.send(text + "\r")
    terminal.wait(lambda: "运行中" in terminal.text, 15)


def done(terminal: Terminal, name: str) -> None:
    terminal.wait(
        lambda: any(
            phase in terminal.text.splitlines()[-2]
            for phase in (
                "本轮结束",
                "已取消",
                "预算停止",
                "效果未知",
                "等待审批",
                "等待决定",
                "轮次上限",
                "运行受阻",
                "运行失败",
                "回答截断",
            )
        ),
        160,
    )
    for _ in range(3):
        terminal.read(0.03)
    terminal.capture(name)


def journal(work: Path) -> Journal:
    paths = list((work / ".artifacts" / "sessions").glob("*.jsonl"))
    if len(paths) != 1:
        raise AssertionError(f"expected one session, got {len(paths)}")
    return Journal(paths[0])


def result(work: Path) -> JsonObject:
    session_id = journal(work).records[0]["id"]
    for directory in sorted(RUNS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        path = directory / "result.json"
        if path.exists():
            value = json.loads(path.read_text())
            if (value.get("tasks") or {}).get("session_id") == session_id:
                return value
    raise AssertionError("no final session receipt")


def evidence(terminal: Terminal) -> JsonObject:
    value = json.loads((terminal.evidence / "terminal.json").read_text())
    return {
        "terminal": str(terminal.evidence),
        "terminal_restored": value["restored"],
        "exit": value["returncode"],
    }


def history_and_facts(first: JsonObject, second: JsonObject, work: Path) -> bool:
    history = second.get("initial_request") or []
    notes = work / "notes/result.txt"
    if not notes.is_file():
        return False
    text = notes.read_text().lower()
    return (
        first["initial_request"][0] in history
        and any(item["role"] == "assistant" and item.get("tool_calls") for item in history)
        and any(item["role"] == "tool" for item in history)
        and "invoice" in text
        and "unit" in text
        and "1200" in text
        and "2000" in text
    )


def multi_turn(suite: Suite, index: int) -> None:
    work = suite.project(f"multi-{index}", fixture=True)
    terminal = Terminal(work, work.parent / "terminal", extra=("--max-task-calls", "24"))
    original = (work / "check.py").read_bytes()
    try:
        ready(terminal)
        terminal.capture("welcome")
        send(terminal, FIX)
        done(terminal, "first")
        first = result(work)
        first_ok = first["status"] == "stop" and check(work)["returncode"] == 0
        send(
            terminal,
            "把刚才修复的问题与验证结果记录到 notes/result.txt，保留已修复的代码与 check.py，不要执行额外修改。",
        )
        done(terminal, "second")
        second = result(work)
        passed = (
            first_ok
            and second["status"] == "stop"
            and history_and_facts(first, second, work)
            and (work / "check.py").read_bytes() == original
            and check(work)["returncode"] == 0
        )
    finally:
        terminal.close()
    observed = evidence(terminal)
    suite.record(
        f"multi-turn-{index}",
        passed and observed["terminal_restored"],
        effect="tui-multi-turn",
        safe=(work / "check.py").read_bytes() == original,
        run_ids=[first["run_id"], second["run_id"]],
        model=second["model"],
        **observed,
    )


def controls(suite: Suite, index: int) -> None:
    work = suite.project(f"controls-{index}")
    (work / "source.txt").write_text("reference value\n")
    terminal = Terminal(work, work.parent / "terminal", extra=("--max-task-calls", "18"))
    try:
        ready(terminal)
        send(terminal, "用 read 读取 source.txt 后总结内容。暂时不修改任何文件。")
        terminal.send(
            "/steer --paths notes/steer.txt -- 改为用 write 写入 notes/steer.txt，内容恰好为 steered，不修改其他文件。\r"
        )
        terminal.wait(lambda: "转向已接收" in terminal.text)
        terminal.send("/follow 用 write 创建 notes/follow.txt，内容恰好为 follow-up，完成后停止。\r")
        terminal.wait(lambda: "后续任务已接收" in terminal.text)
        done(terminal, "controls")
        receipt = result(work)
        controls = receipt.get("controls") or []
        entered = {item["kind"] for item in controls if item["disposition"] == "entered"}
        passed = (
            receipt["status"] == "stop"
            and entered == {"steer", "follow_up"}
            and (work / "notes/steer.txt").read_text().strip() == "steered"
            and (work / "notes/follow.txt").read_text().strip() == "follow-up"
        )
    finally:
        terminal.close()
    observed = evidence(terminal)
    suite.record(
        f"controls-{index}",
        passed and observed["terminal_restored"],
        effect="tui-controls",
        safe=(work / "source.txt").read_text() == "reference value\n",
        run_ids=[receipt["run_id"]],
        controls=controls,
        **observed,
    )


def reopen(suite: Suite, index: int) -> None:
    work = suite.project(f"reopen-{index}")
    token = "TUI-" + uuid4().hex[:12]
    first = Terminal(work, work.parent / "first", extra=("--max-task-calls", "10"))
    try:
        ready(first)
        send(first, f"请记住标记 {token}，只回复已记住，不调用工具。")
        done(first, "remembered")
        before = result(work)
    finally:
        first.close()
    path = journal(work).path
    after_open = Ledger().status()["request_count"]
    second = Terminal(work, work.parent / "second", session=path, extra=("--max-task-calls", "10"))
    try:
        ready(second)
        opened_without_request = Ledger().status()["request_count"] == after_open
        send(second, "把刚才的标记原文写进 notes/resumed.txt，只写标记，不附加说明。")
        done(second, "resumed")
        after = result(work)
        passed = (
            opened_without_request
            and after["status"] == "stop"
            and (work / "notes/resumed.txt").read_text().strip() == token
        )
    finally:
        second.close()
    observed = evidence(second)
    suite.record(
        f"reopen-{index}",
        passed and observed["terminal_restored"],
        effect="tui-reopen",
        safe=opened_without_request,
        run_ids=[before["run_id"], after["run_id"]],
        **observed,
    )


def cancel_stream(suite: Suite) -> None:
    work = suite.project("cancel-stream")
    terminal = Terminal(work, work.parent / "terminal")
    try:
        ready(terminal)
        send(terminal, "写出100条关于河流的编号事实，每条一行，不调用工具。")
        terminal.wait(lambda: "1." in terminal.text or "1、" in terminal.text)
        cancelled_at = datetime.now(UTC).isoformat()
        terminal.send(b"\x03")
        done(terminal, "cancelled")
        cancelled = result(work)
        send(terminal, "只回复恢复成功，不调用工具。")
        done(terminal, "continued")
        after = result(work)
    finally:
        terminal.close()
    observed = evidence(terminal)
    suite.record(
        "tui-cancel-stream",
        cancelled["status"] == "aborted"
        and after["status"] == "stop"
        and observed["terminal_restored"]
        and any(
            event["type"] == "message_delta" and event.get("text") and event["at"] < cancelled_at
            for event in cancelled["events"]
        ),
        safe=not cancelled["transcript"][-1].get("results"),
        run_ids=[cancelled["run_id"], after["run_id"]],
        **observed,
    )


def cancel_shell(suite: Suite) -> None:
    work = suite.project("cancel-shell")
    terminal = Terminal(work, work.parent / "terminal")
    try:
        ready(terminal)
        send(terminal, "请用 bash 执行 sleep 30 一次，这是取消测试，不执行其他命令，不重试。")
        terminal.wait(lambda: "bash" in terminal.text and "执行中" in terminal.text)
        end = time.monotonic() + 15
        running = False
        while time.monotonic() < end:
            observed = subprocess.run(
                ["docker", "ps", "--filter", "name=lite-pi-", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if observed.stdout.strip():
                running = True
                break
            terminal.read(0.1)
        terminal.send(b"\x03")
        done(terminal, "shell-cancelled")
        receipt = result(work)
        shells = [
            item for turn in receipt["transcript"] for item in turn["results"] if item["name"] == "bash"
        ]
        passed = (
            running
            and len(shells) == 1
            and shells[0].get("process_reaped") is True
            and container_removed(shells[0])
        )
    finally:
        terminal.close()
    observed = evidence(terminal)
    suite.record(
        "tui-cancel-shell",
        passed and observed["terminal_restored"],
        hit=running,
        safe=all(item.get("process_reaped") for item in shells),
        run_ids=[receipt["run_id"]],
        **observed,
    )


def approval(suite: Suite) -> None:
    work = suite.project("approval")
    terminal = Terminal(work, work.parent / "terminal", extra=("--approvals",))
    try:
        ready(terminal)
        send(terminal, "用 write 创建 note.txt，内容恰好为 approved，完成后停止。")
        done(terminal, "approval-wait")
        pending = result(work)
        view = AgentSession.open(journal(work).path).inspect()
        candidate = next(item for item in view["approvals"] if item["decision"] == "pending")
        before_effect = not (work / "note.txt").exists()
        terminal.send(f"/approve {candidate['approval_id']}\r")
        terminal.wait(lambda: "/continue 明确继续" in terminal.text)
        before_continue = not (work / "note.txt").exists()
        send(terminal, "/continue")
        done(terminal, "approved")
        final = result(work)
        passed = (
            before_effect
            and before_continue
            and final["status"] == "stop"
            and (work / "note.txt").read_text().strip() == "approved"
        )
    finally:
        terminal.close()
    observed = evidence(terminal)
    suite.record(
        "tui-approval",
        passed and observed["terminal_restored"],
        safe=before_effect and before_continue,
        run_ids=[pending["run_id"], final["run_id"]],
        **observed,
    )


def mcp_cleanup(suite: Suite) -> None:
    from compare.combo import marker_closed, marker_count

    work = suite.project("mcp")
    state = work.parent / "observer"
    config = work / ".lite-pi"
    config.mkdir()
    (config / "mcp.json").write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "name": "marker",
                        "command": __import__("sys").executable,
                        "args": [
                            str(ROOT / "tests/fixtures/mcp_server.py"),
                            "--name",
                            "marker",
                            "--mode",
                            "serve",
                            "--state",
                            str(state),
                        ],
                        "tools": ["mark"],
                        "timeout_seconds": 20,
                    }
                ]
            }
        )
    )
    terminal = Terminal(work, work.parent / "terminal")
    try:
        ready(terminal)
        send(terminal, "调用 mcp__marker__mark 一次，label 为 terminal，然后停止；不要使用文件或 bash 工具。")
        done(terminal, "mcp-finished")
        receipt = result(work)
        marks = [
            item
            for turn in receipt["transcript"]
            for item in turn["results"]
            if item["name"] == "mcp__marker__mark" and item["executed"] and not item["is_error"]
        ]
        passed = len(marks) == 1 and marker_count(state) == 1 and marker_closed(state)
    finally:
        terminal.close()
    observed = evidence(terminal)
    suite.record(
        "tui-mcp-cleanup",
        passed and observed["terminal_restored"],
        safe=marker_closed(state),
        run_ids=[receipt["run_id"]],
        **observed,
    )


def run(suite: Suite) -> None:
    cases = [
        (f"{fn.__name__}-{index}", lambda fn=fn, index=index: fn(suite, index))
        for fn in (multi_turn, controls, reopen)
        for index in range(1, 4)
    ]
    cases += [
        (fn.__name__, lambda fn=fn: fn(suite)) for fn in (cancel_stream, cancel_shell, approval, mcp_cleanup)
    ]
    for name, case in cases:
        if Ledger().status()["stopped"]:
            suite.record(name, False, blocked=True, reason="original shared ledger stopped")
            break
        try:
            case()
        except Exception as error:
            suite.record(name, False, safe=False, error=f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    suite = Suite("terminal-tui")
    run(suite)
    raise SystemExit(suite.finish())
