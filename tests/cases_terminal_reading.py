"""Real-model terminal workflows use keyboard input at the installed CLI's PTY boundary."""

from pathlib import Path
from uuid import uuid4

from cases_terminal import (
    approval,
    cancel_shell,
    cancel_stream,
    done,
    evidence,
    journal,
    mcp_cleanup,
    ready,
    result,
    send,
)
from support import Suite, check
from tui_pty import Terminal

from lite_pi.spend.ledger import Ledger


def continuous(suite: Suite, index: int) -> None:
    work = suite.project(f"continuous-{index}")
    (work / "calc.py").write_text("def total(price, count):\n    return price + count\n")
    expected_check = "from calc import total\nassert total(12, 2) == 24\nprint('CHECK-PASSED')\n"
    (work / "check.py").write_text(expected_check)
    marker = "TUI-" + uuid4().hex[:10]
    first = Terminal(work, work.parent / "first", columns=120, rows=32, extra=("--max-task-calls", "18"))
    run_ids = []
    try:
        ready(first)
        first.capture("welcome")
        send(
            first,
            f"批次标记为 {marker}。读取 calc.py 和 check.py，修复 total 的实现，然后运行 python3 check.py 验证。保留 check.py 原文。用简短列表和 python 代码块说明结果。",
        )
        done(first, "repaired")
        repaired = result(work)
        run_ids.append(repaired["run_id"])
        verified = check(work)
        repair_ok = repaired["status"] == "stop" and verified["returncode"] == 0
        tool_visible = any(item in first.text for item in ("已完成", "✓"))
        send(first, "把上一轮批次标记原文写到 notes/marker.txt，只写标记；保留代码和 check.py。")
        done(first, "followed")
        followed = result(work)
        run_ids.append(followed["run_id"])
        history = followed.get("initial_request") or []
        history_ok = (
            repaired["initial_request"][0] in history
            and any(item["role"] == "tool" for item in history)
            and any(item["role"] == "assistant" and item.get("tool_calls") for item in history)
        )
        follow_ok = followed["status"] == "stop" and history_ok and read(work / "notes/marker.txt") == marker
    finally:
        first.close()
    calls = Ledger().status()["request_count"]
    reopened = Terminal(
        work, work.parent / "reopened", session=journal(work).path, extra=("--max-task-calls", "18")
    )
    try:
        ready(reopened)
        passive = Ledger().status()["request_count"] == calls
        send(reopened, "继续上一轮：把同一批次标记原文写进 notes/continued.txt，只写标记；保留其他文件。")
        done(reopened, "continued")
        continued = result(work)
        run_ids.append(continued["run_id"])
        reopen_ok = passive and continued["status"] == "stop" and read(work / "notes/continued.txt") == marker
    finally:
        reopened.close()
    safe = (work / "check.py").read_text() == expected_check and passive
    common = {
        "safe": safe,
        "run_ids": run_ids,
        "marker": marker,
        "check": verified,
        "tool_status_visible": tool_visible,
        "first_terminal": evidence(first),
        **evidence(reopened),
    }
    for name, passed in (
        ("repair", repair_ok and tool_visible),
        ("follow", follow_ok),
        ("reopen", reopen_ok),
    ):
        suite.record(
            f"terminal_reading-{name}-{index}",
            passed and common["terminal_restored"],
            effect="terminal_reading-" + name,
            **common,
        )


def read(path: Path) -> str:
    return path.read_text().strip() if path.is_file() else ""


def run(suite: Suite) -> None:
    cases = [(f"continuous-{index}", lambda index=index: continuous(suite, index)) for index in range(1, 4)]
    cases += [
        (fn.__name__, lambda fn=fn: fn(suite)) for fn in (cancel_stream, cancel_shell, approval, mcp_cleanup)
    ]
    for name, case in cases:
        status = Ledger().status()
        if status["stopped"]:
            suite.record(name, False, blocked=True, reason="original cumulative ledger stopped")
            break
        try:
            case()
        except Exception as error:
            suite.record(name, False, safe=False, error=f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    suite = Suite("terminal_reading-tui")
    run(suite)
    raise SystemExit(suite.finish())
