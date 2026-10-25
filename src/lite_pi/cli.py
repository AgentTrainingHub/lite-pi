"""Argument parsing and result presentation over the public AgentSession API."""

import argparse
import json
import signal
import sqlite3
import sys
from collections.abc import Callable
from types import FrameType

from lite_pi.cli_args import parse
from lite_pi.journal import Journal, JournalError
from lite_pi.knowledge.tools import KNOWLEDGE_TOOLS
from lite_pi.planning import PLANNING_TOOLS
from lite_pi.session import CODE_TOOLS, AgentSession, RunLimits
from lite_pi.tools.contracts import JsonObject

CLI_OUTPUT_TOKENS = 640


def _progress(event: JsonObject) -> None:
    if event["type"] == "message_delta":
        print(event["text"], end="", file=sys.stderr, flush=True)
        return
    if event["type"] == "tool_start":
        print(
            json.dumps(
                {"event": "tool_start", "run_id": event["run_id"], "tool": event["name"]},
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
    elif event["type"] == "tool_end":
        result = event["result"]
        print(
            json.dumps(
                {
                    "event": "tool_end",
                    "run_id": event["run_id"],
                    "tool": result["name"],
                    "is_error": result["is_error"],
                    "executed": result["executed"],
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser, args = parse(argv)
    if args.command == "chat":
        return _chat(parser, args)
    try:
        result, status = HANDLERS[args.command](args)
    except (JournalError, ValueError, OSError, sqlite3.Error) as error:
        result, status = {"status": "rejected", "reason": str(error)}, 2
    print(json.dumps(result, ensure_ascii=False))
    return status


def _index(args: argparse.Namespace) -> tuple[JsonObject, int]:
    result = AgentSession(args.root, embedding_dir=args.embedding_dir).rebuild_index(mode=args.mode)
    return result, 2 if result.get("is_error") else 0


def _retrieval_trace(args: argparse.Namespace) -> tuple[JsonObject, int]:
    return AgentSession(args.root, embedding_dir=args.embedding_dir).knowledge(), 0


def _plan(args: argparse.Namespace) -> tuple[JsonObject, int]:
    session = AgentSession.open(args.session)
    if not args.change:
        return session.inspect().get("workflow") or {}, 0
    change = json.loads(args.change)
    required = {"expected_version", "reason", "steps", "invalidated_steps"}
    if (
        not isinstance(change, dict)
        or set(change) != required
        or not isinstance(change["invalidated_steps"], list)
    ):
        raise ValueError("invalid_plan_change_fields")
    state = session.revise_plan(
        expected_version=change["expected_version"],
        reason=change["reason"],
        steps=change["steps"],
        invalidated_steps=tuple(change["invalidated_steps"]),
    )
    return state, 0


def _answer(args: argparse.Namespace) -> tuple[JsonObject, int]:
    return AgentSession.open(args.session).answer(
        args.id,
        args.text or "",
        expected_version=args.version,
        cancel=args.cancel,
    ), 0


def _delegate(_args: argparse.Namespace) -> tuple[JsonObject, int]:
    return {
        "status": "rejected",
        "reason": "delegate is a session tool; pass --delegate to code",
    }, 2


def _approve(args: argparse.Namespace) -> tuple[JsonObject, int]:
    receipt = AgentSession.open(args.session).decide(args.id, args.decision)
    accepted = {"approved", "rejected", "revoked", "expired", "superseded"}
    return receipt, 0 if receipt.get("decision") in accepted else 2


def _memory_confirm(args: argparse.Namespace) -> tuple[JsonObject, int]:
    receipt = AgentSession.open(args.session).confirm_memory(
        args.id,
        supersedes=args.supersedes,
        ttl_seconds=args.ttl,
    )
    return receipt, 0 if receipt.get("ok") else 2


def _memory_revoke(args: argparse.Namespace) -> tuple[JsonObject, int]:
    receipt = AgentSession.open(args.session).revoke_memory(args.id)
    return receipt, 0 if receipt.get("ok") else 2


def _memory_resolve(args: argparse.Namespace) -> tuple[JsonObject, int]:
    receipt = AgentSession.open(args.session).resolve_memory(args.id)
    return receipt, 0 if receipt.get("ok") else 2


def _code_tools(args: argparse.Namespace) -> tuple[str, ...]:
    tools = CODE_TOOLS + (("delegate",) if args.delegate else ())
    if args.planning:
        tools += PLANNING_TOOLS
    if args.knowledge:
        tools += KNOWLEDGE_TOOLS
    return tools


def _code(args: argparse.Namespace) -> tuple[JsonObject, int]:
    tools = _code_tools(args)
    journal = Journal.create(args.session, args.root, tools=tools)
    session = AgentSession(args.root, tools=tools, journal=journal, embedding_dir=args.embedding_dir)
    return _run_result(_run_interruptible(session, lambda: session.submit(args.task, **_run_limits(args))))


def _resume(args: argparse.Namespace) -> tuple[JsonObject, int]:
    session = AgentSession.open(args.session, embedding_dir=args.embedding_dir)
    return _run_result(
        _run_interruptible(session, lambda: session.resume(leaf_id=args.leaf, **_run_limits(args)))
    )


def _branch(args: argparse.Namespace) -> tuple[JsonObject, int]:
    record = AgentSession.open(args.session).branch(args.parent, args.task)
    return {
        "status": "branched",
        "id": record["id"],
        "parent_id": record["parent_id"],
        "leaf": record["id"],
    }, 0


def _submit(args: argparse.Namespace) -> tuple[JsonObject, int]:
    session = AgentSession(args.root)
    return _run_result(_run_interruptible(session, lambda: session.submit(args.task, on_event=_progress)))


def _interrupt(args: argparse.Namespace) -> tuple[JsonObject, int]:
    session = AgentSession(args.root)
    return _run_result(_run_interruptible(session, lambda: session.interrupt(args.task, on_event=_progress)))


def _run_result(result: JsonObject) -> tuple[JsonObject, int]:
    receipt = {
        key: result[key]
        for key in (
            "run_id",
            "status",
            "raw_stop_reason",
            "text",
            "model",
            "usage_unknown",
            "reason",
            "checked_files",
            "evidence",
        )
    }
    if result["status"] == "awaiting_question":
        workflow = result["tasks"]["workflow"]
        question = workflow["question"]
        receipt["question"] = {
            "id": question["id"],
            "text": question["text"],
            "options": question["options"],
            "expected_version": workflow["version"],
        }
    status = (
        0
        if result["status"] in {"stop", "aborted", "awaiting_approval", "awaiting_question", "queued"}
        else 1
    )
    return receipt, status


HANDLERS: dict[str, Callable[[argparse.Namespace], tuple[JsonObject, int]]] = {
    "index": _index,
    "retrieval-trace": _retrieval_trace,
    "plan": _plan,
    "answer": _answer,
    "delegate": _delegate,
    "approve": _approve,
    "memory-confirm": _memory_confirm,
    "memory-revoke": _memory_revoke,
    "memory-resolve": _memory_resolve,
    "code": _code,
    "resume": _resume,
    "branch": _branch,
    "submit": _submit,
    "interrupt": _interrupt,
}


def _chat(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.error("交互模式需要 TTY；单次任务请使用 submit 或 code")
    from lite_pi.tui.app import run_chat

    tools = _code_tools(args)
    limits = _task_limits(args)
    limits["max_turns"] = args.max_turns
    limits["max_tokens"] = CLI_OUTPUT_TOKENS
    try:
        return run_chat(args.root, args.session, tools, limits, embedding_dir=args.embedding_dir)
    except (JournalError, ValueError, OSError, sqlite3.Error) as error:
        from lite_pi.tui.text import plain

        print(plain(f"启动失败：{error}"), file=sys.stderr)
        return 2


def _run_limits(args: argparse.Namespace) -> RunLimits:
    limits = _task_limits(args)
    limits["max_turns"] = args.max_turns
    limits["max_tokens"] = CLI_OUTPUT_TOKENS
    limits["on_event"] = _progress
    return limits


def _task_limits(args: argparse.Namespace) -> RunLimits:
    limits: RunLimits = {}
    if getattr(args, "context_tokens", None) is not None:
        limits["context_tokens"] = args.context_tokens
    if getattr(args, "approvals", False):
        limits["approvals"] = True
    if getattr(args, "max_task_calls", None) is not None:
        limits["max_task_calls"] = args.max_task_calls
    if getattr(args, "max_task_seconds", None) is not None:
        limits["max_task_seconds"] = args.max_task_seconds
    if getattr(args, "max_task_output", None) is not None:
        limits["max_task_output_tokens"] = args.max_task_output
    if getattr(args, "max_retrievals", None) is not None:
        limits["max_retrievals"] = args.max_retrievals
    return limits


def _run_interruptible(session: AgentSession, operation: Callable[[], JsonObject]) -> JsonObject:
    previous = signal.getsignal(signal.SIGINT)

    def handle(_signum: int, _frame: FrameType | None) -> None:
        session.cancel()

    signal.signal(signal.SIGINT, handle)
    try:
        return operation()
    finally:
        signal.signal(signal.SIGINT, previous)
