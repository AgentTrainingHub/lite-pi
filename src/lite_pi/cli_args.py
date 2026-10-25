"""CLI grammar and numeric validation; does not start a session."""

import argparse
import math
import sys
from pathlib import Path


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是有限正数")
    return number


def parse(argv: list[str] | None) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(prog="lite-pi")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("submit", "interrupt", "code", "resume", "branch"):
        sub = commands.add_parser(name)
        if name in {"submit", "interrupt", "code"}:
            sub.add_argument("--root", type=Path, default=Path.cwd())
        if name in {"code", "resume", "branch"}:
            sub.add_argument("--session", type=Path, required=True)
        if name in {"code", "resume"}:
            sub.add_argument("--embedding-dir", type=Path)
            sub.add_argument("--max-turns", type=positive_int, default=6)
            sub.add_argument("--approvals", action="store_true")
            sub.add_argument("--max-task-calls", type=positive_int)
            sub.add_argument("--max-task-seconds", type=positive_float)
            sub.add_argument("--max-task-output", type=positive_int)
            sub.add_argument("--max-retrievals", type=positive_int)
            sub.add_argument("--context-tokens", type=positive_int)
        if name == "code":
            sub.add_argument("--planning", action="store_true")
            sub.add_argument("--knowledge", action="store_true")
            sub.add_argument("--delegate", action="store_true")
        if name == "resume":
            sub.add_argument("--leaf")
        if name == "branch":
            sub.add_argument("--from", dest="parent", required=True)
        if name != "resume":
            sub.add_argument("task")
    approve = commands.add_parser("approve")
    approve.add_argument("--session", type=Path, required=True)
    approve.add_argument("--id", required=True)
    approve.add_argument("--decision", required=True, choices=("approve", "reject", "revoke"))
    plan = commands.add_parser("plan", help="查看持久计划及证据")
    plan.add_argument("--session", type=Path, required=True)
    plan.add_argument("--change", help="包含 expected_version、reason、steps、invalidated_steps 的 JSON")
    answer = commands.add_parser("answer", help="回答或取消持久澄清，不自动执行任务")
    answer.add_argument("--session", type=Path, required=True)
    answer.add_argument("--id", required=True)
    answer.add_argument(
        "--version", type=positive_int, required=True, help="等待回执中的 question.expected_version"
    )
    answer_input = answer.add_mutually_exclusive_group(required=True)
    answer_input.add_argument("--text")
    answer_input.add_argument("--cancel", action="store_true")
    commands.add_parser("delegate")
    for action in ("confirm", "revoke", "resolve"):
        memory = commands.add_parser(f"memory-{action}")
        memory.add_argument("--session", type=Path, required=True)
        memory.add_argument("--id", required=True)
        if action == "confirm":
            memory.add_argument("--supersedes")
            memory.add_argument("--ttl", type=positive_float)
    for action in ("index", "retrieval-trace"):
        knowledge = commands.add_parser(action)
        knowledge.add_argument("--root", type=Path, default=Path.cwd())
        knowledge.add_argument("--embedding-dir", type=Path)
        if action == "index":
            knowledge.add_argument("--mode", choices=("bm25", "semantic", "hybrid"), default="bm25")
    chat = commands.add_parser("chat", help="持续交互式终端")
    chat.add_argument("--root", type=Path, default=Path.cwd())
    chat.add_argument("--session", type=Path)
    chat.add_argument("--max-turns", type=positive_int, default=8)
    chat.add_argument("--max-task-calls", type=positive_int)
    chat.add_argument("--max-task-seconds", type=positive_float)
    chat.add_argument("--max-task-output", type=positive_int)
    chat.add_argument("--max-retrievals", type=positive_int)
    chat.add_argument("--context-tokens", type=positive_int)
    chat.add_argument("--approvals", action="store_true")
    chat.add_argument("--knowledge", action="store_true")
    chat.add_argument("--delegate", action="store_true")
    chat.add_argument("--planning", action="store_true")
    chat.add_argument("--embedding-dir", type=Path)
    parsed = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(parsed or ["chat"])
    return parser, args
