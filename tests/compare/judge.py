"""Objective judgments from files, check processes, and request receipts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from lite_pi.spend.ledger import Ledger
from lite_pi.tools.contracts import JsonObject

FIX_TASK = (
    "python3 check.py fails. Fix the implementation so the check passes. "
    "Do not modify check.py. Use the existing unit conversion."
)
HELDOUT_TASK = "check.py 现在失败。请让它通过，不要修改 check.py。请阅读项目文件，修改实现后再运行检查。"


def sha_tree(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            hashes[relative] = "symlink:" + os.readlink(path)
            continue
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def check_unchanged(before: dict[str, str], after: dict[str, str]) -> bool:
    return before.get("check.py") == after.get("check.py") and "check.py" in before


def implementation_changed(before: dict[str, str], after: dict[str, str]) -> bool:
    return any(
        path != "check.py" and before.get(path) != after.get(path) for path in set(before) | set(after)
    )


def tool_feedback(messages: list[JsonObject]) -> bool:
    saw_tool = False
    for item in messages:
        role = item.get("role")
        if role == "tool":
            saw_tool = True
        elif saw_tool and role == "assistant":
            return True
    return False


def usage_since(offset: int, run_ids: set[str]) -> dict[str, int]:
    path = Ledger().calls_path
    prompt = completion = calls = unknown = 0
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            handle.seek(offset)
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("run_id") not in run_ids:
                    continue
                calls += 1
                usage = item.get("usage") if isinstance(item.get("usage"), dict) else {}
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                    prompt += prompt_tokens
                    completion += completion_tokens
                else:
                    unknown += 1
    return {
        "calls": calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "unknown_usage": unknown,
    }


def call_offset() -> int:
    path = Ledger().calls_path
    return path.stat().st_size if path.exists() else 0


def product_messages(result: JsonObject) -> list[JsonObject]:
    messages = result.get("messages")
    if not isinstance(messages, list):
        return []
    return [item for item in messages if isinstance(item, dict)]


def product_run_ids(results: list[JsonObject]) -> set[str]:
    found: set[str] = set()
    for result in results:
        run_id = result.get("run_id")
        if isinstance(run_id, str):
            found.add(run_id)
    return found
