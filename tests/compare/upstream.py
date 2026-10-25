"""Same fixture repair through the frozen Pi CLI proxy and the public Python session."""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

from support import check

from compare.judge import FIX_TASK, check_unchanged, implementation_changed, sha_tree
from lite_pi.journal import Journal
from lite_pi.session import CODE_TOOLS, AgentSession
from lite_pi.tools.contracts import JsonObject

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = os.environ.get("LITE_PI_UPSTREAM")
PI_CLI = (
    Path(UPSTREAM_ROOT).expanduser() / "packages/coding-agent/dist/bundle/cli.js" if UPSTREAM_ROOT else None
)
FIXTURE = ROOT / "baselines" / "coding" / "fixture"


def pi_available() -> bool:
    return PI_CLI is not None and PI_CLI.is_file() and shutil.which("node") is not None


def run_lite(parent: Path) -> JsonObject:
    work = parent / "project"
    shutil.copytree(FIXTURE, work)
    before = sha_tree(work)
    journal = Journal.create(parent / "session.jsonl", work, tools=CODE_TOOLS)
    started = time.monotonic()
    result = AgentSession(work, tools=CODE_TOOLS, journal=journal).submit(
        FIX_TASK, max_turns=12, max_tokens=2048, max_task_calls=12
    )
    checked = check(work)
    after = sha_tree(work)
    return {
        "arm": "lite-pi",
        "run_id": result.get("run_id"),
        "model": result.get("model"),
        "status": result.get("status"),
        "elapsed": round(time.monotonic() - started, 3),
        "check": checked["returncode"],
        "passed": result.get("status") == "stop"
        and checked["returncode"] == 0
        and check_unchanged(before, after)
        and implementation_changed(before, after),
        "safe": check_unchanged(before, after),
    }


def run_pi(parent: Path, index: int, port: int) -> JsonObject:
    sys.path.insert(0, str(ROOT / "scripts"))
    import compare_upstream

    receipt = compare_upstream.run_case(
        "fix", index, FIX_TASK, "read,edit,write,bash", 12, port, output=parent
    )
    return {
        "arm": "upstream-pi",
        "run_id": receipt.get("run_id"),
        "elapsed": receipt.get("elapsed_s"),
        "check": receipt.get("check_returncode"),
        "passed": bool(receipt.get("pass")),
        "safe": receipt.get("env_has_deepseek_key") is False
        and "check.py" not in (receipt.get("changed_files") or []),
        "proxy_calls": receipt.get("proxy_calls"),
        "tool_note": "both arms use read, edit, write, bash; 12 calls, 2048 output tokens per request",
    }
