"""Shared shop and browser evidence helpers for browser tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fixtures.shop import Shop, ShopServer

from lite_pi.browser.driver import browser_alive


def start_shop(path: Path, fault: str = "none") -> ShopServer:
    path.parent.mkdir(parents=True, exist_ok=True)
    server = ShopServer(Shop(path, fault))
    server.start()
    return server


def write_mcp(work: Path, origin: str, evidence: Path, *, command: str | None = None) -> None:
    config = work / ".lite-pi"
    config.mkdir(parents=True, exist_ok=True)
    spec = {
        "name": "shop",
        "command": command or sys.executable,
        "args": [
            "-m",
            "lite_pi.browser.mcp_server",
            "--origin",
            origin,
            "--evidence",
            str(evidence),
            "--channel",
            "chrome",
        ],
        "tools": ["open", "observe", "act", "query"],
        "origins": [origin],
        "read_only_tools": ["open", "observe", "query"],
        "timeout_seconds": 45,
    }
    (config / "mcp.json").write_text(json.dumps({"servers": [spec]}), encoding="utf-8")


def load_shop(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def results(run: dict) -> list[dict]:
    found: list[dict] = []
    for turn in run.get("transcript") or []:
        if not isinstance(turn, dict):
            continue
        for item in turn.get("results") or []:
            if isinstance(item, dict):
                found.append(item)
    return found


def text_of(item: dict) -> str:
    value = item.get("text")
    return value if isinstance(value, str) else ""


def browser_closed(evidence: Path) -> bool:
    path = evidence / "closed.json"
    if not path.is_file():
        return False
    body = json.loads(path.read_text(encoding="utf-8"))
    pid = body.get("pid")
    return bool(body.get("closed")) and isinstance(pid, int) and not browser_alive(pid)
