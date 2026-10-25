"""Explicit local protocol peer configuration, independent process observations."""

import sys
from pathlib import Path

from lite_pi.tools.contracts import JsonObject

FIXTURE = Path(__file__).parent / "fixtures/mcp_server.py"


def fixture_server(
    name: str, state: Path, *, mode: str = "serve", tools: list[str] | None = None, timeout: float = 8
) -> JsonObject:
    return {
        "name": name,
        "command": sys.executable,
        "args": [str(FIXTURE), "--name", name, "--mode", mode, "--state", str(state)],
        "tools": ["mark"] if tools is None else tools,
        "timeout_seconds": timeout,
    }


def observed_pid(state: Path) -> int:
    return int((state / "pid").read_text())
