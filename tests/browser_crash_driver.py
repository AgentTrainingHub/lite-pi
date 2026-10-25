"""Crash a scripted browser action after the intent is stored. Not a real-model run."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

from lite_pi.journal import Journal
from lite_pi.provider import ModelResult
from lite_pi.session import AgentSession


def main() -> None:
    root = Path(sys.argv[1])
    journal = Journal(Path(sys.argv[2]))
    session = AgentSession(root, tools=journal.authorized_tools, journal=journal)
    call = {
        "id": "call-act",
        "type": "function",
        "function": {
            "name": "mcp__shop__act",
            "arguments": json.dumps(
                {"observation": "obs-1", "node_id": "submit-notebook", "action": "click"}
            ),
        },
    }
    result = ModelResult(
        "stop",
        {"role": "assistant", "content": "", "tool_calls": [call]},
        raw_stop_reason="tool_calls",
        actual_model="deterministic-fixture",
        attempts=1,
        usage={"prompt_tokens": 11, "completion_tokens": 4},
    )
    with patch("lite_pi.loop.complete", return_value=result):
        session.submit("order", max_turns=2, max_task_calls=2)


if __name__ == "__main__":
    from crash_injection import install

    install()
    main()
