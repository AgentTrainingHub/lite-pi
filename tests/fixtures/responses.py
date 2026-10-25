"""Controlled model values for deterministic contracts only."""

import json

from lite_pi.provider import ModelResult


def model(content: str, *, calls: list | None = None, attempts: int = 1) -> ModelResult:
    message: dict = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return ModelResult(
        "stop",
        message,
        raw_stop_reason="tool_calls" if calls else "stop",
        actual_model="deterministic-fixture",
        attempts=attempts,
        usage={"prompt_tokens": 11, "completion_tokens": 4},
    )


def tool_call(name: str, arguments: dict, identity: str) -> dict:
    return {
        "id": identity,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
