"""Raw argument and tool-response fixture constructors."""

from lite_pi.provider import ModelResult


def call(name: str, arguments: str = "{}", identity: str = "call-boundary") -> dict:
    return {"id": identity, "type": "function", "function": {"name": name, "arguments": arguments}}


def response(*calls: dict, finish: str = "tool_calls") -> ModelResult:
    return ModelResult(
        finish,
        {"role": "assistant", "content": "", "tool_calls": list(calls)},
        raw_stop_reason=finish,
        actual_model="deterministic-fixture",
        attempts=0,
    )
