"""Normalize provider responses before any assistant action enters the loop."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, cast

from lite_pi.spend.gate import transmit
from lite_pi.spend.pricing import MODEL
from lite_pi.sse import StreamAssembler
from lite_pi.task_budget import RequestBudget
from lite_pi.tools.contracts import JsonObject

type ModelStatus = Literal["stop", "length", "tool_calls", "content_filter", "aborted", "error"]


@dataclass(frozen=True)
class ModelResult:
    status: ModelStatus
    message: JsonObject = field(default_factory=dict)
    raw_stop_reason: str | None = None
    actual_model: str | None = None
    usage: JsonObject | None = None
    evidence: str | None = None
    http_status: int | None = None
    error: str | None = None
    attempts: int = 0


def complete(
    messages: list[JsonObject],
    *,
    definitions: list[JsonObject],
    run_id: str,
    turn: int,
    max_tokens: int,
    interrupt_after_chunks: int | None = None,
    max_attempts: int = 2,
    stream: bool = False,
    on_delta: Callable[[str], None] | None = None,
    stop: Callable[[], bool] | None = None,
    purpose: str | None = None,
    budget: RequestBudget | None = None,
) -> ModelResult:
    payload = {"model": MODEL, "messages": messages, "max_tokens": max_tokens}
    if definitions:
        payload["tools"] = definitions
    streaming = stream or interrupt_after_chunks is not None
    if streaming:
        payload["stream"] = True
    assembler = StreamAssembler()

    def on_chunk(chunk: bytes) -> None:
        delta = assembler.feed(chunk)
        if delta and on_delta is not None:
            on_delta(delta)

    response = transmit(
        payload,
        run_id=run_id,
        purpose=purpose or f"session-turn-{turn}",
        max_output=max_tokens,
        stop_after_chunks=interrupt_after_chunks,
        max_attempts=max_attempts,
        on_chunk=on_chunk if streaming else None,
        stop=stop,
        budget=budget,
    )
    common = {
        "usage": response["usage"],
        "evidence": response["evidence"],
        "http_status": response["http_status"],
        "attempts": response["attempts"],
    }
    error = response["settled"].get("error")
    if error == "aborted":
        message, _finish, model = _assemble_stream(response["body"], partial=True)
        return ModelResult("aborted", message, actual_model=model, error=error, **common)
    if streaming and response["http_status"] == 200:
        try:
            message, finish, model = _assemble_stream(response["body"])
            _validate_message(message, finish)
            if not isinstance(finish, str):
                raise ValueError("missing or unknown finish reason")
            return ModelResult(cast(ModelStatus, finish), message, finish, model, **common)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            return ModelResult("error", error=f"invalid_response: {exc}", **common)
    if error or response["http_status"] != 200:
        return ModelResult("error", error=error or f"HTTP {response['http_status']}", **common)
    try:
        body = json.loads(response["body"])
        choice = body["choices"][0]
        message = choice["message"]
        finish = choice["finish_reason"]
        _validate_message(message, finish)
        if not isinstance(finish, str):
            raise ValueError("missing or unknown finish reason")
        return ModelResult(cast(ModelStatus, finish), message, finish, body.get("model"), **common)
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        return ModelResult("error", error=f"invalid_response: {exc}", **common)


def _validate_message(message: JsonObject, finish: str | None) -> None:
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("invalid assistant message")
    if message.get("content") is not None and not isinstance(message["content"], str):
        raise ValueError("invalid assistant content")
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise ValueError("invalid tool calls")
    ids = set()
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not call["id"]:
            raise ValueError("missing call ID")
        if call["id"] in ids or not isinstance(call.get("function"), dict):
            raise ValueError("invalid call identity")
        function = call["function"]
        if (
            not isinstance(function.get("name"), str)
            or not function["name"]
            or not isinstance(function.get("arguments"), str)
        ):
            raise ValueError("invalid function call")
        ids.add(call["id"])
    if finish not in {"stop", "length", "tool_calls", "content_filter"}:
        raise ValueError("missing or unknown finish reason")
    if calls and finish not in {"tool_calls", "length"}:
        raise ValueError("tool calls conflict with finish reason")
    if finish == "tool_calls" and not calls:
        raise ValueError("empty tool call response")


def _assemble_stream(
    raw: bytes | bytearray, *, partial: bool = False
) -> tuple[JsonObject, str | None, str | None]:
    assembler = StreamAssembler()
    assembler.feed(bytes(raw))
    assembler.finish(partial=partial)
    return assembler.message(), assembler.finish_reason, assembler.model
