"""Scripted HTTP boundary with an isolated ledger; never live-model evidence."""

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

from lite_pi.spend.ledger import Ledger


def response(
    content: str, *, calls: list[dict] | None = None, tokens: int = 4, status: int = 200, stream: bool = True
) -> dict:
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [dict(call, index=i) for i, call in enumerate(calls)]
    payload = {
        "id": "deterministic-response",
        "model": "deepseek-flash",
        "choices": [{"delta": message, "finish_reason": "tool_calls" if calls else "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": tokens, "total_tokens": 11 + tokens},
    }
    if not stream:
        payload["choices"] = [{"message": message, "finish_reason": "tool_calls" if calls else "stop"}]
    return {
        "http_status": status,
        "body": (
            b"data: " + json.dumps(payload).encode() + b"\n\ndata: [DONE]\n\n"
            if stream
            else json.dumps(payload).encode()
        )
        if status == 200
        else b'{"error":"fixture failure"}',
        "content_type": "text/event-stream" if status == 200 else "application/json",
        "error": None,
    }


@contextmanager
def transport(base: Path, responses: list[dict] | Callable[..., dict]) -> Iterator[MagicMock]:
    base.mkdir(parents=True)
    ledger = Ledger(base / "ledger.json", base / "calls.jsonl")
    ledger.bootstrap(Decimal("100"), currency="CNY", is_available=True)
    with (
        patch("lite_pi.spend.gate.load_api_key", return_value="deterministic-not-a-secret"),
        patch("lite_pi.spend.gate.Ledger", return_value=ledger),
        patch("lite_pi.spend.gate.EVIDENCE_ROOT", base / "calls"),
        patch("lite_pi.spend.gate._refresh", return_value=None),
        patch("lite_pi.spend.gate._read_response", side_effect=responses) as wire,
    ):
        yield wire
