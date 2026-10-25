"""Verify a saved Judge episode against its request, response and spend receipts."""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path

from lite_pi.spend.gate import EVIDENCE_ROOT
from lite_pi.sse import StreamAssembler
from lite_pi.tools.contracts import JsonObject


def load(path: Path) -> JsonObject:
    value: JsonObject = json.loads(path.read_text())
    return value


def verify_episode(directory: Path, prompt: str) -> tuple[JsonObject, Decimal, int]:
    expected = {"prompt": prompt, "tools": [], "max_calls": 1, "max_output": 512, "context_tokens": 4096}
    if load(directory / "request-input.json") != expected:
        raise ValueError("judge_input_changed")
    result = load(directory / "result.json")
    budget = load(directory / "budget.json")
    attempts = json.loads((directory / "attempts.json").read_text())
    if (
        result.get("tools") != []
        or budget["max_calls"] != 1
        or budget["max_output"] != 512
        or budget["reserved_calls"] != 0
        or budget["reserved_output"] != 0
        or budget["spent_calls"] + budget["unknown_calls"] != len(attempts)
        or len(attempts) != 1
        or any(t["tool_calls"] or t["results"] for t in result["transcript"])
    ):
        raise ValueError("judge_episode_boundary_mismatch")
    charged = Decimal(0)
    unknown = 0
    spent_output = 0
    unknown_output = 0
    for attempt in attempts:
        identity = attempt["reservation_id"]
        if not isinstance(identity, str) or not re.fullmatch(r"[a-f0-9]{32}", identity):
            raise ValueError("invalid_judge_reservation")
        root = EVIDENCE_ROOT / identity
        receipt, request = load(root / "receipt.json"), load(root / "request.json")
        if (
            attempt["run_id"] != result["run_id"]
            or receipt["run_id"] != result["run_id"]
            or receipt["reservation_id"] != identity
            or receipt["settled_cny"] != attempt["settled_cny"]
            or receipt["usage_unknown"] != attempt["usage_unknown"]
            or request.get("tools")
            or request["model"] != "deepseek-flash"
            or request["max_tokens"] != 512
            or [m["content"] for m in request["messages"] if m["role"] == "user"] != [prompt]
        ):
            raise ValueError("judge_request_receipt_mismatch")
        if receipt["http_status"] == 200 and not receipt.get("error"):
            stream = StreamAssembler()
            stream.feed((root / "response.bin").read_bytes())
            stream.finish()
            if (
                stream.model != result["model"]
                or stream.finish_reason not in ("stop", "length", "content_filter")
                or result["status"] != stream.finish_reason
                or stream.calls
                or stream.message()["content"] != result["text"]
            ):
                raise ValueError("judge_response_mismatch")
        elif result["status"] not in ("error", "aborted", "budget"):
            raise ValueError("judge_failed_response_status_mismatch")
        charged += Decimal(receipt["settled_cny"])
        unknown += bool(receipt["usage_unknown"])
        if receipt["usage_unknown"]:
            unknown_output += request["max_tokens"]
        else:
            spent_output += receipt["usage"]["completion_tokens"]
    if (
        budget["spent_output"] != spent_output
        or budget["unknown_output"] != unknown_output
        or budget["unknown_calls"] != unknown
        or budget["spent_calls"] != len(attempts) - unknown
    ):
        raise ValueError("judge_budget_projection_mismatch")
    return result, charged, unknown
