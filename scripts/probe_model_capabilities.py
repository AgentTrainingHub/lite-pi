"""Model capability probes. Real deepseek-flash calls go through the spend gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lite_pi.spend.gate import SpendBlocked, transmit  # noqa: E402

OUT = ROOT / ".artifacts" / "acceptance" / ("baseline-capability-" + uuid4().hex)


def judge_completion(result: dict) -> dict:
    body = json.loads(result["body"].decode())
    choice = body["choices"][0]
    message = choice["message"]
    usage = result["usage"] or {}
    model = body.get("model") or ""
    return {
        "http_status": result["http_status"],
        "finish_reason": choice.get("finish_reason"),
        "actual_model": model,
        "has_usage": "prompt_tokens" in usage and "completion_tokens" in usage,
        "content_present": bool(message.get("content")),
        "model_is_flash": "flash" in model,
        "reservation_id": result["reservation_id"],
        "evidence": result["evidence"],
        "settled_cny": (result["settled"] or {}).get("settled_cny"),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"probes": []}
    try:
        plain = transmit(
            {
                "model": "deepseek-flash",
                "messages": [{"role": "user", "content": "Reply with the single word OK."}],
                "max_tokens": 64,
                "stream": False,
            },
            run_id="baseline-capability",
            purpose="non-stream",
        )
        streamed = transmit(
            {
                "model": "deepseek-flash",
                "messages": [{"role": "user", "content": "Reply with the single word OK."}],
                "max_tokens": 64,
                "stream": True,
            },
            run_id="baseline-capability",
            purpose="stream",
        )
        tooled = transmit(
            {
                "model": "deepseek-flash",
                "messages": [{"role": "user", "content": "Call echo_token with token BASELINE."}],
                "max_tokens": 256,
                "stream": False,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "echo_token",
                            "description": "Echo a token.",
                            "parameters": {
                                "type": "object",
                                "properties": {"token": {"type": "string"}},
                                "required": ["token"],
                            },
                        },
                    }
                ],
                "tool_choice": {"type": "function", "function": {"name": "echo_token"}},
            },
            run_id="baseline-capability",
            purpose="tool-call",
        )
    except SpendBlocked as blocked:
        report["status"] = "blocked"
        report["reason"] = blocked.reason
        (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False))
        return 2

    plain_judge = judge_completion(plain)
    plain_judge["name"] = "non-stream"
    plain_judge["pass"] = (
        plain_judge["http_status"] == 200
        and plain_judge["finish_reason"] == "stop"
        and plain_judge["has_usage"]
        and plain_judge["model_is_flash"]
        and plain_judge["content_present"]
    )
    stream_text = streamed["body"].decode(errors="replace")
    stream_judge = {
        "name": "stream",
        "http_status": streamed["http_status"],
        "saw_sse": "data:" in stream_text,
        "has_usage": streamed["usage"] is not None and "prompt_tokens" in (streamed["usage"] or {}),
        "actual_model": None,
        "reservation_id": streamed["reservation_id"],
        "evidence": streamed["evidence"],
        "settled_cny": (streamed["settled"] or {}).get("settled_cny"),
    }
    for line in stream_text.splitlines():
        if line.startswith("data:") and "model" in line:
            try:
                event = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            if event.get("model"):
                stream_judge["actual_model"] = event["model"]
                break
    stream_judge["model_is_flash"] = "flash" in (stream_judge["actual_model"] or "")
    stream_judge["pass"] = (
        stream_judge["http_status"] == 200
        and stream_judge["saw_sse"]
        and stream_judge["has_usage"]
        and stream_judge["model_is_flash"]
    )
    tool_body = json.loads(tooled["body"].decode())
    tool_choice = tool_body["choices"][0]
    tool_calls = (tool_choice.get("message") or {}).get("tool_calls") or []
    names = [call.get("function", {}).get("name") for call in tool_calls]
    tool_judge = {
        "name": "tool-call",
        "http_status": tooled["http_status"],
        "finish_reason": tool_choice.get("finish_reason"),
        "tool_names": names,
        "has_usage": tooled["usage"] is not None,
        "actual_model": tool_body.get("model"),
        "reservation_id": tooled["reservation_id"],
        "evidence": tooled["evidence"],
        "settled_cny": (tooled["settled"] or {}).get("settled_cny"),
        "forced_tool_choice": True,
    }
    tool_judge["pass"] = (
        tool_judge["http_status"] == 200
        and "echo_token" in names
        and tool_judge["has_usage"]
        and "flash" in (tool_judge["actual_model"] or "")
    )
    if tool_judge["http_status"] == 200 and "echo_token" not in names:
        tool_judge["coverage"] = "not-covered"
    report["probes"] = [plain_judge, stream_judge, tool_judge]
    report["status"] = "pass" if all(item["pass"] for item in report["probes"]) else "fail"
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "probes": [
                    {key: item[key] for key in ("name", "pass", "http_status", "settled_cny")}
                    for item in report["probes"]
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
