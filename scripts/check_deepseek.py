"""Metered connectivity check. One real deepseek-flash request through the spend ledger."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lite_pi.spend.gate import SpendBlocked, transmit  # noqa: E402


def main() -> int:
    try:
        result = transmit(
            {
                "model": "deepseek-flash",
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 64,
                "stream": False,
            },
            run_id="setup-check",
            purpose="connectivity",
        )
    except SpendBlocked as blocked:
        print(json.dumps({"status": "blocked", "reason": blocked.reason}, ensure_ascii=False))
        return 2
    usage = result["usage"] or {}
    body = json.loads(result["body"].decode()) if result["body"] else {}
    choice = (body.get("choices") or [{}])[0]
    receipt = {
        "status": "connected" if result["http_status"] == 200 else "blocked",
        "http_status": result["http_status"],
        "requested_model": "deepseek-flash",
        "actual_model": body.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
        "reservation_id": result["reservation_id"],
        "evidence": result["evidence"],
        "scope": "metered connectivity",
    }
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if receipt["status"] == "connected" else 1


if __name__ == "__main__":
    sys.exit(main())
