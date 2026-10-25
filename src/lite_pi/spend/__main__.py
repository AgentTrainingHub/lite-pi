"""Spend ledger commands: init, status."""

from __future__ import annotations

import json
import sys

from lite_pi.spend.gate import fetch_balance
from lite_pi.spend.ledger import Ledger


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "status"
    if command == "init":
        balance = fetch_balance()
        result = Ledger().bootstrap(
            balance["total_balance"],
            currency=balance["currency"],
            is_available=balance["is_available"],
        )
        print(
            json.dumps({"balance_cny": str(balance["total_balance"]), **result["status"]}, ensure_ascii=False)
        )
        return 0 if not result["status"]["stopped"] else 2
    if command == "status":
        status = Ledger().status()
        print(json.dumps(status, ensure_ascii=False))
        return 0 if status and not status["stopped"] else 2
    raise SystemExit("Usage: python3 -m lite_pi.spend init|status")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
