"""Cross-process spend ledger. Restarting a process does not reset the round."""

from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from lite_pi.process_identity import process_identity
from lite_pi.runtime import PATHS
from lite_pi.spend.pricing import (
    ALLOWANCE_CAP_CNY,
    INPUT_HIT_CNY,
    INPUT_MISS_CNY,
    MAX_IN_FLIGHT,
    MODEL,
    OUTPUT_CNY,
    RESERVE_FLOOR_CNY,
    TEAM_MAX_IN_FLIGHT,
    allowance_cny,
    q,
)
from lite_pi.storage import atomic_write, locked
from lite_pi.tools.contracts import JsonObject

LEDGER_PATH = PATHS.state / "spend" / "ledger.json"
CALLS_PATH = PATHS.state / "spend" / "calls.jsonl"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _dec(value: object) -> Decimal:
    return Decimal(str(value))


class Ledger:
    def __init__(self, path: Path | None = None, calls_path: Path | None = None) -> None:
        self.path = path or LEDGER_PATH
        self.calls_path = calls_path or CALLS_PATH

    def _update[T](self, fn: Callable[[JsonObject | None], tuple[JsonObject | None, T]]) -> T:
        with locked(self.path.with_suffix(".lock")):
            raw = self.path.read_text() if self.path.exists() else None
            if raw is not None and not raw.strip():
                raise ValueError("empty ledger; refusing to reset spend")
            data = json.loads(raw) if raw is not None else None
            if data is not None and (
                not isinstance(data, dict)
                or not isinstance(data.get("round_id"), str)
                or not data["round_id"].strip()
            ):
                raise ValueError("invalid ledger or round; refusing to reset spend")
            if data:
                self._recover_orphans(data)
            data, result = fn(data)
            updated = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
            if data is not None and updated != raw:
                atomic_write(self.path, updated.encode())
            return result

    def _recover_orphans(self, data: JsonObject) -> None:
        # Keep every unknown charge; release only the dead process's concurrency slot.
        live = []
        for item in data["in_flight"]:
            pid = item.get("owner_pid")
            if pid is None:
                raise ValueError("legacy in-flight reservation requires manual reconciliation")
            identity = process_identity(pid)
            if identity is None or identity != item.get("owner_identity"):
                reserved = _dec(item["reserved_cny"])
                data["spent_cny"] = q(_dec(data["spent_cny"]) + reserved)
                data["reserved_cny"] = q(_dec(data["reserved_cny"]) - reserved)
                data["unknown_usage_calls"] += 1
                data["request_count"] += 1
                data.setdefault("recovered_calls", []).append(
                    {
                        **item,
                        "error": "owner-exited",
                        "usage_unknown": True,
                        "settled_cny": item["reserved_cny"],
                        "settled_at": _now(),
                    }
                )
            else:
                live.append(item)
        data["in_flight"] = live
        self._apply_floor(data)

    def _append_call(self, record: JsonObject) -> None:
        self.calls_path.parent.mkdir(parents=True, exist_ok=True)
        with self.calls_path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def bootstrap(self, balance_cny: Decimal, *, currency: str, is_available: bool) -> JsonObject:
        def apply(data: JsonObject | None) -> tuple[JsonObject | None, JsonObject]:
            if data is not None:
                data["last_balance_cny"] = q(balance_cny)
                data["balance_spent_cny"] = data["spent_cny"]
                data["balance_checks"].append(
                    {"at": _now(), "total_balance_cny": q(balance_cny), "source": "reopen"}
                )
                self._apply_floor(data)
                return data, {"created": False, "status": self._public(data)}
            allowance = allowance_cny(balance_cny)
            created = {
                "round_id": uuid4().hex,
                "currency": currency,
                "model": MODEL,
                "b0_cny": q(balance_cny),
                "allowance_cny": q(allowance),
                "allowance_cap_cny": q(ALLOWANCE_CAP_CNY),
                "reserve_floor_cny": q(RESERVE_FLOOR_CNY),
                "prices_cny_per_token": {
                    "input_miss_peak": q(INPUT_MISS_CNY),
                    "input_hit_peak": q(INPUT_HIT_CNY),
                    "output_peak": q(OUTPUT_CNY),
                    "reservation": "peak cache-miss input and peak output",
                },
                "max_in_flight": MAX_IN_FLIGHT,
                "spent_cny": q(Decimal("0")),
                "balance_spent_cny": q(Decimal("0")),
                "reserved_cny": q(Decimal("0")),
                "unknown_usage_calls": 0,
                "request_count": 0,
                "in_flight": [],
                "balance_checks": [
                    {
                        "at": _now(),
                        "total_balance_cny": q(balance_cny),
                        "is_available": is_available,
                        "source": "bootstrap",
                    }
                ],
                "last_balance_cny": q(balance_cny),
                "stopped": allowance <= 0 or not is_available,
                "stop_reason": None if allowance > 0 and is_available else "allowance-or-unavailable",
                "created_at": _now(),
            }
            self._apply_floor(created)
            return created, {"created": True, "status": self._public(created)}

        return self._update(apply)

    def refresh_balance(self, balance_cny: Decimal, *, is_available: bool, source: str) -> JsonObject:
        def apply(data: JsonObject | None) -> tuple[JsonObject | None, JsonObject]:
            if not data:
                raise ValueError("Ledger is not bootstrapped.")
            data["last_balance_cny"] = q(balance_cny)
            data["balance_spent_cny"] = data["spent_cny"]
            data["balance_checks"].append(
                {
                    "at": _now(),
                    "total_balance_cny": q(balance_cny),
                    "is_available": is_available,
                    "source": source,
                }
            )
            if not is_available:
                data["stopped"] = True
                data["stop_reason"] = "unavailable"
            elif data.get("stop_reason") == "unavailable":
                data["stopped"] = False
                data["stop_reason"] = None
            self._apply_floor(data)
            return data, self._public(data)

        return self._update(apply)

    def require_status(self) -> JsonObject:
        status = self.status()
        if status is None:
            raise ValueError("spend ledger has not been initialized")
        return status

    def status(self) -> JsonObject | None:
        def apply(data: JsonObject | None) -> tuple[JsonObject | None, JsonObject | None]:
            return data, None if data is None else self._public(data)

        return self._update(apply)

    def try_reserve(self, *, run_id: str, purpose: str, input_tokens: int, output_tokens: int) -> JsonObject:
        reserve = _dec(input_tokens) * INPUT_MISS_CNY + _dec(output_tokens) * OUTPUT_CNY

        def apply(data: JsonObject | None) -> tuple[JsonObject | None, JsonObject]:
            if not data:
                return data, {"ok": False, "reason": "no-ledger"}
            reason = self._block_reason(data, reserve)
            if reason:
                return data, {"ok": False, "reason": reason}
            if len(data["in_flight"]) >= int(data["max_in_flight"]):
                return data, {"ok": False, "reason": "busy"}
            reservation_id = uuid4().hex
            record = {
                "reservation_id": reservation_id,
                "run_id": run_id,
                "purpose": purpose,
                "input_tokens_est": input_tokens,
                "output_tokens_est": output_tokens,
                "reserved_cny": q(reserve),
                "started_at": _now(),
                "owner_pid": os.getpid(),
                "owner_identity": process_identity(os.getpid()),
            }
            data["in_flight"].append(record)
            data["reserved_cny"] = q(_dec(data["reserved_cny"]) + reserve)
            return data, {"ok": True, **record}

        return self._update(apply)

    def reserve_wait(
        self,
        *,
        timeout: float = 180,
        stop: Callable[[], bool] | None = None,
        run_id: str,
        purpose: str,
        input_tokens: int,
        output_tokens: int,
    ) -> JsonObject:
        deadline = time.monotonic() + timeout
        while True:
            if stop is not None and stop():
                return {"ok": False, "reason": "cancelled"}
            result = self.try_reserve(
                run_id=run_id, purpose=purpose, input_tokens=input_tokens, output_tokens=output_tokens
            )
            if result.get("ok") or result.get("reason") != "busy":
                return result
            if time.monotonic() >= deadline:
                return {"ok": False, "reason": "busy-timeout"}
            time.sleep(0.05)

    def settle(
        self,
        reservation_id: str,
        *,
        usage: JsonObject | None,
        http_status: int | None,
        actual_model: str | None,
        response_id: str | None,
        error: str | None = None,
    ) -> JsonObject:
        def apply(data: JsonObject | None) -> tuple[JsonObject | None, JsonObject]:
            if not data:
                return data, {"ok": False, "reason": "no-ledger"}
            match = next(
                (item for item in data["in_flight"] if item["reservation_id"] == reservation_id), None
            )
            if match is None:
                return data, {"ok": False, "reason": "unknown-reservation"}
            reserved = _dec(match["reserved_cny"])
            cost, unknown = _cost_from_usage(usage, reserved)
            data["in_flight"] = [
                item for item in data["in_flight"] if item["reservation_id"] != reservation_id
            ]
            data["reserved_cny"] = q(_dec(data["reserved_cny"]) - reserved)
            data["spent_cny"] = q(_dec(data["spent_cny"]) + cost)
            data["request_count"] = int(data["request_count"]) + 1
            if unknown:
                data["unknown_usage_calls"] = int(data["unknown_usage_calls"]) + 1
            if _dec(data["spent_cny"]) + _dec(data["reserved_cny"]) >= _dec(data["allowance_cny"]):
                data["stopped"] = True
                data["stop_reason"] = "allowance"
            self._apply_floor(data)
            record = {
                "settled_at": _now(),
                "reservation_id": reservation_id,
                "run_id": match["run_id"],
                "purpose": match["purpose"],
                "http_status": http_status,
                "actual_model": actual_model,
                "response_id": response_id,
                "usage": usage,
                "reserved_cny": match["reserved_cny"],
                "settled_cny": q(cost),
                "usage_unknown": unknown,
                "reservation_exceeded_cny": q(max(Decimal(0), cost - reserved)),
                "error": error,
                "request_index": data["request_count"],
            }
            data.setdefault("settlements", []).append(record)
            return data, record

        record = self._update(apply)
        if record.get("ok") is False:
            return record
        self._append_call(record)
        return record

    def cancel(self, reservation_id: str, *, reason: str) -> JsonObject:
        """Release a reservation only when no request was sent."""

        def apply(data: JsonObject | None) -> tuple[JsonObject | None, JsonObject]:
            if not data:
                return data, {"ok": False, "reason": "no-ledger"}
            match = next(
                (item for item in data["in_flight"] if item["reservation_id"] == reservation_id), None
            )
            if match is None:
                return data, {"ok": False, "reason": "unknown-reservation"}
            reserved = _dec(match["reserved_cny"])
            data["in_flight"] = [
                item for item in data["in_flight"] if item["reservation_id"] != reservation_id
            ]
            data["reserved_cny"] = q(_dec(data["reserved_cny"]) - reserved)
            return data, {"ok": True, "released_cny": q(reserved), "reason": reason}

        return self._update(apply)

    def set_max_in_flight(self, limit: int) -> JsonObject:
        if limit not in (MAX_IN_FLIGHT, TEAM_MAX_IN_FLIGHT):
            raise ValueError("In-flight limit must be 1, or 2 for an explicit team check.")

        def apply(data: JsonObject | None) -> tuple[JsonObject | None, JsonObject]:
            if not data:
                raise ValueError("Ledger is not bootstrapped.")
            data["max_in_flight"] = limit
            return data, self._public(data)

        return self._update(apply)

    def _apply_floor(self, data: JsonObject) -> None:
        if data.get("stop_reason") in {"allowance", "unavailable", "allowance-or-unavailable"}:
            return
        balance = _dec(data["last_balance_cny"]) - (
            _dec(data["spent_cny"]) - _dec(data.get("balance_spent_cny", "0"))
        )
        reserved = _dec(data["reserved_cny"])
        floor = _dec(data["reserve_floor_cny"])
        if balance - reserved < floor:
            data["stopped"] = True
            data["stop_reason"] = "floor"
        elif data.get("stop_reason") == "floor":
            data["stopped"] = False
            data["stop_reason"] = None

    def _block_reason(self, data: JsonObject, reserve: Decimal) -> str | None:
        if data.get("stopped"):
            return data.get("stop_reason") or "stopped"
        spent = _dec(data["spent_cny"])
        reserved = _dec(data["reserved_cny"])
        allowance = _dec(data["allowance_cny"])
        if spent + reserved + reserve > allowance:
            return "allowance"
        balance = _dec(data["last_balance_cny"]) - (
            _dec(data["spent_cny"]) - _dec(data.get("balance_spent_cny", "0"))
        )
        if balance - (reserved + reserve) < _dec(data["reserve_floor_cny"]):
            return "floor"
        return None

    def _public(self, data: JsonObject) -> JsonObject:
        return {
            "round_id": data["round_id"],
            "b0_cny": data["b0_cny"],
            "allowance_cny": data["allowance_cny"],
            "spent_cny": data["spent_cny"],
            "reserved_cny": data["reserved_cny"],
            "last_balance_cny": data["last_balance_cny"],
            "request_count": data["request_count"],
            "unknown_usage_calls": data["unknown_usage_calls"],
            "in_flight": len(data["in_flight"]),
            "max_in_flight": data["max_in_flight"],
            "stopped": data["stopped"],
            "stop_reason": data["stop_reason"],
            "model": data["model"],
        }


def _cost_from_usage(usage: JsonObject | None, reserved: Decimal) -> tuple[Decimal, bool]:
    if not usage:
        return reserved, True
    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if hit is not None and miss is not None:
        input_cost = _dec(hit) * INPUT_HIT_CNY + _dec(miss) * INPUT_MISS_CNY
    elif prompt is not None:
        input_cost = _dec(prompt) * INPUT_MISS_CNY
    else:
        return reserved, True
    if completion is None:
        return reserved, True
    output_cost = _dec(completion) * OUTPUT_CNY
    return input_cost + output_cost, False
