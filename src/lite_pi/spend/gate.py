"""The only DeepSeek HTTP path. Credentials stay in this process and off subprocess environments."""

from __future__ import annotations

import http.client
import json
import socket
import ssl
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from threading import Event, Thread
from typing import cast

from lite_pi.runtime import PATHS
from lite_pi.spend.connection import CancellableHTTPSConnection
from lite_pi.spend.envfile import resolve_config
from lite_pi.spend.ledger import Ledger
from lite_pi.spend.pricing import (
    CHARS_PER_TOKEN_ESTIMATE,
    ENDPOINT,
    INPUT_ESTIMATE_MARGIN,
    MAX_OUTPUT_TOKENS,
    MODEL,
)
from lite_pi.spend.stream import StreamCapture
from lite_pi.storage import atomic_write
from lite_pi.task_budget import BudgetExceeded, RequestBudget
from lite_pi.tools.contracts import JsonObject

EVIDENCE_ROOT = PATHS.state / "spend" / "calls"
RETRYABLE = {429, 500, 502, 503, 504}
HOST = "api.deepseek.com"


class SpendBlocked(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def load_api_key() -> str:
    try:
        values = resolve_config()
    except (OSError, ValueError) as error:
        raise SpendBlocked(str(error)) from error
    if values["DEEPSEEK_MODEL"] != MODEL:
        raise SpendBlocked("Configured model is not deepseek-flash.")
    if values["DEEPSEEK_API_BASE"] != ENDPOINT:
        raise SpendBlocked("Configured endpoint is not https://api.deepseek.com.")
    key = values["DEEPSEEK_API_KEY"]
    if not key:
        raise SpendBlocked("Missing DeepSeek configuration.")
    return key


def prepare_payload(payload: JsonObject, *, max_output: int = MAX_OUTPUT_TOKENS) -> JsonObject:
    prepared: JsonObject = json.loads(json.dumps(payload))
    if prepared.get("model") != MODEL:
        raise ValueError("only deepseek-flash is allowed")
    prepared["thinking"] = {"type": "disabled"}
    prepared.pop("reasoning_effort", None)
    requested = prepared.get("max_tokens")
    limit = min(MAX_OUTPUT_TOKENS, max_output, int(requested) if requested is not None else MAX_OUTPUT_TOKENS)
    if limit < 1:
        raise ValueError("max_tokens")
    prepared["max_tokens"] = limit
    prepared.pop("max_completion_tokens", None)
    if prepared.get("stream"):
        options = dict(prepared.get("stream_options") or {})
        options["include_usage"] = True
        prepared["stream_options"] = options
    return prepared


def estimate_input_tokens(payload: JsonObject) -> int:
    encoded = json.dumps(payload, ensure_ascii=False)
    ascii_chars = sum(ord(char) < 128 for char in encoded)
    # Non-ASCII text and JSON punctuation tokenize more densely than English prose.
    estimate = (
        Decimal(ascii_chars) / Decimal(CHARS_PER_TOKEN_ESTIMATE) + Decimal(len(encoded) - ascii_chars)
    ) * INPUT_ESTIMATE_MARGIN
    return max(1, int(estimate) + 1)


def fetch_balance(api_key: str | None = None) -> JsonObject:
    key = api_key or load_api_key()
    status, raw = _http("GET", "/user/balance", None, key, timeout=30)
    if status != 200:
        raise SpendBlocked(f"Balance request failed with HTTP {status}.")
    body = json.loads(raw.decode())
    infos = body.get("balance_infos") or body.get("balances") or []
    cny = next((item for item in infos if item.get("currency") == "CNY"), None)
    if cny is None or cny.get("total_balance") in (None, ""):
        raise SpendBlocked("Balance response has no CNY total.")
    return {
        "is_available": bool(body.get("is_available")),
        "currency": "CNY",
        "total_balance": Decimal(str(cny["total_balance"])),
        "granted_balance": str(cny.get("granted_balance", "")),
        "topped_up_balance": str(cny.get("topped_up_balance", "")),
    }


class RequestNotSent(RuntimeError):
    """Cancellation was observed before the HTTP request began."""


def _cancelled_response(attempts: int) -> JsonObject:
    return {
        "http_status": None,
        "body": b"",
        "content_type": "application/json",
        "usage": None,
        "reservation_id": None,
        "settled": {"error": "aborted"},
        "evidence": None,
        "attempts": attempts,
    }


def transmit(
    payload: JsonObject,
    *,
    run_id: str,
    purpose: str,
    ledger: Ledger | None = None,
    max_attempts: int = 2,
    max_output: int = MAX_OUTPUT_TOKENS,
    on_chunk: Callable[[bytes], None] | None = None,
    stop_after_chunks: int | None = None,
    stop: Callable[[], bool] | None = None,
    budget: RequestBudget | None = None,
) -> JsonObject:
    book = ledger or Ledger()
    prepared = prepare_payload(payload, max_output=max_output)
    key = load_api_key()
    outcome = None
    settled = None
    reservation_id = None
    evidence = None
    if max_attempts not in (1, 2):
        raise ValueError("max_attempts must be 1 or 2")

    def stopped() -> bool:
        return bool(stop is not None and stop()) or (
            budget is not None and budget.stop_reason() == "time_budget"
        )

    for attempt in range(1, max_attempts + 1):
        if stopped():
            return _cancelled_response(attempt - 1)
        task_reservation = budget.reserve(purpose, int(prepared["max_tokens"])) if budget else None
        if budget is not None and task_reservation is None:
            raise BudgetExceeded(budget.stop_reason() or "task_output", attempts=attempt - 1)
        sent = False
        usage = None
        try:
            try:
                reservation = book.reserve_wait(
                    run_id=run_id,
                    purpose=purpose,
                    input_tokens=estimate_input_tokens(prepared),
                    output_tokens=int(prepared["max_tokens"]),
                    stop=stopped,
                )
            except (OSError, ValueError) as error:
                raise SpendBlocked("ledger unavailable: " + str(error)) from error
            if not reservation.get("ok"):
                if reservation.get("reason") == "cancelled":
                    return _cancelled_response(attempt - 1)
                if outcome is None:
                    raise SpendBlocked(reservation.get("reason") or "blocked")
                break
            reservation_id = reservation["reservation_id"]
            if stopped():
                book.cancel(reservation_id, reason="cancelled-before-send")
                return _cancelled_response(attempt - 1)
            evidence = EVIDENCE_ROOT / reservation_id
            try:
                evidence.mkdir(parents=True, exist_ok=False)
                _write_json(evidence / "request.json", _redact(prepared, key))
            except OSError as error:
                book.cancel(reservation_id, reason="evidence-unavailable-before-send")
                raise SpendBlocked("request evidence unavailable: " + str(error)) from error
            sent = True
            try:
                outcome = _read_response(
                    prepared,
                    key,
                    on_chunk=on_chunk,
                    stop_after_chunks=stop_after_chunks,
                    stop=stopped,
                    partial_path=evidence / "response.partial",
                )
            except RequestNotSent:
                sent = False
                book.cancel(reservation_id, reason="cancelled-before-send")
                return _cancelled_response(attempt - 1)
            except (TimeoutError, OSError, http.client.HTTPException) as error:
                outcome = {
                    "http_status": None,
                    "body": b"",
                    "content_type": "application/json",
                    "error": type(error).__name__,
                }
            except BaseException as error:
                book.settle(
                    reservation_id,
                    usage=None,
                    http_status=None,
                    actual_model=None,
                    response_id=None,
                    error=type(error).__name__,
                )
                raise
            outcome["attempts"] = attempt
            usage = _usage_from(outcome["body"])
            settled = book.settle(
                reservation_id,
                usage=usage,
                http_status=outcome["http_status"],
                actual_model=_model_from(outcome["body"]),
                response_id=_response_id(outcome["body"]),
                error=outcome.get("error"),
            )
            _write_bytes(evidence / "response.bin", _redact_bytes(outcome["body"], key))
            _write_json(
                evidence / "receipt.json",
                {
                    "reservation_id": reservation_id,
                    "run_id": run_id,
                    "purpose": purpose,
                    "attempt": attempt,
                    "http_status": outcome["http_status"],
                    "usage": usage,
                    "settled_cny": settled.get("settled_cny"),
                    "usage_unknown": settled.get("usage_unknown"),
                    "error": outcome.get("error"),
                    "stream": bool(prepared.get("stream")),
                },
            )
            if int(settled.get("request_index") or 0) % 20 == 0:
                refresh_error = _refresh(book, key)
                if refresh_error:
                    _write_json(evidence / "balance-error.json", {"error": refresh_error})
            status = outcome["http_status"]
            if outcome.get("error") == "aborted" or (status is not None and status not in RETRYABLE):
                break
        finally:
            if budget is not None and task_reservation is not None:
                if sent:
                    budget.settle(task_reservation, usage, 1)
                else:
                    budget.release(task_reservation)
    if outcome is None or settled is None:
        raise RuntimeError("transmit ended without a response outcome and settlement")
    return {
        "http_status": outcome["http_status"],
        "body": outcome["body"],
        "content_type": outcome["content_type"],
        "usage": _usage_from(outcome["body"]),
        "reservation_id": reservation_id,
        "settled": settled,
        "evidence": str(evidence),
        "attempts": outcome["attempts"],
    }


def _refresh(book: Ledger, key: str) -> str | None:
    try:
        balance = fetch_balance(key)
    except (SpendBlocked, OSError, http.client.HTTPException, json.JSONDecodeError) as error:
        return type(error).__name__
    book.refresh_balance(balance["total_balance"], is_available=balance["is_available"], source="every-20")
    return None


def _read_response(
    payload: JsonObject,
    key: str,
    *,
    on_chunk: Callable[[bytes], None] | None,
    stop_after_chunks: int | None = None,
    stop: Callable[[], bool] | None = None,
    partial_path: Path | None = None,
) -> JsonObject:
    status, raw, content_type, _chunks, error = _http_stream(
        "POST",
        "/chat/completions",
        payload,
        key,
        timeout=120,
        on_chunk=on_chunk,
        stop_after_chunks=stop_after_chunks,
        stop=stop,
        partial_path=partial_path,
    )
    return {
        "http_status": status,
        "body": raw,
        "content_type": content_type,
        "error": error,
        "attempts": 1,
    }


def _http(method: str, path: str, payload: JsonObject | None, key: str, *, timeout: int) -> tuple[int, bytes]:
    status, raw, _content_type, _chunks, _aborted = _http_stream(method, path, payload, key, timeout=timeout)
    return status, raw


def _http_stream(
    method: str,
    path: str,
    payload: JsonObject | None,
    key: str,
    *,
    timeout: int,
    on_chunk: Callable[[bytes], None] | None = None,
    stop_after_chunks: int | None = None,
    stop: Callable[[], bool] | None = None,
    partial_path: Path | None = None,
) -> tuple[int, bytes, str, list[bytes], str | None]:
    context = ssl.create_default_context()
    if stop is not None and stop():
        raise RequestNotSent("cancelled before HTTP send")
    connection = CancellableHTTPSConnection(HOST, timeout=timeout, context=context)
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    headers = {"Authorization": "Bearer " + key, "Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
        if payload and payload.get("stream"):
            headers["Accept"] = "text/event-stream"
    capture = StreamCapture(key, partial_path)
    transport_error = None
    status, content_type = 0, "application/json"
    done = Event()
    active_socket = None

    def interrupt_read() -> None:
        # shutdown interrupts a blocked SSL read; close() alone can wait for that read.
        while not done.wait(0.05):
            if stop is not None and stop():
                target = active_socket or connection.sock
                if target is not None:
                    try:
                        target.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        continue  # Retry if TLS has just replaced the raw socket.
                    return

    watcher = Thread(target=interrupt_read, name="model-cancel") if stop is not None else None
    if watcher is not None:
        watcher.start()
    try:
        if stop is not None and stop():
            raise RequestNotSent("cancelled before HTTP send")
        connection.request(method, path, body=body, headers=headers)
        active_socket = connection.sock
        response = connection.getresponse()
        status = response.status
        content_type = response.getheader("Content-Type") or "application/json"
        if partial_path is not None:
            _write_json(
                partial_path.with_suffix(".headers.json"),
                {"http_status": status, "content_type": content_type},
            )
        transport_error = capture.read(response, on_chunk, stop, stop_after_chunks)
        capture.mark(
            "aborted" if transport_error == "aborted" else "closed" if transport_error else "finished",
            len(capture.chunks),
        )
    except (OSError, http.client.HTTPException):
        if stop is None or not stop():
            raise
        transport_error = "aborted"
        capture.mark("aborted", len(capture.chunks))
    finally:
        done.set()
        connection.close()
        if watcher is not None:
            watcher.join()
    return status, b"".join(capture.chunks), content_type, capture.chunks, transport_error


def _usage_from(raw: bytes) -> JsonObject | None:
    if not raw:
        return None
    text = raw.decode(errors="replace")
    if text.lstrip().startswith("{"):
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            return None
        direct_usage = body.get("usage")
        return direct_usage if isinstance(direct_usage, dict) else None
    usage: JsonObject | None = None
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
    return usage


def _model_from(raw: bytes) -> str | None:
    parsed = _first_json(raw)
    if not isinstance(parsed, dict):
        return None
    model = parsed.get("model")
    return model if isinstance(model, str) else None


def _response_id(raw: bytes) -> str | None:
    parsed = _first_json(raw)
    if not isinstance(parsed, dict):
        return None
    value = parsed.get("id")
    return value if isinstance(value, str) else None


def _first_json(raw: bytes) -> object:
    text = raw.decode(errors="replace").lstrip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    for line in text.splitlines():
        if line.startswith("data:") and line[5:].strip() not in {"", "[DONE]"}:
            try:
                return json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
    return None


def _redact(value: JsonObject, key: str) -> JsonObject:
    encoded = json.dumps(value, ensure_ascii=False)
    return cast(JsonObject, json.loads(encoded.replace(key, "[redacted]")))


def _redact_bytes(raw: bytes, key: str) -> bytes:
    return raw.replace(key.encode(), b"[redacted]")


def _write_json(path: Path, value: object) -> None:
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def _write_bytes(path: Path, raw: bytes) -> None:
    atomic_write(path, raw)
