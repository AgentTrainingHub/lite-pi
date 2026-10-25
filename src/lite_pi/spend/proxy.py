"""Local OpenAI-compatible proxy. Upstream Pi uses a dummy key; this process adds the real one."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TypedDict

from lite_pi.spend.gate import SpendBlocked, transmit
from lite_pi.spend.pricing import MAX_OUTPUT_TOKENS
from lite_pi.tools.contracts import JsonObject


class ProxyState(TypedDict):
    run_id: str
    purpose: str
    max_calls: int | None
    calls: int


_state: ProxyState = {"run_id": "proxy", "purpose": "upstream", "max_calls": None, "calls": 0}
_lock = threading.Lock()


def begin_run(run_id: str, purpose: str, max_calls: int) -> None:
    with _lock:
        _state.update({"run_id": run_id, "purpose": purpose, "max_calls": max_calls, "calls": 0})


def end_run() -> JsonObject:
    with _lock:
        snapshot = dict(_state)
        _state["max_calls"] = None
        return snapshot


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._send_json(200, {"ok": True})
            return
        self._send_json(404, {"error": "not-found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/control/begin":
            body = json.loads(raw.decode() or "{}")
            begin_run(str(body["run_id"]), str(body.get("purpose") or "upstream"), int(body["max_calls"]))
            self._send_json(200, {"ok": True})
            return
        if path == "/control/end":
            self._send_json(200, end_run())
            return
        if path not in {"/chat/completions", "/v1/chat/completions"}:
            self._send_json(404, {"error": "not-found"})
            return
        with _lock:
            if _state["max_calls"] is not None and _state["calls"] >= _state["max_calls"]:
                self._send_json(402, {"error": {"message": "run call cap", "type": "budget"}})
                return
            _state["calls"] += 1
            run_id = _state["run_id"]
            purpose = _state["purpose"]
        try:
            payload = json.loads(raw.decode())
        except json.JSONDecodeError:
            self._send_json(400, {"error": {"message": "invalid json"}})
            return
        try:
            result = transmit(
                payload, run_id=run_id, purpose=purpose, max_output=MAX_OUTPUT_TOKENS, max_attempts=1
            )
        except SpendBlocked as blocked:
            self._send_json(402, {"error": {"message": blocked.reason, "type": "budget"}})
            return
        except ValueError as error:
            self._send_json(400, {"error": {"message": str(error)}})
            return
        status = result["http_status"] or 502
        body = result["body"]
        self.send_response(status)
        self.send_header("Content-Type", result["content_type"])
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: JsonObject) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def serve(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
