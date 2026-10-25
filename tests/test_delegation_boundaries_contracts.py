"""Boundary regressions: controlled transport and real local sockets, no paid calls."""

import http.client
import json
import socket
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch

from deterministic_transport import response, transport
from fixtures.delegation import delegate_call, delegate_many, delegate_results, scripted
from fixtures.responses import tool_call

from lite_pi.journal import Journal
from lite_pi.session import AgentSession
from lite_pi.spend.gate import _http_stream


class DelegationBoundaryContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.work = self.root / "project"
        self.work.mkdir()
        (self.work / "a.txt").write_text("ALPHA")
        (self.work / "b.txt").write_text("OUTSIDE-CANARY")

    def tearDown(self):
        self.tmp.cleanup()

    def parent(self, tools=("read", "delegate")):
        journal = Journal.create(self.root / "session.jsonl", self.work, tools=tools)
        return AgentSession(self.work, tools=tools, journal=journal)

    def test_delegate_cannot_grant_read_to_parent_without_read(self):
        session = self.parent(("delegate",))
        with transport(
            self.root / "wire", [response("", calls=[delegate_call("look", ["a.txt"], "d")])]
        ) as wire:
            result = session.submit("look", max_turns=1, max_task_calls=3)
        item = delegate_results(result)[0]
        self.assertEqual(item["reason"], "parent_read_required")
        self.assertEqual(wire.call_count, 1)
        self.assertNotIn("ALPHA", str(item))

    def test_large_file_rejects_one_survey_and_keeps_sibling(self):
        (self.work / "large.txt").write_text("x" * 51201)
        replies = [
            response("", calls=[delegate_many([("large", ["large.txt"]), ("small", ["a.txt"])], "d")]),
            response("FACT a.txt: ALPHA"),
        ]
        with transport(self.root / "wire", replies):
            result = self.parent().submit("look", max_turns=1, max_task_calls=4)
        surveys = delegate_results(result)[0]["surveys"]
        self.assertEqual([s["status"] for s in surveys], ["rejected", "complete"])
        self.assertIn("unreadable", surveys[0]["reason"])

    def test_parent_scope_bounds_survey_and_host_proof(self):
        session = self.parent()
        session.steer("stay in a", paths=["a.txt"])
        replies = [response("", calls=[delegate_call("look outside", ["b.txt"], "d")])]
        with transport(self.root / "wire", replies) as wire:
            result = session.submit("look", max_turns=1, max_task_calls=4)
        item = delegate_results(result)[0]
        self.assertEqual(item["surveys"][0]["reason"], "path_scope")
        self.assertNotIn("OUTSIDE-CANARY", str(item))
        self.assertEqual(wire.call_count, 1)

    def test_child_scope_survives_steering_and_context_is_selected(self):
        (self.work / "AGENTS.md").write_text("CONTEXT-OUTSIDE-CANARY")
        parent = self.parent()
        book = parent._book(approvals=False, approval_ttl=None, claim=True, max_task_calls=5)
        child = AgentSession(self.work, tools=("read",))

        def observe(event):
            if event["type"] == "run_start":
                child.steer("read b instead")

        replies = [response("", calls=[tool_call("read", {"path": "b.txt"}, "r")]), response("done")]
        try:
            with transport(self.root / "wire", replies):
                result = child.investigate(
                    "read a", scope=("a.txt",), budget=book.budget, cancel=Event(), on_event=observe
                )
        finally:
            book.close()
        items = [item for turn in result["transcript"] for item in turn["results"]]
        self.assertEqual(items[0]["reason"], "path_scope")
        self.assertIsNone(result["context"]["instructions"]["adopted"])
        self.assertNotIn("OUTSIDE-CANARY", str(result))

    def test_child_restores_cancel_and_history_after_investigation(self):
        parent = self.parent()
        book = parent._book(approvals=False, approval_ttl=None, claim=True, max_task_calls=5)
        child = AgentSession(self.work, tools=("read",))
        cancel = Event()
        try:
            with transport(self.root / "wire", [response("private child context")]):
                child.investigate("a", scope=("a.txt",), budget=book.budget, cancel=cancel)
            child.cancel()
            self.assertFalse(cancel.is_set())
            self.assertEqual(child._messages, [])
        finally:
            book.close()

    def test_child_events_are_delivered_on_parent_thread(self):
        owner = threading.get_ident()
        received = []
        replies = scripted(
            [
                response("", calls=[delegate_many([("a", ["a.txt"]), ("b", ["b.txt"])], "d")]),
                response("one"),
                response("two"),
            ]
        )

        def observe(event):
            if event["type"] == "delegate_child":
                received.append(threading.get_ident())

        with transport(self.root / "wire", replies):
            result = self.parent().submit("look", max_turns=1, max_task_calls=5, on_event=observe)
        self.assertTrue(received)
        self.assertEqual(set(received), {owner})
        self.assertEqual(
            [event["seq"] for event in result["events"]], list(range(1, len(result["events"]) + 1))
        )

    def test_cancel_interrupts_a_blocked_http_body(self):
        headers_sent = Event()
        release = Event()
        cancel = Event()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Length", "100000")
                self.end_headers()
                self.wfile.flush()
                headers_sent.set()
                release.wait(5)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever)
        worker.start()

        def stop_after_headers():
            headers_sent.wait(3)
            cancel.set()

        trigger = threading.Thread(target=stop_after_headers)
        trigger.start()
        started = time.monotonic()
        try:
            with (
                patch("lite_pi.spend.gate.HOST", f"127.0.0.1:{server.server_port}"),
                patch(
                    "lite_pi.spend.gate.CancellableHTTPSConnection",
                    side_effect=lambda host, **kw: http.client.HTTPConnection(host, timeout=kw["timeout"]),
                ),
            ):
                result = _http_stream(
                    "POST", "/", {"stream": True}, "deterministic", timeout=5, stop=cancel.is_set
                )
            self.assertEqual(result[-1], "aborted")
            self.assertLess(time.monotonic() - started, 1.5)
        finally:
            release.set()
            trigger.join()
            server.shutdown()
            server.server_close()
            worker.join()
        self.assertFalse(any(t.name == "model-cancel" for t in threading.enumerate()))

    def test_revoked_parent_read_cannot_be_delegated(self):
        extension = self.work / ".lite-pi/extensions/deny/extension.py"
        extension.parent.mkdir(parents=True)
        extension.write_text(
            'def register(api):\n    api.on("tool_call", lambda event: {"revoke": "read"})\n'
        )
        replies = [response("", calls=[delegate_call("look", ["a.txt"], "d")])]
        with transport(self.root / "wire", replies) as wire:
            result = self.parent().submit("look", max_turns=1, max_task_calls=3)
        self.assertEqual(delegate_results(result)[0]["reason"], "parent_read_required")
        self.assertEqual(wire.call_count, 1)

    def test_binary_evidence_does_not_claim_child_wrote(self):
        (self.work / "binary.txt").write_bytes(b"\xff\xfe")
        replies = [
            response("", calls=[delegate_call("look", ["binary.txt"], "d")]),
            response("cannot decode"),
        ]
        with transport(self.root / "wire", replies):
            result = self.parent().submit("look", max_turns=1, max_task_calls=3)
        item = delegate_results(result)[0]
        self.assertTrue(item["is_error"])
        self.assertEqual(item["reason"], "evidence_unavailable")
        self.assertFalse(item["child_write_detected"])

    def test_directory_read_error_can_be_corrected_by_model(self):
        replies = [
            response("", calls=[tool_call("read", {"path": "."}, "dir")]),
            response("", calls=[tool_call("read", {"path": "a.txt"}, "file")]),
            response("ALPHA"),
        ]
        with transport(self.root / "wire", replies):
            result = self.parent(("read",)).submit("read", max_turns=3, max_task_calls=3)
        self.assertEqual(result["status"], "stop")
        items = [item for turn in result["transcript"] for item in turn["results"]]
        self.assertTrue(items[0]["is_error"])
        self.assertEqual(items[1]["content"], "ALPHA")

    def test_cancel_interrupts_stalled_tls_handshake(self):
        cancel, release, received = Event(), Event(), Event()
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen()
        server.settimeout(3)

        def stall():
            connection, _ = server.accept()
            with connection:
                connection.settimeout(3)
                if connection.recv(4096):
                    received.set()
                    cancel.set()
                    release.wait(5)

        worker = threading.Thread(target=stall)
        worker.start()
        started = time.monotonic()
        try:
            with patch("lite_pi.spend.gate.HOST", f"127.0.0.1:{server.getsockname()[1]}"):
                result = _http_stream("POST", "/", {}, "deterministic", timeout=5, stop=cancel.is_set)
            self.assertTrue(received.is_set())
            self.assertEqual(result[-1], "aborted")
            self.assertLess(time.monotonic() - started, 1.5)
        finally:
            release.set()
            worker.join(5)
            server.close()
        self.assertFalse(worker.is_alive())
        self.assertFalse(any(t.name == "model-cancel" for t in threading.enumerate()))

    def test_acceptance_requires_reading_original_target_after_delegation(self):
        from cases_delegation import LOOKUP, _parent_verified

        def read_turn(path, content):
            return {
                "tool_calls": ["read"],
                "provider": {"message": {"tool_calls": [tool_call("read", {"path": path}, "r")]}},
                "results": [{"name": "read", "tool_call_id": "r", "executed": True, "content": content}],
            }

        delegate_turn = {"results": [{"name": "delegate", "executed": True}]}
        for turn in [read_turn("check.py", LOOKUP), read_turn("src/lookup.py", "changed")]:
            self.assertFalse(_parent_verified({"transcript": [delegate_turn, turn]}))
        good = read_turn("src/lookup.py", LOOKUP)
        self.assertFalse(_parent_verified({"transcript": [good, delegate_turn]}))
        self.assertTrue(_parent_verified({"transcript": [delegate_turn, good]}))

    def test_cancel_judge_rejects_completed_child_and_completed_stream(self):
        from cases_delegation import _cancelled_child_request

        evidence = self.root / "request"
        evidence.mkdir()
        hit = [{"evidence": str(evidence), "child_run": "child"}]
        survey = {"run_id": "child", "status": "complete"}
        receipt = {"run_id": "child", "error": "aborted"}
        (evidence / "receipt.json").write_text(json.dumps(receipt))
        (evidence / "stream.jsonl").write_text(json.dumps({"kind": "aborted"}) + "\n")
        (evidence / "response.bin").write_bytes(b"data: partial")
        self.assertFalse(_cancelled_child_request(hit, [{"surveys": [survey]}]))
        survey["status"] = "aborted"
        self.assertTrue(_cancelled_child_request(hit, [{"surveys": [survey]}]))
        (evidence / "response.bin").write_bytes(b"data: [DONE]")
        self.assertFalse(_cancelled_child_request(hit, [{"surveys": [survey]}]))
