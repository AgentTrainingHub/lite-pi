"""Deterministic interaction contracts. Scripted responses are not live-model evidence."""

import io
import json
import signal
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from lite_pi.cli import _progress, _run_interruptible
from lite_pi.journal import Journal
from lite_pi.provider import ModelResult, _assemble_stream
from lite_pi.session import AgentSession


def call(name: str, arguments: str, identity: str) -> dict:
    return {"id": identity, "type": "function", "function": {"name": name, "arguments": arguments}}


def model(content: str = "", *calls: dict, finish: str = "stop") -> ModelResult:
    message: dict = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = list(calls)
        finish = "tool_calls"
    return ModelResult(
        finish,
        message,
        raw_stop_reason=finish,
        actual_model="deterministic-fixture",
        attempts=0,
    )


class InteractionContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lite-pi-interaction-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.work = self.base / "project"
        self.work.mkdir()
        runs = patch("lite_pi.session.RUNS", self.base / "runs")
        runs.start()
        self.addCleanup(runs.stop)

    def test_sse_events_keep_partial_text_and_split_tool_calls(self):
        head = b'data: {"model":"deepseek-flash","choices":[{"delta":{"content":"par"}}]}\n'
        unfinished = head + b'data: {"choices":[{"delta":{"content":"tial"}}]'
        finished = (
            head
            + b'data: {"choices":[{"delta":{"content":"tial"}}]}\n'
            + b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1",'
            + b'"function":{"name":"read","arguments":"{\\"path\\""}}]}}]}\n'
            + b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\\"a\\"}"}}]},'
            + b'"finish_reason":"tool_calls"}]}\n'
        )
        partial, _finish, model_name = _assemble_stream(unfinished, partial=True)
        self.assertEqual(partial["content"], "par")
        self.assertEqual(model_name, "deepseek-flash")
        message, finish, _model_name = _assemble_stream(finished)
        self.assertEqual(message["content"], "partial")
        self.assertEqual(finish, "tool_calls")
        self.assertEqual(message["tool_calls"][0]["id"], "c1")
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"], '{"path":"a"}')

    def test_published_events_are_independent_and_text_is_not_repeated(self):
        seen = []

        def complete(_messages, **kwargs):
            kwargs["on_delta"]("hel")
            kwargs["on_delta"]("lo")
            return model("hello")

        def observe(event):
            seen.append(event)
            event["text"] = "changed"

        with patch("lite_pi.loop.complete", side_effect=complete):
            result = AgentSession(self.work).submit("fixture", on_event=observe)
        stored = [event["text"] for event in result["events"] if event["type"] == "message_delta"]
        self.assertEqual(stored, ["hel", "lo"])
        self.assertEqual("".join(stored), result["text"])
        self.assertEqual(
            [event["type"] for event in result["events"] if event["type"] == "run_end"], ["run_end"]
        )
        self.assertIn("turn_start", [event["type"] for event in result["events"]])
        self.assertTrue(all(event["text"] == "changed" for event in seen if "text" in event))
        displayed = io.StringIO()
        with patch("sys.stderr", displayed):
            for event in result["events"]:
                _progress(event)
        self.assertEqual(displayed.getvalue(), "hello")

    def test_steer_and_follow_up_enter_different_requests(self):
        journal = Journal.create(self.base / "session.jsonl", self.work, tools=("write",))
        session = AgentSession(self.work, tools=("write",), journal=journal)
        requests = []
        script = [
            model(
                "",
                call("write", json.dumps({"path": "notes/a.txt", "content": "A\n"}), "write-a"),
            ),
            model(
                "",
                call("write", json.dumps({"path": "notes/b.txt", "content": "B\n"}), "write-b"),
            ),
            model("done"),
            model(
                "",
                call("write", json.dumps({"path": "notes/c.txt", "content": "C\n"}), "write-c"),
            ),
            model("followed"),
            model("resumed"),
        ]

        def complete(messages, **_kwargs):
            requests.append(deepcopy(messages))
            return script[len(requests) - 1]

        def observe(event):
            if event["type"] == "tool_end" and event["result"]["tool_call_id"] == "write-a":
                session.steer("STEER-ONE", paths=("notes/a.txt",))
                session.steer("STEER-TWO", paths=("notes/a.txt",))
                session.follow_up("FOLLOW-LATER")

        with patch("lite_pi.loop.complete", side_effect=complete):
            result = session.submit("fixture", max_turns=6, on_event=observe)
            resumed = session.resume(max_turns=1)
        self.assertNotIn("STEER-ONE", json.dumps(requests[0]))
        self.assertIn("STEER-ONE", json.dumps(requests[1]))
        self.assertNotIn("STEER-TWO", json.dumps(requests[1]))
        self.assertNotIn("FOLLOW-LATER", json.dumps(requests[1]))
        self.assertIn("STEER-TWO", json.dumps(requests[2]))
        self.assertNotIn("FOLLOW-LATER", json.dumps(requests[2]))
        self.assertIn("FOLLOW-LATER", json.dumps(requests[3]))
        self.assertNotIn("FOLLOW-LATER", json.dumps(requests[3][:-1]))
        self.assertEqual((self.work / "notes/c.txt").read_text(), "C\n")
        self.assertIn("STEER-ONE", json.dumps(requests[5]))
        self.assertEqual((self.work / "notes/a.txt").read_text(), "A\n")
        self.assertFalse((self.work / "notes/b.txt").exists())
        scoped = result["transcript"][1]["results"][0]
        self.assertEqual(scoped["reason"], "path_scope")
        self.assertFalse(scoped["executed"])
        entered = [item for item in result["controls"] if item["disposition"] == "entered"]
        self.assertEqual([item["text"] for item in entered], ["STEER-ONE", "STEER-TWO", "FOLLOW-LATER"])
        self.assertTrue(
            all(item["submitted_at"] <= item["accepted_at"] <= item["entered_at"] for item in entered)
        )
        self.assertEqual(resumed["text"], "resumed")
        self.assertTrue(any(record["body"].get("content") == "FOLLOW-LATER" for record in journal.records))

    def test_cancel_keeps_completed_write_and_duplicate_cancel_is_inert(self):
        session = AgentSession(self.work, tools=("write",))
        script = [
            model(
                "",
                call("write", json.dumps({"path": "keep.txt", "content": "kept\n"}), "keep"),
                call("write", json.dumps({"path": "drop.txt", "content": "dropped\n"}), "drop"),
            ),
            model("second"),
        ]
        calls = {"n": 0}

        def complete(_messages, **kwargs):
            calls["n"] += 1
            if kwargs.get("stop") and kwargs["stop"]():
                return model("unused")
            return script[calls["n"] - 1]

        def observe(event):
            if event["type"] == "tool_end" and event["result"]["tool_call_id"] == "keep":
                session.follow_up("DO-NOT-RETRY")
                first = session.cancel()
                second = session.cancel()
                self.assertTrue(first["first"])
                self.assertFalse(second["first"])

        with patch("lite_pi.loop.complete", side_effect=complete):
            result = session.submit("fixture", on_event=observe)
            rejected = session.steer("REJECTED-AFTER")
            self.assertEqual(rejected["disposition"], "rejected_after_cancel")
            self.assertEqual(session.control.cancel_count, 2)
            again = session.submit("next", max_turns=1)
        endings = [event for event in result["events"] if event["type"] == "run_end"]
        self.assertEqual(len(endings), 1)
        self.assertEqual(result["status"], "aborted")
        self.assertEqual((self.work / "keep.txt").read_text(), "kept\n")
        self.assertFalse((self.work / "drop.txt").exists())
        self.assertEqual(result["transcript"][0]["results"][1]["reason"], "cancelled")
        self.assertFalse(result["transcript"][0]["results"][1]["executed"])
        dropped = next(item for item in result["controls"] if item["text"] == "DO-NOT-RETRY")
        self.assertEqual(dropped["disposition"], "dropped_on_cancel")
        self.assertNotIn("DO-NOT-RETRY", json.dumps(again["initial_request"]))
        self.assertNotIn("REJECTED-AFTER", json.dumps(again["initial_request"]))
        self.assertEqual(again["text"], "second")

    def test_aborted_stream_does_not_start_a_tool(self):
        session = AgentSession(self.work, tools=("write",))

        def complete(_messages, **kwargs):
            kwargs["on_delta"]("partial")
            self.assertTrue(kwargs["stop"]())
            return ModelResult(
                "aborted",
                {"role": "assistant", "content": "partial", "tool_calls": [call("write", "{}", "late")]},
                error="aborted",
                attempts=0,
            )

        def observe(event):
            if event["type"] == "message_delta":
                session.cancel()

        with patch("lite_pi.loop.complete", side_effect=complete):
            result = session.submit("fixture", on_event=observe)
        self.assertEqual(result["status"], "aborted")
        self.assertEqual(result["text"], "partial")
        self.assertEqual(result["transcript"][0]["results"], [])
        self.assertFalse(any(event["type"].startswith("tool_") for event in result["events"]))
        self.assertEqual(list(self.work.iterdir()), [])

    def test_sigint_uses_the_session_cancel(self):
        session = AgentSession(self.work)

        def operation():
            signal.raise_signal(signal.SIGINT)
            return session.control.cancel.is_set()

        self.assertTrue(_run_interruptible(session, operation))
