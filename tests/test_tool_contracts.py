"""Deterministic host contracts. Scripted responses are never live-model evidence."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from fixtures.tool_calls import call, response

from lite_pi.session import AgentSession
from lite_pi.tools.registry import ToolRegistry
from lite_pi.tools.workspace import Workspace


class ToolContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lite-pi-contract-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.work = self.base / "project"
        self.work.mkdir()
        (self.work / "sample.txt").write_text("verified content\n")
        self.registry = ToolRegistry(
            Workspace(self.work),
            ("read",),
            max_bytes=1024,
            shell_timeout=1,
            cancel_bash_after=None,
            evidence=self.base / "evidence",
        )
        # Offline artifacts must stay outside the product's real-model run directory.
        runs = patch("lite_pi.session.RUNS", self.base / "runs")
        runs.start()
        self.addCleanup(runs.stop)

    def assert_rejected(self, result, name, reason, identity="call-boundary"):
        self.assertEqual(result["name"], name)
        self.assertEqual(result["tool_call_id"], identity)
        self.assertEqual(result["reason"], reason)
        self.assertTrue(result["is_error"])
        self.assertFalse(result["executed"])
        self.assertEqual(list(self.work.iterdir()), [self.work / "sample.txt"])
        self.assertEqual((self.work / "sample.txt").read_text(), "verified content\n")

    def test_unknown_tool_rejected_before_argument_parsing(self):
        result = self.registry.execute(call("missing", "not JSON"))
        self.assert_rejected(result, "missing", "unknown_tool")
        self.assertEqual(result["text"], "Tool missing not found")

    def test_registered_but_unauthorized_write_has_no_effect(self):
        result = self.registry.execute(call("write", '{"path":"new.txt","content":"forbidden"}'))
        self.assert_rejected(result, "write", "tool_not_authorized")
        self.assertEqual([item["function"]["name"] for item in self.registry.advertised()], ["read"])

    def test_unknown_tool_does_not_disable_next_valid_call(self):
        self.registry.execute(call("missing"))
        result = self.registry.execute(call("read", '{"path":"sample.txt"}', "call-next"))
        self.assertTrue(result["executed"])
        self.assertFalse(result["is_error"])
        self.assertEqual(result["tool_call_id"], "call-next")
        self.assertEqual(result["content"], "verified content\n")

    def run_sequence(self, responses):
        requests = []

        def complete(messages, **_kwargs):
            requests.append(deepcopy(messages))
            self.assertLessEqual(len(requests), len(responses), "unexpected extra model turn")
            return responses[len(requests) - 1]

        with patch("lite_pi.loop.complete", side_effect=complete):
            result = AgentSession(self.work, tools=("read",)).submit("contract fixture", max_turns=4)
        self.assertEqual(len(requests), len(responses))
        self.assertEqual(result["request_count"], 0)
        self.assertEqual(result["model"], "deterministic-fixture")
        self.assertEqual(json.loads((Path(result["evidence"]) / "result.json").read_text()), result)
        return result, requests

    def test_unknown_result_is_paired_and_next_turn_recovers(self):
        result, requests = self.run_sequence(
            [
                response(call("missing")),
                response(call("read", '{"path":"sample.txt"}', "call-read")),
                response(finish="stop"),
            ]
        )
        rejected = result["transcript"][0]["results"][0]
        self.assert_rejected(rejected, "missing", "unknown_tool")
        self.assertEqual(
            requests[1][-1],
            {"role": "tool", "tool_call_id": "call-boundary", "content": "Tool missing not found"},
        )
        self.assertEqual(requests[2][-1]["tool_call_id"], "call-read")
        self.assertIn("verified content", requests[2][-1]["content"])
        self.assertEqual(result["status"], "stop")

    def test_unknown_call_does_not_suppress_valid_call_in_same_batch(self):
        result, requests = self.run_sequence(
            [
                response(call("missing"), call("read", '{"path":"sample.txt"}', "call-read")),
                response(finish="stop"),
            ]
        )
        rejected, accepted = result["transcript"][0]["results"]
        self.assert_rejected(rejected, "missing", "unknown_tool")
        self.assertTrue(accepted["executed"])
        self.assertFalse(accepted["is_error"])
        self.assertEqual(
            [message["tool_call_id"] for message in requests[1][-2:]], ["call-boundary", "call-read"]
        )
        self.assertEqual(result["status"], "stop")

    def test_unknown_tool_respects_turn_limit(self):
        with patch("lite_pi.loop.complete", return_value=response(call("missing"))) as model:
            result = AgentSession(self.work, tools=("read",)).submit("contract fixture", max_turns=1)
        self.assertEqual(model.call_count, 1)
        self.assertEqual(result["status"], "turn_limit")
        self.assert_rejected(result["transcript"][0]["results"][0], "missing", "unknown_tool")


if __name__ == "__main__":
    unittest.main()
