"""Deterministic ownership and lifecycle contracts; no paid model responses."""

import json
from unittest.mock import patch

from fixtures.extensions import ExtensionFixture, alive, model
from fixtures.tool_calls import call
from mcp_support import fixture_server, observed_pid

from lite_pi.extensions import ExtensionError, ExtensionHost, load_extensions
from lite_pi.mcp_host import McpError, McpHost
from lite_pi.session import AgentSession
from lite_pi.tools.edit import write_file
from lite_pi.tools.workspace import Workspace


class DesignContracts(ExtensionFixture):
    def test_final_evidence_and_event_follow_cleanup(self):
        events = []
        cleanup_observed = []
        original_close = McpHost.close

        def close(host):
            self.assertFalse(list((self.base / "runs").glob("*/result.json")))
            self.assertFalse(list((self.base / "runs").glob("*/events.jsonl")))
            original_close(host)
            cleanup_observed.append(True)

        def observe(event):
            events.append(event)
            if event["type"] == "run_end":
                self.assertEqual(cleanup_observed, [True])
                path = self.base / "runs" / event["run_id"]
                saved = json.loads((path / "result.json").read_text())
                self.assertEqual(saved["events"], events)
                persisted = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
                self.assertEqual(persisted, events)

        with patch("lite_pi.loop.complete", return_value=model()), patch.object(McpHost, "close", close):
            result = AgentSession(self.work).submit("run", on_event=observe)
        self.assertEqual(result["events"], events)
        self.assertEqual([event["seq"] for event in events], list(range(1, len(events) + 1)))
        self.assertEqual(sum(event["type"] == "run_end" for event in events), 1)

    def test_observer_failure_is_reported_and_original_exception_survives(self):
        events = []
        failure = RuntimeError("consumer failed")

        def observe(event):
            events.append(event)
            if event["type"] == "message_end":
                raise failure

        with patch("lite_pi.loop.complete", return_value=model()):
            with self.assertRaises(RuntimeError) as caught:
                AgentSession(self.work).submit("run", on_event=observe)
        self.assertIs(caught.exception, failure)
        saved = json.loads(next((self.base / "runs").glob("*/result.json")).read_text())
        self.assertEqual(saved["status"], "error")
        self.assertIn("consumer failed", saved["reason"])
        self.assertEqual(saved["events"][-1], events[-1])
        self.assertEqual(events[-1]["status"], "error")

    def test_failed_install_reaps_peer_and_saves_diagnostic(self):
        observer = self.base / "peer"
        config = self.work / ".lite-pi/mcp.json"
        config.parent.mkdir()
        config.write_text(json.dumps({"servers": [fixture_server("alpha", observer, tools=["missing"])]}))
        with patch("lite_pi.loop.complete") as complete, self.assertRaisesRegex(McpError, "missing_mcp_tool"):
            AgentSession(self.work).submit("run")
        complete.assert_not_called()
        self.assertFalse(alive(observed_pid(observer)))
        saved = json.loads(next((self.base / "runs").glob("*/result.json")).read_text())
        self.assertEqual(saved["status"], "error")
        self.assertTrue(saved["mcp"]["servers"][0]["closed"])

    def test_duplicate_config_is_rejected_before_any_process_starts(self):
        observer = self.base / "peer"
        config = self.work / ".lite-pi/mcp.json"
        config.parent.mkdir()
        spec = fixture_server("alpha", observer)
        config.write_text(json.dumps({"servers": [spec, spec]}))
        with self.assertRaisesRegex(McpError, "duplicate_mcp_server"):
            AgentSession(self.work).submit("run")
        self.assertFalse(observer.exists())

    def test_extension_works_with_only_declared_host_operations(self):
        self.extension(
            "independent",
            """
def register(api):
    def execute(args):
        result = api.write('result.txt', 'written')
        api.revoke_tool('independent')
        return result
    api.register_tool('independent', 'write then revoke', {'type':'object'}, execute)
""",
        )
        workspace = Workspace(self.work)
        loaded = load_extensions(self.work, workspace)
        revocations = []
        loaded.bind(
            ExtensionHost(lambda path, content: write_file(workspace, path, content), revocations.append)
        )
        result = loaded.tools[0].execute({})
        self.assertIn("written", result["text"])
        self.assertEqual((self.work / "result.txt").read_text(), "written")
        self.assertEqual(revocations, ["independent"])
        loaded.bind(None)
        with self.assertRaisesRegex(ExtensionError, "extension_api_inactive"):
            loaded.tools[0].execute({})

    def test_advertised_schema_cannot_change_execution_validation(self):
        self.extension(
            "strict",
            """
def register(api):
    api.register_tool('strict', 'string only',
                      {'type':'object','properties':{'value':{'type':'string'}},'required':['value']},
                      lambda args: {'text': args['value']})
""",
        )
        registry = self.registry()
        advertised = registry.advertised()
        advertised[0]["function"]["parameters"]["properties"]["value"]["type"] = "integer"
        outcome = registry.execute(call("strict", '{"value": 42}', "invalid"))
        self.assertFalse(outcome["executed"])
        self.assertIn("Validation failed", outcome["text"])
        result = registry.execute(call("strict", '{"value": "ok"}', "valid"))
        self.assertEqual(result["text"], "ok")
