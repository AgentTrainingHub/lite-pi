"""Deterministic extensions contracts. Scripted responses and local MCP processes are not live-model evidence."""

import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fixtures.extensions import ExtensionFixture, alive, model
from fixtures.tool_calls import call, response
from mcp_support import fixture_server, observed_pid

from lite_pi.extensions import ExtensionError, LoadedExtensions, load_extensions
from lite_pi.mcp_host import McpError, McpHost, load_mcp_config
from lite_pi.session import AgentSession
from lite_pi.tools.registry import ToolRegistry
from lite_pi.tools.workspace import Workspace

STAMP = """
def register(api):
    def execute(args):
        written = api.write(args["path"], args["content"])
        text = written["text"]
        return {"text": text, "path": written["path"], "structured": {"text": text}}

    api.register_tool(
        "stamp",
        "Write a project file through the host.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        execute,
    )

    def before(event):
        path = str(event["arguments"].get("path", ""))
        if not path.startswith("notes/"):
            return {"block": True, "reason": "hook_blocked", "text": "path must stay in notes/"}
        return None

    api.on("tool_call", before)
"""


class ExtensionContracts(ExtensionFixture):
    def test_init_failure_discards_pending_tools(self):
        self.extension("a_good", STAMP)
        self.extension("b_bad", "def register(api):\n    raise RuntimeError('boom')\n")
        with patch("lite_pi.loop.complete", return_value=model()) as complete:
            with self.assertRaises(ExtensionError) as caught:
                AgentSession(self.work, tools=()).submit("do not run")
        self.assertIn("b_bad", str(caught.exception))
        self.assertEqual(complete.call_count, 0)
        self.assertFalse((self.work / "notes").exists())
        shutil.rmtree(self.work / ".lite-pi" / "extensions" / "b_bad")
        loaded = load_extensions(self.work, Workspace(self.work))
        self.assertEqual([tool.name for tool in loaded.tools], ["stamp"])

    def test_symlink_extension_is_rejected(self):
        target = self.base / "outside.py"
        target.write_text(STAMP)
        path = self.work / ".lite-pi" / "extensions" / "linked"
        path.mkdir(parents=True)
        (path / "extension.py").symlink_to(target)
        with self.assertRaises(ExtensionError) as caught:
            load_extensions(self.work, Workspace(self.work))
        self.assertIn("invalid_extension_file", str(caught.exception))

    def test_hook_rechecks_arguments_scope_and_revoke(self):
        self.extension("stamp", STAMP)
        self.extension(
            "z_guard",
            """
def register(api):
    def before(event):
        if event["name"] != "stamp":
            return None
        args = dict(event["arguments"])
        if args.get("content") == "rewrite":
            args["extra"] = True
            return {"arguments": args}
        if args.get("content") == "escape":
            return {"arguments": {"path": "outside.txt", "content": "no"}}
        if args.get("content") == "explode":
            raise RuntimeError("hook failed")
        if args.get("content") == "drop":
            return {"revoke": "stamp"}
        return None

    def after(event):
        if event["result"].get("text", "").startswith("---"):
            api.revoke_tool("stamp")
        text = "shown"
        return {"text": text, "structured": {"text": text}}

    api.on("tool_call", before)
    api.on("tool_result", after)
""",
        )
        requests = []

        def complete(messages, **kwargs):
            requests.append((messages, kwargs["definitions"]))
            if len(requests) == 1:
                return response(
                    call(
                        "stamp", '{"path":"notes/extension.txt","content":"EXTENSION-STAMP\\n"}', "call-stamp"
                    )
                )
            return model()

        with patch("lite_pi.loop.complete", side_effect=complete):
            result = AgentSession(self.work, tools=()).submit("stamp the note", max_turns=3)
        self.assertEqual((self.work / "notes" / "extension.txt").read_text(), "EXTENSION-STAMP\n")
        self.assertIn("stamp", [item["function"]["name"] for item in requests[0][1]])
        self.assertNotIn("bash", result["tools"])
        shown = result["transcript"][0]["results"][0]
        self.assertEqual(shown["text"], "shown")
        self.assertEqual(shown["structured"]["text"], "shown")
        raw = json.loads(Path(shown["raw_evidence"]).read_text())
        self.assertIn("EXTENSION-STAMP", raw["result"]["text"])
        self.assertNotEqual(raw["result"]["text"], "shown")
        registry = self.registry()
        blocked = registry.execute(call("stamp", '{"path":"outside.txt","content":"x"}', "call-block"))
        self.assertEqual(blocked["reason"], "hook_blocked")
        self.assertFalse((self.work / "outside.txt").exists())
        registry.scope = ("notes",)
        invalid = registry.execute(
            call("stamp", '{"path":"notes/a.txt","content":"rewrite"}', "call-invalid")
        )
        self.assertEqual(invalid["reason"], "hook_arguments_invalid")
        self.assertFalse((self.work / "notes" / "a.txt").exists())
        escaped = registry.execute(call("stamp", '{"path":"notes/a.txt","content":"escape"}', "call-scope"))
        self.assertEqual(escaped["reason"], "path_scope")
        self.assertFalse((self.work / "outside.txt").exists())
        native_registry = ToolRegistry(
            Workspace(self.work),
            ("write",),
            max_bytes=1024,
            shell_timeout=1,
            cancel_bash_after=None,
            evidence=self.base / "evidence-native",
        )
        native_registry.attach(load_extensions(self.work, Workspace(self.work)), None)
        native_registry.scope = ("notes",)
        native_scope = native_registry.execute(
            call("write", '{"path":"outside.txt","content":"x"}', "call-native-scope")
        )
        self.assertEqual(native_scope["reason"], "path_scope")
        failed = registry.execute(call("stamp", '{"path":"notes/a.txt","content":"explode"}', "call-hook"))
        self.assertEqual(failed["reason"], "hook_error")
        dropped = registry.execute(call("stamp", '{"path":"notes/a.txt","content":"drop"}', "call-drop"))
        self.assertEqual(dropped["reason"], "tool_revoked")
        again = registry.execute(
            call("stamp", '{"path":"notes/a.txt","content":"EXTENSION-STAMP\\n"}', "call-again")
        )
        self.assertEqual(again["reason"], "tool_revoked")
        native = registry.execute(call("write", '{"path":"notes/native.txt","content":"x"}', "call-write"))
        self.assertEqual(native["reason"], "tool_not_authorized")
        self.assertFalse((self.work / "notes" / "native.txt").exists())

    def test_result_conflict_keeps_original_and_context_is_bounded(self):
        self.extension(
            "stamp",
            """
def register(api):
    def execute(args):
        written = api.write(args["path"], args["content"])
        text = written["text"]
        return {"text": text, "path": written["path"], "structured": {"text": text}}

    def after(event):
        return {"text": "one", "structured": {"text": "two"}}

    def context(event):
        longest = max(len(str(item.get("content", ""))) for item in event["messages"])
        (api._workspace.root / "seen.txt").write_text(str(longest))
        if event["messages"] and event["messages"][0].get("content") == "replace":
            return {"messages": []}
        return {"text": "C" * 5000}

    api.register_tool(
        "stamp",
        "Write a project file through the host.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        execute,
    )
    api.on("tool_result", after)
    api.on("context", context)
""",
        )
        registry = self.registry()
        outcome = registry.execute(call("stamp", '{"path":"notes/keep.txt","content":"KEEP"}', "call-keep"))
        self.assertIn("KEEP", outcome["text"])
        self.assertEqual(outcome["structured"]["text"], outcome["text"])
        raw = json.loads(Path(outcome["raw_evidence"]).read_text())
        self.assertEqual(raw["result"]["text"], outcome["text"])
        self.assertEqual(registry.hook_log[-1]["status"], "conflict")
        self.assertEqual(registry.hook_log[-1]["source"], ".lite-pi/extensions/stamp/extension.py")
        notes = registry.context_messages([{"role": "user", "content": "U" * 9000}])
        self.assertLessEqual(len(notes[0]["content"]), 4200)
        self.assertIn("Extension context from .lite-pi/extensions/stamp/extension.py:", notes[0]["content"])
        self.assertIn("This note does not grant tools.", notes[0]["content"])
        self.assertLessEqual(int((self.work / "seen.txt").read_text()), 8000)
        rejected = registry.context_messages([{"role": "user", "content": "replace"}])
        self.assertEqual(rejected, [])
        self.assertEqual(registry.hook_log[-1]["status"], "context_rejected")

    def test_context_note_is_sent_without_replacing_history(self):
        self.extension(
            "note",
            """
def register(api):
    def context(event):
        return {"text": "CTX-TOKEN"}
    api.on("context", context)
""",
        )
        seen = []

        def complete(messages, **_kwargs):
            seen.append(messages)
            return model("done")

        with patch("lite_pi.loop.complete", side_effect=complete):
            result = AgentSession(self.work, tools=()).submit("original task", max_turns=1)
        self.assertEqual(result["status"], "stop")
        self.assertTrue(any("CTX-TOKEN" in m.get("content", "") for m in seen[0] if m["role"] == "system"))
        self.assertEqual(seen[0][-1], {"role": "user", "content": "original task"})
        self.assertIn({"role": "user", "content": "original task"}, result["messages"])
        self.assertNotIn("CTX-TOKEN", json.dumps(result["messages"]))


def server(name: str, mode: str = "serve", tools: list[str] | None = None, timeout: float = 8) -> dict:
    return fixture_server(name, Path("unused"), mode=mode, tools=tools, timeout=timeout)


class McpContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lite-pi-extensions-mcp-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.work = self.base / "project"
        self.work.mkdir()
        self.evidence = self.base / "evidence"
        self.evidence.mkdir()

    def open(self, servers: list[dict]) -> tuple[ToolRegistry, McpHost]:
        for spec in servers:
            spec["args"][-1] = str(self.evidence / "mcp" / spec["name"])
        config = self.work / ".lite-pi"
        config.mkdir(parents=True, exist_ok=True)
        (config / "mcp.json").write_text(json.dumps({"servers": servers}))
        host = McpHost.open(self.work, self.evidence)
        self.addCleanup(host.close)
        registry = ToolRegistry(
            Workspace(self.work),
            (),
            max_bytes=1024,
            shell_timeout=1,
            cancel_bash_after=None,
            evidence=self.evidence,
        )
        registry.attach(
            LoadedExtensions(),
            host,
        )
        return registry, host

    def calls(self, name: str) -> list[str]:
        path = self.evidence / "mcp" / name / "calls.jsonl"
        if not path.exists():
            return []
        return [line for line in path.read_text().splitlines() if line]

    def test_one_server_call_and_resource_round_trip(self):
        registry, host = self.open([server("alpha")])
        report = host.report()
        self.assertEqual(report["sdk"], "mcp==1.30.0")
        self.assertEqual(report["sdk_protocol"], "2025-11-25")
        self.assertIn("http", report["unsupported"])
        peer = report["servers"][0]
        self.assertEqual(peer["negotiated_protocol"], "2025-11-25")
        self.assertEqual(peer["server"]["name"], "alpha")
        names = [item["function"]["name"] for item in registry.advertised()]
        self.assertIn("mcp__alpha__mark", names)
        self.assertIn("mcp__alpha__read_resource", names)
        self.assertNotIn("bash", names)
        marked = registry.execute(call("mcp__alpha__mark", '{"label":"alpha"}', "call-mark"))
        self.assertTrue(marked["executed"])
        self.assertEqual((self.work / "notes" / "alpha.txt").read_text(), "ALPHA-MARK\n")
        listed = registry.execute(call("mcp__alpha__list_resources", "{}", "call-list"))
        self.assertIn("fixture://guide", listed["text"])
        read = registry.execute(call("mcp__alpha__read_resource", '{"uri":"fixture://guide"}', "call-read"))
        self.assertIn("ALPHA-GUIDE", read["text"])
        self.assertEqual(read["structured"]["text"], read["text"])
        self.assertIn("does not grant tools", read["text"])
        self.assertEqual(read["mcp_attempts"], 1)
        self.assertNotIn("bash", [item["function"]["name"] for item in registry.advertised()])
        host.close()
        self.assertFalse(alive(observed_pid(self.evidence / "mcp/alpha")))

    def test_two_servers_use_qualified_names(self):
        registry, host = self.open([server("alpha"), server("beta")])
        registry.execute(call("mcp__alpha__mark", '{"label":"alpha"}', "call-a"))
        registry.execute(call("mcp__beta__mark", '{"label":"beta"}', "call-b"))
        self.assertEqual((self.work / "notes" / "alpha.txt").read_text(), "ALPHA-MARK\n")
        self.assertEqual((self.work / "notes" / "beta.txt").read_text(), "BETA-MARK\n")
        pids = [observed_pid(self.evidence / "mcp" / item["name"]) for item in host.report()["servers"]]
        host.close()
        self.assertTrue(all(not alive(pid) for pid in pids))

    def test_withdrawal_and_schema_change_block_stale_calls(self):
        registry, _host = self.open([server("alpha", mode="control", tools=["mark", "shift", "drop"])])
        registry.execute(call("mcp__alpha__shift", "{}", "call-shift"))
        stale = registry.execute(call("mcp__alpha__mark", '{"label":"alpha"}', "call-stale"))
        self.assertFalse(stale["executed"])
        self.assertIn("Validation failed", stale["text"])
        self.assertNotIn("mark", self.calls("alpha"))
        registry.execute(call("mcp__alpha__drop", "{}", "call-drop"))
        revoked = registry.execute(
            call("mcp__alpha__mark", '{"label":"alpha","generation":"2"}', "call-gone")
        )
        self.assertEqual(revoked["reason"], "tool_revoked")
        self.assertEqual(self.calls("alpha"), ["shift", "drop"])
        self.assertNotIn("mcp__alpha__mark", [item["function"]["name"] for item in registry.advertised()])

    def test_disconnect_is_not_retried(self):
        registry, _host = self.open([server("alpha", mode="crash")])
        failed = registry.execute(call("mcp__alpha__mark", '{"label":"alpha"}', "call-crash"))
        self.assertEqual(failed["reason"], "mcp_call_failed")
        self.assertEqual(failed["mcp_attempts"], 1)
        self.assertEqual(failed["effect"], "unknown")
        self.assertEqual((self.work / "notes" / "crash.txt").read_text(), "crashed\n")
        again = registry.execute(call("mcp__alpha__mark", '{"label":"alpha"}', "call-retry"))
        self.assertEqual(again["reason"], "tool_revoked")
        self.assertEqual(self.calls("alpha"), ["mark"])

    def test_timeout_keeps_partial_output_and_kills_the_process(self):
        registry, host = self.open([server("alpha", mode="slow", tools=["slow"], timeout=1)])
        pid = observed_pid(self.evidence / "mcp/alpha")
        failed = registry.execute(call("mcp__alpha__slow", "{}", "call-slow"))
        self.assertEqual(failed["reason"], "mcp_call_failed")
        self.assertEqual(failed["mcp_attempts"], 1)
        self.assertEqual(failed["effect"], "unknown")
        self.assertEqual((self.work / "notes" / "slow-partial.txt").read_text(), "PARTIAL\n")
        self.assertEqual(self.calls("alpha"), ["slow"])
        again = registry.execute(call("mcp__alpha__slow", "{}", "call-slow-again"))
        self.assertEqual(again["reason"], "tool_revoked")
        self.assertEqual(self.calls("alpha"), ["slow"])
        host.close()
        self.assertFalse(alive(pid))

    def test_large_result_keeps_full_evidence(self):
        registry, _host = self.open([server("alpha", mode="large", tools=["large"])])
        result = registry.execute(call("mcp__alpha__large", "{}", "call-large"))
        self.assertTrue(result["executed"])
        self.assertLess(len(result["text"]), 9000)
        self.assertEqual(result["structured"]["text"], result["text"])
        full = Path(result["structured"]["full_output"])
        self.assertEqual(len(full.read_text()), 100_000)
        raw = json.loads(Path(result["raw_evidence"]).read_text())
        self.assertLess(len(raw["result"]["text"]), 9000)
        self.assertEqual(full.read_text(), "L" * 100_000)
        second = registry.execute(call("mcp__alpha__large", "{}", "call-second"))
        self.assertNotEqual(second["structured"]["full_output"], str(full))
        self.assertEqual(Path(second["structured"]["full_output"]).read_text(), "R" * 100_000)
        self.assertEqual(full.read_text(), "L" * 100_000)

    def test_init_failure_names_the_server_and_closes_the_sibling(self):
        with self.assertRaises(McpError) as caught:
            self.open([server("alpha"), server("beta", mode="fail-init")])
        self.assertIn("beta", str(caught.exception))
        self.assertIn("refused to initialize", str(caught.exception))
        alpha_pid = int((self.evidence / "mcp" / "alpha" / "pid").read_text())
        self.assertFalse(alive(alpha_pid))

    def test_paginated_tools_resources_and_remote_schema(self):
        registry, _host = self.open([server("alpha", mode="pages")])
        result = registry.execute(call("mcp__alpha__mark", '{"label":"alpha","extra":true}', "paged"))
        self.assertNotIn("additionalProperties", registry.allowed["mcp__alpha__mark"].schema)
        self.assertFalse(result["is_error"])
        result = registry.execute(call("mcp__alpha__list_resources", "{}", "pages"))
        self.assertIn("fixture://guide", result["text"])
        self.assertEqual(self.calls("alpha").count("list_resources"), 2)

    def test_full_remote_result_preserves_error_structured_and_image(self):
        registry, _host = self.open([server("alpha", mode="structured")])
        result = registry.execute(call("mcp__alpha__mark", '{"label":"alpha"}', "structured"))
        self.assertTrue(result["is_error"])
        self.assertTrue(result["executed"])
        raw = json.loads(Path(result["structured"]["remote_evidence"]).read_text())
        self.assertEqual(raw["structuredContent"], {"written": 1, "status": "partial"})
        self.assertTrue(raw["isError"])
        self.assertEqual(raw["content"][1]["data"], "aW1hZ2U=")

    def test_resources_require_peer_capability(self):
        registry, _host = self.open([server("alpha", mode="no-resources")])
        self.assertEqual(list(registry.allowed), ["mcp__alpha__mark"])

    def test_init_timeout_closes_process_without_pid_evidence(self):
        spec = server("alpha", mode="hang-init", timeout=1)
        config = self.work / ".lite-pi"
        config.mkdir()
        observer = self.base / "independent-observer"
        spec["args"][-1] = str(observer)
        (config / "mcp.json").write_text(json.dumps({"servers": [spec]}))
        with self.assertRaisesRegex(McpError, "mcp_timeout"):
            McpHost.open(self.work, self.evidence)
        self.assertFalse(alive(observed_pid(observer)))
        self.assertFalse((self.evidence / "mcp/alpha/pid").exists())
        self.assertFalse(any(t.name == "mcp-alpha" for t in threading.enumerate()))

    def test_cancel_during_initialize_preserves_cancel_and_reaps_peer(self):
        spec = server("alpha", mode="hang-init", timeout=10)
        config = self.work / ".lite-pi"
        config.mkdir()
        observer = self.base / "observer"
        spec["args"][-1] = str(observer)
        (config / "mcp.json").write_text(json.dumps({"servers": [spec]}))
        session = AgentSession(self.work)
        done = threading.Event()

        def interrupt():
            while not done.wait(0.01):
                if (observer / "pid").exists():
                    session.cancel()
                    return

        worker = threading.Thread(target=interrupt)
        worker.start()
        try:
            with self.assertRaisesRegex(McpError, "mcp_start_cancelled"):
                session.submit("run")
        finally:
            done.set()
            worker.join(2)
        self.assertTrue(session.control.cancel.is_set())
        self.assertFalse(alive(observed_pid(observer)))

    def test_cancel_inflight_closes_both_peers_without_repeating(self):
        registry, host = self.open([server("alpha", mode="slow", tools=["slow"]), server("beta")])
        cancel = threading.Event()
        registry.cancel_event = cancel
        for peer in host.servers:
            peer.cancel_event = cancel
        observed = threading.Event()

        def interrupt():
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if (self.work / "notes/slow-partial.txt").exists():
                    observed.set()
                    cancel.set()
                    return
                time.sleep(0.01)

        worker = threading.Thread(target=interrupt)
        worker.start()
        started = time.monotonic()
        result = registry.execute(call("mcp__alpha__slow", "{}", "cancel"))
        worker.join(6)
        host.close()
        self.assertTrue(observed.is_set())
        self.assertLess(time.monotonic() - started, 7)
        self.assertEqual(result["effect"], "unknown")
        self.assertEqual(self.calls("alpha"), ["slow"])
        self.assertTrue(all(peer["closed"] for peer in host.report()["servers"]))
        for name in ("alpha", "beta"):
            self.assertFalse(alive(observed_pid(self.evidence / "mcp" / name)))


class ExtensionBoundaryContracts(unittest.TestCase):
    setUp = ExtensionFixture.setUp
    extension = ExtensionFixture.extension
    registry = ExtensionFixture.registry

    def test_extension_api_checks_actual_write_scope(self):
        self.extension(
            "hidden",
            """
def register(api):
    def execute(args):
        return api.write('outside.txt', 'unexpected')
    api.register_tool('hidden', 'write', {'type':'object'}, execute)
""",
        )
        registry = self.registry()
        registry.scope = ("notes",)
        result = registry.execute(call("hidden", "{}", "scope"))
        self.assertTrue(result["is_error"])
        self.assertFalse((self.work / "outside.txt").exists())

    def test_extension_ancestor_link_and_hardlink_are_rejected(self):
        outside = self.base / "outside"
        (outside / "extensions/demo").mkdir(parents=True)
        script = outside / "extensions/demo/extension.py"
        script.write_text("def register(api): pass\n")
        (self.work / ".lite-pi").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ExtensionError):
            load_extensions(self.work, Workspace(self.work))
        (self.work / ".lite-pi").unlink()
        target = self.work / ".lite-pi/extensions/demo"
        target.mkdir(parents=True)
        os.link(script, target / "extension.py")
        with self.assertRaises(ExtensionError):
            load_extensions(self.work, Workspace(self.work))

    def test_model_cannot_install_host_executables(self):
        from lite_pi.tools.edit import write_file

        for name in (".lite-pi/extensions/new/extension.py", ".LITE-PI/mcp.json"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                write_file(Workspace(self.work), name, "danger")

    def test_extension_exception_reports_unknown_effect_and_keeps_identity(self):
        self.extension(
            "broken",
            """
def register(api):
    def execute(args):
        api.write('partial.txt', 'partial')
        raise RuntimeError('failure after write')
    api.register_tool('broken', 'write then fail', {'type':'object'}, execute)
""",
        )
        result = self.registry().execute(call("broken", "{}", "broken-call"))
        self.assertEqual(result["reason"], "tool_execution_failed")
        self.assertEqual(result["effect"], "unknown")
        self.assertEqual(result["tool_call_id"], "broken-call")
        self.assertEqual((self.work / "partial.txt").read_text(), "partial")

    def test_unknown_effect_stops_batch_and_pairs_remaining_calls(self):
        self.extension(
            "broken",
            """
def register(api):
    def execute(args):
        api.write('partial.txt', 'partial')
        raise ValueError('failure after write')
    api.register_tool('broken', 'write then fail', {'type':'object'}, execute)
""",
        )
        scripted = response(
            call("broken", "{}", "one"), call("write", '{"path":"later.txt","content":"no"}', "two")
        )
        with patch("lite_pi.loop.complete", return_value=scripted) as complete:
            result = AgentSession(self.work, tools=("write",)).submit("run", max_turns=3)
        self.assertEqual(result["status"], "effect_unknown")
        self.assertEqual(complete.call_count, 1)
        self.assertFalse((self.work / "later.txt").exists())
        self.assertEqual(
            [m["tool_call_id"] for m in result["messages"] if m["role"] == "tool"], ["one", "two"]
        )

    def test_bad_schema_fails_before_model_and_result_identity_is_host_owned(self):
        self.extension(
            "bad",
            """
def register(api):
    api.register_tool('bad', 'invalid', {'type':'object', 'properties': {'x': {'type': 'nonsense'}}}, lambda a: {})
""",
        )
        with patch("lite_pi.loop.complete") as complete, self.assertRaises(ExtensionError):
            AgentSession(self.work).submit("run")
        complete.assert_not_called()
        shutil.rmtree(self.work / ".lite-pi/extensions/bad")
        self.extension(
            "identity",
            """
def register(api):
    api.register_tool('identity', 'identity', {'type':'object'}, lambda a: {'text': 'ok', 'name': 'write', 'tool_call_id': 'forged'})
""",
        )
        result = self.registry().execute(call("identity", "{}", "original"))
        self.assertEqual((result["name"], result["tool_call_id"]), ("identity", "original"))

    def test_mcp_configuration_rejects_ancestor_link_and_invalid_settings(self):
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "mcp.json").write_text('{"servers": []}')
        (self.work / ".lite-pi").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(McpError):
            load_mcp_config(self.work)
        (self.work / ".lite-pi").unlink()
        (self.work / ".lite-pi").mkdir()
        config = self.work / ".lite-pi/mcp.json"
        os.link(outside / "mcp.json", config)
        with self.assertRaises(McpError):
            load_mcp_config(self.work)
        config.unlink()
        for change in (
            {"name": "a__b"},
            {"timeout_seconds": float("nan")},
            {"command": None},
            {"transport": "http"},
        ):
            with self.subTest(change=change):
                spec = {**server("alpha"), **change}
                config.write_text(json.dumps({"servers": [spec]}))
                with self.assertRaises(McpError):
                    McpHost.open(self.work, self.base / "evidence")

    def test_context_notes_count_toward_request_budget(self):
        self.extension(
            "notes",
            """
def register(api):
    api.on('context', lambda event: {'text': 'N' * 4000})
""",
        )
        with patch("lite_pi.loop.complete") as complete:
            result = AgentSession(self.work).submit("short", max_tokens=50, context_tokens=200)
        self.assertEqual(result["status"], "compaction_failed")
        complete.assert_not_called()

    def test_shell_cannot_publish_host_configuration(self):
        from lite_pi.tools.shell_workspace import ShellWorkspace

        self.extension("trusted", "def register(api): pass")
        shell = ShellWorkspace(self.work, self.base / "guest")
        self.assertFalse((shell.path / ".lite-pi").exists())
        target = shell.path / ".LITE-PI/extensions/evil"
        target.mkdir(parents=True)
        (target / "extension.py").write_text("danger")
        with self.assertRaisesRegex(ValueError, "protected project path"):
            shell.publish({})
        self.assertFalse((self.work / ".lite-pi/extensions/evil").exists())

    def test_invalid_resume_does_not_initialize_extensions(self):
        from lite_pi.journal import Journal, JournalError

        self.extension(
            "side_effect",
            """
from pathlib import Path
def register(api):
    Path(__file__).with_name('executed').write_text('yes')
""",
        )
        journal = Journal.create(self.base / "journal.jsonl", self.work)
        with self.assertRaises(JournalError):
            AgentSession(self.work, journal=journal).resume(leaf_id="missing")
        self.assertFalse((self.work / ".lite-pi/extensions/side_effect/executed").exists())

    def test_invalid_extension_result_preserves_effect_and_schema_ownership(self):
        self.extension(
            "invalid",
            """
def register(api):
    schema = {'type': 'object', 'properties': {'value': {'type': 'string'}}}
    def execute(args):
        api.write('partial.txt', 'done')
        return {'text': 'one', 'structured': {'text': 'two'}}
    api.register_tool('invalid', 'invalid return', schema, execute)
    schema['properties']['value']['type'] = 'integer'
""",
        )
        registry = self.registry()
        result = registry.execute(call("invalid", '{"value":"text"}', "original"))
        self.assertEqual(result["reason"], "tool_result_invalid")
        self.assertEqual(result["effect"], "unknown")
        self.assertTrue(result["executed"])
        self.assertTrue(Path(result["raw_evidence"]).exists())
        self.assertEqual((self.work / "partial.txt").read_text(), "done")

    def test_cleanup_failure_is_visible_in_final_event_and_evidence(self):
        events = []
        with (
            patch("lite_pi.loop.complete", return_value=model()),
            patch("lite_pi.mcp_host.McpHost.close", side_effect=McpError("cleanup test failure")),
            self.assertRaises(McpError),
        ):
            AgentSession(self.work).submit("run", on_event=events.append)
        ends = [e for e in events if e["type"] == "run_end"]
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0]["status"], "effect_unknown")
        saved = json.loads(next((self.base / "runs").glob("*/result.json")).read_text())
        self.assertEqual(saved["status"], "effect_unknown")
        self.assertEqual(saved["events"][-1], ends[0])


if __name__ == "__main__":
    unittest.main()
