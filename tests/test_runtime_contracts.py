"""Runtime ownership and failure semantics; no network model or shared spend ledger."""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import anyio

from lite_pi.knowledge.db import KnowledgeError
from lite_pi.knowledge.tools import _rebuild
from lite_pi.lifecycle import close_resources
from lite_pi.mcp_transport import OwnedStdio
from lite_pi.outcomes import ToolControl
from lite_pi.process_identity import process_alive, process_identity
from lite_pi.retrieval.budget import RetrievalMeter
from lite_pi.retrieval.embedding import validate_vector
from lite_pi.runtime import RuntimePaths
from lite_pi.session import AgentSession
from lite_pi.spend.gate import _write_json, estimate_input_tokens
from lite_pi.sse import StreamAssembler
from lite_pi.tui.controller import Controller


class RuntimeContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def test_runtime_paths_are_independent_of_installation_and_cwd(self):
        with (
            patch("lite_pi.runtime.Path.home", return_value=self.root),
            patch.dict(os.environ, {"HOME": str(self.root)}, clear=True),
        ):
            self.assertEqual(RuntimePaths.load().state, self.root / ".lite-pi")
            config = self.root / ".config/lite-pi/runtime.toml"
            config.parent.mkdir(parents=True)
            config.write_text(f'state_dir = "{self.root}/saved"\nenv_file = "{self.root}/credentials"\n')
            self.assertEqual(RuntimePaths.load().credentials, self.root / "credentials")
            with patch.dict(os.environ, {"LITE_PI_HOME": str(self.root / "override")}):
                self.assertEqual(RuntimePaths.load().state, self.root / "override")
                self.assertEqual(RuntimePaths.load().credentials, self.root / "credentials")
            config.write_text('state_dir = "state"\n')
            self.assertEqual(RuntimePaths.load().state, config.parent / "state")
            config.write_text("state_dir = 3\n")
            with self.assertRaisesRegex(ValueError, "state_dir must be"):
                RuntimePaths.load()
            config.write_text("state_dir = \n")
            with self.assertRaisesRegex(ValueError, "runtime.toml: invalid TOML"):
                RuntimePaths.load()

    def test_pid_reuse_releases_slot_but_permission_error_does_not(self):
        identity = process_identity(os.getpid())
        self.assertTrue(process_alive(os.getpid(), identity))
        self.assertFalse(process_alive(os.getpid(), identity + "-previous"))
        with patch("lite_pi.process_identity.os.kill", side_effect=PermissionError):
            self.assertTrue(process_alive(123, "unverifiable"))

    def test_cleanup_calls_every_resource_in_order(self):
        closed = []

        def bad():
            closed.append("bad")
            raise OSError("close failed")

        failures = close_resources(
            [("first", lambda: closed.append("first")), ("bad", bad), ("last", lambda: closed.append("last"))]
        )
        self.assertEqual(closed, ["first", "bad", "last"])
        self.assertEqual([failure.resource for failure in failures], ["bad"])

    def test_session_preserves_primary_and_multiple_cleanup_failures(self):
        work = self.root / "project"
        work.mkdir()
        primary = ValueError("original provider failure")
        with (
            patch("lite_pi.session.RUNS", self.root / "runs"),
            patch("lite_pi.loop.complete", side_effect=primary),
            patch("lite_pi.mcp_host.McpHost.close", side_effect=OSError("mcp cleanup failed")),
            patch(
                "lite_pi.extensions.LoadedExtensions.bind",
                side_effect=[None, RuntimeError("extension cleanup failed")],
            ),
            self.assertRaises(ValueError) as caught,
        ):
            AgentSession(work, tools=()).submit("hello")
        self.assertIs(caught.exception, primary)
        notes = str(primary.__notes__)
        self.assertIn("mcp cleanup failed", notes)
        self.assertIn("extension cleanup failed", notes)
        outcomes = list((self.root / "runs").glob("*/result.json"))
        self.assertTrue(outcomes)
        saved = json.loads(outcomes[0].read_text())
        self.assertIn("original provider failure", saved["reason"])
        self.assertTrue(saved["cleanup_errors"])

    def test_cleanup_failure_is_not_replaced_by_the_callers_handled_exception(self):
        from fixtures.responses import model

        work = self.root / "project"
        work.mkdir()
        with (
            patch("lite_pi.session.RUNS", self.root / "runs"),
            patch("lite_pi.loop.complete", return_value=model("done")),
            patch("lite_pi.mcp_host.McpHost.close", side_effect=OSError("mcp cleanup failed")),
        ):
            try:
                raise KeyError("caller is handling this")
            except KeyError:
                with self.assertRaises(OSError) as caught:
                    AgentSession(work, tools=()).submit("hello")
        self.assertIn("mcp cleanup failed", str(caught.exception))
        self.assertNotIn("mcp_cleanup_failed", str(getattr(caught.exception, "__notes__", [])))

    def test_task_finish_preserves_release_failure_when_close_also_fails(self):
        from unittest.mock import Mock

        from lite_pi.tasks import TaskBook

        book = TaskBook.__new__(TaskBook)
        primary = ValueError("release failed")
        book.gate = Mock()
        book.gate.release.side_effect = primary
        with (
            patch.object(book, "close", side_effect=OSError("close failed")),
            self.assertRaises(ValueError) as caught,
        ):
            book.finish()
        self.assertIs(caught.exception, primary)
        self.assertIn("close failed", str(primary.__notes__))

    def test_task_initialization_preserves_schema_failure_after_cleanup(self):
        from lite_pi.journal import Journal
        from lite_pi.project_gate import ProjectGate
        from lite_pi.tasks import TaskBook
        from lite_pi.tools.workspace import Workspace

        project = self.root / "project"
        project.mkdir()
        journal = Journal.create(self.root / "session.jsonl", project, tools=())
        primary = ValueError("schema failed")
        original_close = ProjectGate.close

        def fail_close(gate):
            original_close(gate)
            raise OSError("gate close failed")

        with (
            patch("lite_pi.tasks.migrate", side_effect=primary),
            patch.object(ProjectGate, "close", fail_close),
            self.assertRaises(ValueError) as caught,
        ):
            TaskBook(
                journal,
                Workspace(project),
                tools=(),
                approvals=False,
                max_calls=None,
                max_output=None,
                max_seconds=None,
                approval_ttl=None,
                claim=True,
            )
        self.assertIs(caught.exception, primary)
        self.assertIn("gate close failed", str(primary.__notes__))

    def test_loop_close_race_still_releases_mcp_handle(self):
        import io
        from unittest.mock import Mock

        from lite_pi.mcp_host import McpServer

        server = McpServer.__new__(McpServer)
        server._closed = False
        server._closing = threading.Event()
        server.loop = Mock()
        server.loop.is_running.return_value = True
        server.loop.is_closed.return_value = True
        server.loop.call_soon_threadsafe.side_effect = RuntimeError("Event loop is closed")
        server._transport = Mock()
        server.thread = None
        server._err = io.StringIO()
        server.close()
        server._transport.kill.assert_called_once()
        self.assertTrue(server._closed and server.dead and server._err.closed)

    def test_settlement_records_reservation_overshoot(self):
        from decimal import Decimal

        from lite_pi.spend.ledger import Ledger

        ledger = Ledger(self.root / "ledger.json", self.root / "calls.jsonl")
        ledger.bootstrap(Decimal(100), currency="CNY", is_available=True)
        reserved = ledger.try_reserve(run_id="fixture", purpose="mechanism", input_tokens=1, output_tokens=1)
        settled = ledger.settle(
            reserved["reservation_id"],
            usage={
                "prompt_tokens": 1000,
                "completion_tokens": 1000,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 1000,
            },
            http_status=200,
            actual_model="fixture",
            response_id="fixture",
        )
        self.assertGreater(Decimal(settled["reservation_exceeded_cny"]), 0)
        self.assertEqual(ledger.require_status()["in_flight"], 0)

    def test_stream_fragments_unicode_and_malformed_events(self):
        data = (
            "data: " + json.dumps({"choices": [{"delta": {"content": "中文"}}]}, ensure_ascii=False) + "\n\n"
        ).encode()
        parser = StreamAssembler()
        emitted = "".join(parser.feed(bytes([byte])) for byte in data * 120)
        parser.finish()
        self.assertEqual(emitted, "中文" * 120)
        self.assertEqual(parser.message()["content"], emitted)
        parser.feed(b"data: {bad}\n")
        with self.assertRaisesRegex(ValueError, "invalid SSE"):
            parser.finish()
        partial = StreamAssembler()
        partial.feed(b"data: {unfinished")
        partial.finish(partial=True)
        self.assertEqual(partial.message()["content"], "")

    def test_chinese_reservation_and_atomic_evidence(self):
        self.assertGreater(estimate_input_tokens({"content": "中" * 1000}), 1000)
        target = self.root / "request.json"
        _write_json(target, {"original": True})
        with patch("lite_pi.storage.os.replace", side_effect=OSError("publish failed")):
            with self.assertRaises(OSError):
                _write_json(target, {"replacement": True})
        self.assertEqual(json.loads(target.read_text()), {"original": True})

    def test_control_fields_reject_malformed_types(self):
        for invalid in ({"blocks_run": "true"}, {"effect": "lost"}, {"halt": "unknown-status"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ToolControl.read(invalid)
        state = ToolControl.read({"is_error": True, "effect": "unknown", "blocks_run": True})
        self.assertTrue(state.blocks_run)
        self.assertEqual(state.effect, "unknown")

    def test_bad_vector_is_a_domain_error(self):
        for value in (None, {}, "oops", [True, 0], [float("nan"), 0]):
            with self.subTest(value=value), self.assertRaises(KnowledgeError):
                validate_vector(value, 2)

    def test_failed_rebuild_does_not_consume_success_quota(self):
        class Index:
            def rebuild(self, *, mode):
                return {"is_error": True, "reason": "temporary_source_failure"}

        meter = RetrievalMeter(3)
        for _ in range(3):
            self.assertTrue(_rebuild(Index(), meter, {})["is_error"])
        self.assertEqual(meter.rebuilds, 0)

    def test_tui_shutdown_has_a_deadline(self):
        stop = threading.Event()
        worker = threading.Thread(target=stop.wait, daemon=True)
        worker.start()
        controller = Controller.__new__(Controller)
        controller.worker = worker
        controller.failed = False
        with patch.object(controller, "cancel"):
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "收尾超时"):
                controller.join(0.02)
            self.assertLess(time.monotonic() - started, 1)
        stop.set()
        worker.join(1)

    def test_tui_quit_restores_terminal_when_worker_does_not_cancel(self):
        from tui_pty import Terminal

        project = self.root / "project"
        project.mkdir()
        terminal = Terminal(project, self.root / "pty", controlled=True)
        try:
            terminal.wait(lambda: "lite-pi" in terminal.text)
            terminal.send("BLOCK_WORKER\r")
            terminal.wait(lambda: "worker blocked" in terminal.text)
            terminal.send("/quit\r")
            terminal.wait(lambda: terminal.process.poll() is not None, timeout=20)
            self.assertIn("收尾超时", terminal.raw.decode(errors="replace"))
            import termios

            actual, expected = termios.tcgetattr(terminal.slave), list(terminal.before)
            actual[3] &= ~getattr(termios, "PENDIN", 0)
            expected[3] &= ~getattr(termios, "PENDIN", 0)
            self.assertEqual(actual, expected)
        finally:
            terminal.close()

    def test_stdio_shutdown_kills_a_stubborn_owned_process(self):
        async def check():
            transport = OwnedStdio()
            pid_file = self.root / "pid"
            program = "import os,signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path('pid').write_text(str(os.getpid()));time.sleep(60)"
            with (self.root / "stderr").open("w") as stderr:
                async with transport.connect(
                    sys.executable, ["-c", program], {"PATH": "/usr/bin:/bin"}, self.root, stderr
                ):
                    with anyio.fail_after(5):
                        while not pid_file.exists():
                            await anyio.sleep(0.01)
                pid = int(pid_file.read_text())
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)

        anyio.run(check)

    def test_cli_rejects_invalid_budget_before_model_access(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "lite_pi",
                "code",
                "--session",
                str(self.root / "session.jsonl"),
                "--root",
                str(self.root),
                "--max-turns",
                "0",
                "hello",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("必须是正整数", proc.stderr)
        self.assertFalse((self.root / "session.jsonl").exists())

    def test_chat_context_window_reaches_session_limits(self):
        from lite_pi.cli import main

        with (
            patch("lite_pi.cli.sys.stdin.isatty", return_value=True),
            patch("lite_pi.cli.sys.stdout.isatty", return_value=True),
            patch("lite_pi.tui.app.run_chat", return_value=0) as launch,
        ):
            self.assertEqual(main(["chat", "--root", str(self.root), "--context-tokens", "4096"]), 0)
        self.assertEqual(launch.call_args.args[3]["context_tokens"], 4096)
        self.assertEqual(launch.call_args.args[3]["max_tokens"], 640)

    def test_chat_rejects_invalid_context_window_before_terminal_start(self):
        proc = subprocess.run(
            [sys.executable, "-m", "lite_pi", "chat", "--context-tokens", "0"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("必须是正整数", proc.stderr)
        self.assertNotIn("交互模式需要 TTY", proc.stderr)

    def test_mcp_host_closes_later_servers_after_os_error(self):
        from unittest.mock import Mock

        from lite_pi.mcp_host import McpHost

        first, second = Mock(), Mock()
        first.name, second.name = "first", "second"
        primary = OSError("first server close failed")
        first.close.side_effect = primary
        second.close.side_effect = RuntimeError("second server close failed")
        with self.assertRaises(OSError) as caught:
            McpHost([first, second]).close()
        self.assertIs(caught.exception, primary)
        first.close.assert_called_once()
        second.close.assert_called_once()
        self.assertIn("second server close failed", str(primary.__notes__))
