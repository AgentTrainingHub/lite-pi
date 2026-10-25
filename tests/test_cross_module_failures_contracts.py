"""Cross-module fault contracts; all model responses and HTTP are controlled."""

import os
import stat
import threading
from decimal import Decimal
from unittest.mock import patch

from compare import arms, macos_run, upstream
from fixtures.recovery import RecoveryFixture
from fixtures.responses import model, tool_call

from lite_pi.session import AgentSession
from lite_pi.spend.gate import RequestNotSent, _http_stream, transmit
from lite_pi.spend.ledger import Ledger


class CrossModuleFailureContracts(RecoveryFixture):
    def test_graph_setup_error_survives_task_book_close_failure(self) -> None:
        parent = self.base / "graph-setup"
        parent.mkdir()
        with (
            patch.object(arms, "TaskBook") as book,
            patch.object(arms, "ToolRegistry", side_effect=ValueError("registry failed")),
            self.assertRaisesRegex(ValueError, "registry failed") as caught,
        ):
            book.return_value.gate.acquire.return_value = {"state": "running"}
            book.return_value.close.side_effect = RuntimeError("book close failed")
            arms.run_graph(parent, upstream.FIXTURE, run_id="setup-failure")
        self.assertIn("book close failed", str(caught.exception.__notes__))

    def test_mcp_observer_rejects_forged_workspace_marks(self) -> None:
        work = self.base / "mcp-project"

        def child(*args):
            state = work / "mcp-state"
            state.mkdir()
            (state / "calls.jsonl").write_text("mark\n")
            return {"returncode": 0, "mark_calls": 0}

        with patch.object(macos_run, "_python", side_effect=child):
            observed = macos_run._mcp(self.base / "venv", {}, work)
        self.assertFalse(observed["mark_recorded"])
        self.assertFalse(observed["pid_observed"])
        self.assertIsNone(observed["pid_alive"])

    def test_mcp_observer_requires_one_actual_call_and_process_exit(self) -> None:
        state = self.base / "mcp-observer"
        state.mkdir()
        (state / "pid").write_text("12345")
        for count, calls, expected in ((1, 1, True), (2, 2, False), (1, 0, False)):
            with self.subTest(count=count, calls=calls):
                (state / "calls.jsonl").write_text("mark\n" * count)
                with (
                    patch.object(macos_run, "_python", return_value={"mark_calls": calls}),
                    patch("compare.combo.os.kill", side_effect=ProcessLookupError),
                ):
                    observed = macos_run._mcp(self.base / "venv", {}, self.base / f"mcp-{count}-{calls}")
                self.assertEqual(observed["mark_recorded"], expected)
                self.assertTrue(observed["pid_observed"])
                self.assertFalse(observed["pid_alive"])
        with (
            patch.object(macos_run, "_python", return_value={"mark_calls": 1}),
            patch("compare.combo.os.kill", side_effect=PermissionError("cannot inspect")),
            self.assertRaises(PermissionError),
        ):
            macos_run._mcp(self.base / "venv", {}, self.base / "mcp-permission")

    def test_passing_files_cannot_mask_unresolved_run_status(self) -> None:
        for status, expected in (("effect_unknown", False), ("stop", True)):
            with self.subTest(status=status):
                parent = self.base / status
                parent.mkdir()
                with (
                    patch.object(upstream.AgentSession, "submit", return_value={"status": status}),
                    patch.object(upstream, "check", return_value={"returncode": 0}),
                    patch.object(upstream, "implementation_changed", return_value=True),
                ):
                    receipt = upstream.run_lite(parent)
                self.assertEqual(receipt["passed"], expected)

    def test_directory_sync_failure_after_publish_keeps_unknown_effect(self) -> None:
        registry = self.registry("write")
        fsync = os.fsync
        failed = False

        def fail_directory_once(fd):
            nonlocal failed
            if not failed and stat.S_ISDIR(os.fstat(fd).st_mode):
                failed = True
                raise OSError("directory sync failed after publication")
            return fsync(fd)

        with patch("lite_pi.tools.workspace.os.fsync", side_effect=fail_directory_once):
            result = registry.execute(tool_call("write", {"path": "out.txt", "content": "once"}, "w"))
        self.assertEqual((self.work / "out.txt").read_text(), "once")
        self.assertTrue(result["executed"])
        self.assertEqual(result["effect"], "unknown")

    def test_raw_evidence_failure_locks_memory_session(self) -> None:
        session = AgentSession(self.work, tools=("write",))
        call = tool_call("write", {"path": "out.txt", "content": "once"}, "w")
        with (
            patch("lite_pi.loop.complete", return_value=model("", calls=[call])),
            patch("lite_pi.tools.registry.ToolRegistry._write_raw", side_effect=OSError("disk full")),
        ):
            result = session.submit("write", max_turns=1)
        self.assertEqual((self.work / "out.txt").read_text(), "once")
        self.assertEqual(result["status"], "effect_unknown")
        self.assertEqual(result["reason"], "tool_evidence_failed")
        with patch("lite_pi.loop.complete") as provider:
            again = session.submit("again")
        provider.assert_not_called()
        self.assertEqual(again["status"], "effect_unknown")

    def test_consumer_exception_cannot_clear_observed_unknown_effect(self) -> None:
        session = AgentSession(self.work, tools=("write",))
        call = tool_call("write", {"path": "out.txt", "content": "once"}, "w")

        def observe(event):
            if event["type"] == "tool_end":
                raise RuntimeError("consumer failed")

        with (
            patch("lite_pi.loop.complete", return_value=model("", calls=[call])),
            patch("lite_pi.tools.registry.ToolRegistry._write_raw", side_effect=OSError("disk full")),
            self.assertRaisesRegex(RuntimeError, "consumer failed"),
        ):
            session.submit("write", max_turns=1, on_event=observe)
        with patch("lite_pi.loop.complete") as provider:
            again = session.submit("again")
        provider.assert_not_called()
        self.assertEqual(again["status"], "effect_unknown")

    def test_cancel_while_waiting_releases_task_reservation_without_http(self) -> None:
        ledger = Ledger(self.base / "ledger.json", self.base / "calls.jsonl")
        ledger.bootstrap(Decimal("100"), currency="CNY", is_available=True)
        occupied = ledger.try_reserve(run_id="busy", purpose="test", input_tokens=1, output_tokens=1)
        self.assertTrue(occupied["ok"])
        budget = self.book(self.journal(), max_calls=1).budget
        cancelled = threading.Event()
        with (
            patch("lite_pi.spend.gate.load_api_key", return_value="fixture"),
            patch("lite_pi.spend.ledger.time.sleep", side_effect=lambda _: cancelled.set()),
            patch("lite_pi.spend.gate._read_response") as wire,
        ):
            result = transmit(
                {"model": "deepseek-flash", "messages": [], "max_tokens": 16},
                run_id="waiting",
                purpose="test",
                ledger=ledger,
                stop=cancelled.is_set,
                budget=budget,
            )
        wire.assert_not_called()
        self.assertEqual(result["settled"]["error"], "aborted")
        self.assertEqual(result["attempts"], 0)
        self.assertEqual(budget.snapshot()["spent_calls"], 0)
        self.assertEqual(budget.snapshot()["reserved_calls"], 0)
        ledger.cancel(occupied["reservation_id"], reason="test-finished")
        self.assertEqual(ledger.status()["request_count"], 0)

    def test_shell_failure_and_consumer_exception_lock_memory_session(self) -> None:
        def observe(event):
            if event["type"] == "tool_end":
                raise RuntimeError("consumer failed")

        for reason in ("cleanup_failed", "runtime_error", "workspace_sync_failed"):
            with self.subTest(reason=reason):
                session = AgentSession(self.work, tools=("bash",))
                call = tool_call("bash", {"command": "touch result.txt"}, "shell")
                outcome = {
                    "name": "bash",
                    "executed": True,
                    "is_error": True,
                    "reason": reason,
                    "effect": "unknown",
                    "blocks_run": True,
                    "text": reason,
                }
                with (
                    patch("lite_pi.loop.complete", return_value=model("", calls=[call])),
                    patch("lite_pi.tools.registry.ToolRegistry.execute", return_value=outcome),
                    self.assertRaisesRegex(RuntimeError, "consumer failed"),
                ):
                    session.submit("run command", max_turns=1, on_event=observe)
                with patch("lite_pi.loop.complete") as provider:
                    again = session.submit("continue")
                provider.assert_not_called()
                self.assertEqual(again["status"], "effect_unknown")

    def test_cancel_after_global_slot_releases_both_reservations(self) -> None:
        ledger = Ledger(self.base / "ledger.json", self.base / "calls.jsonl")
        ledger.bootstrap(Decimal("100"), currency="CNY", is_available=True)
        budget = self.book(self.journal(), max_calls=1).budget
        cancelled = threading.Event()

        def acquire(**kwargs):
            kwargs.pop("stop")
            result = ledger.try_reserve(**kwargs)
            cancelled.set()
            return result

        with (
            patch("lite_pi.spend.gate.load_api_key", return_value="fixture"),
            patch.object(ledger, "reserve_wait", side_effect=acquire),
            patch("lite_pi.spend.gate._read_response") as wire,
        ):
            result = transmit(
                {"model": "deepseek-flash", "messages": [], "max_tokens": 16},
                run_id="waiting",
                purpose="test",
                ledger=ledger,
                stop=cancelled.is_set,
                budget=budget,
            )
        wire.assert_not_called()
        self.assertEqual(result["attempts"], 0)
        self.assertEqual(ledger.status()["in_flight"], 0)
        self.assertEqual(ledger.status()["request_count"], 0)
        self.assertEqual(budget.snapshot()["reserved_calls"], 0)

    def test_transport_does_not_create_connection_after_cancel(self) -> None:
        with patch("lite_pi.spend.gate.CancellableHTTPSConnection") as connection:
            with self.assertRaises(RequestNotSent):
                _http_stream("POST", "/chat/completions", {}, "fixture", timeout=1, stop=lambda: True)
        connection.assert_not_called()

    def test_deadline_while_waiting_releases_reservations_without_http(self) -> None:
        ledger = Ledger(self.base / "ledger.json", self.base / "calls.jsonl")
        ledger.bootstrap(Decimal("100"), currency="CNY", is_available=True)
        occupied = ledger.try_reserve(run_id="busy", purpose="test", input_tokens=1, output_tokens=1)
        budget = self.book(self.journal(), max_calls=1).budget
        expired = False

        def expire(_seconds: float) -> None:
            nonlocal expired
            expired = True

        with (
            patch("lite_pi.spend.gate.load_api_key", return_value="fixture"),
            patch.object(budget, "stop_reason", side_effect=lambda: "time_budget" if expired else None),
            patch("lite_pi.spend.ledger.time.sleep", side_effect=expire),
            patch("lite_pi.spend.gate._read_response") as wire,
        ):
            result = transmit(
                {"model": "deepseek-flash", "messages": [], "max_tokens": 16},
                run_id="waiting",
                purpose="test",
                ledger=ledger,
                budget=budget,
            )
        wire.assert_not_called()
        self.assertEqual(result["attempts"], 0)
        self.assertEqual(budget.snapshot()["reserved_calls"], 0)
        ledger.cancel(occupied["reservation_id"], reason="test-finished")
        self.assertEqual(ledger.status()["request_count"], 0)

    def test_sent_callback_failure_settles_unknown_and_keeps_original_error(self) -> None:
        ledger = Ledger(self.base / "ledger.json", self.base / "calls.jsonl")
        ledger.bootstrap(Decimal("100"), currency="CNY", is_available=True)
        budget = self.book(self.journal(), max_calls=1).budget
        with (
            patch("lite_pi.spend.gate.load_api_key", return_value="fixture"),
            patch("lite_pi.spend.gate.EVIDENCE_ROOT", self.base / "http"),
            patch("lite_pi.spend.gate._read_response", side_effect=RuntimeError("consumer failed")) as wire,
            self.assertRaisesRegex(RuntimeError, "consumer failed"),
        ):
            transmit(
                {"model": "deepseek-flash", "messages": [], "max_tokens": 16},
                run_id="last-permitted-call",
                purpose="test",
                ledger=ledger,
                budget=budget,
            )
        wire.assert_called_once()
        self.assertEqual(ledger.status()["in_flight"], 0)
        self.assertEqual(ledger.status()["request_count"], 1)
        self.assertEqual(ledger.status()["unknown_usage_calls"], 1)
        self.assertEqual(budget.snapshot()["unknown_calls"], 1)
        self.assertEqual(budget.snapshot()["reserved_calls"], 0)
