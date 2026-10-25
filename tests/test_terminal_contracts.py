"""Deterministic terminal host contracts and actual PTY keyboard tests; no paid responses."""

import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Unpack
from unittest.mock import patch
from uuid import uuid4

from fixtures.responses import model, tool_call
from tui_pty import ROOT, Terminal

from lite_pi.control import RunControl
from lite_pi.journal import Journal
from lite_pi.provider import ModelResult
from lite_pi.session import CODE_TOOLS, RunLimits
from lite_pi.spend.ledger import Ledger
from lite_pi.tools.contracts import JsonObject
from lite_pi.tui.app import ChatApp, resume_command
from lite_pi.tui.catalog import Catalog
from lite_pi.tui.controller import Controller, Update
from lite_pi.tui.display import Screen, plain, welcome


class TuiHostContracts(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name).resolve()
        self.catalog = Catalog(self.work, CODE_TOOLS)
        self.session = self.catalog.create()

    def controller(self, **limits: Unpack[RunLimits]) -> Controller:
        return Controller(self.catalog, self.session, limits)

    def finish(self, c: Controller) -> None:
        end = time.monotonic() + 10
        while c.busy and time.monotonic() < end:
            c.drain()
            time.sleep(0.01)
        self.assertFalse(c.busy)

    def test_closed_run_controls_are_returned_without_replay(self) -> None:
        control = RunControl()
        a = control.submit("steer", "one")
        control.submit("follow_up", "two")
        result = control.return_pending()
        self.assertEqual([v["text"] for v in result], ["one", "two"])
        self.assertEqual(result[0]["id"], a["id"])
        self.assertTrue(all(v["disposition"] == "not_sent" for v in result))
        self.assertIsNone(control.drain("steer"))
        self.assertIsNone(control.drain("follow_up"))
        control.begin_run()
        self.assertIsNone(control.drain("steer"))

    def test_display_sanitizes_control_sequences_and_preserves_unicode(self) -> None:
        text = "中文🙂 é\x1b[2J\x1b]52;c;secret\x07\r\u202e"
        safe = plain(text)
        self.assertNotIn("\x1b", safe)
        self.assertNotIn("\x07", safe)
        self.assertNotIn("\r", safe)
        self.assertNotIn("\u202e", safe)
        self.assertIn("中文🙂 é", safe)
        self.assertEqual(plain("a\nb\tc"), "a\nb\tc")

    def test_compaction_failure_has_a_readable_terminal_status(self) -> None:
        screen = Screen(phase="compaction_failed")
        self.assertEqual(screen.phase_label, "摘要失败，历史已保留")
        screen.phase = "superseded"
        self.assertEqual(screen.phase_label, "候选已失效，请继续")

    def test_screen_does_not_duplicate_streaming_text_and_expands_tools(self) -> None:
        screen = Screen()
        screen.event({"type": "turn_start"})
        screen.event({"type": "message_delta", "text": "hello"})
        screen.event({"type": "message_end", "text": "hello"})
        screen.event({"type": "tool_start", "name": "read", "tool_call_id": "a"})
        screen.event(
            {
                "type": "tool_end",
                "result": {
                    "name": "read",
                    "tool_call_id": "a",
                    "executed": True,
                    "is_error": True,
                    "text": "missing",
                },
            }
        )
        self.assertEqual(screen.transcript().count("hello"), 1)
        self.assertIn("失败", screen.transcript())
        screen.expanded = True
        self.assertIn('"tool_call_id": "a"', screen.transcript())

    def test_opening_preserves_saved_authority_and_does_not_call_provider(self) -> None:
        path = self.work / "read.jsonl"
        Journal.create(path, self.work, tools=("read",))
        with patch("lite_pi.loop.complete") as provider:
            opened = self.catalog.open(path)
            self.assertEqual(opened.tools, ("read",))
            self.assertEqual(opened.inspect(), {})
        provider.assert_not_called()

    def test_session_navigation_rejects_foreign_and_linked_history(self) -> None:
        foreign = self.work / "other"
        foreign.mkdir()
        path = self.work / "foreign.jsonl"
        Journal.create(path, foreign, tools=CODE_TOOLS)
        with self.assertRaisesRegex(ValueError, "其他项目"):
            self.catalog.open(path)
        link = self.work / "link.jsonl"
        link.symlink_to(self.session.journal.path)
        with self.assertRaisesRegex(ValueError, "符号链接"):
            self.catalog.open(link)
        self.session.journal.path.write_text("broken")
        found, errors = self.catalog.list_sessions()
        self.assertEqual(found, [])
        self.assertTrue(errors)

    def test_inspection_does_not_claim_or_change_approval_candidate(self) -> None:
        call = tool_call("write", {"path": "x.txt", "content": "x"}, "w")
        with patch("lite_pi.loop.complete", return_value=model("", calls=[call])):
            result = self.session.submit("write", approvals=True, max_turns=1)
        self.assertEqual(result["status"], "awaiting_approval")
        before = self.session.inspect()
        for _ in range(3):
            self.assertEqual(before, self.session.inspect())
        self.assertFalse((self.work / "x.txt").exists())
        candidate = before["approvals"][0]["approval_id"]
        decision = self.session.decide(candidate, "reject")
        self.assertEqual(decision["decision"], "rejected")

    def test_multiple_submissions_use_one_session_and_unknown_command_keeps_draft(self) -> None:
        c = self.controller(max_turns=2)
        with patch("lite_pi.loop.complete", return_value=model("answer")) as provider:
            c.submit("one")
            self.finish(c)
            c.submit("two")
            self.finish(c)
        self.assertEqual(provider.call_count, 2)
        users = [r["body"]["content"] for r in self.session.journal.chain() if r["kind"] == "user"]
        self.assertEqual(users, ["one", "two"])
        with self.assertRaisesRegex(ValueError, "未知命令"):
            c.submit("/wat")
        with patch("lite_pi.loop.complete") as provider:
            c.submit("/status")
            c.submit("/sessions")
            c.submit("/help")
        provider.assert_not_called()

    def test_final_drain_race_restores_late_input_and_rejects_further_controls(self) -> None:
        c = self.controller()
        c.operation = "op"
        c._accepting = True
        c.worker = threading.Thread(target=lambda: None)
        c.worker.start()
        c.session.steer("late")
        c._publish("op", "result", {"status": "stop"})
        with self.assertRaises(ValueError):
            c._control("steer", "too late")
        c.drain()
        self.assertEqual(c.restored_draft, "late")
        self.assertIsNone(c.session.control.drain("steer"))

    def test_late_events_cannot_change_current_screen(self) -> None:
        c = self.controller()
        c.operation = "old"
        c.command("/new", "")
        c.queue.put(Update("old", "event", {"type": "message_delta", "text": "stale"}))
        self.assertFalse(c.drain())
        self.assertNotIn("stale", c.screen.transcript())

    def test_startup_cancel_and_unexpected_worker_failure_are_visible(self) -> None:
        c = self.controller(max_turns=2)
        started = threading.Event()
        release = threading.Event()

        def complete(*args: object, stop: Callable[[], bool] | None = None, **kwargs: object) -> ModelResult:
            started.set()
            release.wait(3)
            return model("done")

        with patch("lite_pi.loop.complete", side_effect=complete):
            c.start("wait")
            self.assertTrue(started.wait(3))
            c.cancel()
            release.set()
            self.finish(c)
        self.assertTrue(c.cancel_requested)
        with patch("lite_pi.loop.complete", side_effect=SystemExit("broken extension")):
            c.start("again")
            self.finish(c)
        self.assertTrue(c.failed)
        self.assertTrue(c.closing)
        self.assertIn("SystemExit", c.screen.transcript())

    def test_unknown_effect_and_final_receipt_survive_worker_error(self) -> None:
        c = self.controller()

        def fail(_text: str, **limits: Unpack[RunLimits]) -> JsonObject:
            observe = limits["on_event"]
            observe(
                {
                    "type": "tool_end",
                    "result": {"name": "write", "effect": "unknown", "executed": True, "is_error": True},
                }
            )
            observe(
                {
                    "type": "run_end",
                    "result": {
                        "status": "effect_unknown",
                        "evidence": "/receipt",
                        "model": "actual",
                        "text": "",
                    },
                }
            )
            raise OSError("cleanup broke")

        with patch.object(c.session, "submit", side_effect=fail):
            c.start("write")
            self.finish(c)
        self.assertEqual(c.screen.phase, "effect_unknown")
        self.assertEqual(c.screen.result["evidence"], "/receipt")
        self.assertIn("效果未知", c.screen.transcript())
        self.assertIn("cleanup broke", c.screen.transcript())
        self.assertTrue(c.failed and c.closing)

    def test_final_event_delivers_receipt_even_when_session_raises(self) -> None:
        events = []
        with patch("lite_pi.loop.complete", side_effect=OSError("offline")):
            with self.assertRaises(OSError):
                self.session.submit("ask", on_event=events.append)
        end = events[-1]
        self.assertEqual(end["type"], "run_end")
        receipt = end["result"]
        self.assertEqual(receipt["status"], "error")
        self.assertTrue(Path(receipt["evidence"]).joinpath("result.json").is_file())

    def test_resume_command_quotes_shell_expansion_and_control_bytes(self) -> None:
        for name in ("space ' $(echo injected)", "control\x1b]52;c;secret\x07", "line\nnext"):
            project = self.work / name
            command = resume_command(project, project / "session")
            self.assertNotIn("\x1b", command)
            self.assertNotIn("\x07", command)
            self.assertNotIn("\n", command)
            shell = subprocess.run(
                ["/bin/bash", "-c", "set -- " + command + "; printf '%s' \"$4\""],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(shell.stdout, str(project))
        self.assertEqual(shlex.split(resume_command(self.work, self.work / "s"))[3], str(self.work))

    def test_pump_failure_exits_and_preserves_cleanup_error_and_signals(self) -> None:
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import DummyInput
        from prompt_toolkit.output import DummyOutput

        c = self.controller()
        original = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}

        async def check() -> None:
            with create_app_session(input=DummyInput(), output=DummyOutput()):
                app = ChatApp(c)
                future = asyncio.get_running_loop().create_future()

                async def run_async(*, pre_run: Callable[[], None], **kwargs: object) -> None:
                    pre_run()
                    await future

                def exit_app(*, exception: BaseException) -> None:
                    future.set_exception(exception)

                with (
                    patch.object(app.app, "run_async", side_effect=run_async),
                    patch.object(app.app, "exit", side_effect=exit_app),
                    patch.object(c, "drain", side_effect=RuntimeError("render failed")),
                    patch.object(c, "join", side_effect=OSError("join failed")),
                ):
                    with self.assertRaisesRegex(RuntimeError, "render failed") as raised:
                        await app.run()
                    self.assertIn("join failed", " ".join(raised.exception.__notes__))

        asyncio.run(check())
        self.assertEqual(original, {sig: signal.getsignal(sig) for sig in original})

    def test_pty_initialization_and_capture_failure_release_descriptors(self) -> None:
        import pty
        from unittest.mock import Mock

        master, slave = pty.openpty()
        with (
            patch("tui_pty.pty.openpty", return_value=(master, slave)),
            patch("tui_pty.subprocess.Popen", side_effect=OSError("spawn failed")),
        ):
            with self.assertRaisesRegex(OSError, "spawn failed"):
                Terminal(self.work, self.work / "failed-pty", controlled=True)
        for fd in (master, slave):
            with self.assertRaises(OSError):
                os.fstat(fd)
        terminal = Terminal(self.work, self.work / "capture-failed", controlled=True)
        terminal.capture = Mock(side_effect=OSError("capture failed"))
        with self.assertRaisesRegex(OSError, "capture failed"):
            terminal.close()
        self.assertIsNotNone(terminal.process.poll())
        for fd in (terminal.master, terminal.slave):
            with self.assertRaises(OSError):
                os.fstat(fd)

    def test_welcome_sizes_and_fields(self) -> None:
        for width in (120, 80, 40, 20):
            output = "".join(t for _, t in welcome(width, self.work, "new", CODE_TOOLS, "0.1.0"))
            if width >= 40:
                self.assertIn("lite-pi", output)
            if width >= 60:
                self.assertIn("▄▄▄", output)
            if width < 40:
                self.assertIn("Ctrl+D", output)


class TuiPtyContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.base = ROOT / ".artifacts/contracts-pty" / uuid4().hex
        self.base.mkdir(parents=True)
        self.work = self.base / "project"
        self.work.mkdir()
        self.before = Ledger().status()["request_count"]
        self.terminal = None

    def tearDown(self) -> None:
        if self.terminal is not None:
            self.terminal.close()
        self.assertEqual(Ledger().status()["request_count"], self.before)
        if self.terminal is not None:
            report = json.loads((self.terminal.evidence / "terminal.json").read_text())
            self.assertTrue(report["restored"])

    def launch(self, *, color: bool = True) -> Terminal:
        self.terminal = Terminal(self.work, self.base / "terminal", controlled=True, color=color)
        self.terminal.wait(lambda: "Enter 提交" in self.terminal.text)
        return self.terminal

    def test_formal_input_multiline_paste_and_three_turns(self) -> None:
        t = self.launch()
        t.capture("welcome")
        t.send("\x1b[200~第一行\n第二行\x1b[201~")
        for _ in range(5):
            t.read()
        self.assertNotIn("controlled:", t.text)
        t.capture("pasted")
        t.send("\r")
        t.wait(lambda: "本轮结束" in t.text)
        t.send("two\r")
        t.wait(lambda: "controlled:two" in t.text)
        t.send("three\r")
        t.wait(lambda: "controlled:three" in t.text)
        t.capture("three-turns")
        journals = list((self.work / ".artifacts/sessions").glob("*.jsonl"))
        users = [r["body"]["content"] for r in Journal(journals[0]).chain() if r["kind"] == "user"]
        self.assertEqual(users, ["第一行\n第二行", "two", "three"])

    def test_cancel_returns_to_input_then_continues(self) -> None:
        t = self.launch()
        t.send("WAIT\r")
        t.wait(lambda: "运行中" in t.text)
        t.send(b"\x03")
        t.wait(lambda: "已取消" in t.text)
        t.send("after-cancel\r")
        t.wait(lambda: "controlled:after-cancel" in t.text)
        t.capture("continued")

    def test_resize_draft_and_monochrome_safe_output(self) -> None:
        t = self.launch(color=False)
        t.send("草稿🙂 é")
        t.resize(40, 12)
        t.wait(lambda: "草稿" in t.text)
        t.capture("40x12")
        t.resize(120, 32)
        t.wait(lambda: "草稿🙂" in t.text)
        t.capture("120x32")
        t.send(b"\x03")
        t.send("/unknown\r")
        t.wait(lambda: "未知命令" in t.text and "/unknown" in t.text)
        self.assertIn("/unknown", t.text)

    def test_close_signal_restores_terminal(self) -> None:
        t = self.launch()
        os.kill(t.process.pid, signal.SIGTERM)
        t.wait(lambda: t.process.poll() is not None)
        self.assertEqual(t.process.returncode, 0)

    def test_hangup_while_busy_cancels_and_restores_terminal(self) -> None:
        t = self.launch()
        t.send("WAIT\r")
        t.wait(lambda: "运行中" in t.text)
        os.kill(t.process.pid, signal.SIGHUP)
        t.wait(lambda: t.process.poll() is not None)
        self.assertEqual(t.process.returncode, 0)

    def test_edit_history_unicode_and_control_sequence_safety(self) -> None:
        t = self.launch()
        t.send("abX\x1b[D\x7f中\r")
        t.wait(lambda: "controlled:a中X" in t.text)
        t.send("\x1b[A")
        for _ in range(3):
            t.read()
        self.assertIn("> a中X", t.text)
        t.send(b"\x03")
        t.send("HOSTILE\r")
        t.wait(lambda: "hostile:" in t.text)
        t.wait(lambda: "\\u001b" in t.text)
        self.assertNotIn(b"\x1b]52;c;c2VjcmV0", t.raw)
        self.assertNotIn(b"\x1b]0;injected-title", t.raw)
        t.capture("safe-output")

    def test_burst_output_keeps_draft_and_scroll_keys_work(self) -> None:
        t = self.launch()
        t.send("BURST\r")
        t.wait(lambda: "line 1 " in t.text)
        t.send("未提交🙂")
        t.wait(lambda: "本轮结束" in t.text and "未提交🙂" in t.text)
        t.capture("burst-draft")
        t.send("\x1b[5~")
        for _ in range(3):
            t.read()
        self.assertIn("未提交🙂", t.text)
        t.capture("scrolled")

    def test_render_failure_exits_and_restores_terminal(self) -> None:
        self.work = self.base / "render-failure"
        self.work.mkdir()
        t = self.launch()
        t.send("fail\r")
        t.wait(lambda: t.process.poll() is not None)
        self.assertNotEqual(t.process.returncode, 0)
        self.assertIn("controlled render failure", t.raw.decode(errors="replace"))

    def test_startup_failure_sanitizes_path_and_restores_terminal(self) -> None:
        self.work = self.base / "missing\x1b]52;c;fake\x07"
        self.terminal = Terminal(self.work, self.base / "terminal")
        t = self.terminal
        t.wait(lambda: t.process.poll() is not None)
        self.assertEqual(t.process.returncode, 2)
        self.assertNotIn(b"\x1b]52;c;fake", t.raw)
        self.assertTrue(b"\\u001b" in t.raw or b"\\x1b" in t.raw)

    def test_fatal_error_chain_is_readable_after_exit(self) -> None:
        t = self.launch()
        t.send("FATAL\r")
        t.wait(lambda: t.process.poll() is not None)
        self.assertEqual(t.process.returncode, 1)
        self.assertIn(b"SystemExit: original fatal", t.raw)
        self.assertNotIn(b"\x1b]52;c;fake", t.raw)

    def test_non_tty_rejects_without_requests(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "lite_pi", "chat", "--root", str(self.work)],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("TTY", result.stderr)
