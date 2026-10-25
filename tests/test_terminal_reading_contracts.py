"""Terminal behavior through the existing CLI/PTY seam; controlled responses are deterministic."""

import json
import platform
import re
import time
import unittest
from pathlib import Path
from uuid import uuid4

from tui_pty import ROOT, Terminal

from lite_pi.journal import Journal
from lite_pi.session import CODE_TOOLS
from lite_pi.spend.ledger import Ledger


class ReadingPtyContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.base = ROOT / ".artifacts/terminal_reading/contracts" / uuid4().hex
        self.work = self.base / "project"
        self.work.mkdir(parents=True)
        self.before = Ledger().status()["request_count"]
        self.terminal: Terminal | None = None

    def tearDown(self) -> None:
        if self.terminal is not None:
            self.terminal.close()
            status = json.loads((self.terminal.evidence / "terminal.json").read_text())
            self.assertTrue(status["restored"])
        self.assertEqual(Ledger().status()["request_count"], self.before)

    def launch(self, **options: object) -> Terminal:
        self.terminal = Terminal(self.work, self.base / "terminal", controlled=True, **options)
        self.terminal.wait(lambda: "Enter" in self.terminal.text)
        return self.terminal

    def test_status_amount_fits_the_actual_terminal_cells(self) -> None:
        t = self.launch()
        self.assertIn("CNY", next(line for line in t.text.splitlines() if "约 " in line))

    def test_assistant_markdown_and_role_styles_are_readable(self) -> None:
        t = self.launch(columns=100, rows=32)
        t.send("TUI_MARKDOWN\r")
        t.wait(lambda: "TUI-DONE" in t.text)
        self.assertIn("金额检查", t.text)
        self.assertIn("乘法计算", t.text)
        self.assertNotIn("**乘法计算**", t.text)
        self.assertNotIn("`total()`", t.text)
        self.assertIn("  total = 12 * 2", t.text)
        self.assertIn("› 你", t.text)
        self.assertIn("● lite-pi", t.text)
        self.assertEqual(t.text.count("TUI-DONE"), 1)
        bold = [c for row in t.screen.buffer.values() for c in row.values() if c.data == "乘"]
        self.assertTrue(any(c.bold for c in bold))
        code_row = next(
            row for row in t.screen.buffer.values() if "total =" in "".join(c.data for c in row.values())
        )
        self.assertFalse(any(c.underscore for c in code_row.values()))
        t.capture("markdown-and-roles")

    def test_reading_anchor_survives_live_output_resize_and_draft(self) -> None:
        t = self.launch()
        t.send("TUI_STREAM\r")
        t.wait(lambda: "段落030" in t.text)
        t.send("\x1b[5~")
        for _ in range(3):
            t.read(0.03)
        anchor = re.findall(r"段落\d{3}", t.text)[0]
        t.send("草稿é🙂")
        t.wait(lambda: "有新内容" in t.text)
        self.assertIn(anchor, t.text)
        t.resize(60, 24)
        t.wait(lambda: anchor in t.text and "草稿" in t.text, 1)
        self.assertIn(anchor, t.text)
        self.assertIn("草稿", t.text)
        t.capture("reading-anchor")
        t.send("\x1b[1;5F")
        t.wait(lambda: "段落159" in t.text)
        self.assertNotIn("有新内容", t.text)
        self.assertIn("草稿", t.text)
        t.capture("latest-with-draft")

    def test_markdown_decoding_cannot_emit_terminal_controls(self) -> None:
        t = self.launch()
        t.send("TUI_UNTRUSTED\r")
        t.wait(lambda: "TUI-SAFE" in t.text)
        self.assertNotIn(b"\x1b]52;c;evil", t.raw)
        self.assertNotIn("\u202e".encode(), t.raw)
        self.assertIn("状态伪装", t.text)
        self.assertIn("+ safe", t.text)
        self.assertIn("- old", t.text)
        t.capture("safe-markdown")

    def test_tool_summary_can_expand_without_reexecuting(self) -> None:
        (self.work / "source.txt").write_text("\n".join(f"data-{i:02d}" for i in range(12)))
        t = self.launch(columns=100, rows=80)
        t.send("TUI_TOOLS\r")
        t.wait(lambda: "TUI-TOOLS-DONE" in t.text)
        self.assertIn("read · source.txt", t.text)
        self.assertIn("已完成", t.text)
        self.assertNotIn("data-11", t.text)
        self.assertNotIn("tool_call_id", t.text)
        t.capture("tool-summary")
        t.send(b"\x0f")
        t.wait(lambda: "tool_call_id" in t.text)
        self.assertIn("data-11", t.text)
        t.capture("tool-details")
        t.send(b"\x0f")
        t.wait(lambda: "tool_call_id" not in t.text)

    def test_details_read_original_evidence_after_hook_rewrites_preview(self) -> None:
        extension = self.work / ".lite-pi/extensions/large"
        extension.mkdir(parents=True)
        (extension / "extension.py").write_text(
            "def register(api):\n    api.register_tool('large', 'large result', {'type': 'object'}, lambda args: {'text': 'ORIGINAL-EVIDENCE-MARKER'})\n    api.on('tool_result', lambda event: {'text': 'changed preview'})\n"
        )
        t = self.launch(columns=100, rows=60)
        t.send("TUI_LARGE\r")
        t.wait(lambda: "TUI-TOOLS-DONE" in t.text)
        self.assertIn("changed preview", t.text)
        self.assertNotIn("ORIGINAL-EVIDENCE-MARKER", t.text)
        t.send(b"\x0f")
        t.wait(lambda: "ORIGINAL-EVIDENCE-MARKER" in t.text)
        self.assertIn("原始工具输出", t.text)
        t.capture("original-evidence")

    def test_single_long_line_marks_summary_cut_and_details_keep_readable_receipt(self) -> None:
        extension = self.work / ".lite-pi/extensions/large"
        extension.mkdir(parents=True)
        (extension / "extension.py").write_text(
            "def register(api):\n    api.register_tool('large', 'long line', {'type': 'object'}, lambda args: {'text': 'x' * 5500 + 'LAST-ORIGINAL-MARKER'})\n"
        )
        t = self.launch()
        t.send("TUI_LARGE\r")
        t.wait(lambda: "TUI-TOOLS-DONE" in t.text)
        self.assertIn("显示摘要已截断", t.text)
        t.send(b"\x0f")
        t.wait(lambda: "raw_stop_reason" in t.text)
        t.send("\x1b[5~")
        t.wait(lambda: "LAST-ORIGINAL-MARKER" in re.sub(r"\s+|↪", "", t.text))
        self.assertNotIn("显示预览已截断", t.text)
        t.capture("full-readable-original")

    def test_large_tool_details_keep_input_responsive_and_expose_evidence_limit(self) -> None:
        extension = self.work / ".lite-pi/extensions/large"
        extension.mkdir(parents=True)
        (extension / "extension.py").write_text(
            "def register(api):\n    api.register_tool('large', 'large result', {'type': 'object'}, lambda args: {'text': 'large-line\\n' * 200000})\n"
        )
        t = self.launch()
        t.send("TUI_LARGE\r")
        t.wait(lambda: "TUI-TOOLS-DONE" in t.text)
        t.wait(lambda: "本轮结束" in t.text.splitlines()[-2])
        started = time.monotonic()
        t.send(b"\x0f")
        t.send("大结果草稿")
        t.wait(lambda: "大结果草稿" in t.text, 1)
        self.assertLess(time.monotonic() - started, 1)
        # Following the latest content shows the final run receipt; the tool is above it.
        t.wait(lambda: "raw_stop_reason" in t.text)
        t.capture("latest-run-details")
        t.send("\x1b[5~")
        t.wait(lambda: "证据不可读" in t.text, 1)
        self.assertIn("完整原始证据", t.text)
        self.assertIn("大结果草稿", t.text)
        t.capture("bounded-large-details")

    def test_detail_reader_rejects_missing_external_and_symlink_evidence(self) -> None:
        from lite_pi.session import RUNS
        from lite_pi.tui.evidence import original

        run = RUNS / ("run-" + uuid4().hex)
        (run / "tool-raw").mkdir(parents=True)
        raw = run / "tool-raw/01-call.json"
        outside = self.base / "private.txt"
        outside.write_text("PRIVATE-CANARY")
        raw.write_text(json.dumps({"result": {"output_path": str(outside)}}))
        self.assertIn("不可读", original(str(raw)))
        self.assertNotIn("PRIVATE-CANARY", original(str(raw)))
        link = run / "linked.log"
        link.symlink_to(outside)
        raw.write_text(json.dumps({"result": {"output_path": str(link)}}))
        self.assertIn("不可读", original(str(raw)))
        self.assertNotIn("PRIVATE-CANARY", original(str(raw)))
        self.assertIn("不可读", original(str(run / "tool-raw/missing.json")))
        raw.write_text("[]")
        self.assertIn("不可读", original(str(raw)))
        raw.write_text(json.dumps({"result": {"output_path": {"bad": "type"}}}))
        self.assertIn("不可读", original(str(raw)))

    def test_reopened_history_expansion_is_not_new_business_content(self) -> None:
        (self.work / "source.txt").write_text("original source")
        t = self.launch()
        t.send("TUI_TOOLS\r")
        t.wait(lambda: "TUI-TOOLS-DONE" in t.text)
        t.close()
        path = next((self.work / ".artifacts/sessions").glob("*.jsonl"))
        self.terminal = Terminal(self.work, self.base / "reopened", controlled=True, session=path)
        t = self.terminal
        t.wait(lambda: "Enter" in t.text)
        t.send("\x1b[5~\x0f")
        t.wait(lambda: "tool_call_id" in t.text)
        self.assertNotIn("有新内容", t.text)

    def test_collapsing_hidden_tool_position_returns_to_its_title(self) -> None:
        from lite_pi.tui.messages import Entry, ToolContent, ToolState
        from lite_pi.tui.viewport import TranscriptControl

        expanded = True
        entry = Entry(
            ToolContent("read", "source.txt", ToolState.DONE, "\n".join(f"line{i}" for i in range(40)))
        )
        view = TranscriptControl(lambda: [entry.block(expanded)])
        view.create_content(80, 6)
        view.page(-1)
        self.assertGreater(view.anchor.line, 3)
        expanded = False
        view.create_content(80, 6)
        self.assertEqual(view.rows[view.top].anchor.line, -1)

    def test_collapsing_wrapped_tool_position_returns_to_its_title(self) -> None:
        from lite_pi.tui.messages import Entry, Role, TextContent, ToolContent, ToolState
        from lite_pi.tui.viewport import Anchor, TranscriptControl

        expanded = True
        entry = Entry(ToolContent("read", "source.txt", ToolState.DONE, "x" * 5500))
        after = Entry(TextContent(Role.USER, "after\n" * 20))
        view = TranscriptControl(lambda: [entry.block(expanded), after.block(expanded)])
        view.following = False
        view.anchor = Anchor(entry.identity, 1, 3000)
        view.create_content(80, 6)
        self.assertGreater(view.anchor.column, 1000)
        expanded = False
        view.create_content(80, 6)
        self.assertEqual(view.rows[view.top].anchor.line, -1)

    def test_compact_status_and_approval_preserve_rejected_draft(self) -> None:
        t = self.launch(extra=("--approvals",))
        t.send("TUI_APPROVE\r")
        t.wait(lambda: "待审批" in t.text)
        self.assertIn("Enter", t.text)
        self.assertNotIn("本次结束", t.text)
        self.assertNotIn(str(Path.home()), t.text)
        t.send("尚未批准的新要求\r")
        t.wait(lambda: "尚未批准的新要求" in t.text)
        for _ in range(3):
            t.read(0.05)
        self.assertIn("尚未批准的新要求", t.text)
        self.assertFalse((self.work / "notes/approved.txt").exists())
        t.resize(40, 12)
        t.wait(lambda: "尚未批准" in t.text)
        self.assertTrue("待审批" in t.text or "等待审批" in t.text)
        t.capture("approval-compact")

    def test_open_fence_stays_readable_until_closed(self) -> None:
        t = self.launch()
        t.send("TUI_FENCE\r")
        t.wait(lambda: "x = 1" in t.text)
        self.assertIn("```python", t.text)
        t.wait(lambda: "TUI-FENCE-CLOSED" in t.text)
        self.assertNotIn("```python", t.text)
        self.assertIn("x = 1", t.text)

    def test_long_history_remains_responsive_during_64k_stream(self) -> None:
        path = self.work / ".artifacts/sessions/long.jsonl"
        history = Journal.create(path, self.work, tools=CODE_TOOLS)
        for i in range(500):
            history.append("user", {"content": f"历史提问{i:03d}"})
            history.append("assistant", {"role": "assistant", "content": f"历史回答{i:03d}，保留事实。"})
        t = self.launch(session=path)
        t.send("TUI_LOAD\r")
        t.wait(lambda: "load-line " in t.text)
        measurements = {}
        started = time.monotonic()
        t.send("负载草稿🙂")
        t.wait(lambda: "负载草稿" in t.text, 1)
        measurements["typing_seconds"] = time.monotonic() - started
        started = time.monotonic()
        t.send("\x1b[5~")
        t.wait(lambda: "有新内容" in t.text, 1)
        measurements["paging_seconds"] = time.monotonic() - started
        started = time.monotonic()
        t.resize(100, 32)
        t.wait(lambda: "负载草稿" in t.text and "Ctrl+End" in t.text, 1)
        measurements["resize_seconds"] = time.monotonic() - started
        t.capture("load-reading")
        (self.base / "latency.json").write_text(
            json.dumps(
                {
                    "system": platform.platform(),
                    "history_blocks": 1000,
                    "active_characters": 65536,
                    "events_per_second": 50,
                    "measurements": measurements,
                },
                indent=2,
            )
            + "\n"
        )
        self.assertTrue(all(value < 1 for value in measurements.values()))

    def test_monochrome_and_small_window_keep_meaning_and_draft(self) -> None:
        t = self.launch(color=False)
        t.send("TUI_MARKDOWN\r")
        t.wait(lambda: "TUI-DONE" in t.text)
        self.assertTrue(
            all(
                c.fg == "default" and c.bg == "default"
                for row in t.screen.buffer.values()
                for c in row.values()
            )
        )
        t.send("保留草稿")
        t.resize(30, 9)
        t.wait(lambda: "窗口过小" in t.text and "保留草稿" in t.text, 1)
        t.resize(80, 24)
        t.wait(lambda: "TUI-DONE" in t.text and "保留草稿" in t.text, 1)

    def test_reopening_unknown_effect_keeps_visible_block_and_draft(self) -> None:
        extension = self.work / ".lite-pi/extensions/broken"
        extension.mkdir(parents=True)
        (extension / "extension.py").write_text(
            "def register(api):\n    def execute(args):\n        api.write('partial.txt', 'partial')\n        raise RuntimeError('failure after write')\n    api.register_tool('broken', 'write then fail', {'type': 'object'}, execute)\n"
        )
        t = self.launch()
        t.send("TUI_UNKNOWN\r")
        t.wait(lambda: "效果未知" in t.text)
        self.assertEqual((self.work / "partial.txt").read_text(), "partial")
        t.close()
        path = next((self.work / ".artifacts/sessions").glob("*.jsonl"))
        self.terminal = Terminal(self.work, self.base / "reopened", controlled=True, session=path)
        t = self.terminal
        t.wait(lambda: "Enter" in t.text)
        self.assertIn("效果未知", t.text.splitlines()[-2])
        t.send("不要重复执行\r")
        t.wait(lambda: "不要重复执行" in t.text)
        t.wait(lambda: "拒绝" in t.text)
        self.assertIn("不要重复执行", t.text)
        t.capture("unknown-reopened")

    def test_malformed_tool_arguments_do_not_crash_presentation(self) -> None:
        t = self.launch()
        t.send("TUI_INVALID\r")
        t.wait(lambda: "TUI-INVALID-DONE" in t.text)
        self.assertIn("未执行", t.text)
        self.assertIn("Validation failed", t.text)


if __name__ == "__main__":
    unittest.main()
