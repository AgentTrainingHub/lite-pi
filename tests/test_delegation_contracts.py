"""Deterministic delegation contracts. Scripted replies are not real-model hits."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock

from deterministic_transport import response, transport
from fixtures.delegation import child_report, delegate_call, delegate_many, delegate_results, scripted
from fixtures.responses import tool_call

from lite_pi.delegate import DelegationTable
from lite_pi.journal import Journal
from lite_pi.project_gate import gate_db_path
from lite_pi.session import AgentSession
from lite_pi.tools.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = """
def register(api):
    api.register_tool(
        "note",
        "Leave a note.",
        {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        lambda args: {"text": "noted", "is_error": False},
    )
"""


class DelegationContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.work = self.root / "project"
        self.work.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_child_does_not_load_host_surface(self) -> None:
        self._file("src/lookup.py", "def lookup(name):\n    return table[name]\n")
        self._extension()
        self._mcp()
        journal = Journal.create(self.root / "child.jsonl", self.work, tools=("read",))
        parent = AgentSession(self.work, tools=("read",), journal=journal)
        book = parent._book(approvals=False, approval_ttl=None, claim=True, max_task_calls=4)
        try:
            with transport(self.root / "http", [response("seen")]) as wire:
                seen = AgentSession(self.work, tools=("read",)).investigate(
                    "read src/lookup.py",
                    scope=("src/lookup.py",),
                    budget=book.budget,
                    cancel=Event(),
                )
        finally:
            book.close()
        self.assertEqual(wire.call_count, 1)
        self.assertEqual(seen["tools"], ["read"])
        self.assertEqual(seen["extensions"], [])
        self.assertFalse((self.work / "mcp-started").exists())
        self.assertNotIn("note", seen["tools"])

    def test_plain_submit_starts_configured_mcp(self) -> None:
        self._file("src/lookup.py", "def lookup(name):\n    return 1\n")
        self._mcp()
        with self.assertRaises(Exception):
            AgentSession(self.work, tools=("read",)).submit("look", max_turns=1)
        self.assertTrue((self.work / "mcp-started").exists())

    def test_late_report_does_not_replace_a_sealed_generation(self) -> None:
        table = DelegationTable(
            Workspace(self.work),
            Event(),
            lambda _event: None,
            "parent",
            investigate=AgentSession(self.work, tools=("read",)).investigate,
            can_read=lambda: True,
            parent_scope=lambda: None,
        )
        generation = table.open_generation()
        table.seal(generation)
        report = child_report(generation)
        self.assertFalse(table.accept(report))
        self.assertEqual(table.receipts, {})
        self.assertEqual(table.dropped, [report.survey_id])

    def test_close_reports_a_live_thread(self) -> None:
        table = DelegationTable(
            Workspace(self.work),
            Event(),
            lambda _event: None,
            "parent",
            investigate=AgentSession(self.work, tools=("read",)).investigate,
            can_read=lambda: True,
            parent_scope=lambda: None,
        )
        hold = Event()

        def block() -> None:
            hold.wait(2)

        thread = threading.Thread(target=block, name="delegate-stuck")
        thread.start()
        table.threads.append(thread)
        with self.assertRaises(RuntimeError) as caught:
            table.close(timeout=0.05)
        self.assertIn("delegate_cleanup_failed", str(caught.exception))
        hold.set()
        thread.join(1)
        self.assertFalse(thread.is_alive())

    def test_delegate_without_parent_budget_does_not_copy_one(self) -> None:
        self._file("src/lookup.py", "def lookup(name):\n    return table[name]\n")
        calls = scripted([response("", calls=[delegate_call("only", ["src/lookup.py"], "c1")])])
        session = AgentSession(self.work, tools=("read", "delegate"))
        with transport(self.root / "http", calls):
            result = session.submit("survey", max_turns=1, max_tokens=128)
        item = delegate_results(result)[0]
        self.assertEqual(item["reason"], "shared_budget_required")
        self.assertEqual(item["surveys"], [])
        self.assertFalse((self.work / "src/lookup.py").read_text().endswith("changed\n"))

    def test_shared_budget_is_not_copied_per_child(self) -> None:
        self._file("src/a.py", "alpha\n")
        self._file("src/b.py", "beta\n")
        calls = scripted(
            [
                response(
                    "",
                    calls=[
                        delegate_many(
                            [("read a", ["src/a.py"]), ("read b", ["src/b.py"])],
                            "c1",
                        )
                    ],
                ),
                response("FACT src/a.py: alpha"),
            ]
        )
        journal = Journal.create(self.root / "budget.jsonl", self.work, tools=("read", "delegate"))
        session = AgentSession(self.work, tools=("read", "delegate"), journal=journal)
        with transport(self.root / "http", calls) as wire:
            result = session.submit("survey both", max_turns=1, max_tokens=128, max_task_calls=2)
        item = delegate_results(result)[0]
        statuses = {survey["status"] for survey in item["surveys"]}
        self.assertIn("budget", statuses)
        self.assertEqual(wire.call_count, 2)
        database = sqlite3.connect(journal.path.with_name(journal.path.name + ".tasks.sqlite"))
        database.row_factory = sqlite3.Row
        rows = list(database.execute("SELECT * FROM budget"))
        database.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["max_calls"], 2)
        self.assertLessEqual(rows[0]["spent_calls"], 2)
        self.assertTrue(item["narrative_is_not_evidence"])
        self.assertFalse(item["child_write_detected"])

    def test_parallel_reads_overlap_without_a_second_writer(self) -> None:
        self._file("src/a.py", "alpha\n")
        self._file("src/b.py", "beta\n")
        entered = Event()
        lock = Lock()
        state = {"n": 0, "inside": 0, "peak": 0}

        def read(*_args, **_kwargs):
            with lock:
                state["n"] += 1
                current = state["n"]
            if current == 1:
                return response(
                    "",
                    calls=[
                        delegate_many(
                            [("read a", ["src/a.py"]), ("read b", ["src/b.py"])],
                            "c1",
                        )
                    ],
                )
            with lock:
                state["inside"] += 1
                state["peak"] = max(state["peak"], state["inside"])
                if state["inside"] >= 2:
                    entered.set()
            if not entered.wait(3):
                raise TimeoutError("children did not overlap")
            try:
                return response("seen")
            finally:
                with lock:
                    state["inside"] -= 1

        journal = Journal.create(self.root / "parallel.jsonl", self.work, tools=("read", "delegate"))
        session = AgentSession(self.work, tools=("read", "delegate"), journal=journal)
        with transport(self.root / "http", read):
            result = session.submit("survey both", max_turns=1, max_tokens=128, max_task_calls=6)
        item = delegate_results(result)[0]
        self.assertEqual(item["peak_concurrency"], 2)
        self.assertTrue(item["overlapped"])
        self.assertGreaterEqual(state["peak"], 2)
        self.assertTrue(item["joined"])
        gate = sqlite3.connect(gate_db_path(self.work))
        modes = [row[0] for row in gate.execute("SELECT mode FROM workers")]
        gate.close()
        self.assertTrue(modes)
        self.assertTrue(all(mode == "write" for mode in modes))

    def test_missing_path_does_not_cancel_the_sibling(self) -> None:
        self._file("src/lookup.py", "return table[name]\n")
        calls = scripted(
            [
                response(
                    "",
                    calls=[
                        delegate_many(
                            [("missing", ["missing.py"]), ("read lookup", ["src/lookup.py"])],
                            "c1",
                        )
                    ],
                ),
                response("FACT src/lookup.py: return table[name]"),
            ]
        )
        result = self._submit(calls, max_task_calls=6)
        item = delegate_results(result)[0]
        statuses = {survey["status"]: survey for survey in item["surveys"]}
        self.assertEqual(statuses["rejected"]["reason"], "path_missing")
        self.assertEqual(statuses["complete"]["tools"], ["read"])
        self.assertEqual(item["failure_policy"], "continue_siblings")
        proof = next(proof for proof in item["proofs"] if proof["path"] == "src/lookup.py")
        digest = hashlib.sha256((self.work / "src/lookup.py").read_bytes()).hexdigest()
        self.assertEqual(proof["sha256"], digest)
        self.assertIn("return table[name]", proof["excerpt"])

    def test_completion_claim_is_not_evidence(self) -> None:
        text = "def lookup(name):\n    return table[name]\n"
        self._file("src/lookup.py", text)
        calls = scripted(
            [
                response("", calls=[delegate_call("fix it", ["src/lookup.py"], "c1")]),
                response("已完成，src/lookup.py 已修改为 return 0\nFACT src/lookup.py: return 0"),
            ]
        )
        result = self._submit(calls, max_task_calls=6)
        item = delegate_results(result)[0]
        survey = item["surveys"][0]
        self.assertTrue(item["narrative_is_not_evidence"])
        self.assertEqual(survey["completion_claim"], "unsupported")
        self.assertTrue(any("quote_not_in_file" in line for line in survey["unsupported"]))
        self.assertFalse(item["child_write_detected"])
        self.assertEqual((self.work / "src/lookup.py").read_text(), text)
        proof = item["proofs"][0]
        self.assertEqual(proof["sha256"], hashlib.sha256(text.encode()).hexdigest())
        self.assertIn("return table[name]", proof["excerpt"])
        self.assertNotIn("return 0", proof["excerpt"])

    def test_child_cannot_use_shell_mcp_or_paths_outside_the_survey(self) -> None:
        self._file("src/lookup.py", "return table[name]\n")
        self._file("secret.txt", "CANARY-SECRET-delegation\n")
        self._extension()
        calls = scripted(
            [
                response("", calls=[delegate_call("read the function", ["src/lookup.py"], "c1")]),
                response(
                    "",
                    calls=[
                        tool_call("bash", {"command": "echo hi"}, "bash-1"),
                        tool_call("write", {"path": "secret.txt", "content": "nope"}, "write-1"),
                        tool_call("read", {"path": "secret.txt"}, "read-1"),
                        tool_call("mcp__marker__note", {}, "mcp-1"),
                        tool_call("note", {"text": "nope"}, "note-1"),
                    ],
                ),
                response("FACT src/lookup.py: return table[name]"),
            ]
        )
        result = self._submit(calls, tools=("read", "edit", "write", "bash", "delegate"), max_task_calls=8)
        item = delegate_results(result)[0]
        survey = item["surveys"][0]
        self.assertEqual(survey["tools"], ["read"])
        self.assertIn("bash:tool_not_authorized", survey["refusals"])
        self.assertIn("write:tool_not_authorized", survey["refusals"])
        self.assertIn("read:path_scope", survey["refusals"])
        self.assertIn("mcp__marker__note:unknown_tool", survey["refusals"])
        self.assertIn("note:unknown_tool", survey["refusals"])
        self.assertIn("note", result["tools"])
        self.assertNotIn("CANARY-SECRET-delegation", json.dumps(item))
        self.assertEqual((self.work / "secret.txt").read_text(), "CANARY-SECRET-delegation\n")
        self.assertFalse(item["child_write_detected"])

    def test_parent_cancel_before_children_leaves_no_survey(self) -> None:
        self._file("src/lookup.py", "return table[name]\n")
        calls = scripted([response("", calls=[delegate_call("read", ["src/lookup.py"], "c1")])])
        journal = Journal.create(self.root / "cancel.jsonl", self.work, tools=("read", "delegate"))
        session = AgentSession(self.work, tools=("read", "delegate"), journal=journal)

        def on_event(event):
            if event.get("type") == "tool_start" and event.get("name") == "delegate":
                session.cancel()

        with transport(self.root / "http", calls) as wire:
            result = session.submit(
                "survey", max_turns=1, max_tokens=128, max_task_calls=4, on_event=on_event
            )
        self.assertEqual(result["status"], "aborted")
        item = delegate_results(result)[0]
        self.assertFalse(item["executed"])
        self.assertEqual(item["reason"], "cancelled")
        self.assertEqual(wire.call_count, 1)
        self.assertEqual((self.work / "src/lookup.py").read_text(), "return table[name]\n")

    def test_parent_cancel_drops_an_in_flight_child(self) -> None:
        self._file("src/lookup.py", "return table[name]\n")
        journal = Journal.create(self.root / "inflight.jsonl", self.work, tools=("read", "delegate"))
        session = AgentSession(self.work, tools=("read", "delegate"), journal=journal)
        state = {"n": 0}
        lock = Lock()

        def read(*_args, **_kwargs):
            with lock:
                state["n"] += 1
                current = state["n"]
            if current == 1:
                return response("", calls=[delegate_call("read", ["src/lookup.py"], "c1")])
            session.cancel()
            return response("已完成")

        with transport(self.root / "http", read):
            result = session.submit("survey", max_turns=1, max_tokens=128, max_task_calls=4)
        self.assertEqual(result["status"], "aborted")
        item = delegate_results(result)[0]
        self.assertEqual(item["surveys"][0]["status"], "aborted")
        self.assertTrue(item["joined"])
        self.assertEqual((self.work / "src/lookup.py").read_text(), "return table[name]\n")

    def test_third_survey_is_rejected_by_the_schema(self) -> None:
        self._file("src/lookup.py", "return table[name]\n")
        calls = scripted(
            [
                response(
                    "",
                    calls=[
                        delegate_many(
                            [
                                ("one", ["src/lookup.py"]),
                                ("two", ["src/lookup.py"]),
                                ("three", ["src/lookup.py"]),
                            ],
                            "c1",
                        )
                    ],
                )
            ]
        )
        result = self._submit(calls, max_task_calls=4)
        item = delegate_results(result)[0]
        self.assertIn("Validation failed", item["text"])
        self.assertFalse(item["executed"])

    def test_crashed_delegate_is_not_replayed(self) -> None:
        self._file("src/lookup.py", "return table[name]\n")
        journal = Journal.create(
            self.root / "crash.jsonl", self.work, tools=("read", "edit", "write", "bash", "delegate")
        )
        env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "LITE_PI_CRASH_AFTER": "intent"}
        if "HOME" in os.environ:
            env["HOME"] = os.environ["HOME"]
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tests/delegation_crash_driver.py"),
                str(self.work),
                str(journal.path),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        (self.root / "crash.err").write_text(proc.stderr)
        self.assertEqual(proc.returncode, 86)
        self.assertNotIn("DEEPSEEK_API_KEY", proc.stderr)
        task_db = sqlite3.connect(journal.path.with_name(journal.path.name + ".tasks.sqlite"))
        task_db.row_factory = sqlite3.Row
        intent = task_db.execute("SELECT name, status, replay FROM intents").fetchone()
        task_db.close()
        self.assertEqual(intent["name"], "delegate")
        self.assertEqual(intent["status"], "intent")
        self.assertEqual(intent["replay"], "unsafe")
        resumed = AgentSession.open(journal.path).resume(max_turns=1, max_task_calls=4)
        self.assertEqual(resumed["status"], "effect_unknown")
        self.assertEqual(resumed["reason"], "effect_unverified")
        self.assertEqual((self.work / "src/lookup.py").read_text(), "return table[name]\n")

    def test_cli_delegate_command_does_not_start_a_child(self) -> None:
        env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
        if "HOME" in os.environ:
            env["HOME"] = os.environ["HOME"]
        proc = subprocess.run(
            [sys.executable, "-m", "lite_pi", "delegate"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.loads(proc.stdout)["status"], "rejected")
        help_text = subprocess.run(
            [sys.executable, "-m", "lite_pi", "code", "--help"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertIn("--delegate", help_text.stdout)

    def _submit(self, calls, *, max_task_calls: int, tools: tuple[str, ...] = ("read", "delegate")) -> dict:
        journal = Journal.create(self.root / "session.jsonl", self.work, tools=tools)
        session = AgentSession(self.work, tools=tools, journal=journal)
        with transport(self.root / "http", calls):
            return session.submit("survey", max_turns=1, max_tokens=128, max_task_calls=max_task_calls)

    def _file(self, relative: str, text: str) -> None:
        path = self.work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _extension(self) -> None:
        path = self.work / ".lite-pi/extensions/note"
        path.mkdir(parents=True)
        (path / "extension.py").write_text(EXTENSION, encoding="utf-8")

    def _mcp(self) -> None:
        directory = self.work / ".lite-pi"
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "servers": [
                {
                    "name": "marker",
                    "command": sys.executable,
                    "args": ["-c", "open('mcp-started','w').write('x')"],
                    "tools": [],
                    "timeout_seconds": 1,
                }
            ]
        }
        (directory / "mcp.json").write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
