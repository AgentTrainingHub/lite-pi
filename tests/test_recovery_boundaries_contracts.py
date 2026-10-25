"""Recovery failure regressions: scripted transport, real SQLite and child processes."""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from deterministic_transport import response, transport
from fixtures.recovery import RecoveryFixture
from fixtures.responses import model, tool_call

from lite_pi.journal import Journal
from lite_pi.project_gate import ProjectGate
from lite_pi.session import AgentSession
from lite_pi.tasks import StaleOwner, task_db_path


class RecoveryBoundaryContracts(RecoveryFixture):
    def test_reopened_scope_blocks_outside_write(self) -> None:
        journal = self.journal("write")
        session = AgentSession.open(journal.path)
        session.steer("Only allowed/", paths=("allowed/",))
        with patch("lite_pi.loop.complete", return_value=model("ack")):
            session.submit("remember scope", max_turns=1)
        reopened = AgentSession.open(journal.path)
        call = tool_call("write", {"path": "outside.txt", "content": "NO"}, "outside")
        with patch("lite_pi.loop.complete", return_value=model("", calls=[call])):
            result = reopened.submit("write outside", max_turns=1)
        self.assertFalse((self.work / "outside.txt").exists())
        self.assertEqual(reopened.control.scope, ("allowed/",))
        self.assertEqual(result["transcript"][0]["results"][0]["executed"], False)

    def test_unknown_result_survives_reopen_and_prevents_model(self) -> None:
        journal = self.journal("write")
        book = self.book(journal)
        gate = self.admit(
            book,
            self.registry("write").authorize(tool_call("write", {"path": "x", "content": "x"}, "unknown")),
        )
        book.commit(
            gate.intent_id, {"text": "uncertain", "effect": "unknown", "executed": True, "is_error": True}
        )
        book.close()
        for _ in range(2):
            with patch("lite_pi.loop.complete") as provider:
                result = AgentSession.open(journal.path).resume()
            provider.assert_not_called()
            self.assertEqual(result["status"], "effect_unknown")
            self.assertEqual(result["tasks"]["intents"][0]["status"], "effect_unknown")

    def test_call_identity_is_scoped_to_assistant_and_branch(self) -> None:
        (self.work / "token").write_text("OK")
        journal = self.journal("read")
        book = self.book(journal)
        registry = self.registry("read")
        call = tool_call("read", {"path": "token"}, "repeated")
        for index in range(3):
            if index == 2:
                journal.branch(journal.records[0]["id"], "branch")
            self.assistant(journal, call)
            gate = book.admit(registry.authorize(call), journal.leaf_id())
            self.assertIsNone(book.recover(registry, journal.incomplete_tail()))
            self.assertIsNone(journal.incomplete_tail())
            self.assertIsNotNone(gate.intent_id)
        intents = book.snapshot()["intents"]
        self.assertEqual(len(intents), 3)
        self.assertEqual(sum(r["kind"] == "tool" for r in journal.records), 3)
        self.assertEqual(len([r for r in journal.chain() if r["kind"] == "tool"]), 1)

    def test_approval_object_changes_and_insert_is_atomic(self) -> None:
        journal = self.journal("write")
        book = self.book(journal, approvals=True)
        registry = self.registry("write")
        gate = self.admit(
            book,
            registry.authorize(
                tool_call("write", {"path": "target", "content": "new", "overwrite": True}, "change")
            ),
        )
        book.decide(gate.outcome["approval_id"], "approve")
        (self.work / "target").write_text("third party")
        result = book.recover(registry, journal.incomplete_tail())
        self.assertEqual(result["status"], "superseded")
        self.assertEqual((self.work / "target").read_text(), "third party")
        other = self.book(self.journal("write"), approvals=True)
        other._db.execute(
            "CREATE TRIGGER fail_approval BEFORE INSERT ON approvals BEGIN SELECT RAISE(ABORT, 'fixture failure'); END"
        )
        with self.assertRaisesRegex(sqlite3.IntegrityError, "fixture failure"):
            self.admit(
                other, registry.authorize(tool_call("write", {"path": "no", "content": "no"}, "atomic"))
            )
        self.assertEqual(other.snapshot()["intents"], [])

    def test_retry_each_attempt_has_its_own_budget_and_unknown_usage(self) -> None:
        for cap in (1, 2):
            with self.subTest(cap=cap):
                journal = self.journal()
                replies = [response("", status=500), response("done", tokens=3)]
                with transport(self.base / f"http-{cap}", replies) as wire:
                    result = AgentSession.open(journal.path).submit(
                        "run", max_tokens=32, max_task_calls=cap, max_task_output_tokens=100
                    )
                budget = result["tasks"]["budget"]
                self.assertEqual(wire.call_count, cap)
                self.assertEqual(result["request_count"], cap)
                self.assertEqual(budget["unknown_calls"], 1)
                self.assertEqual(budget["unknown_output"], 32)
                self.assertEqual(budget["spent_calls"], cap - 1)
                self.assertEqual(budget["spent_output"], 3 if cap == 2 else 0)
                self.assertEqual(budget["reserved_calls"], 0)
                self.assertEqual(result["status"], "budget" if cap == 1 else "stop")

    def test_stale_budget_owner_cannot_reserve_or_release(self) -> None:
        journal = self.journal()
        first = self.book(journal)
        reserved = first.budget.reserve("turn", 32)
        second = self.book(journal)
        for action in (
            lambda: first.budget.reserve("turn", 32),
            lambda: first.budget.release(reserved),
            lambda: first.budget.settle(reserved, {"completion_tokens": 1}, 1),
        ):
            with self.assertRaises(StaleOwner):
                action()
        self.assertEqual(second.budget.snapshot()["unknown_calls"], 1)

    def test_cleanup_failure_is_visible_and_connections_close(self) -> None:
        journal = self.journal()
        runs = self.base / "runs"
        events = []
        with (
            patch("lite_pi.session.RUNS", runs),
            patch("lite_pi.loop.complete", return_value=model("done")),
            patch.object(ProjectGate, "release", side_effect=RuntimeError("fixture release failure")),
        ):
            with self.assertRaisesRegex(RuntimeError, "fixture release failure"):
                AgentSession.open(journal.path).submit("run", on_event=events.append)
        result = json.loads(next(runs.glob("*/result.json")).read_text())
        self.assertEqual(result["status"], "effect_unknown")
        self.assertIn("task_cleanup_failed", result["reason"])
        self.assertEqual(events[-1]["status"], "effect_unknown")
        # Closing a gate must not skip the task database even if gate cleanup fails.
        book = self.book(self.journal())
        book.gate.acquire("read")
        with patch.object(book.gate, "release", side_effect=RuntimeError("fail")):
            with self.assertRaises(RuntimeError):
                book.close()
        for db in (book._db, book.gate._gate):
            with self.assertRaises(sqlite3.ProgrammingError):
                db.execute("SELECT 1")
        book.close()

    def test_constructor_failure_closes_acquired_connections(self) -> None:
        journal = self.journal()
        from lite_pi.task_storage import connect

        db = connect(task_db_path(journal))
        with (
            patch("lite_pi.tasks.connect", return_value=db),
            patch("lite_pi.tasks.ProjectGate", side_effect=RuntimeError("gate open failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "gate open failed"):
                self.book(journal)
        with self.assertRaises(sqlite3.ProgrammingError):
            db.execute("SELECT 1")

    def test_final_workers_are_closed_and_unsafe_extensions_are_exclusive(self) -> None:
        journal = self.journal()
        extension = self.work / ".lite-pi/extensions/unsafe/extension.py"
        extension.parent.mkdir(parents=True)
        extension.write_text(
            "def register(api):\n    api.register_tool('unsafe', 'effect', {'type':'object'}, lambda args: {'text':'ok'})\n"
        )
        held = self.book(self.journal("read"))
        held.gate.acquire("read")
        with patch("lite_pi.loop.complete") as provider:
            result = AgentSession.open(journal.path).submit("run")
        self.assertEqual(result["status"], "queued")
        provider.assert_not_called()
        held.close()
        with patch("lite_pi.loop.complete", return_value=model("done")):
            result = AgentSession.open(journal.path).submit("run")
        self.assertEqual(result["tasks"]["workers"][-1]["mode"], "write")
        self.assertEqual(result["tasks"]["workers"][-1]["state"], "closed")

    def test_submit_resolves_pending_tools_before_appending_user(self) -> None:
        (self.work / "token").write_text("OK")
        journal = self.journal("read")
        book = self.book(journal)
        self.admit(book, self.registry("read").authorize(tool_call("read", {"path": "token"}, "pending")))
        book.close()
        with patch("lite_pi.loop.complete", return_value=model("done")):
            result = AgentSession.open(journal.path).submit("new request", max_turns=1)
        self.assertEqual(result["status"], "stop")
        messages = Journal(journal.path).messages()
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool", "user", "assistant"])

    def test_approved_effect_crash_does_not_repeat_in_new_process(self) -> None:
        journal = self.journal("write")
        book = self.book(journal, approvals=True)
        gate = self.admit(
            book,
            self.registry("write").authorize(
                tool_call("write", {"path": "target", "content": "once"}, "approve-crash")
            ),
        )
        book.decide(gate.outcome["approval_id"], "approve")
        book.close()
        script = f"import sys; sys.path.insert(0, {str(Path.cwd() / 'tests')!r}); from crash_injection import install; install(); from pathlib import Path; from lite_pi.session import AgentSession; AgentSession.open(Path({str(journal.path)!r})).resume(max_task_calls=1)"
        env = {"PATH": "/usr/bin:/bin", "HOME": os.environ["HOME"], "LITE_PI_CRASH_AFTER": "effect"}
        # Do not change limits after approval: it would supersede the candidate.
        script = script.replace(".resume(max_task_calls=1)", ".resume()")
        child = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=20
        )
        self.assertEqual(child.returncode, 86, child.stderr)
        target = self.work / "target"
        stamp = target.stat().st_mtime_ns
        with patch("lite_pi.loop.complete", return_value=model("done")):
            result = AgentSession.open(journal.path).resume(max_turns=1)
        self.assertEqual(result["status"], "stop")
        self.assertEqual(target.stat().st_mtime_ns, stamp)
        self.assertEqual(Journal(journal.path).records[-2]["body"]["outcome"]["effect"], "reconciled")

    def test_legacy_identity_migrates_and_ambiguity_preserves_database(self) -> None:
        for ambiguous in (False, True):
            journal = self.journal("read")
            book = self.book(journal)
            call = tool_call("read", {"path": "token"}, "legacy")
            self.admit(book, self.registry("read").authorize(call))
            book.close()
            with sqlite3.connect(task_db_path(journal)) as db:
                definition = db.execute("SELECT sql FROM sqlite_master WHERE name='intents'").fetchone()[0]
                definition = (
                    definition.replace("CREATE TABLE intents", "CREATE TABLE old_intents")
                    .replace("assistant_id TEXT NOT NULL,", "")
                    .replace("UNIQUE(session_id, assistant_id, call_id)", "UNIQUE(session_id, call_id)")
                )
                db.execute(definition)
                columns = [r[1] for r in db.execute("PRAGMA table_info(old_intents)")]
                fields = ",".join(columns)
                db.execute(f"INSERT INTO old_intents ({fields}) SELECT {fields} FROM intents")
                db.execute("DROP TABLE intents")
                db.execute("ALTER TABLE old_intents RENAME TO intents")
            if ambiguous:
                journal.branch(journal.records[0]["id"], "other branch")
                self.assistant(journal, call)
                with self.assertRaisesRegex(ValueError, "ambiguous legacy"):
                    self.book(journal)
                with sqlite3.connect(task_db_path(journal)) as db:
                    self.assertNotIn("assistant_id", [r[1] for r in db.execute("PRAGMA table_info(intents)")])
                    self.assertEqual(db.execute("SELECT COUNT(*) FROM intents").fetchone()[0], 1)
            else:
                opened = self.book(journal)
                self.assertIsNone(opened.recover(self.registry("read"), journal.incomplete_tail()))

    def test_retry_safe_extension_still_requires_approval_and_exclusive_slot(self) -> None:
        extension = self.work / ".lite-pi/extensions/idempotent/extension.py"
        extension.parent.mkdir(parents=True)
        extension.write_text(
            "def register(api):\n    def execute(args): return api.write('changed.txt', 'changed')\n    api.register_tool('idempotent', 'idempotent write', {'type':'object'}, execute, replay='safe')\n"
        )
        journal = self.journal()
        held = self.book(self.journal("read"))
        held.gate.acquire("read")
        with patch("lite_pi.loop.complete") as provider:
            queued = AgentSession.open(journal.path).submit("run", approvals=True)
        provider.assert_not_called()
        self.assertEqual(queued["status"], "queued")
        held.close()
        with patch(
            "lite_pi.loop.complete", return_value=model("", calls=[tool_call("idempotent", {}, "safe-write")])
        ):
            waiting = AgentSession.open(journal.path).submit("run", approvals=True)
        self.assertEqual(waiting["status"], "awaiting_approval")
        self.assertFalse((self.work / "changed.txt").exists())

    def test_approval_version_reaches_actual_file_commit(self) -> None:
        for name in ("write", "edit"):
            target = self.work / name
            target.write_text("before")
            journal = self.journal(name)
            registry = self.registry(name)
            book = self.book(journal, approvals=True)
            args = (
                {"path": name, "content": "after", "overwrite": True}
                if name == "write"
                else {"path": name, "old": "before", "new": "after"}
            )
            gate = self.admit(book, registry.authorize(tool_call(name, args, "cas")))
            book.decide(gate.outcome["approval_id"], "approve")
            original = book._mark

            def mark(identity: str, status: str, reason: str | None = None) -> None:
                original(identity, status, reason)
                if status == "intent":
                    target.write_text("before concurrent")

            with patch.object(book, "_mark", side_effect=mark):
                book.recover(registry, journal.incomplete_tail())
            self.assertEqual(target.read_text(), "before concurrent")
            outcome = journal.records[-1]["body"]["outcome"]
            self.assertFalse(outcome["executed"])
            self.assertIn("approved file version changed", outcome["text"])
            receipt = book.decide(gate.outcome["approval_id"], "revoke")
            self.assertFalse(receipt["executed"])

    def test_concurrent_first_open_and_admission_use_one_database(self) -> None:
        script = f"""
import sys, time
from pathlib import Path
from lite_pi.project_gate import ProjectGate
sys.stdin.readline()
gate = ProjectGate(Path({str(self.work)!r}), sys.argv[1])
try:
    admission = gate.acquire('read')
    print(admission['state'], flush=True)
    time.sleep(.1)
finally: gate.close()
"""
        children = [
            subprocess.Popen(
                [sys.executable, "-c", script, str(i)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for i in range(8)
        ]
        try:
            for child in children:
                child.stdin.write("start\n")
                child.stdin.flush()
            outputs = [child.communicate(timeout=15) for child in children]
            for child, (out, err) in zip(children, outputs, strict=True):
                self.assertEqual(child.returncode, 0, err)
                self.assertIn(out.strip(), {"running", "queued"})
            self.assertTrue(any(out.strip() == "queued" for out, _ in outputs))
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.communicate()

    def test_revoking_started_approval_preserves_unknown_execution(self) -> None:
        journal = self.journal("write")
        book = self.book(journal, approvals=True)
        gate = self.admit(
            book,
            self.registry("write").authorize(
                tool_call("write", {"path": "target", "content": "new"}, "started")
            ),
        )
        approval_id = gate.outcome["approval_id"]
        book.decide(approval_id, "approve")
        intent_id = book.snapshot()["intents"][0]["intent_id"]
        book._mark(intent_id, "intent")
        receipt = book.decide(approval_id, "revoke")
        self.assertIsNone(receipt["executed"])
        self.assertEqual(book.snapshot()["intents"][0]["status"], "intent")
