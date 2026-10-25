"""Deterministic recovery contracts. Scripted model replies are not live-model evidence."""

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

from deterministic_transport import response, transport
from fixtures.recovery import RecoveryFixture
from fixtures.responses import model, tool_call

from lite_pi.compaction import estimate_request
from lite_pi.extensions import HookRegistration
from lite_pi.journal import Journal
from lite_pi.project_gate import READ_LIMIT, gate_db_path
from lite_pi.resources import load_context, request_prefix
from lite_pi.session import AgentSession
from lite_pi.tasks import StaleOwner, task_db_path
from lite_pi.tools.contracts import HaltAdmission, RunAdmission
from lite_pi.tools.edit import write_file
from lite_pi.tools.workspace import Workspace


class RecoveryContracts(RecoveryFixture):
    def test_final_arguments_are_stored_and_invalid_calls_are_not(self) -> None:
        journal = self.journal("read", "write")
        registry = self.registry("read", "write")
        registry._hooks.append(
            HookRegistration(
                "tool_call",
                lambda payload: {"arguments": {**payload["arguments"], "path": "renamed.txt"}},
                "contract",
            )
        )
        prepared = registry.authorize(tool_call("read", {"path": "original.txt"}, "call-final"))
        self.assertEqual(prepared.arguments["path"], "renamed.txt")
        self.assertEqual(prepared.replay, "safe")
        book = self.book(journal)
        gate = self.admit(book, prepared)
        self.assertIsInstance(gate, RunAdmission)
        stored = book.snapshot()["intents"][0]
        self.assertEqual(stored["arguments"]["path"], "renamed.txt")
        self.assertEqual(stored["status"], "intent")
        invalid = registry.authorize(tool_call("read", {"path": 4}, "call-invalid"))
        self.assertFalse(invalid.get("ready"))
        self.assertEqual(len(book.snapshot()["intents"]), 1)

    def test_explicit_incomplete_leaf_can_be_recovered(self) -> None:
        (self.work / "token.txt").write_text("token")
        for historical in (False, True):
            with self.subTest(historical=historical):
                journal = self.journal("read")
                registry = self.registry("read")
                book = self.book(journal)
                call = tool_call("read", {"path": "token.txt"}, "pending")
                self.admit(book, registry.authorize(call))
                leaf = journal.leaf_id()
                book.close()
                if historical:
                    journal.branch(journal.records[0]["id"], "another branch")
                opened = AgentSession.open(journal.path)
                with patch("lite_pi.loop.complete", return_value=model("done")):
                    result = opened.resume(leaf_id=leaf, max_turns=1)
                self.assertEqual(result["status"], "stop")
                recovered = Journal(journal.path)
                self.assertIsNone(recovered.incomplete_tail())
                self.assertEqual(sum(r["kind"] == "tool" for r in recovered.chain()), 1)

    def test_unknown_effect_survives_cancel_and_blocks_next_memory_session_submit(self) -> None:
        session = AgentSession(self.work, tools=("write",))
        calls = [
            tool_call("write", {"path": "out.txt", "content": "once"}, "first"),
            tool_call("write", {"path": "other.txt", "content": "twice"}, "second"),
        ]

        def lose_result(prepared):
            (self.work / "out.txt").write_text("once")
            session.cancel()
            return {
                "name": "write",
                "tool_call_id": prepared.tool_call_id,
                "text": "result lost",
                "executed": True,
                "is_error": True,
                "effect": "unknown",
                "reason": "lost",
            }

        with (
            patch("lite_pi.loop.complete", return_value=model("", calls=calls)),
            patch("lite_pi.tools.registry.ToolRegistry.complete", side_effect=lose_result) as executed,
        ):
            first = session.submit("write", max_turns=2)
            self.assertEqual(executed.call_count, 1)
        self.assertEqual(first["status"], "effect_unknown")
        self.assertEqual(first["reason"], "lost")
        with patch("lite_pi.loop.complete") as provider:
            second = session.submit("try again", max_turns=2)
        provider.assert_not_called()
        self.assertEqual(second["status"], "effect_unknown")
        self.assertFalse((self.work / "other.txt").exists())

    def test_public_session_allows_call_id_reuse_across_assistant_turns(self) -> None:
        (self.work / "token.txt").write_text("token")
        journal = self.journal("read")
        call = tool_call("read", {"path": "token.txt"}, "reused")
        session = AgentSession(self.work, tools=("read",), journal=journal)
        with patch(
            "lite_pi.loop.complete",
            side_effect=[model("", calls=[call]), model("", calls=[call]), model("done")],
        ):
            result = session.submit("read twice", max_turns=3)
        self.assertEqual(result["status"], "stop")
        opened = AgentSession.open(journal.path)
        with patch("lite_pi.loop.complete", return_value=model("continued")):
            continued = opened.submit("continue", max_turns=1)
        self.assertEqual(continued["status"], "stop")
        projected = Journal(journal.path).messages()
        self.assertEqual(sum(m["role"] == "tool" and m["tool_call_id"] == "reused" for m in projected), 2)

    def test_same_assistant_duplicate_ids_remain_invalid(self) -> None:
        from lite_pi.journal import JournalError

        journal = self.journal("read")
        call = tool_call("read", {"path": "token.txt"}, "same")
        journal.append("assistant", {"content": "", "tool_calls": [call, call]})
        with self.assertRaisesRegex(JournalError, "duplicate_call_id"):
            journal.messages()

    def test_invalid_preview_is_tool_feedback_and_session_can_continue(self) -> None:
        directory = self.work / "directory"
        directory.mkdir()
        outside = self.base / "outside.txt"
        outside.write_text("private")
        (self.work / "link").symlink_to(outside)
        for index, target in enumerate((str(outside), "directory", "link", ".env")):
            with self.subTest(target=target):
                journal = self.journal("write")
                session = AgentSession(self.work, tools=("write",), journal=journal)
                calls = [
                    model("", calls=[tool_call("write", {"path": target, "content": "bad"}, "bad")]),
                    model(
                        "",
                        calls=[tool_call("write", {"path": f"valid-{index}.txt", "content": "ok"}, "good")],
                    ),
                    model("done"),
                ]
                with patch("lite_pi.loop.complete", side_effect=calls):
                    result = session.submit("write", max_turns=3)
                self.assertEqual(result["status"], "stop")
                failure = result["transcript"][0]["results"][0]
                self.assertEqual(failure["reason"], "invalid_target")
                self.assertFalse(failure["executed"])
                self.assertEqual((self.work / f"valid-{index}.txt").read_text(), "ok")
                self.assertIsNone(Journal(journal.path).incomplete_tail())
                self.assertEqual(outside.read_text(), "private")

    def test_unreadable_effect_on_recovery_remains_unknown(self) -> None:
        journal = self.journal("write")
        registry = self.registry("write")
        book = self.book(journal)
        call = tool_call("write", {"path": "target.txt", "content": "once"}, "uncertain")
        self.admit(book, registry.authorize(call))
        (self.work / "target.txt").mkdir()
        halted = book.recover(registry, journal.incomplete_tail())
        self.assertEqual(halted["status"], "effect_unknown")
        self.assertTrue((self.work / "target.txt").is_dir())

    def test_extension_path_does_not_imply_local_file_authority(self) -> None:
        from lite_pi.recovery import effect_image

        image = effect_image(Workspace(self.work), "custom_remote", {"path": "https://service/item"})
        self.assertIsNone(image.before_sha256)
        self.assertIsNone(image.after_sha256)
        self.assertIn("https://service/item", image.preview)

    def test_retry_requires_both_replay_policies(self) -> None:
        (self.work / "note.txt").write_text("TOKEN-18\n")
        journal = self.journal("read")
        registry = self.registry("read")
        call = tool_call("read", {"path": "note.txt"}, "call-read")
        self.assistant(journal, call)
        book = self.book(journal)
        self.admit(book, registry.authorize(call))
        registry.allowed["read"] = type(registry.allowed["read"])(
            "read",
            registry.allowed["read"].description,
            registry.allowed["read"].schema,
            registry.allowed["read"].execute,
            "unsafe",
        )
        halted = book.recover(registry, journal.incomplete_tail())
        self.assertEqual(halted["status"], "blocked")
        outcome = next(record["body"]["outcome"] for record in journal.records if record["kind"] == "tool")
        self.assertFalse(outcome["executed"])
        self.assertNotIn("TOKEN-18", outcome["text"])

    def test_safe_retry_runs_once_and_a_missing_tool_stops(self) -> None:
        (self.work / "note.txt").write_text("TOKEN-18\n")
        journal = self.journal("read")
        registry = self.registry("read")
        call = tool_call("read", {"path": "note.txt"}, "call-safe")
        self.assistant(journal, call)
        book = self.book(journal)
        self.admit(book, registry.authorize(call))
        self.assertIsNone(book.recover(registry, journal.incomplete_tail()))
        self.assertEqual(journal.records[-1]["body"]["outcome"]["executed"], True)
        self.assertIn("TOKEN-18", journal.records[-1]["body"]["content"])
        self.assertIsNone(journal.incomplete_tail())
        self.assertEqual(sum(record["kind"] == "tool" for record in journal.records), 1)

        other = self.journal("read")
        missing = self.registry("read")
        other_call = tool_call("read", {"path": "note.txt"}, "call-missing")
        self.assistant(other, other_call)
        other_book = self.book(other)
        self.admit(other_book, missing.authorize(other_call))
        del missing.allowed["read"]
        halted = other_book.recover(missing, other.incomplete_tail())
        self.assertEqual(halted["status"], "blocked")
        self.assertEqual((self.work / "note.txt").read_text(), "TOKEN-18\n")

    def test_saved_result_is_appended_once(self) -> None:
        (self.work / "note.txt").write_text("TOKEN-18\n")
        journal = self.journal("read")
        registry = self.registry("read")
        call = tool_call("read", {"path": "note.txt"}, "call-saved")
        self.assistant(journal, call)
        book = self.book(journal)
        gate = self.admit(book, registry.authorize(call))
        book._save_result(gate.intent_id, registry.complete(registry.authorize(call)))
        self.assertEqual(book.snapshot()["intents"][0]["status"], "result_saved")
        self.assertFalse(any(record["kind"] == "tool" for record in journal.records))
        book.close()
        resumed = self.book(journal)
        self.assertIsNone(resumed.recover(registry, journal.incomplete_tail()))
        self.assertEqual(sum(record["kind"] == "tool" for record in journal.records), 1)
        self.assertEqual(resumed.snapshot()["intents"][0]["status"], "settled")
        self.assertIsNone(journal.incomplete_tail())

    def test_queryable_effects_are_not_repeated_and_unknown_stays_unknown(self) -> None:
        target = self.work / "out.txt"
        target.write_text("SAME")
        stamp = target.stat().st_mtime_ns
        journal = self.journal("write", "bash")
        registry = self.registry("write", "bash")
        matched = tool_call("write", {"path": "out.txt", "content": "SAME", "overwrite": True}, "call-match")
        self.assistant(journal, matched)
        book = self.book(journal)
        self.admit(book, registry.authorize(matched))
        self.assertIsNone(book.recover(registry, journal.incomplete_tail()))
        self.assertEqual(target.stat().st_mtime_ns, stamp)
        self.assertEqual(target.read_text(), "SAME")
        self.assertEqual(journal.records[-1]["body"]["outcome"]["effect"], "reconciled")

        absent_journal = self.journal("write")
        absent_call = tool_call("write", {"path": "missing.txt", "content": "NEW"}, "call-absent")
        self.assistant(absent_journal, absent_call)
        absent = self.book(absent_journal)
        self.admit(absent, self.registry("write").authorize(absent_call))
        halted = absent.recover(self.registry("write"), absent_journal.incomplete_tail())
        self.assertEqual(halted["status"], "awaiting_decision")
        self.assertFalse((self.work / "missing.txt").exists())

        mismatch = self.work / "changed.txt"
        mismatch.write_text("BEFORE")
        mismatch_journal = self.journal("write")
        mismatch_call = tool_call(
            "write", {"path": "changed.txt", "content": "AFTER", "overwrite": True}, "call-mismatch"
        )
        self.assistant(mismatch_journal, mismatch_call)
        mismatch_book = self.book(mismatch_journal)
        self.admit(mismatch_book, self.registry("write").authorize(mismatch_call))
        mismatch.write_text("THIRD")
        unknown = mismatch_book.recover(self.registry("write"), mismatch_journal.incomplete_tail())
        self.assertEqual(unknown["status"], "effect_unknown")
        self.assertEqual(mismatch.read_text(), "THIRD")
        self.assertEqual(mismatch_book.blocking()["status"], "effect_unknown")

        bash_journal = self.journal("bash")
        bash_call = tool_call("bash", {"command": "printf HI"}, "call-bash")
        self.assistant(bash_journal, bash_call)
        bash_book = self.book(bash_journal)
        self.admit(bash_book, self.registry("bash").authorize(bash_call))
        bash_halt = bash_book.recover(self.registry("bash"), bash_journal.incomplete_tail())
        self.assertEqual(bash_halt["status"], "effect_unknown")

    def test_approval_receipts_bind_the_candidate(self) -> None:
        journal = self.journal("write")
        registry = self.registry("write")
        call = tool_call("write", {"path": "out.txt", "content": "MARKER"}, "call-approve")
        self.assistant(journal, call)
        book = self.book(journal, approvals=True)
        gate = self.admit(book, registry.authorize(call))
        self.assertIsInstance(gate, HaltAdmission)
        self.assertIn("MARKER", gate.outcome["preview"])
        self.assertFalse((self.work / "out.txt").exists())
        approval_id = gate.outcome["approval_id"]
        first = book.decide(approval_id, "approve")
        second = book.decide(approval_id, "approve")
        self.assertEqual(first, second)
        self.assertEqual(first["decision"], "approved")
        self.assertFalse(first["executed"])
        self.assertIsNone(book.recover(registry, journal.incomplete_tail()))
        self.assertEqual((self.work / "out.txt").read_text(), "MARKER")
        written = (self.work / "out.txt").stat().st_mtime_ns
        self.assertIsNone(journal.incomplete_tail())
        self.assertEqual((self.work / "out.txt").stat().st_mtime_ns, written)
        revoked = book.decide(approval_id, "revoke")
        self.assertTrue(revoked["executed"])
        self.assertEqual((self.work / "out.txt").read_text(), "MARKER")

    def test_reject_revoke_expiry_and_scope_change(self) -> None:
        journal = self.journal("write")
        registry = self.registry("write")
        call = tool_call("write", {"path": "out.txt", "content": "NO"}, "call-reject")
        self.assistant(journal, call)
        book = self.book(journal, approvals=True)
        gate = self.admit(book, registry.authorize(call))
        receipt = book.decide(gate.outcome["approval_id"], "reject")
        self.assertEqual(receipt["decision"], "rejected")
        halted = book.recover(registry, journal.incomplete_tail())
        self.assertEqual(halted["status"], "rejected")
        self.assertFalse((self.work / "out.txt").exists())

        revoke_journal = self.journal("write")
        revoke_call = tool_call("write", {"path": "revoked.txt", "content": "NO"}, "call-revoke")
        self.assistant(revoke_journal, revoke_call)
        revoke_book = self.book(revoke_journal, approvals=True)
        revoke_gate = self.admit(revoke_book, registry.authorize(revoke_call))
        self.assertEqual(
            revoke_book.decide(revoke_gate.outcome["approval_id"], "revoke")["decision"], "revoked"
        )
        self.assertEqual(revoke_book.recover(registry, revoke_journal.incomplete_tail())["status"], "revoked")
        self.assertFalse((self.work / "revoked.txt").exists())

        expired_journal = self.journal("write")
        expired_call = tool_call("write", {"path": "expired.txt", "content": "NO"}, "call-expired")
        self.assistant(expired_journal, expired_call)
        expired = self.book(expired_journal, approvals=True, approval_ttl=0)
        expired_gate = self.admit(expired, registry.authorize(expired_call))
        self.assertEqual(
            expired.decide(expired_gate.outcome["approval_id"], "approve")["decision"], "expired"
        )
        self.assertFalse((self.work / "expired.txt").exists())

        scoped = self.journal("write")
        scoped_call = tool_call("write", {"path": "scoped.txt", "content": "NO"}, "call-scope")
        self.assistant(scoped, scoped_call)
        scoped_book = self.book(scoped, approvals=True)
        scoped_gate = self.admit(scoped_book, registry.authorize(scoped_call))
        scoped_book.note_scope(("other.txt",))
        self.assertEqual(scoped_book.snapshot()["approvals"][0]["decision"], "superseded")
        decision = scoped_book.decide(scoped_gate.outcome["approval_id"], "approve")
        self.assertEqual(decision["decision"], "superseded")
        self.assertFalse((self.work / "scoped.txt").exists())

    def test_stale_owner_cannot_commit(self) -> None:
        journal = self.journal("read")
        registry = self.registry("read")
        call = tool_call("read", {"path": "note.txt"}, "call-owner")
        first = self.book(journal)
        gate = self.admit(first, registry.authorize(call))
        second = self.book(journal)
        with self.assertRaises(StaleOwner):
            first.commit(gate.intent_id, {"text": "late", "executed": True, "is_error": False})
        self.assertEqual(second.snapshot()["intents"][0]["status"], "intent")
        self.assertFalse(any(record["kind"] == "tool" for record in journal.records))

        late_journal = self.journal("read")
        late = self.book(late_journal)
        late_gate = self.admit(late, registry.authorize(tool_call("read", {"path": "note.txt"}, "call-late")))
        ready = threading.Event()
        errors: list[BaseException] = []

        def commit_late() -> None:
            ready.wait(2)
            try:
                late.commit(late_gate.intent_id, {"text": "late", "executed": True, "is_error": False})
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=commit_late)
        worker.start()
        replacement = self.book(late_journal)
        ready.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], StaleOwner)
        self.assertEqual(replacement.snapshot()["intents"][0]["status"], "intent")

    def test_read_cap_write_exclusion_and_same_file(self) -> None:
        self.assertEqual(READ_LIMIT, 2)
        journals = [self.journal("read") for _ in range(3)]
        books = [self.book(item) for item in journals]
        self.assertEqual(books[0].gate.acquire("read")["state"], "running")
        self.assertEqual(books[1].gate.acquire("read")["state"], "running")
        queued = books[2].gate.acquire("read")
        self.assertEqual(
            queued,
            {"worker_id": queued["worker_id"], "state": "queued", "reason": "read_limit", "mode": "read"},
        )
        self.assertEqual(books[2].gate.dismiss(queued["worker_id"])["state"], "cancelled")
        books[0].close()
        follower = self.book(self.journal("read"))
        self.assertEqual(follower.gate.acquire("read")["state"], "running")
        for item in (*books, follower):
            item.close()

        writer = self.book(self.journal("write"))
        reader = self.book(self.journal("read"))
        held = writer.gate.acquire("write")
        self.assertEqual(held["state"], "running")
        self.assertEqual(reader.gate.acquire("read")["reason"], "write_busy")
        with sqlite3.connect(gate_db_path(self.work)) as gate:
            gate.execute("UPDATE workers SET pid=? WHERE worker_id=?", (2_000_000_000, held["worker_id"]))
        self.assertEqual(self.book(self.journal("write")).gate.acquire("write")["state"], "running")

        workspace = Workspace(self.work)
        payloads = ["A" * 4000, "B" * 4000]
        errors: list[BaseException] = []

        def write(payload: str) -> None:
            try:
                write_file(workspace, "shared.txt", payload, overwrite=True)
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=write, args=(payload,)) for payload in payloads]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)
        self.assertFalse(errors)
        self.assertIn((self.work / "shared.txt").read_text(), set(payloads))

    def test_budget_is_reserved_settled_and_not_released_twice(self) -> None:
        journal = self.journal("read")
        book = self.book(journal, max_calls=1, max_output=100)
        turn = book.budget.reserve("turn", 32)
        self.assertIsNotNone(turn)
        self.assertIsNone(book.budget.reserve("compaction", 32))
        book.budget.settle(str(turn), {"completion_tokens": 4}, 1)
        budget = book.snapshot()["budget"]
        self.assertEqual(budget["spent_calls"], 1)
        self.assertEqual(budget["spent_output"], 4)
        self.assertEqual(budget["unknown_output"], 0)
        self.assertIsNone(book.budget.reserve("turn", 8))

        unknown_journal = self.journal("read")
        unknown = self.book(unknown_journal, max_output=100)
        reservation = unknown.budget.reserve("turn", 32)
        unknown.budget.settle(str(reservation), None, 1)
        unknown_budget = unknown.snapshot()["budget"]
        self.assertEqual(unknown_budget["spent_output"], 0)
        self.assertEqual(unknown_budget["unknown_output"], 32)
        self.assertTrue(unknown_budget["output_unknown"])
        self.assertEqual(unknown_budget["unknown_calls"], 1)
        self.assertEqual(unknown_budget["spent_calls"], 0)

        held = self.journal("read")
        owner = self.book(held, max_calls=1)
        owner.budget.reserve("turn", 16)
        owner.close()
        resumed = self.book(held, max_calls=1)
        reaped = resumed.snapshot()["budget"]
        self.assertEqual(reaped["unknown_calls"], 1)
        self.assertEqual(reaped["reserved_calls"], 0)
        self.assertIsNone(resumed.budget.reserve("turn", 8))
        again = self.book(held, max_calls=1)
        self.assertEqual(again.snapshot()["budget"]["unknown_calls"], 1)
        self.assertIsNone(again.budget.reserve("turn", 8))

        timed_journal = self.journal("read")
        timed = self.book(timed_journal, max_seconds=30)
        with sqlite3.connect(task_db_path(timed_journal)) as db:
            db.execute("UPDATE budget SET started_at=?", (time.time() - 60,))
        self.assertEqual(timed.budget.stop_reason(), "time_budget")
        self.assertIsNone(timed.budget.reserve("turn", 8))

    def test_public_session_recovers_and_keeps_approval(self) -> None:
        (self.work / "note.txt").write_text("TOKEN-18\n")
        journal = Journal.create(self.base / "session.jsonl", self.work, tools=("read", "write"))
        call = tool_call("read", {"path": "note.txt"}, "call-session")
        self.assistant(journal, call)
        registry = self.registry("read", "write")
        book = self.book(journal)
        self.admit(book, registry.authorize(call))
        book.close()
        session = AgentSession(self.work, tools=("read", "write"), journal=Journal(journal.path))
        with patch("lite_pi.loop.complete", return_value=model("TOKEN-18")):
            recovered = session.resume(max_turns=1)
        self.assertEqual(recovered["status"], "stop")
        self.assertIn("TOKEN-18", recovered["text"])
        self.assertEqual(sum(record["kind"] == "tool" for record in Journal(journal.path).records), 1)

        approved = Journal.create(self.base / "approval.jsonl", self.work, tools=("read", "write"))
        approval = AgentSession(self.work, tools=("read", "write"), journal=approved)
        with transport(
            self.base / "approval-http",
            [
                response(
                    "", calls=[tool_call("write", {"path": "approved.txt", "content": "MARKER"}, "call-w")]
                )
            ],
        ):
            waiting = approval.submit("write the file", approvals=True, max_task_calls=1, max_turns=2)
        self.assertEqual(waiting["status"], "awaiting_approval")
        self.assertFalse((self.work / "approved.txt").exists())
        self.assertIn("MARKER", waiting["text"])
        approval_id = waiting["tasks"]["approvals"][0]["approval_id"]
        self.assertIsNotNone(Journal(approved.path).incomplete_tail())
        receipt = approval.decide(approval_id, "approve")
        self.assertEqual(receipt["decision"], "approved")
        resumed = approval.resume()
        self.assertEqual((self.work / "approved.txt").read_text(), "MARKER")
        self.assertEqual(resumed["status"], "budget")
        stamp = (self.work / "approved.txt").stat().st_mtime_ns
        again = approval.resume()
        self.assertEqual((self.work / "approved.txt").stat().st_mtime_ns, stamp)
        self.assertEqual(again["status"], "budget")

    def test_cli_resume_prints_a_halt_before_another_model_call(self) -> None:
        journal = Journal.create(self.base / "cli-approval.jsonl", self.work, tools=("read", "write"))
        session = AgentSession(self.work, tools=("read", "write"), journal=journal)
        with patch(
            "lite_pi.loop.complete",
            return_value=model(
                "", calls=[tool_call("write", {"path": "held.txt", "content": "HELD"}, "call-cli")]
            ),
        ):
            waiting = session.submit("write the file", approvals=True, max_task_calls=1, max_turns=2)
        self.assertEqual(waiting["status"], "awaiting_approval")
        proc = subprocess.run(
            [sys.executable, "-m", "lite_pi", "resume", "--session", str(journal.path), "--max-turns", "1"],
            cwd=Path(__file__).resolve().parents[1],
            env={
                "PATH": "/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
                "HOME": os.environ.get("HOME", ""),
            },
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["status"], "awaiting_approval")
        self.assertEqual(payload["reason"], "awaiting_approval")
        self.assertIsNone(payload["raw_stop_reason"])
        self.assertFalse((self.work / "held.txt").exists())

    def test_summary_and_turn_share_the_task_budget(self) -> None:
        journal = Journal.create(self.base / "summary.jsonl", self.work, tools=("read",))
        session = AgentSession(self.work, tools=("read",), journal=journal)
        with transport(self.base / "first-http", [response("noted")]) as wire:
            opened = session.submit("word " * 2000, max_turns=1, max_tokens=64, max_task_calls=2)
        self.assertEqual(opened["status"], "stop")
        self.assertEqual(wire.call_count, 1)
        history = Journal(journal.path).messages() + [{"role": "user", "content": "Continue"}]
        reserve = 64
        prefix = request_prefix(load_context(self.work))
        recent = estimate_request(prefix + history[-1:], [], reserve)
        full = estimate_request(prefix + history, [], reserve)
        # The current host prompt and read schema must fit beside the kept turn.
        context_tokens = recent + reserve + min(512, max(1, int(reserve * 0.8))) + 1000
        self.assertGreater(full + reserve, context_tokens)
        with transport(self.base / "summary-http", [response("pending task")]) as wire:
            result = session.submit(
                "Continue", max_turns=2, max_tokens=reserve, context_tokens=context_tokens
            )
        self.assertEqual(result["status"], "budget")
        self.assertEqual(wire.call_count, 1)
        self.assertEqual(result["tasks"]["budget"]["spent_calls"], 2)
