"""Knowledge boundary regressions: real files/SQLite, with no model API calls."""

import json
import os
import sqlite3
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from lite_pi.knowledge.db import KnowledgeDB, KnowledgeError
from lite_pi.memory.ledger import MemoryLedger
from lite_pi.retrieval.budget import RetrievalMeter
from lite_pi.retrieval.index import CorpusIndex
from lite_pi.tools.workspace import Workspace


class KnowledgeBoundaryContracts(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / "project"
        self.work.mkdir()
        self.now = datetime(2026, 10, 14, tzinfo=UTC)

    def test_answer_judge_rejects_unverified_claims_and_wrong_facts(self):
        from retrieval_judge import contract_answer

        report = {
            "parameter": "user_id",
            "parameter_type": "str",
            "return_type": "str",
            "citations": ["cite:chk_good"],
        }
        self.assertTrue(contract_answer(json.dumps(report), {"chk_good"}, {"chk_good"}))
        self.assertTrue(
            contract_answer("```json\n" + json.dumps(report) + "\n```", {"chk_good"}, {"chk_good"})
        )
        self.assertTrue(
            contract_answer(json.dumps({**report, "citations": ["chk_good"]}), {"chk_good"}, {"chk_good"})
        )
        self.assertFalse(
            contract_answer(
                json.dumps(report).replace(
                    '"parameter": "user_id"', '"parameter":"id","parameter":"user_id"'
                ),
                {"chk_good"},
                {"chk_good"},
            )
        )
        self.assertFalse(
            contract_answer(json.dumps({**report, "refund_rate": "17%"}), {"chk_good"}, {"chk_good"})
        )
        self.assertFalse(
            contract_answer(json.dumps({**report, "tests_passed": True}), {"chk_good"}, {"chk_good"})
        )
        self.assertFalse(
            contract_answer(json.dumps({**report, "parameter": "id"}), {"chk_good"}, {"chk_good"})
        )
        self.assertFalse(contract_answer(json.dumps(report) + " 运行测试成功", {"chk_good"}, {"chk_good"}))
        self.assertFalse(contract_answer(json.dumps(report), set(), {"chk_good"}))

    def test_gap_judge_rejects_invented_claims(self):
        from retrieval_judge import gap_answer

        report = {"status": "insufficient_evidence", "citations": []}
        self.assertTrue(gap_answer(json.dumps(report)))
        self.assertFalse(gap_answer(json.dumps({**report, "refund_rate": 17})))
        self.assertFalse(gap_answer(json.dumps({**report, "citations": ["cite:chk_fake"]})))
        self.assertFalse(gap_answer("无法确认，但费率可能是17%"))

    def database(self):
        db = KnowledgeDB.open(self.work)
        self.addCleanup(db.close)
        return db

    def candidate(self, ledger, text):
        ref = ledger.record_user_source("fixture:" + text, text)
        return ledger.propose(
            object_key="convention", statement=text, source_kind="user_message", source_ref=ref
        )["memory"]["memory_id"]

    def test_database_rejects_external_directory_and_files(self):
        outside = self.root / "outside"
        outside.mkdir()
        control = self.work / ".lite-pi"
        control.symlink_to(outside, target_is_directory=True)
        with self.assertRaises((ValueError, OSError)):
            KnowledgeDB.open(self.work)
        self.assertEqual(list(outside.iterdir()), [])
        control.unlink()
        control.mkdir()
        target = outside / "untouched"
        target.write_bytes(b"private")
        for suffix in ("", "-wal", "-shm", "-journal"):
            for link_kind in ("symbolic", "hard"):
                with self.subTest(suffix=suffix, link_kind=link_kind):
                    linked = control / ("knowledge.sqlite" + suffix)
                    if link_kind == "symbolic":
                        linked.symlink_to(target)
                    else:
                        os.link(target, linked)
                    with self.assertRaises(KnowledgeError):
                        KnowledgeDB.open(self.work)
                    self.assertEqual(target.read_bytes(), b"private")
                    linked.unlink()

    def test_concurrent_database_initialization(self):
        def initialize(_):
            db = KnowledgeDB.open(self.work)
            try:
                return db.connection.execute("PRAGMA user_version").fetchone()[0]
            finally:
                db.close()

        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(list(pool.map(initialize, range(4))), [3] * 4)

    def test_three_way_conflict_has_one_active_and_db_enforces_it(self):
        db = self.database()
        ledger = MemoryLedger(db)
        a, b, c = [self.candidate(ledger, text) for text in ("A", "B", "C")]
        self.assertTrue(ledger.confirm(a, now=self.now)["ok"])
        self.assertEqual(ledger.confirm(b, now=self.now)["reason"], "conflict")
        self.assertEqual(ledger.confirm(c, now=self.now)["reason"], "conflict")
        self.assertEqual(ledger.read(run_id="before", now=self.now)["memories"], [])
        self.assertTrue(ledger.resolve(a, now=self.now)["ok"])
        self.assertEqual([m["memory_id"] for m in ledger.read(run_id="after", now=self.now)["memories"]], [a])
        self.assertFalse(ledger.resolve(c, now=self.now)["ok"])
        with self.assertRaises(sqlite3.IntegrityError):
            with db.transaction() as tx:
                tx.execute("UPDATE memories SET status='active' WHERE memory_id=?", (c,))

    def test_expired_active_does_not_conflict_and_conflict_retains_ttl(self):
        ledger = MemoryLedger(self.database())
        a, b, c = [self.candidate(ledger, text) for text in ("A", "B", "C")]
        ledger.confirm(a, ttl_seconds=1, now=self.now)
        later = self.now + timedelta(seconds=2)
        self.assertTrue(ledger.confirm(b, now=later)["ok"])
        self.assertEqual(ledger.confirm(c, ttl_seconds=1, now=later)["reason"], "conflict")
        resolved = ledger.resolve(c, now=later + timedelta(seconds=2))
        self.assertFalse(resolved["ok"])
        self.assertEqual(resolved["memory"]["status"], "expired")
        self.assertTrue(ledger.resolve(b, now=later + timedelta(seconds=2))["ok"])

    def test_user_source_must_be_host_registered(self):
        db = self.database()
        ledger = MemoryLedger(db)
        fake = ledger.propose(
            object_key="rule", statement="fake", source_kind="user_message", source_ref="invented-message"
        )["memory"]["memory_id"]
        self.assertEqual(ledger.confirm(fake)["reason"], "unknown_source")
        ref = ledger.record_user_source("journal:session:user-record", "Use snake_case")
        real = ledger.propose(
            object_key="rule", statement="Use snake_case", source_kind="user_message", source_ref=ref
        )["memory"]["memory_id"]
        self.assertTrue(ledger.confirm(real)["ok"])
        source = db.connection.execute("SELECT * FROM memory_sources WHERE source_ref=?", (ref,)).fetchone()
        self.assertEqual(source["content"], "Use snake_case")
        self.assertEqual(source["identity"], "journal:session:user-record")

    def test_legacy_unverified_active_source_cannot_be_adopted(self):
        db = self.database()
        ledger = MemoryLedger(db)
        candidate = ledger.propose(
            object_key="legacy", statement="unverified", source_kind="user_message", source_ref="user"
        )["memory"]["memory_id"]
        with db.transaction() as tx:
            tx.execute("UPDATE memories SET status='active' WHERE memory_id=?", (candidate,))
        with self.assertRaisesRegex(KnowledgeError, "unverified_user_source"):
            ledger.adopt_for_run("test")
        self.assertTrue(ledger.revoke(candidate)["ok"])
        self.assertEqual(ledger.adopt_for_run("after-revoke"), ())

    def prepare_index(self):
        (self.work / ".lite-pi").mkdir()
        (self.work / "a.md").write_text("alpha first")
        (self.work / "b.md").write_text("alpha second")
        specs = [{"path": "a.md", "version": "1"}, {"path": "b.md", "version": "1"}]
        manifest = self.work / ".lite-pi/corpus.json"
        manifest.write_text(json.dumps({"sources": specs}))
        index = CorpusIndex(self.database(), Workspace(self.work))
        self.assertEqual(index.rebuild()["reason"], "rebuilt")
        return index, manifest, specs

    def test_manifest_membership_and_versions_invalidate_index(self):
        index, manifest, specs = self.prepare_index()
        for changed in (
            specs[:1],
            [dict(specs[0], version="2"), specs[1]],
            specs + [{"path": "c.md", "version": "1"}],
        ):
            manifest.write_text(json.dumps({"sources": changed}))
            result = index.search("alpha", run_id="probe", limit=4, meter=RetrievalMeter(4))
            self.assertEqual(result["reason"], "index_stale")
            self.assertEqual(result["hits"], [])
        manifest.unlink()
        self.assertEqual(
            index.search("alpha", run_id="missing", limit=4, meter=RetrievalMeter(4))["hits"], []
        )

    def test_query_limit_is_part_of_repeat_identity(self):
        index, _, _ = self.prepare_index()
        meter = RetrievalMeter(4)
        one = index.search("alpha", run_id="probe", limit=1, meter=meter)
        two = index.search("alpha", run_id="probe", limit=4, meter=meter)
        self.assertEqual(len(one["hits"]), 1)
        self.assertEqual(len(two["hits"]), 2)
        self.assertNotEqual(two["reason"], "repeat")
        self.assertEqual(index.search("alpha", run_id="probe", limit=4, meter=meter)["reason"], "repeat")


if __name__ == "__main__":
    unittest.main()
