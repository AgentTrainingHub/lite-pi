"""Persistent ledger identity and budget continuity; no network calls."""

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from uuid import UUID

from lite_pi.spend.ledger import Ledger


class SpendIdentityContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ledger = Ledger(self.root / "ledger.json", self.root / "calls.jsonl")

    def bootstrap(self, ledger):
        return ledger.bootstrap(Decimal("100"), currency="CNY", is_available=True)

    def test_identity_is_created_once_per_ledger(self):
        first = self.bootstrap(self.ledger)
        identity = first["status"]["round_id"]
        self.assertTrue(first["created"])
        self.assertEqual(UUID(identity).version, 4)
        reopened = Ledger(self.ledger.path, self.ledger.calls_path)
        again = self.bootstrap(reopened)
        self.assertFalse(again["created"])
        self.assertEqual(again["status"]["round_id"], identity)
        other = self.bootstrap(Ledger(self.root / "other.json", self.root / "other-calls.jsonl"))
        self.assertNotEqual(other["status"]["round_id"], identity)

    def test_existing_identity_spend_and_reservation_survive_reopen(self):
        self.bootstrap(self.ledger)
        reservation = self.ledger.try_reserve(
            run_id="continuity", purpose="mechanism", input_tokens=100, output_tokens=100
        )
        data = json.loads(self.ledger.path.read_text())
        data.update(
            round_id="legacy-ledger",
            spent_cny="2.50000000",
            allowance_cny="5.00000000",
            allowance_cap_cny="5.00000000",
            request_count=7,
            unknown_usage_calls=1,
        )
        self.ledger.path.write_text(json.dumps(data))
        reopened = Ledger(self.ledger.path, self.ledger.calls_path)
        self.assertFalse(self.bootstrap(reopened)["created"])
        after = json.loads(self.ledger.path.read_text())
        for key in (
            "round_id",
            "spent_cny",
            "allowance_cny",
            "allowance_cap_cny",
            "request_count",
            "unknown_usage_calls",
            "reserved_cny",
            "in_flight",
        ):
            self.assertEqual(after[key], data[key], key)
        reopened.settle(
            reservation["reservation_id"],
            usage=None,
            http_status=None,
            actual_model=None,
            response_id=None,
            error="connection interrupted",
        )
        status = reopened.require_status()
        self.assertEqual(status["round_id"], data["round_id"])
        self.assertEqual(status["request_count"], 8)
        self.assertEqual(status["unknown_usage_calls"], 2)
        self.assertEqual(status["in_flight"], 0)
        self.assertEqual(Decimal(status["spent_cny"]), Decimal("2.5") + Decimal(reservation["reserved_cny"]))

    def test_invalid_identity_never_reinitializes_a_ledger(self):
        for invalid in (None, 0, [], {}, "", "   "):
            with self.subTest(identity=invalid):
                raw = json.dumps({"round_id": invalid})
                self.ledger.path.write_text(raw)
                with self.assertRaisesRegex(ValueError, "refusing to reset spend"):
                    self.bootstrap(self.ledger)
                self.assertEqual(self.ledger.path.read_text(), raw)

    def test_exhausted_existing_budget_remains_exhausted(self):
        self.bootstrap(self.ledger)
        data = json.loads(self.ledger.path.read_text())
        data.update(spent_cny=data["allowance_cny"], round_id="existing-budget")
        self.ledger.path.write_text(json.dumps(data))
        reopened = Ledger(self.ledger.path, self.ledger.calls_path)
        self.assertFalse(self.bootstrap(reopened)["created"])
        denied = reopened.try_reserve(
            run_id="over-budget", purpose="mechanism", input_tokens=1, output_tokens=1
        )
        self.assertFalse(denied["ok"])
        self.assertEqual(reopened.require_status()["spent_cny"], data["spent_cny"])
