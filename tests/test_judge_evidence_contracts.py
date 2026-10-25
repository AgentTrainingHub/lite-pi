"""Judge report reconstruction uses controlled transport, never human or live evidence."""

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from deterministic_transport import response, transport

from lite_pi.spend.ledger import Ledger


class JudgeEvidenceContracts(unittest.TestCase):
    def test_recompute_uses_raw_episodes_and_rejects_changed_projection(self):
        self.check_recompute("human-only")

    def test_mixed_recompute_preserves_reference_sources(self):
        self.check_recompute("mixed")

    def check_recompute(self, policy):
        from compare.calibrated_judge import recompute, run

        with TemporaryDirectory() as raw:
            base = Path(raw)
            references = json.loads(Path("tests/fixtures/evaluation/judge-references.json").read_text())
            if policy == "human-only":
                for row in references:
                    row.update(
                        human_label="A",
                        reviewer="controlled test only",
                        human_reason="synthetic",
                        confirmed_at="2026-10-19",
                    )
            reference_file = base / "references.json"
            reference_file.write_text(json.dumps(references))
            paired_file = base / "paired.json"
            paired_file.write_text('{"status":"complete"}')
            ledger = Ledger(base / "wire/ledger.json", base / "wire/calls.jsonl")
            answers = [response('{"winner":"A","score_a":4,"score_b":0,"reason":"test"}') for _ in range(24)]
            answers[-1]["body"] = answers[-1]["body"].replace(b'"stop"', b'"content_filter"')
            with (
                transport(base / "wire", answers),
                patch("compare.calibrated_judge.Ledger", return_value=ledger),
                patch("compare.model_episode.Ledger", return_value=ledger),
                patch("compare.judge_receipts.EVIDENCE_ROOT", base / "wire/calls"),
            ):
                report = run(base / "judge", reference_file, paired_file, policy=policy)
                before = ledger.require_status()
                rebuilt = recompute(base / "judge")
                self.assertEqual(rebuilt, report)
                self.assertEqual(recompute(Path(os.path.relpath(base / "judge"))), report)
                self.assertEqual(before, ledger.require_status())
                self.assertEqual(rebuilt["status"], "complete")
                self.assertEqual(rebuilt["reference_policy"], policy)
                self.assertEqual(rebuilt["splits"]["calibration"]["agreement"], 5 if policy == "mixed" else 6)
                if policy == "mixed":
                    self.assertEqual(rebuilt["reference_counts"], {"human": 4, "ai": 8})
                    self.assertFalse(rebuilt["human_calibration_complete"])
                self.assertEqual(rebuilt["splits"]["evaluation"]["invalid"], 1)
                projection = base / "judge/observations.json"
                saved = projection.read_text()
                rows = json.loads(saved)
                rows[0]["judgment"]["winner"] = "B"
                projection.write_text(json.dumps(rows))
                with self.assertRaisesRegex(ValueError, "projection_mismatch"):
                    recompute(base / "judge")
                projection.write_text(saved)
                episode = base / "judge/cal-01-0"
                for filename, field, value, error in (
                    ("result.json", "status", "budget", "response_mismatch"),
                    ("budget.json", "spent_output", 999999, "budget_projection_mismatch"),
                ):
                    with self.subTest(field=field):
                        path = episode / filename
                        original = path.read_text()
                        changed = json.loads(original)
                        changed[field] = value
                        path.write_text(json.dumps(changed))
                        with self.assertRaisesRegex(ValueError, error):
                            recompute(base / "judge")
                        path.write_text(original)
                prompt = base / "judge/cal-01-0/request-input.json"
                data = json.loads(prompt.read_text())
                data["prompt"] = "different task"
                prompt.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "input_changed"):
                    recompute(base / "judge")
