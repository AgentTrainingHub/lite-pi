"""Reference provenance is independent of model agreement and human calibration."""

import json
import unittest
from pathlib import Path

from compare.calibrated_judge import prompt_for, summarize, validate_references


class JudgeReferenceContracts(unittest.TestCase):
    def setUp(self):
        self.rows = json.loads(Path("tests/fixtures/evaluation/judge-references.json").read_text())

    def test_mixed_references_require_explicit_policy_and_complete_ai_review(self):
        with self.assertRaisesRegex(ValueError, "human_reference_pending"):
            validate_references(self.rows)
        refs = validate_references(self.rows, policy="mixed")
        self.assertEqual([r.source for r in refs].count("human"), 4)
        self.assertEqual([r.source for r in refs].count("ai"), 8)
        for field in ("ai_label", "ai_reason", "ai_reviewer", "ai_reviewed_at", "ai_review_authorization"):
            with self.subTest(field=field):
                rows = json.loads(json.dumps(self.rows))
                rows[4][field] = ""
                with self.assertRaisesRegex(ValueError, "ai_reference_pending"):
                    validate_references(rows, policy="mixed")
        with self.assertRaisesRegex(ValueError, "unknown_reference_policy"):
            validate_references(self.rows, policy="anything")

    def test_partial_human_review_cannot_fall_back_to_ai(self):
        self.rows[4]["human_label"] = "A"
        with self.assertRaisesRegex(ValueError, "human_reference_pending"):
            validate_references(self.rows, policy="mixed")

    def test_mixed_report_separates_sources_and_never_leaks_labels_into_prompt(self):
        refs = validate_references(self.rows, policy="mixed")
        observations = []
        for ref in refs:
            for swapped in (False, True):
                winner = {"A": "B", "B": "A"}.get(ref.label, ref.label) if swapped else ref.label
                observations.append(
                    {
                        "id": ref.id,
                        "run_id": f"{ref.id}-{swapped}",
                        "swapped": swapped,
                        "judgment": {"winner": winner},
                        "evidence": "controlled",
                    }
                )
            for private in (ref.reviewer, ref.reason, ref.reviewed_at, ref.authorization):
                if private:
                    self.assertNotIn(private, prompt_for(ref, False))
        report = summarize(refs, observations)
        self.assertFalse(report["human_calibration_complete"])
        self.assertEqual(report["reference_counts"], {"human": 4, "ai": 8})
        self.assertEqual(report["by_source"]["human"]["calibration"]["agreement"], 8)
        self.assertEqual(report["by_source"]["human"]["evaluation"]["total"], 0)
        self.assertEqual(report["by_source"]["ai"]["evaluation"]["agreement"], 12)
        self.assertTrue(all("human" not in row for row in report["results"]))
