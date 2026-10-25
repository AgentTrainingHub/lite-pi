"""Deterministic evaluation mechanisms; never counted as live Judge agreement."""

import unittest
from dataclasses import replace
from decimal import Decimal

from compare.paired import Attempt, Observation, PairingError, summarize


class EvaluationContracts(unittest.TestCase):
    def observations(self):
        records = []
        for task in ("a", "b", "c"):
            for repeat in range(3):
                for arm in ("solo", "team"):
                    run = f"{task}-{repeat}-{arm}"
                    records.append(
                        Observation(
                            task,
                            repeat,
                            arm,
                            run,
                            (),
                            (Attempt(run, run, Decimal("0.01"), Decimal("0.01")),),
                            1,
                            0,
                            1.0 if arm == "solo" else 2.0,
                            arm == "team",
                            True,
                            run,
                        )
                    )
        return records

    def test_task_level_pairs_keep_failed_runs_and_reproduce_interval(self):
        rows = self.observations()
        report = summarize(rows, ("a", "b", "c"), 3)
        self.assertEqual(report["pairs"], 9)
        self.assertEqual(report["metrics"]["success"]["mean"], 1)
        self.assertEqual(report["metrics"]["success"]["interval95"], [1, 1])
        self.assertEqual(report["metrics"]["elapsed"]["mean"], 1)
        self.assertEqual(report, summarize(rows, ("a", "b", "c"), 3))

    def test_missing_duplicate_and_child_omission_are_errors(self):
        rows = self.observations()
        for invalid in (
            rows[:-1],
            rows + [rows[0]],
            [replace(rows[0], child_run_ids=("child",), budget_calls=2), *rows[1:]],
        ):
            with self.subTest(length=len(invalid)), self.assertRaises(PairingError):
                summarize(invalid, ("a", "b", "c"), 3)

    def test_unknown_usage_suppresses_cost_comparison(self):
        rows = self.observations()
        rows[0] = replace(
            rows[0],
            unknown_calls=1,
            attempts=(Attempt(rows[0].parent_run_id, "unknown", None, Decimal("0.1")),),
        )
        report = summarize(rows, ("a", "b", "c"), 3)
        self.assertIsNone(report["metrics"]["known_cost"])
        self.assertEqual(report["unknown_usage_calls"], 1)

    def test_objective_failure_cannot_be_overridden_by_textual_success(self):
        rows = self.observations()
        rows[1] = replace(rows[1], objective_valid=False)
        report = summarize(rows, ("a", "b", "c"), 3)
        self.assertLess(report["metrics"]["success"]["mean"], 1)

    def test_bootstrap_resamples_tasks_instead_of_individual_repeats(self):
        rows = self.observations()
        rows = [
            replace(r, elapsed=1 + {"a": 1, "b": 2, "c": 4}[r.task]) if r.arm == "team" else r for r in rows
        ]
        report = summarize(rows, ("a", "b", "c"), 3)
        self.assertAlmostEqual(report["metrics"]["elapsed"]["mean"], 7 / 3)
        self.assertEqual(report["metrics"]["elapsed"]["interval95"], [1, 4])

    def test_judge_contract_rejects_invalid_and_preserves_objective_gate(self):
        from compare.calibrated_judge import admissible, parse_judgment

        good = parse_judgment('{"winner":"A","score_a":4,"score_b":0,"reason":"有依据"}')
        self.assertEqual(good.winner, "A")
        self.assertFalse(admissible(False, good))
        self.assertTrue(admissible(True, good))
        for text in (
            '{"winner":"A","score_a":0,"score_b":4,"reason":"错"}',
            '{"winner":"A","score_a":true,"score_b":0,"reason":"错"}',
            '{"winner":"B","winner":"A","score_a":4,"score_b":0,"reason":"重复"}',
        ):
            with self.assertRaises(ValueError):
                parse_judgment(text)

    def test_judge_reference_requires_actual_human_confirmation_and_independent_splits(self):
        from compare.calibrated_judge import validate_references

        rows = [
            {
                "id": f"{split}-{i}",
                "split": split,
                "facts": f"fact-{split}-{i}",
                "answer_a": "a",
                "answer_b": "b",
                "human_label": None,
                "reviewer": None,
                "human_reason": None,
                "confirmed_at": None,
            }
            for split in ("calibration", "evaluation")
            for i in range(6)
        ]
        with self.assertRaisesRegex(ValueError, "human_reference_pending"):
            validate_references(rows)
        for row in rows:
            row.update(
                human_label="A",
                reviewer="controlled test only",
                human_reason="test fact",
                confirmed_at="2026-10-18",
            )
        validate_references(rows)
        rows[-1] = {**rows[0], "id": "other", "split": "evaluation"}
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate_references(rows)

    def test_zero_request_child_is_retained_when_its_receipt_declares_zero(self):
        rows = self.observations()
        rows[1] = replace(rows[1], child_run_ids=("child",), child_call_counts=(("child", 0),), success=False)
        report = summarize(rows, ("a", "b", "c"), 3)
        self.assertEqual(report["pairs"], 9)

    def test_raw_receipts_reject_a_tampered_projection_or_conditions(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from compare.paired_manifest import decode_manifest
        from compare.paired_run import evaluate, run_arm
        from deterministic_transport import response, transport

        from lite_pi.spend.ledger import Ledger

        with TemporaryDirectory() as raw:
            base = Path(raw)
            manifest = decode_manifest(
                json.loads(Path("tests/fixtures/evaluation/paired-manifest.json").read_text())
            )
            ledger = Ledger(base / "wire/ledger.json", base / "wire/calls.jsonl")
            with (
                transport(base / "wire", [response("done")]),
                patch("compare.paired_run.Ledger", return_value=ledger),
                patch("compare.paired_evidence.EVIDENCE_ROOT", base / "wire/calls"),
            ):
                record = run_arm(base / "grams-0-solo", manifest.tasks[0], "solo", 0, manifest)
            from compare.paired_run import encode_observation

            projection = json.loads(json.dumps(encode_observation(record)))
            projection["budget_calls"] = 0
            projection["attempts"] = []
            (base / "observations.json").write_text(json.dumps([projection]))
            with patch("compare.paired_evidence.EVIDENCE_ROOT", base / "wire/calls"):
                report = evaluate(base, manifest)
            self.assertEqual(report["error"], "observation_projection_mismatch")
            initial = base / "grams-0-solo/initial.json"
            changed = json.loads(initial.read_text())
            changed["max_calls"] = 99
            initial.write_text(json.dumps(changed))
            with patch("compare.paired_evidence.EVIDENCE_ROOT", base / "wire/calls"):
                report = evaluate(base, manifest)
            self.assertEqual(report["error"], "shared_conditions_changed")

    def test_diagnosis_changes_only_assembly_and_rejects_unavailable_identity(self):
        from compare.rag_diagnosis import assemble

        hits = [
            {"chunk_id": "right", "source": "limits.md", "text": "cap=17"},
            {"chunk_id": "other", "source": "note.md", "text": "unknown"},
        ]
        self.assertEqual(assemble(hits, ("other",)), [hits[1]])
        self.assertEqual(assemble(hits, ("right", "other")), hits)
        with self.assertRaises(ValueError):
            assemble(hits, ("fabricated",))

    def test_manifest_rejects_identity_and_budget_errors_before_execution(self):
        import json
        from pathlib import Path

        from compare.paired_manifest import decode_manifest

        raw = json.loads(Path("tests/fixtures/evaluation/paired-manifest.json").read_text())
        decode_manifest(raw)
        for key, value in (("arms", ["solo", "solo"]), ("max_calls", True), ("model", "other"), ("pairs", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                decode_manifest({**raw, key: value})
        with self.assertRaises(ValueError):
            decode_manifest({**raw, "tasks": [raw["tasks"][0]] * 3})

    def test_verifier_failure_preserves_paid_receipts_and_partial_state(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from compare.paired_manifest import decode_manifest
        from compare.paired_run import run_arm
        from deterministic_transport import response, transport

        from lite_pi.spend.ledger import Ledger

        with TemporaryDirectory() as raw:
            base = Path(raw)
            manifest = decode_manifest(
                json.loads(Path("tests/fixtures/evaluation/paired-manifest.json").read_text())
            )
            ledger = Ledger(base / "wire/ledger.json", base / "wire/calls.jsonl")
            with (
                transport(base / "wire", [response("done")]),
                patch("compare.paired_run.Ledger", return_value=ledger),
                patch("compare.paired_run.check", side_effect=RuntimeError("verifier interrupted")),
            ):
                with self.assertRaisesRegex(RuntimeError, "verifier interrupted"):
                    run_arm(base / "episode", manifest.tasks[0], "solo", 0, manifest)
            self.assertEqual(len(json.loads((base / "episode/attempts.json").read_text())), 1)
            self.assertTrue((base / "episode/budget.json").is_file())
            self.assertEqual(
                json.loads((base / "episode/failure.json").read_text())["reason"], "verifier interrupted"
            )

    def test_judge_reports_false_accept_abstention_invalid_and_order_conflict(self):
        from compare.calibrated_judge import Reference
        from compare.calibrated_judge import summarize as judge_summary

        refs = [
            {"id": "c", "split": "calibration", "human_label": "A"},
            {"id": "e", "split": "evaluation", "human_label": "B"},
        ]

        def observation(identity, swapped, winner):
            return {
                "id": identity,
                "run_id": f"{identity}-{swapped}",
                "swapped": swapped,
                "judgment": {"winner": winner} if winner else None,
                "evidence": "controlled",
            }

        report = judge_summary(
            tuple(
                Reference(
                    r["id"],
                    r["split"],
                    "facts",
                    "a",
                    "b",
                    r["human_label"],
                    "controlled",
                    "test",
                    "2026-10-18",
                )
                for r in refs
            ),
            [
                observation("c", False, "B"),
                observation("c", True, "B"),
                observation("e", False, None),
                observation("e", True, "abstain"),
            ],
        )
        self.assertEqual(report["splits"]["calibration"]["false_accept"], 1)
        self.assertEqual(report["splits"]["calibration"]["agreement"], 1)
        self.assertEqual(report["splits"]["calibration"]["order_conflict"], 2)
        self.assertEqual(report["splits"]["evaluation"]["invalid"], 1)
        self.assertEqual(report["splits"]["evaluation"]["abstain"], 1)

    def test_judge_rejects_path_escape_and_swapped_reference_overlap(self):
        import json
        from pathlib import Path

        from compare.calibrated_judge import validate_references

        rows = json.loads(Path("tests/fixtures/evaluation/judge-references.json").read_text())
        for row in rows:
            row.update(
                human_label=row["proposed_label"],
                reviewer="controlled",
                human_reason="test",
                confirmed_at="2026-10-18",
            )
        rows[0]["id"] = "../escape"
        with self.assertRaisesRegex(ValueError, "unsafe_reference_identity"):
            validate_references(rows)
        rows[0]["id"] = "safe"
        rows[-1].update(facts=rows[0]["facts"], answer_a=rows[0]["answer_b"], answer_b=rows[0]["answer_a"])
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate_references(rows)

    def test_tool_free_episode_failure_preserves_input_and_attempts(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from compare.model_episode import run_episode
        from deterministic_transport import response, transport
        from support import save

        from lite_pi.spend.ledger import Ledger

        with TemporaryDirectory() as raw:
            base = Path(raw)
            episode = base / "episode"
            episode.mkdir()
            ledger = Ledger(base / "wire/ledger.json", base / "wire/calls.jsonl")

            def fail_result(path, value):
                if path.name == "result.json":
                    raise OSError("completion record interrupted")
                save(path, value)

            with (
                transport(base / "wire", [response("done")]),
                patch("compare.model_episode.Ledger", return_value=ledger),
                patch("compare.model_episode.save", side_effect=fail_result),
            ):
                with self.assertRaisesRegex(OSError, "completion record interrupted"):
                    run_episode(episode, "no tools")
            self.assertTrue((episode / "request-input.json").is_file())
            self.assertEqual(len(json.loads((episode / "attempts.json").read_text())), 1)
            self.assertEqual(json.loads((episode / "budget.json").read_text())["spent_calls"], 1)

    def test_diagnostic_disk_failure_does_not_replace_the_original_error(self):
        from pathlib import Path
        from unittest.mock import patch

        from compare.evidence_failures import record_failure

        original = RuntimeError("original effect failure")
        with patch("compare.evidence_failures.save", side_effect=OSError("disk unavailable")):
            record_failure(Path("failure.json"), original)
        self.assertEqual(str(original), "original effect failure")
        self.assertIn("disk unavailable", " ".join(original.__notes__))

    def test_diagnosis_without_improvement_is_a_complete_comparison(self):
        from compare.rag_diagnosis import summarize_pairs

        rows = []
        for i in range(3):
            arms = [
                {
                    "arm": arm,
                    "run_id": f"{i}-{arm}",
                    "valid": False,
                    "safe": True,
                    "status": "stop",
                    "budget": {"reserved_calls": 0, "spent_calls": 1, "unknown_calls": 0},
                }
                for arm in ("baseline", "intervention")
            ]
            rows.append({"repetition": i, "arms": arms, "improved": False})
        report = summarize_pairs(rows)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["improved_pairs"], 0)
