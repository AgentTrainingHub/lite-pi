"""Independent integrated checks reject wrong facts, fabricated citations and stale candidates."""

import unittest
from copy import deepcopy

from integrated_checks import approved_candidate, artifact_matches, citation_ids, plan_verified

from lite_pi.recovery import candidate_version


def verified_workflow():
    definitions = (
        ("sources", ["docs/limits.md"], [], ["source-read", "search"]),
        ("build", ["docs/limits.md"], ["sources"], ["source-read", "write"]),
        ("check", ["result.txt"], ["build"], ["result-read"]),
    )
    intents = []
    for identity, name, arguments in (
        ("source-read", "read", {"path": "docs/limits.md"}),
        ("search", "search_docs", {"query": "QUERY_PAGE_CAP", "mode": "hybrid"}),
        ("write", "write", {"path": "result.txt"}),
        ("result-read", "read", {"path": "result.txt"}),
    ):
        intents.append(
            {
                "intent_id": identity,
                "name": name,
                "arguments": arguments,
                "status": "settled",
                "call_id": identity,
                "result": {
                    "intent_id": identity,
                    "name": name,
                    "tool_call_id": identity,
                    "executed": True,
                    "is_error": False,
                    "requirement_version": 1,
                    "file_versions": {arguments["path"]: "a" * 64} if "path" in arguments else {},
                },
            }
        )
    by_id = {i["intent_id"]: i for i in intents}
    return {
        "intents": intents,
        "workflow": {
            "requirement_version": 1,
            "plan": {
                "steps": [
                    {
                        "id": identity,
                        "inputs": inputs,
                        "depends_on": dependencies,
                        "input_versions": dict.fromkeys(inputs, "a" * 64),
                        "valid": True,
                        "evidence_epoch": 1,
                        "checks": [
                            {"intent_id": receipt, "name": by_id[receipt]["name"], "passed": True}
                            for receipt in evidence
                        ],
                    }
                    for identity, inputs, dependencies, evidence in definitions
                ]
            },
        },
    }


class IntegratedContracts(unittest.TestCase):
    def test_artifacts_require_correct_fact_and_actual_citation(self):
        refs = {"cite:actual"}
        self.assertTrue(artifact_matches('{"limit":17,"citation":"cite:actual"}', "JSON", refs))
        self.assertTrue(artifact_matches("limit=17\ncite:actual\n", "TEXT", refs))
        for text in (
            '{"limit":18,"citation":"cite:actual"}',
            '{"limit":17,"citation":"cite:invented"}',
            '{"limit":17,"citation":["cite:actual"]}',
            '{"limit":18,"limit":17,"citation":"cite:actual"}',
        ):
            self.assertFalse(artifact_matches(text, "JSON", refs))
        self.assertFalse(artifact_matches("limit=17 cite:invented", "TEXT", refs))
        self.assertFalse(artifact_matches("limit=17 cite:actual", "TEXT", refs))
        self.assertFalse(artifact_matches("limit=17\tcite:actual", "TEXT", refs))

    def test_citations_must_come_from_expected_source_and_fact(self):
        runs = [
            {
                "transcript": [
                    {
                        "results": [
                            {
                                "name": "search_docs",
                                "hits": [
                                    {
                                        "chunk_id": "real",
                                        "source": "docs/limits.md",
                                        "text": "QUERY_PAGE_CAP = 17",
                                    },
                                    {
                                        "chunk_id": "other",
                                        "source": "docs/history.md",
                                        "text": "QUERY_PAGE_CAP = 17",
                                    },
                                    {
                                        "chunk_id": "old",
                                        "source": "docs/limits.md",
                                        "text": "QUERY_PAGE_CAP = 18",
                                    },
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
        self.assertEqual(citation_ids(runs), {"cite:real"})

    def test_approval_is_bound_to_current_path_fact_and_requirement(self):
        arguments = {"path": "result.txt", "content": "limit=17\ncite:real"}
        version = candidate_version("write", arguments, None, "current-constraints")
        state = {
            "approvals": [
                {
                    "decision": "pending",
                    "intent_id": "write1",
                    "approval_id": "a1",
                    "candidate_version": version,
                }
            ],
            "intents": [
                {
                    "intent_id": "write1",
                    "name": "write",
                    "status": "awaiting_approval",
                    "candidate_version": version,
                    "arguments": arguments,
                }
            ],
            "workflow": {"requirement_version": 2},
        }
        task = {"constraint": "current-constraints", "scope": None, "workflow": deepcopy(state["workflow"])}
        self.assertEqual(
            approved_candidate(state, "TEXT", {"cite:real"}, current_task=task)["approval_id"], "a1"
        )
        changed = deepcopy(state)
        changed["approvals"][0]["candidate_version"] = "obsolete"
        with self.assertRaisesRegex(AssertionError, "current authorized"):
            approved_candidate(changed, "TEXT", {"cite:real"}, current_task=task)
        state["workflow"]["requirement_version"] = 3
        with self.assertRaisesRegex(AssertionError, "current authorized"):
            approved_candidate(state, "TEXT", {"cite:real"}, current_task=task)

    def test_plan_preserves_frozen_inputs_and_dependencies(self):
        state = verified_workflow()
        self.assertTrue(plan_verified(state))
        for step, field, value in ((0, "inputs", []), (1, "depends_on", []), (2, "inputs", ["other.txt"])):
            changed = deepcopy(state)
            changed["workflow"]["plan"]["steps"][step][field] = value
            self.assertFalse(plan_verified(changed))

    def test_plan_checks_require_the_observed_execution_receipts(self):
        state = verified_workflow()
        changes = (
            ("status", "awaiting_approval"),
            ("result", None),
        )
        for field, value in changes:
            changed = deepcopy(state)
            changed["intents"][0][field] = value
            self.assertFalse(plan_verified(changed))
        for field, value in (
            ("executed", False),
            ("is_error", True),
            ("intent_id", "other-session-receipt"),
            ("tool_call_id", "another-call"),
            ("requirement_version", 0),
            ("file_versions", {}),
        ):
            changed = deepcopy(state)
            changed["intents"][0]["result"][field] = value
            self.assertFalse(plan_verified(changed))
        for step, removed in ((0, "search"), (1, "write"), (2, "result-read")):
            changed = deepcopy(state)
            checks = changed["workflow"]["plan"]["steps"][step]["checks"]
            changed["workflow"]["plan"]["steps"][step]["checks"] = [
                c for c in checks if c["intent_id"] != removed
            ]
            self.assertFalse(plan_verified(changed))
        changed = deepcopy(state)
        changed["intents"][1]["arguments"]["mode"] = "bm25"
        self.assertFalse(plan_verified(changed))
