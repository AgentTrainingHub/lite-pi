"""Hybrid retrieval public-session contracts; controlled inference/HTTP is mechanism evidence only."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deterministic_transport import response, transport

from lite_pi.journal import Journal
from lite_pi.retrieval.embedding import Encoding, Representation
from lite_pi.session import AgentSession

FIXTURE = Path(__file__).parent / "fixtures/retrieval"


class ControlledEncoder:
    representation = Representation()

    def __init__(self, directory):
        pass

    def encode(self, texts, *, query, stopped):
        if stopped():
            raise ValueError("cancelled")
        vectors = []
        for text in texts:
            v = [0.0] * 384
            v[0 if query or "核对" in text else 1] = 1.0
            vectors.append(tuple(v))
        return Encoding(tuple(vectors), 0.0, 0.0, tuple(10 for _ in texts))

    def close(self):
        pass


def tool_call(mode, query="核对 未知"):
    return {
        "id": "search",
        "type": "function",
        "function": {
            "name": "search_docs",
            "arguments": json.dumps({"query": query, "mode": mode, "limit": 4}),
        },
    }


class HybridRetrievalContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lite-pi-hybrid_retrieval-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.work = self.base / "project"
        shutil.copytree(FIXTURE / "docs", self.work / "docs")
        (self.work / ".lite-pi").mkdir()
        shutil.copy2(FIXTURE / "corpus.json", self.work / ".lite-pi/corpus.json")

    def test_missing_embedding_is_explicit_and_bm25_keeps_working(self):
        session = AgentSession(self.work, embedding_dir=self.base / "missing")
        built = session.rebuild_index()
        self.assertEqual(built["reason"], "rebuilt")
        semantic = session.rebuild_index(mode="semantic")
        self.assertTrue(semantic["is_error"])
        self.assertIn("embedding_model_missing", semantic["reason"])
        self.assertEqual(session.knowledge()["generation_id"], built["generation_id"])

    def test_semantic_candidates_are_visible_through_the_public_session(self):
        path = self.base / "session.jsonl"
        journal = Journal.create(path, self.work, tools=("search_docs",))
        session = AgentSession(self.work, tools=("search_docs",), journal=journal)
        with patch("lite_pi.retrieval.index.LocalEncoder", ControlledEncoder):
            built = session.rebuild_index(mode="semantic")
            with transport(
                self.base / "wire", [response("", calls=[tool_call("semantic")]), response("依据资料回答")]
            ):
                result = session.submit("查找恢复规则", max_turns=2)
        outcome = result["transcript"][0]["results"][0]
        self.assertFalse(outcome["is_error"], outcome)
        self.assertEqual(outcome["trace"]["mode"], "semantic")
        self.assertEqual(outcome["generation_id"], built["generation_id"])
        self.assertEqual(outcome["hits"][0]["source"], "docs/recovery.md")
        self.assertEqual(outcome["trace"]["representation"]["dimension"], 384)

    def test_modes_and_repeated_queries_keep_distinct_provenance(self):
        session = AgentSession(self.work, tools=("search_docs",))
        with patch("lite_pi.retrieval.index.LocalEncoder", ControlledEncoder):
            session.rebuild_index(mode="semantic")
            with transport(
                self.base / "cache",
                [
                    response("", calls=[tool_call("semantic")]),
                    response("", calls=[tool_call("bm25")]),
                    response("", calls=[tool_call("semantic")]),
                    response("done"),
                ],
            ):
                result = session.submit("比较检索", max_turns=4)
        outcomes = [t["results"][0] for t in result["transcript"] if t["results"]]
        self.assertEqual([o["trace"]["mode"] for o in outcomes], ["semantic", "bm25", "semantic"])
        self.assertIsNone(outcomes[1]["repeat_of"])
        self.assertEqual(outcomes[2]["repeat_of"], outcomes[0]["query_id"])
        self.assertEqual(outcomes[2]["trace"], outcomes[0]["trace"])

    def test_failed_rebuild_and_representation_change_do_not_fall_back(self):
        from lite_pi.knowledge.db import KnowledgeError

        session = AgentSession(self.work, tools=("search_docs",))
        with patch("lite_pi.retrieval.index.LocalEncoder", ControlledEncoder):
            initial = session.rebuild_index(mode="semantic")
        with patch("lite_pi.retrieval.index.LocalEncoder", side_effect=KnowledgeError("embedding_cancelled")):
            failed = session.rebuild_index(mode="semantic")
        self.assertTrue(failed["is_error"])
        self.assertEqual(session.knowledge()["generation_id"], initial["generation_id"])

        class ChangedEncoder(ControlledEncoder):
            representation = Representation(revision="different-revision")

        with patch("lite_pi.retrieval.index.LocalEncoder", ChangedEncoder):
            with transport(self.base / "changed", [response("", calls=[tool_call("semantic")])]):
                result = session.submit("查询", max_turns=1)
        self.assertEqual(result["transcript"][0]["results"][0]["reason"], "embedding_representation_mismatch")

    def test_bm25_does_not_load_embedding_runtime(self):
        session = AgentSession(self.work, tools=("search_docs",))
        with patch("lite_pi.retrieval.index.LocalEncoder", side_effect=AssertionError("must stay lazy")):
            session.rebuild_index()
            with transport(self.base / "lazy", [response("", calls=[tool_call("bm25", "QUERY_PAGE_CAP")])]):
                result = session.submit("查询", max_turns=1)
        self.assertFalse(result["transcript"][0]["results"][0]["is_error"])

    def hybrid_run(self, *, invalid=False, max_model_calls=None, mutate=None):
        journal = Journal.create(self.base / "hybrid.jsonl", self.work, tools=("search_docs",))
        session = AgentSession(self.work, tools=("search_docs",), journal=journal)
        count = 0
        captured = []

        def respond(payload, *args, **kwargs):
            nonlocal count
            count += 1
            if "Relevance ranker" in payload["messages"][0]["content"]:
                self.assertNotIn("tools", payload)
                candidates = json.loads(payload["messages"][1]["content"])["candidates"]
                captured.extend(candidates)
                if mutate:
                    mutate()
                order = ["invented"] if invalid else [c["chunk_id"] for c in reversed(candidates)]
                return response(json.dumps({"order": order}), stream=bool(payload.get("stream")))
            return response("", calls=[tool_call("hybrid")]) if count == 1 else response("done")

        with patch("lite_pi.retrieval.index.LocalEncoder", ControlledEncoder):
            session.rebuild_index(mode="semantic")
            with transport(self.base / "hybrid-wire", respond) as wire:
                result = session.submit(
                    "混合检索", max_turns=2, max_model_calls=max_model_calls, max_task_calls=4
                )
        return result, captured, wire.call_count

    def test_hybrid_deduplicates_reranks_once_and_charges_parent(self):
        result, candidates, count = self.hybrid_run()
        outcome = result["transcript"][0]["results"][0]
        self.assertFalse(outcome["is_error"], outcome)
        self.assertLessEqual(len(candidates), 8)
        self.assertEqual(len({c["chunk_id"] for c in candidates}), len(candidates))
        trace = outcome["trace"]
        self.assertTrue({c["chunk_id"] for c in trace["bm25"]} & {c["chunk_id"] for c in trace["semantic"]})
        self.assertEqual(
            [h["chunk_id"] for h in outcome["hits"]], list(reversed([c["chunk_id"] for c in candidates]))[:4]
        )
        self.assertEqual(count, 3)
        self.assertEqual(result["request_count"], 2)
        self.assertEqual(result["total_request_count"], 3)
        self.assertEqual(result["tasks"]["budget"]["spent_calls"], 3)

    def test_invalid_rerank_keeps_receipt_without_fallback(self):
        result, _, _ = self.hybrid_run(invalid=True)
        outcome = result["transcript"][0]["results"][0]
        self.assertTrue(outcome["is_error"])
        self.assertEqual(outcome["hits"], [])
        self.assertEqual(outcome["reason"], "rerank_invalid_permutation")
        self.assertIn("evidence", outcome["trace"]["rerank"])

    def test_rerank_cannot_bypass_the_run_call_limit(self):
        result, candidates, count = self.hybrid_run(max_model_calls=1)
        self.assertEqual(count, 1)
        self.assertEqual(candidates, [])
        self.assertEqual(result["transcript"][0]["results"][0]["reason"], "run_calls")

    def test_insufficient_output_budget_prevents_the_http_request(self):
        journal = Journal.create(self.base / "small-output.jsonl", self.work, tools=("search_docs",))
        session = AgentSession(self.work, tools=("search_docs",), journal=journal)
        with transport(self.base / "small-output", []) as wire:
            result = session.submit("查询", max_tokens=64, max_task_output_tokens=1, max_turns=1)
        self.assertEqual(wire.call_count, 0)
        self.assertEqual(result["status"], "budget")
        self.assertEqual(result["tasks"]["budget"]["spent_calls"], 0)

    def test_source_withdrawn_during_rerank_cannot_be_returned(self):
        def mutate():
            path = self.work / ".lite-pi/corpus.json"
            payload = json.loads(path.read_text())
            payload["sources"] = payload["sources"][:-1]
            path.write_text(json.dumps(payload))

        result, _, _ = self.hybrid_run(mutate=mutate)
        outcome = result["transcript"][0]["results"][0]
        self.assertEqual(outcome["hits"], [])
        self.assertEqual(outcome["reason"], "index_stale")
        self.assertIn("evidence", outcome["trace"]["rerank"])

    def test_native_query_cancellation_closes_runtime_and_keeps_empty_result(self):
        from lite_pi.knowledge.db import KnowledgeError

        journal = Journal.create(self.base / "cancel.jsonl", self.work, tools=("search_docs",))
        session = AgentSession(self.work, tools=("search_docs",), journal=journal)
        closed = []

        class CancellingEncoder(ControlledEncoder):
            def encode(self, texts, *, query, stopped):
                if query:
                    session.cancel()
                    raise KnowledgeError("cancelled")
                return super().encode(texts, query=query, stopped=stopped)

            def close(self):
                closed.append(True)

        with patch("lite_pi.retrieval.index.LocalEncoder", CancellingEncoder):
            session.rebuild_index(mode="semantic")
            with transport(self.base / "cancel-wire", [response("", calls=[tool_call("semantic")])]) as wire:
                result = session.submit("查询", max_turns=2)
        self.assertEqual(result["status"], "aborted")
        self.assertEqual(wire.call_count, 1)
        self.assertEqual(len(closed), 2)
        self.assertEqual(result["transcript"][0]["results"][0]["hits"], [])

    def test_invalid_dimensions_preserve_the_previous_generation(self):
        session = AgentSession(self.work)
        built = session.rebuild_index()

        class BrokenEncoder(ControlledEncoder):
            def encode(self, texts, *, query, stopped):
                return Encoding(tuple((1.0,) for _ in texts), 0, 0, tuple(1 for _ in texts))

        with patch("lite_pi.retrieval.index.LocalEncoder", BrokenEncoder):
            result = session.rebuild_index(mode="semantic")
        self.assertEqual(result["reason"], "embedding_dimension_or_nonfinite")
        self.assertEqual(session.knowledge()["generation_id"], built["generation_id"])

    def test_factual_judge_preserves_exact_integer_values_and_rejects_bad_claims(self):
        from cases_hybrid_retrieval import judge

        question = {"id": "id-dev", "expected_source": "limits.md"}
        result = {
            "status": "stop",
            "transcript": [
                {
                    "results": [
                        {
                            "name": "search_docs",
                            "executed": True,
                            "is_error": False,
                            "trace": {"mode": "bm25"},
                            "hits": [{"chunk_id": "a", "source": "docs/limits.md"}],
                        }
                    ]
                }
            ],
        }
        for value, expected in (
            (17, True),
            ("17", True),
            (18, False),
            ("17 extra", False),
            (True, False),
            (17.0, False),
        ):
            with self.subTest(value=repr(value)):
                result["text"] = json.dumps({"status": "supported", "fact": value, "citations": ["cite:a"]})
                self.assertEqual(judge(result, question, "bm25")[0], expected)
        result["text"] = '{"status":"supported","fact":17,"citations":["cite:invented"]}'
        self.assertFalse(judge(result, question, "bm25")[0])
        result["text"] = '{"status":"supported","fact":17,"citations":["cite:a"]}'
        self.assertFalse(judge(result, question, "semantic")[0])

    def test_factual_grammar_checks_units_direction_and_every_claim(self):
        from cases_hybrid_retrieval import fact_matches

        self.assertTrue(fact_matches("14 天", 14, "conf-eval"))
        self.assertFalse(fact_matches("14 天", 14, "id-dev"))
        self.assertFalse(fact_matches("14 天，自动续期", 14, "conf-eval"))
        self.assertTrue(fact_matches("GRAMS_PER_KILOGRAM = 1000；克数转为千克时除以 1000。", 1000, "id-eval"))
        for claim in (
            "GRAMS_PER_KILOGRAM = 1000；克数转为千克时除以 100",
            "GRAMS_PER_KILOGRAM = 1000；千克转克数时除以 1000",
            "GRAMS_PER_KILOGRAM = 1000；克数转为千克时乘以 1000",
            "GRAMS_PER_KILOGRAM = 999；克数转为千克时除以 1000",
            "GRAMS_PER_KILOGRAM = 1000；克数转为千克时除以 1000；常量为 999",
            "GRAMS_PER_KILOGRAM = 1000；克数转为千克时除以 1000；适用于所有版本",
        ):
            with self.subTest(claim=claim):
                self.assertFalse(fact_matches(claim, 1000, "id-eval"))
