"""Deterministic LangGraph flow. Scripted ports are not real-model hits."""

from __future__ import annotations

import inspect
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import compare.flow as flow
from compare.flow import GraphRun, compile_flow, initial_state

from lite_pi.provider import ModelResult
from lite_pi.tools.contracts import JsonObject


class ScriptModel:
    def __init__(self) -> None:
        self.seen: list[list[object]] = []

    def complete(
        self,
        messages,
        definitions,
        *,
        run_id: str,
        turn: int,
        max_tokens: int,
        budget,
    ) -> ModelResult:
        del definitions, run_id, max_tokens, budget
        self.seen.append([item.get("role") for item in messages])
        if turn == 1:
            message: JsonObject = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path": "a.txt"}'},
                    }
                ],
            }
            return ModelResult("tool_calls", message, "tool_calls", "script", attempts=1)
        return ModelResult("stop", {"role": "assistant", "content": "done"}, "stop", "script", attempts=1)


class ScriptTools:
    def __init__(self, text: str = "missing file") -> None:
        self.calls = 0
        self.text = text

    def definitions(self) -> list[JsonObject]:
        return []

    def execute(self, call: JsonObject) -> JsonObject:
        del call
        self.calls += 1
        return {"text": self.text, "is_error": True, "executed": True, "name": "read"}


class StopBudget:
    def __init__(self, reason: str | None) -> None:
        self.reason = reason

    def reserve(self, purpose: str, output_tokens: int) -> str | None:
        del purpose, output_tokens
        return None

    def settle(self, reservation_id: str, usage, attempts: int) -> None:
        del reservation_id, usage, attempts

    def release(self, reservation_id: str) -> None:
        del reservation_id

    def stop_reason(self) -> str | None:
        return self.reason

    def snapshot(self) -> JsonObject:
        return {"reason": self.reason}


class GraphFlowContracts(unittest.TestCase):
    def test_graph_has_model_and_tool_nodes_and_does_not_call_the_product_loop(self) -> None:
        source = inspect.getsource(flow)
        self.assertNotIn("lite_pi.loop", source)
        self.assertNotIn("AgentSession", source)
        app = compile_flow(
            ScriptModel(), ScriptTools(), run_id="det", max_turns=3, max_tokens=32, budget=None
        )
        names = set(app.get_graph().nodes)
        self.assertIn("model", names)
        self.assertIn("tools", names)

    def test_tool_error_is_in_the_resumed_model_state(self) -> None:
        model = ScriptModel()
        tools = ScriptTools()
        closed: list[str] = []
        with TemporaryDirectory() as raw:
            run = GraphRun(
                compile_flow(model, tools, run_id="det", max_turns=4, max_tokens=32, budget=None),
                "thread-1",
                Path(raw),
                closer=lambda: closed.append("closed") or {"spent_calls": 1},
            )
            try:
                paused = run.invoke(initial_state("inspect a.txt"), interrupt_after=["tools"])
                self.assertEqual(paused["tool_node_runs"], 1)
                self.assertIn("tool", [item.get("role") for item in paused["messages"]])
                self.assertEqual(tools.calls, 1)
                finished = run.invoke(None)
            finally:
                run.close()
            self.assertTrue((Path(raw) / "graph-result.json").is_file())
        self.assertEqual(finished["status"], "stop")
        self.assertEqual(finished["model_node_runs"], 2)
        self.assertEqual(model.seen[1][-2:], ["assistant", "tool"])
        self.assertIn("missing file", str(finished["messages"][-2].get("content")))
        self.assertEqual(closed, ["closed"])

    def test_budget_stop_does_not_call_the_model(self) -> None:
        model = ScriptModel()
        app = compile_flow(
            model,
            ScriptTools(),
            run_id="det",
            max_turns=3,
            max_tokens=32,
            budget=StopBudget("task_calls"),
        )
        state = app.invoke(initial_state("inspect"), {"configurable": {"thread_id": "budget"}})
        self.assertEqual(state["status"], "budget")
        self.assertEqual(state["reason"], "task_calls")
        self.assertEqual(model.seen, [])

    def test_close_reports_cleanup_failure(self) -> None:
        def fail() -> None:
            raise RuntimeError("book_close_failed")

        run = GraphRun(
            compile_flow(ScriptModel(), ScriptTools(), run_id="det", max_turns=2, max_tokens=32, budget=None),
            "thread-close",
            closer=fail,
        )
        run.invoke(initial_state("stop"), interrupt_after=["model"])
        with self.assertRaises(RuntimeError) as caught:
            run.close()
        self.assertIn("book_close_failed", str(caught.exception))
        self.assertIn("book_close_failed", run.receipt()["cleanup_error"])


class BatchModel(ScriptModel):
    def __init__(self, status: str = "tool_calls") -> None:
        super().__init__()
        self.status = status

    def complete(self, messages, definitions, **kwargs) -> ModelResult:
        result = super().complete(messages, definitions, **kwargs)
        if kwargs["turn"] != 1:
            return result
        result.message["tool_calls"].append(
            {"id": "c2", "type": "function", "function": {"name": "write", "arguments": "{}"}}
        )
        return replace(result, status=self.status)


class FailureTools(ScriptTools):
    def execute(self, call: JsonObject) -> JsonObject:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("effect happened but result was lost")
        return {"text": "written", "executed": True, "is_error": False}


class GraphSafetyContracts(unittest.TestCase):
    def test_model_exception_retains_completed_tool_checkpoint(self) -> None:
        model = ScriptModel()
        first = model.complete([], [], run_id="det", turn=1, max_tokens=32, budget=None)
        tools = ScriptTools()
        with TemporaryDirectory() as raw:
            run = GraphRun(
                compile_flow(model, tools, run_id="det", max_turns=3, max_tokens=32, budget=None),
                "failure",
                Path(raw),
            )
            with (
                patch.object(model, "complete", side_effect=[first, RuntimeError("model failed")]),
                self.assertRaisesRegex(RuntimeError, "model failed"),
            ):
                run.invoke(initial_state("inspect"))
            with self.assertRaisesRegex(RuntimeError, "graph_failed"):
                run.invoke(None)
            run.close()
            receipt = json.loads((Path(raw) / "graph-result.json").read_text())
        self.assertEqual(tools.calls, 1)
        self.assertEqual(receipt["tool_node_runs"], 1)
        self.assertEqual(len(receipt["outcomes"]), 1)
        self.assertEqual(receipt["status"], "error")
        self.assertIn("model failed", receipt["execution_error"])

    def test_original_error_survives_cleanup_and_receipt_failure(self) -> None:
        def close():
            raise RuntimeError("cleanup failed")

        model = ScriptModel()
        with TemporaryDirectory() as raw:
            evidence = Path(raw) / "file"
            evidence.write_text("cannot be a directory")
            run = GraphRun(
                compile_flow(model, ScriptTools(), run_id="det", max_turns=3, max_tokens=32, budget=None),
                "failure",
                evidence,
                closer=close,
            )
            with (
                patch.object(model, "complete", side_effect=ValueError("model failed")),
                self.assertRaisesRegex(ValueError, "model failed") as caught,
                run,
            ):
                run.invoke(initial_state("inspect"))
        self.assertIn("cleanup failed", run.cleanup_error)
        self.assertIn("FileExistsError", run.evidence_error)
        self.assertIn("cleanup failed", str(caught.exception.__notes__))
        self.assertIn("receipt also failed", str(caught.exception.__notes__))

    def test_uncertain_second_effect_cannot_replay_first_effect(self) -> None:
        tools = FailureTools()
        run = GraphRun(
            compile_flow(BatchModel(), tools, run_id="batch", max_turns=3, max_tokens=32, budget=None),
            "batch",
        )
        first = run.invoke(initial_state("run"), interrupt_after=["tools"])
        self.assertEqual(tools.calls, 1)
        self.assertEqual(len(first["pending"]), 1)
        halted = run.invoke(None)
        self.assertEqual(halted["status"], "effect_unknown")
        self.assertEqual(tools.calls, 2)
        run.invoke(None)
        self.assertEqual(tools.calls, 2)
        run.close()

    def test_length_response_does_not_execute_partial_calls(self) -> None:
        tools = ScriptTools()
        run = GraphRun(
            compile_flow(
                BatchModel("length"), tools, run_id="length", max_turns=3, max_tokens=32, budget=None
            ),
            "length",
        )
        self.assertEqual(run.invoke(initial_state("run"))["status"], "length")
        self.assertEqual(tools.calls, 0)
        run.close()

    def test_external_failure_feedback_resumes_same_checkpoint(self) -> None:
        model = ScriptModel()
        tools = ScriptTools()
        run = GraphRun(
            compile_flow(model, tools, run_id="feedback", max_turns=3, max_tokens=32, budget=None),
            "feedback",
        )
        run.invoke(initial_state("run"), interrupt_after=["boundary"])
        run.followup("check failed: missing conversion")
        final = run.invoke(None)
        self.assertEqual(final["status"], "stop")
        self.assertEqual(tools.calls, 1)
        self.assertEqual(model.seen[-1][-3:], ["assistant", "tool", "user"])
        run.close()
        with self.assertRaisesRegex(RuntimeError, "graph_closed"):
            run.invoke(None)

    def test_unknown_result_stops_remaining_batch(self) -> None:
        class UnknownTools(ScriptTools):
            def execute(self, call: JsonObject) -> JsonObject:
                self.calls += 1
                return {"text": "connection lost", "effect": "unknown", "is_error": True}

        tools = UnknownTools()
        run = GraphRun(
            compile_flow(BatchModel(), tools, run_id="unknown", max_turns=3, max_tokens=32, budget=None),
            "unknown",
        )
        self.assertEqual(run.invoke(initial_state("run"))["status"], "effect_unknown")
        self.assertEqual(tools.calls, 1)
        run.close()

    def test_delegation_requires_child_and_parent_original_read_before_target_edit(self) -> None:
        import hashlib

        from compare.combo import LOOKUP, verified_delegation

        def turn(name, path, **outcome):
            return {
                "provider": {
                    "message": {
                        "tool_calls": [
                            {"id": name, "function": {"name": name, "arguments": json.dumps({"path": path})}}
                        ]
                    }
                },
                "results": [
                    {"name": name, "tool_call_id": name, "executed": True, "is_error": False, **outcome}
                ],
            }

        read = turn("read", "src/lookup.py", content=LOOKUP)
        digest = hashlib.sha256(LOOKUP.encode()).hexdigest()
        delegated = {
            "results": [
                {
                    "name": "delegate",
                    "executed": True,
                    "is_error": False,
                    "joined": True,
                    "proofs": [{"path": "src/lookup.py", "sha256": digest, "before_sha256": digest}],
                    "surveys": [{"run_id": "child"}],
                }
            ]
        }
        with TemporaryDirectory() as raw, patch("compare.combo.RUNS", Path(raw)):
            child = Path(raw) / "child"
            child.mkdir()
            (child / "result.json").write_text(json.dumps({"transcript": [read]}))
            self.assertTrue(
                verified_delegation({"transcript": [turn("write", "note.json"), delegated, read]})
            )
            self.assertFalse(
                verified_delegation({"transcript": [turn("edit", "src/lookup.py"), delegated, read]})
            )
            self.assertFalse(
                verified_delegation({"transcript": [delegated, turn("read", "other.py", content=LOOKUP)]})
            )
            (child / "result.json").write_text(json.dumps({"transcript": []}))
            self.assertFalse(verified_delegation({"transcript": [delegated, read]}))

    def test_failed_capability_calls_are_not_coverage(self) -> None:
        from compare.combo import capabilities

        result = {
            "transcript": [
                {
                    "results": [
                        {"name": name, "executed": True, "is_error": True}
                        for name in (
                            "search_docs",
                            "memory_read",
                            "mcp__marker__mark",
                            "mcp__shop__act",
                            "mcp__shop__query",
                            "delegate",
                        )
                    ]
                }
            ]
        }
        self.assertFalse(any(capabilities(result).values()))


if __name__ == "__main__":
    unittest.main()
