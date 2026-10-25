"""LangGraph coding flow. It calls the product model and tool ports, not the handwritten loop."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from lite_pi.provider import ModelResult, complete
from lite_pi.spend.gate import SpendBlocked
from lite_pi.task_budget import BudgetExceeded, RequestBudget
from lite_pi.tools.contracts import JsonObject
from lite_pi.tools.registry import ToolRegistry


class GraphState(TypedDict):
    messages: list[JsonObject]
    turn: int
    status: str
    reason: str
    request_count: int
    model_node_runs: int
    tool_node_runs: int
    pending: list[JsonObject]
    outcomes: list[JsonObject]


class ModelPort(Protocol):
    def complete(
        self,
        messages: Sequence[JsonObject],
        definitions: Sequence[JsonObject],
        *,
        run_id: str,
        turn: int,
        max_tokens: int,
        budget: RequestBudget | None,
    ) -> ModelResult: ...


class ToolPort(Protocol):
    def definitions(self) -> list[JsonObject]: ...

    def execute(self, call: JsonObject) -> JsonObject: ...


class ProviderModel:
    """Product provider. Streaming stays on the same transmit path as the handwritten loop."""

    def complete(
        self,
        messages: Sequence[JsonObject],
        definitions: Sequence[JsonObject],
        *,
        run_id: str,
        turn: int,
        max_tokens: int,
        budget: RequestBudget | None,
    ) -> ModelResult:
        return complete(
            list(messages),
            definitions=list(definitions),
            run_id=run_id,
            turn=turn,
            max_tokens=max_tokens,
            stream=True,
            budget=budget,
        )


class RegistryTools:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def definitions(self) -> list[JsonObject]:
        return self._registry.advertised()

    def execute(self, call: JsonObject) -> JsonObject:
        return self._registry.execute(call)


def compile_flow(
    model: ModelPort,
    tools: ToolPort,
    *,
    run_id: str,
    max_turns: int,
    max_tokens: int,
    budget: RequestBudget | None,
) -> CompiledStateGraph[GraphState, None, GraphState, GraphState]:
    def model_node(state: GraphState) -> GraphState:
        turn = state["turn"] + 1
        if turn > max_turns:
            return _copy(state, turn=state["turn"], status="turn_limit", reason="turn_limit")
        if budget is not None:
            stopped = budget.stop_reason()
            if stopped:
                return _copy(state, status="budget", reason=stopped)
        try:
            result = model.complete(
                state["messages"],
                tools.definitions(),
                run_id=run_id,
                turn=turn,
                max_tokens=max_tokens,
                budget=budget,
            )
        except BudgetExceeded as error:
            return _copy(
                state,
                turn=turn,
                status="budget",
                reason=error.reason,
                request_count=state["request_count"] + error.attempts,
            )
        except SpendBlocked as error:
            return _copy(state, status="blocked", reason=error.reason)
        message = result.message if result.message else {"role": "assistant", "content": ""}
        messages = [*state["messages"], message]
        calls = message.get("tool_calls")
        status = "running" if isinstance(calls, list) and calls else result.status
        if result.status in {"error", "aborted", "length", "content_filter"}:
            status = result.status
        return _copy(
            state,
            messages=messages,
            turn=turn,
            status=status,
            reason=result.error or "",
            request_count=state["request_count"] + result.attempts,
            model_node_runs=state["model_node_runs"] + 1,
            pending=calls if status == "running" else [],
        )

    def tools_node(state: GraphState) -> GraphState:
        # One effect per checkpoint; an uncertain effect ends the graph and cannot be replayed.
        call, *pending = state["pending"]
        try:
            outcome = tools.execute(call)
        except Exception as error:
            outcome = {
                "text": f"{type(error).__name__}: {error}",
                "is_error": True,
                "effect": "unknown",
                "reason": "tool_exception",
            }
        reason = str(outcome.get("reason") or "")
        halted = (
            outcome.get("effect") == "unknown"
            or bool(outcome.get("halt"))
            or reason in {"cleanup_failed", "runtime_error", "workspace_sync_failed", "isolation_unavailable"}
        )
        message = {"role": "tool", "tool_call_id": call["id"], "content": str(outcome.get("text") or "")}
        return _copy(
            state,
            messages=[*state["messages"], message],
            pending=pending,
            outcomes=[*state["outcomes"], outcome],
            status="effect_unknown" if halted else "running",
            reason=reason,
            tool_node_runs=state["tool_node_runs"] + 1,
        )

    def route(state: GraphState) -> str:
        if state["status"] != "running":
            return "end"
        return "tools" if state["pending"] else "boundary"

    def boundary(state: GraphState) -> GraphState:
        return state

    graph = StateGraph(GraphState)
    graph.add_node("model", model_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, {"tools": "tools", "boundary": "boundary", "end": END})
    graph.add_node("boundary", boundary)
    graph.add_conditional_edges("tools", route, {"tools": "tools", "boundary": "boundary", "end": END})
    graph.add_edge("boundary", "model")
    return graph.compile(checkpointer=MemorySaver())


def initial_state(task: str) -> GraphState:
    return {
        "messages": [{"role": "user", "content": task}],
        "turn": 0,
        "status": "running",
        "reason": "",
        "request_count": 0,
        "model_node_runs": 0,
        "tool_node_runs": 0,
        "pending": [],
        "outcomes": [],
    }


@dataclass
class GraphRun:
    """Owns one compiled graph. The caller closes the task database after the last invoke."""

    app: CompiledStateGraph[GraphState, None, GraphState, GraphState]
    thread_id: str
    evidence: Path | None = None
    closer: Callable[[], JsonObject | None] | None = None
    state: GraphState | None = None
    cleanup_error: str | None = None
    execution_error: str | None = None
    evidence_error: str | None = None
    _closed: bool = field(default=False, init=False)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.close()
        except Exception as cleanup:
            if exc is None:
                raise
            exc.add_note(f"Graph cleanup also failed: {type(cleanup).__name__}: {cleanup}")
            for note in getattr(cleanup, "__notes__", []):
                exc.add_note(note)

    def invoke(
        self,
        state: GraphState | None,
        *,
        interrupt_after: list[str] | None = None,
    ) -> GraphState:
        if self._closed:
            raise RuntimeError("graph_closed")
        if self.execution_error is not None:
            raise RuntimeError("graph_failed")
        try:
            result = self.app.invoke(
                state,
                {"configurable": {"thread_id": self.thread_id}, "recursion_limit": 1000},
                interrupt_after=interrupt_after,
            )
        except BaseException as error:
            self.execution_error = f"{type(error).__name__}: {error}"
            try:
                checkpoint = self.app.get_state({"configurable": {"thread_id": self.thread_id}})
                if checkpoint.values:
                    self.state = _as_state(checkpoint.values)
            except Exception as snapshot_error:
                error.add_note(f"Checkpoint read failed: {type(snapshot_error).__name__}: {snapshot_error}")
            raise
        self.state = _as_state(result)
        return self.state

    def followup(self, task: str) -> None:
        if self.state is None or self._closed or self.execution_error is not None:
            raise RuntimeError("graph_not_active")
        self.state = _copy(self.state, messages=[*self.state["messages"], {"role": "user", "content": task}])
        self.app.update_state({"configurable": {"thread_id": self.thread_id}}, self.state)

    def close(self) -> JsonObject:
        if self._closed:
            return self.receipt()
        self._closed = True
        tasks: JsonObject | None = None
        try:
            if self.closer is not None:
                tasks = self.closer()
        except Exception as error:
            self.cleanup_error = f"{type(error).__name__}: {error}"
            try:
                self._write(tasks)
            except Exception as evidence_error:
                error.add_note(
                    f"Graph receipt also failed: {type(evidence_error).__name__}: {evidence_error}"
                )
            raise
        self._write(tasks)
        return self.receipt()

    def receipt(self) -> JsonObject:
        state = self.state
        return {
            "arm": "langgraph",
            "thread_id": self.thread_id,
            "status": "error" if self.execution_error else None if state is None else state["status"],
            "reason": None if state is None else state["reason"],
            "request_count": 0 if state is None else state["request_count"],
            "model_node_runs": 0 if state is None else state["model_node_runs"],
            "tool_node_runs": 0 if state is None else state["tool_node_runs"],
            "roles": [] if state is None else [item.get("role") for item in state["messages"]],
            "cleanup_error": self.cleanup_error,
            "execution_error": self.execution_error,
            "evidence_error": self.evidence_error,
            "nodes": ["model", "tools", "boundary"],
            "outcomes": [] if state is None else state["outcomes"],
            "messages": [] if state is None else state["messages"],
        }

    def _write(self, tasks: JsonObject | None) -> None:
        if self.evidence is None:
            return
        try:
            self.evidence.mkdir(parents=True, exist_ok=True)
            body = {**self.receipt(), "tasks": tasks}
            (self.evidence / "graph-result.json").write_text(
                json.dumps(body, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as error:
            self.evidence_error = f"{type(error).__name__}: {error}"
            raise


def _copy(state: GraphState, **updates: object) -> GraphState:
    merged: dict[str, object] = dict(state)
    merged.update(updates)
    return _as_state(merged)


_STATE = TypeAdapter(GraphState)


def _as_state(value: object) -> GraphState:
    return _STATE.validate_python(value, strict=True)
