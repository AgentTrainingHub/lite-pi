"""Internal model/tool state machine. Public callers use AgentSession."""

from copy import deepcopy
from dataclasses import asdict

from lite_pi.compaction import compact_if_needed
from lite_pi.control import ControlInput, RunControl
from lite_pi.journal import Journal
from lite_pi.outcomes import RunStatus, ToolControl
from lite_pi.provider import ModelResult, complete
from lite_pi.recovery import EffectPreviewError
from lite_pi.resources import public_context, request_prefix
from lite_pi.run_trace import RunOutcome, RunTrace
from lite_pi.spend.gate import SpendBlocked
from lite_pi.system_prompt import system_message
from lite_pi.task_budget import BudgetExceeded, RequestBudget, RunRequestBudget
from lite_pi.tasks import TaskPort
from lite_pi.tools.contracts import HaltAdmission, JsonObject, PreparedCall
from lite_pi.tools.registry import ToolRegistry


def run(
    messages: list[JsonObject],
    *,
    registry: ToolRegistry,
    trace: RunTrace,
    journal: Journal | None,
    max_turns: int,
    max_tokens: int,
    max_model_calls: int | None,
    interrupt_after_chunks: int | None,
    control: RunControl | None = None,
    stream: bool = True,
    context: JsonObject | None = None,
    context_tokens: int | None = None,
    book: TaskPort | None = None,
    budget: RequestBudget | None = None,
) -> RunOutcome:
    return AgentLoop(
        messages,
        registry=registry,
        trace=trace,
        journal=journal,
        max_turns=max_turns,
        max_tokens=max_tokens,
        max_model_calls=max_model_calls,
        interrupt_after_chunks=interrupt_after_chunks,
        control=control,
        stream=stream,
        context=context,
        context_tokens=context_tokens,
        book=book,
        budget=budget,
    ).execute()


class AgentLoop:
    """Owns mutable run state; each phase advances that state once."""

    def __init__(
        self,
        messages: list[JsonObject],
        *,
        registry: ToolRegistry,
        trace: RunTrace,
        journal: Journal | None,
        max_turns: int,
        max_tokens: int,
        max_model_calls: int | None,
        interrupt_after_chunks: int | None,
        control: RunControl | None = None,
        stream: bool = True,
        context: JsonObject | None = None,
        context_tokens: int | None = None,
        book: TaskPort | None = None,
        budget: RequestBudget | None = None,
    ) -> None:
        self.messages = messages
        self.registry = registry
        self.trace = trace
        self.journal = journal
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.max_model_calls = max_model_calls
        self.interrupt_after_chunks = interrupt_after_chunks
        self.stream = stream
        self.context = context
        self.context_tokens = context_tokens
        self.book = book
        self.control = control or RunControl()
        self.spend_budget = budget if budget is not None else book.budget if book is not None else None
        self.transcript: list[JsonObject] = []
        self.run_id = trace.run_id
        self.initial = deepcopy(messages)
        self.sources = journal.source_ids() if journal else []
        self.status: RunStatus = "turn_limit"
        self.text = ""
        self.raw_stop: str | None = None
        self.reason: str | None = None
        self.last: ModelResult | None = None
        self.requests = 0
        self.pending_effects: dict[str, str] = {}
        self.unqueryable_effect: str | None = None
        self.compactions: list[JsonObject] = []
        self.resource_prefix = request_prefix(context) if context else []
        self.definitions = registry.advertised()

    def emit(self, event: JsonObject) -> None:
        self.trace.emit(event)

    def request_stopped(self) -> bool:
        return self.control.cancel.is_set() or (
            self.spend_budget is not None and self.spend_budget.stop_reason() == "time_budget"
        )

    def inject(self, item: ControlInput) -> None:
        entered = self.control.mark_entered(item["id"])
        self.registry.scope = self.control.scope
        if self.book is not None:
            self.book.note_scope(self.control.scope)
        message = {"role": "user", "content": item["text"]}
        self.messages.append(message)
        if self.journal:
            self.journal.append(
                "user", {"content": item["text"], "control_id": item["id"], "kind": item["kind"]}
            )
        self.emit({"type": "control_entered", "run_id": self.run_id, "control": entered})

    def ready(self) -> bool:
        if self.control.cancel.is_set():
            self.status, self.reason = "aborted", "cancelled"
            return False
        if self.max_model_calls is not None and self.requests >= self.max_model_calls:
            self.status = "budget"
            return False
        reason = self.spend_budget.stop_reason() if self.spend_budget else None
        if reason:
            self.status, self.reason = "budget", reason
            return False
        return True

    def execute(self) -> RunOutcome:
        self.emit(
            {
                "type": "run_start",
                "run_id": self.run_id,
                "history_messages": len(self.initial),
                "source_ids": self.sources,
            }
        )
        for turn in range(1, self.max_turns + 1):
            if self.control.cancel.is_set():
                self.status, self.reason = "aborted", "cancelled"
                break
            waiting = self.book.waiting_question() if self.book else None
            if waiting is not None:
                self.status, self.reason, self.text = waiting["status"], waiting["reason"], waiting["text"]
                break
            steered = self.control.drain("steer")
            if steered:
                self.inject(steered)
            if not self.ready() or not self.model_turn(turn):
                break
            calls, assistant_id = self.record_response(turn)
            if self.status in {"error", "aborted", "content_filter"}:
                break
            if not calls:
                if self.next_input():
                    continue
                break
            if not self.tools_turn(calls, assistant_id, turn):
                break
        return self.finish()

    def model_turn(self, turn: int) -> bool:
        self.emit({"type": "turn_start", "run_id": self.run_id, "turn": turn})
        self.registry.prepare()
        self.definitions = self.registry.advertised()
        prefix = [system_message(self.definitions), *self.resource_prefix]
        notes = self.registry.context_messages(self.messages)
        if self.book is not None:
            notes += self.book.context_messages()
        try:
            updated, info = compact_if_needed(
                self.messages,
                journal=self.journal,
                context_tokens=self.context_tokens,
                reserve_tokens=self.max_tokens,
                prefix=prefix + notes,
                definitions=self.definitions,
                run_id=self.run_id,
                max_attempts=min(2, self.max_model_calls - self.requests)
                if self.max_model_calls is not None
                else 2,
                stop=self.request_stopped,
                budget=self.spend_budget,
            )
            if info is not None:
                self.compactions.append(info)
                self.requests += info.get("attempts") or 0
                if info.get("failed"):
                    self.reason = info["reason"]
                    self.status = "aborted" if self.reason == "cancelled" else "compaction_failed"
                    return False
                if info.get("adopted"):
                    self.messages[:] = updated
            if not self.ready():
                return False
            self.last = complete(
                prefix + notes + self.messages,
                definitions=self.definitions,
                run_id=self.run_id,
                turn=turn,
                max_tokens=self.max_tokens,
                interrupt_after_chunks=self.interrupt_after_chunks,
                max_attempts=min(2, self.max_model_calls - self.requests)
                if self.max_model_calls is not None
                else 2,
                stream=self.stream and self.interrupt_after_chunks is None,
                on_delta=lambda piece: self.emit(
                    {"type": "message_delta", "run_id": self.run_id, "turn": turn, "text": piece}
                ),
                stop=self.request_stopped,
                budget=self.spend_budget,
            )
        except BudgetExceeded as error:
            self.requests += error.attempts
            self.status, self.reason = ("budget", error.reason)
            return False
        except SpendBlocked as error:
            self.status, self.reason = ("blocked", error.reason)
            return False
        return True

    def record_response(self, turn: int) -> tuple[list[JsonObject], str]:
        last = self.last
        if last is None:
            raise RuntimeError("model turn produced no response")
        self.requests += last.attempts
        if (
            self.compactions
            and self.compactions[-1].get("adopted")
            and ("deviation" not in self.compactions[-1])
        ):
            actual = (last.usage or {}).get("prompt_tokens")
            expected = self.compactions[-1].get("continuation_estimate")
            if isinstance(actual, int) and isinstance(expected, int):
                self.compactions[-1]["continuation_prompt_tokens"] = actual
                self.compactions[-1]["deviation"] = actual - expected
            else:
                self.compactions[-1]["continuation_prompt_tokens"] = None
                self.compactions[-1]["deviation"] = None
        self.status, self.raw_stop, self.reason = (last.status, last.raw_stop_reason, last.error)
        self.text = last.message.get("content") or ""
        self.transcript.append(
            {
                "turn": turn,
                "finish_reason": self.raw_stop,
                "provider": asdict(last),
                "tool_calls": [],
                "results": [],
            }
        )
        if self.status in {"error", "aborted", "content_filter"}:
            self.emit(
                {
                    "type": "message_end",
                    "run_id": self.run_id,
                    "turn": turn,
                    "text": self.text,
                    "status": self.status,
                }
            )
            return ([], "")
        assistant = deepcopy(last.message)
        self.messages.append(assistant)
        assistant_id = ""
        if self.journal:
            saved = self.journal.append(
                "assistant",
                {**assistant, "content": assistant.get("content") or "", "finish_reason": self.raw_stop},
            )
            assistant_id = saved["id"]
        calls = assistant.get("tool_calls") or []
        self.transcript[-1]["tool_calls"] = [call["function"].get("name") for call in calls]
        self.emit(
            {
                "type": "message_end",
                "run_id": self.run_id,
                "turn": turn,
                "text": self.text,
                "status": self.status,
            }
        )
        self.emit({"type": "model_response", "run_id": self.run_id, "message": assistant})
        return (calls, assistant_id)

    def next_input(self) -> bool:
        pending_steer = None if self.control.cancel.is_set() else self.control.drain("steer")
        if pending_steer:
            self.inject(pending_steer)
            self.status = "turn_limit"
            return True
        follow_up = None if self.control.cancel.is_set() else self.control.drain("follow_up")
        if follow_up:
            self.inject(follow_up)
            self.status = "turn_limit"
            return True
        return False

    def tools_turn(self, calls: list[JsonObject], assistant_id: str, turn: int) -> bool:
        blocked_tool = None
        halted = False
        for call in calls:
            persisted = False
            if self.control.cancel.is_set() or blocked_tool:
                outcome = {
                    "name": call["function"].get("name"),
                    "tool_call_id": call["id"],
                    "executed": False,
                    "is_error": True,
                    "text": "Run cancelled; tool not executed."
                    if self.control.cancel.is_set()
                    else "Run blocked; tool not executed.",
                    "reason": "cancelled" if self.control.cancel.is_set() else blocked_tool,
                }
            elif self.pending_effects and call["function"].get("name") not in self.pending_effects.values():
                outcome = {
                    "name": call["function"].get("name"),
                    "tool_call_id": call["id"],
                    "executed": False,
                    "is_error": True,
                    "reason": "query_required",
                    "text": "An action has an unknown effect. Only its reconciliation query is allowed.",
                }
            elif self.status == "length":
                outcome = {
                    "name": call["function"].get("name"),
                    "tool_call_id": call["id"],
                    "executed": False,
                    "is_error": True,
                    "reason": "length",
                    "text": "Output truncated; tool not executed. Resend complete arguments.",
                }
            else:
                self.emit(
                    {
                        "type": "tool_start",
                        "run_id": self.run_id,
                        "turn": turn,
                        "name": call["function"].get("name"),
                        "tool_call_id": call["id"],
                    }
                )
                outcome, persisted = _invoke_tool(self.registry, self.book, call, assistant_id)
                self.emit({"type": "tool_end", "run_id": self.run_id, "turn": turn, "result": outcome})
            tool_control = ToolControl.read(outcome)
            if tool_control.resolves_effect is not None:
                self.pending_effects.pop(tool_control.resolves_effect, None)
            if tool_control.effect == "unknown":
                if tool_control.effect_key is not None and tool_control.reconcile_tool is not None:
                    self.pending_effects[tool_control.effect_key] = tool_control.reconcile_tool
                else:
                    blocked_tool = tool_control.reason or "effect_unknown"
                    self.unqueryable_effect = blocked_tool
            if tool_control.blocks_run:
                blocked_tool = tool_control.reason or "tool_blocked"
            if tool_control.halt is None:
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": outcome.get("model_content", outcome["text"]),
                    }
                )
            if self.journal and (not persisted):
                self.journal.append(
                    "tool", {"tool_call_id": call["id"], "content": outcome["text"], "outcome": outcome}
                )
            self.transcript[-1]["results"].append(outcome)
            self.emit({"type": "tool_result", "run_id": self.run_id, "result": outcome})
            if tool_control.halt is not None:
                self.status = tool_control.halt
                self.reason = str(outcome.get("reason") or self.status)
                self.text = str(outcome.get("text") or self.text)
                halted = True
                break
        if halted:
            return False
        if self.control.cancel.is_set():
            self.status, self.reason = ("aborted", "cancelled")
            return False
        if blocked_tool:
            self.status = "effect_unknown" if self.unqueryable_effect else "blocked"
            self.reason = blocked_tool
            return False
        if self.status == "length":
            return False
        self.status = "turn_limit"
        return True

    def finish(self) -> RunOutcome:
        if (
            not self.control.cancel.is_set()
            and self.spend_budget is not None
            and (self.spend_budget.stop_reason() == "time_budget")
        ):
            self.status, self.reason = ("budget", "time_budget")
        if self.pending_effects or self.unqueryable_effect is not None:
            self.status, self.reason = ("effect_unknown", self.unqueryable_effect or "effect_unverified")
        waiting = self.book.waiting_question() if self.book is not None else None
        if waiting is not None and self.status not in {
            "error",
            "aborted",
            "effect_unknown",
            "awaiting_approval",
        }:
            self.status, self.reason, self.text = (waiting["status"], waiting["reason"], waiting["text"])
        return {
            "request_count": self.requests,
            "total_request_count": self.spend_budget.consumed_calls
            if isinstance(self.spend_budget, RunRequestBudget)
            else self.requests,
            "status": self.status,
            "raw_stop_reason": self.raw_stop,
            "text": self.text,
            "reason": self.reason,
            "model": self.last.actual_model if self.last else None,
            "usage": self.last.usage if self.last else None,
            "usage_unknown": self.last is None or self.last.usage is None,
            "transcript": self.transcript,
            "controls": self.control.snapshot(),
            "initial_request": self.initial,
            "source_ids": self.sources,
            "messages": deepcopy(self.messages),
            "tools": list(self.registry.allowed),
            "context": public_context(self.context) if self.context else None,
            "compactions": self.compactions,
            "extensions": self.registry.extension_report,
            "hook_log": self.registry.hook_log,
            "checked_files": any(
                item.get("name") == "read" and (not item["is_error"])
                for turn in self.transcript
                for item in turn["results"]
            ),
        }


def _invoke_tool(
    registry: ToolRegistry, book: TaskPort | None, call: JsonObject, assistant_id: str
) -> tuple[JsonObject, bool]:
    if book is None:
        return registry.execute(call), False
    prepared = registry.authorize(call)
    if not isinstance(prepared, PreparedCall):
        return prepared, False
    if book.question_blocks(prepared):
        return {
            "name": prepared.name,
            "tool_call_id": prepared.tool_call_id,
            "executed": False,
            "is_error": True,
            "reason": "clarification_pending",
            "text": "This action depends on a pending clarification; it was not executed.",
        }, False
    try:
        gate = book.admit(prepared, assistant_id)
    except EffectPreviewError as error:
        return {
            "name": prepared.name,
            "tool_call_id": prepared.tool_call_id,
            "executed": False,
            "is_error": True,
            "reason": "invalid_target",
            "text": str(error),
        }, False
    if isinstance(gate, HaltAdmission):
        return dict(gate.outcome), True
    outcome = registry.complete(prepared)
    return book.commit(gate.intent_id, outcome), True
