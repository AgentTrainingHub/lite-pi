"""The public entry for submit, interruption, durable resume and branch selection."""

from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from threading import Event
from typing import TypedDict, Unpack
from uuid import uuid4

from lite_pi.control import ControlInput, RunControl
from lite_pi.delegate import DelegationTable, delegate_tool
from lite_pi.delegation_report import ChildBinding
from lite_pi.extensions import LoadedExtensions, load_extensions
from lite_pi.journal import Journal
from lite_pi.knowledge.db import KnowledgeDB
from lite_pi.knowledge.prefix import adopted_context
from lite_pi.knowledge.tools import KNOWLEDGE_TOOLS, knowledge_tools
from lite_pi.lifecycle import close_resources
from lite_pi.loop import run
from lite_pi.mcp_host import McpHost
from lite_pi.memory.ledger import MemoryLedger
from lite_pi.planning import PLANNING_TOOLS, plan_tool, question_tool
from lite_pi.resources import load_context
from lite_pi.retrieval.budget import RetrievalMeter
from lite_pi.retrieval.embedding import Mode
from lite_pi.retrieval.index import CorpusIndex
from lite_pi.run_settings import RunSettings
from lite_pi.run_trace import RunOutcome, RunTrace
from lite_pi.runtime import PATHS
from lite_pi.storage import locked
from lite_pi.task_budget import RequestBudget, RunRequestBudget
from lite_pi.tasks import TaskBook, run_halt
from lite_pi.tools.contracts import JsonObject, ToolDefinition
from lite_pi.tools.registry import ToolRegistry
from lite_pi.tools.workspace import Workspace

RUNS = PATHS.state / "runs"
CODE_TOOLS = ("read", "edit", "write", "bash")


def _with_memory(context: JsonObject, knowledge: KnowledgeDB | None, run_id: str) -> JsonObject:
    if knowledge is None:
        return context
    messages, records = adopted_context(MemoryLedger(knowledge), run_id)
    return {**context, "memories": messages, "memory_records": records}


class RunLimits(TypedDict, total=False):
    max_turns: int
    max_tokens: int
    max_bytes: int
    shell_timeout: float
    cancel_bash_after: float | None
    max_model_calls: int | None
    interrupt_after_chunks: int | None
    on_event: Callable[[JsonObject], None] | None
    context_tokens: int | None
    approvals: bool
    max_task_calls: int | None
    max_task_output_tokens: int | None
    max_task_seconds: float | None
    approval_ttl_seconds: float | None
    max_retrievals: int | None


class Unsupported(RuntimeError):
    def __init__(self, operation: str):
        self.operation = operation
        super().__init__(f"{operation} is not supported")


class AgentSession:
    def __init__(
        self,
        project_root: Path,
        *,
        tools: tuple[str, ...] = (),
        journal: Journal | None = None,
        embedding_dir: Path | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve(strict=True)
        if journal and self.project_root != journal.project_root:
            raise ValueError("journal project root differs from session")
        if journal and not set(tools).issubset(journal.authorized_tools):
            raise ValueError("tools exceed persisted session authority")
        if set(tools) & set(PLANNING_TOOLS) and journal is None:
            raise ValueError("planning requires a persisted journal")
        self.tools = tools
        self.embedding_dir = embedding_dir
        self.journal = journal
        self.workspace = Workspace(self.project_root)
        self._messages: list[JsonObject] = []
        self._unresolved_effect = False
        self._child: ChildBinding | None = None
        self._delegation: DelegationTable | None = None
        self.control = RunControl()

    @classmethod
    def open(
        cls, path: Path, *, tools: tuple[str, ...] | None = None, embedding_dir: Path | None = None
    ) -> "AgentSession":
        journal = Journal(path)
        return cls(
            journal.project_root,
            tools=journal.authorized_tools if tools is None else tools,
            journal=journal,
            embedding_dir=embedding_dir,
        )

    def submit(self, task: str, **limits: Unpack[RunLimits]) -> JsonObject:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must not be empty")
        return self._run(task=task, **limits)

    def _investigate_child(
        self,
        task: str,
        *,
        scope: tuple[str, ...],
        budget: RequestBudget,
        cancel: Event,
        on_event: Callable[[JsonObject], None] | None = None,
        max_turns: int = 3,
        max_tokens: int = 512,
    ) -> JsonObject:
        child = AgentSession(self.workspace.root, tools=("read",))
        return child.investigate(
            task,
            scope=scope,
            budget=budget,
            cancel=cancel,
            on_event=on_event,
            max_turns=max_turns,
            max_tokens=max_tokens,
        )

    def investigate(
        self,
        task: str,
        *,
        scope: tuple[str, ...],
        budget: RequestBudget,
        cancel: Event,
        on_event: Callable[[JsonObject], None] | None = None,
        max_turns: int = 3,
        max_tokens: int = 512,
    ) -> JsonObject:
        """Read-only child run. It does not load MCP or extensions and has no task journal."""
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must not be empty")
        if self.journal is not None:
            raise ValueError("child investigation cannot own a task journal")
        if self.tools != ("read",):
            raise ValueError("child investigation tools must be read")
        if not scope or any(not path for path in scope):
            raise ValueError("child scope must be a non-empty tuple of relative paths")
        if max_turns < 1 or max_tokens < 1:
            raise ValueError("child limits must be positive")
        scope = tuple("/".join(self.workspace.parts(path)) for path in scope)
        previous_control, previous_messages = self.control, self._messages
        self.control, self._messages = RunControl(), []
        self.control.adopt(cancel)
        previous = self._child
        self._child = ChildBinding(budget, scope)
        try:
            return self._run(task=task, max_turns=max_turns, max_tokens=max_tokens, on_event=on_event)
        finally:
            self._child = previous
            self.control, self._messages = previous_control, previous_messages

    def interrupt(
        self, task: str, *, max_tokens: int = 256, on_event: Callable[[JsonObject], None] | None = None
    ) -> JsonObject:
        return self._run(task=task, max_tokens=max_tokens, interrupt_after_chunks=2, on_event=on_event)

    def resume(self, *, leaf_id: str | None = None, **limits: Unpack[RunLimits]) -> JsonObject:
        if self.journal is None:
            raise ValueError("resume requires a persisted journal")
        return self._run(task=None, leaf_id=leaf_id, **limits)

    def branch(self, parent_id: str, task: str) -> JsonObject:
        if self.journal is None:
            raise ValueError("branch requires a persisted journal")
        with locked(self.journal.lock_path, blocking=False):
            self.journal.reload()
            return self.journal.branch(parent_id, task)

    def steer(self, text: str, *, paths: tuple[str, ...] | None = None) -> ControlInput:
        if paths is not None:
            for path in paths:
                self.workspace.parts(path)
        return self.control.submit("steer", text, paths)

    def follow_up(self, text: str) -> ControlInput:
        return self.control.submit("follow_up", text, None)

    def cancel(self) -> JsonObject:
        return self.control.cancel_run()

    def return_pending(self) -> list[ControlInput]:
        """After a run ends, return unsent inputs to its caller for editing."""
        return self.control.return_pending()

    def decide(self, approval_id: str, decision: str) -> JsonObject:
        if self.journal is None:
            raise ValueError("approve requires a persisted journal")
        if decision not in {"approve", "reject", "revoke"}:
            raise ValueError("decision must be approve, reject, or revoke")
        with locked(self.journal.lock_path, blocking=False):
            self.journal.reload()
            book = self._book(approvals=False, approval_ttl=None, claim=False)
            try:
                return book.decide(approval_id, decision)
            finally:
                book.close()

    def inspect(self) -> JsonObject:
        """Inspect an idle session without starting tools, requests, or a task claim."""
        if self.journal is None:
            return {}
        with locked(self.journal.lock_path, blocking=False):
            self.journal.reload()
            return TaskBook.inspect(self.journal)

    def answer(
        self, question_id: str, answer: str, *, expected_version: int, cancel: bool = False
    ) -> JsonObject:
        if self.journal is None or "question" not in self.tools:
            raise ValueError("question_authority_required")
        if type(expected_version) is not int or expected_version < 0 or not isinstance(cancel, bool):
            raise ValueError("invalid_answer_version_or_cancel")
        if not isinstance(answer, str) or (not cancel and not answer.strip()):
            raise ValueError("answer_must_not_be_empty")
        with locked(self.journal.lock_path, blocking=False):
            self.journal.reload()
            if self.journal.incomplete_tail() is not None:
                raise ValueError("resume_required_before_answer")
            book = self._book(approvals=False, approval_ttl=None, claim=True)
            try:
                book.workflow.adopt_revision()
                book.workflow.refresh_inputs()
                question = book.workflow.question_for_answer(question_id, expected_version)
                original_answer = answer
                if not cancel and answer.isdecimal() and 1 <= int(answer) <= len(question.options):
                    answer = question.options[int(answer) - 1]
                self.journal.append(
                    "user",
                    {
                        "content": "[取消澄清]" if cancel else answer,
                        "question_id": question.id,
                        "question_version": expected_version,
                        "question_cancelled": cancel,
                        "answer_input": original_answer,
                    },
                )
                book.workflow.adopt_answer()
                return book.workflow.snapshot().payload()
            finally:
                book.close()

    def revise_plan(
        self,
        *,
        expected_version: int,
        reason: str,
        steps: list[JsonObject],
        invalidated_steps: tuple[str, ...],
    ) -> JsonObject:
        if self.journal is None or "plan" not in self.tools:
            raise ValueError("plan_authority_required")
        args: JsonObject = {
            "action": "revise",
            "expected_version": expected_version,
            "reason": reason,
            "steps": steps,
            "invalidated_steps": list(invalidated_steps),
        }
        with locked(self.journal.lock_path, blocking=False):
            self.journal.reload()
            book = self._book(approvals=False, approval_ttl=None, claim=True)
            try:
                book.workflow.validate_revision(args)
                book.prepare_plan_revision()
                self.journal.append(
                    "user", {"content": "Plan revision requested: " + reason, "plan_revision": args}
                )
                book.workflow.execute(args)
                return book.workflow.snapshot().payload()
            finally:
                book.close()

    def _memory_source(self, run_id: str) -> tuple[str, str]:
        if self.journal is not None:
            for record in reversed(self.journal.chain()):
                if record["kind"] == "user":
                    return f"journal:{self.journal.path}:{record['id']}", str(record["body"]["content"])
        else:
            for index in range(len(self._messages) - 1, -1, -1):
                message = self._messages[index]
                if message.get("role") == "user" and isinstance(message.get("content"), str):
                    return f"run:{run_id}:message:{index}", message["content"]
        return run_id, ""

    def rebuild_index(self, *, mode: Mode = "bm25") -> JsonObject:
        database = KnowledgeDB.open(self.project_root)
        index = CorpusIndex(
            database, self.workspace, embedding_dir=self.embedding_dir, stopped=self.control.cancel.is_set
        )
        try:
            return index.rebuild(mode=mode)
        finally:
            index.close()
            database.close()

    def knowledge(self) -> JsonObject:
        """Inspect persisted retrieval provenance without loading an embedding runtime."""
        database = KnowledgeDB.open(self.project_root)
        try:
            index = CorpusIndex(database, self.workspace)
            return {"generation_id": index.current_generation(), "queries": index.queries()}
        finally:
            database.close()

    def confirm_memory(
        self, memory_id: str, *, supersedes: str | None = None, ttl_seconds: float | None = None
    ) -> JsonObject:
        return self._memory(
            lambda ledger: ledger.confirm(memory_id, supersedes=supersedes, ttl_seconds=ttl_seconds)
        )

    def revoke_memory(self, memory_id: str) -> JsonObject:
        return self._memory(lambda ledger: ledger.revoke(memory_id))

    def resolve_memory(self, memory_id: str) -> JsonObject:
        return self._memory(lambda ledger: ledger.resolve(memory_id))

    def _memory(self, operation: Callable[[MemoryLedger], JsonObject]) -> JsonObject:
        database = KnowledgeDB.open(self.project_root)
        try:
            return operation(MemoryLedger(database))
        finally:
            database.close()

    def dismiss(self, worker_id: str) -> JsonObject:
        if self.journal is None:
            raise ValueError("queue control requires a persisted journal")
        with locked(self.journal.lock_path, blocking=False):
            book = self._book(approvals=False, approval_ttl=None, claim=False)
            try:
                return book.gate.dismiss(worker_id)
            finally:
                book.close()

    def _book(
        self,
        *,
        approvals: bool,
        approval_ttl: float | None,
        claim: bool,
        max_task_calls: int | None = None,
        max_task_output_tokens: int | None = None,
        max_task_seconds: float | None = None,
    ) -> TaskBook:
        if self.journal is None:
            raise ValueError("task state requires a persisted journal")
        return TaskBook(
            self.journal,
            self.workspace,
            tools=self.tools,
            approvals=approvals,
            max_calls=max_task_calls,
            max_output=max_task_output_tokens,
            max_seconds=max_task_seconds,
            approval_ttl=approval_ttl,
            claim=claim,
        )

    def _run(
        self, *, task: str | None, leaf_id: str | None = None, **limits: Unpack[RunLimits]
    ) -> JsonObject:
        settings = RunSettings(**limits)
        settings.validate()
        with locked(self.journal.lock_path, blocking=False) if self.journal else nullcontext():
            if self.journal:
                self.journal.reload()
                tail = self.journal.incomplete_tail(leaf_id)
                self.journal.messages(tail["parent_id"] if tail is not None else leaf_id)
                if leaf_id is not None:
                    self.journal.select(leaf_id, allow_incomplete=True)
            if self._delegation is not None:
                leftover = self._delegation
                self._delegation = None
                leftover.close()
            self.control.begin_run()
            child = self._child
            context: JsonObject = (
                {"instructions": {"adopted": None, "shadowed": []}, "skills": []}
                if child is not None
                else load_context(self.project_root)
            )
            extensions = (
                LoadedExtensions()
                if child is not None
                else load_extensions(self.project_root, self.workspace)
            )
            run_id = "run-" + uuid4().hex
            evidence = RUNS / run_id
            evidence.mkdir(parents=True)
            mcp = (
                McpHost([])
                if child is not None
                else McpHost.open(self.project_root, evidence, cancel_event=self.control.cancel)
            )
            trace = RunTrace(run_id, evidence, settings.on_event)
            outcome: RunOutcome = {"status": "error", "text": "", "reason": "run_incomplete"}
            book: TaskBook | None = None
            knowledge: KnowledgeDB | None = None
            index: CorpusIndex | None = None
            delegation: DelegationTable | None = None
            primary: BaseException | None = None
            try:
                extras: tuple[ToolDefinition, ...] = ()
                if self.journal:
                    book = self._book(
                        approvals=settings.approvals,
                        approval_ttl=settings.approval_ttl_seconds,
                        claim=True,
                        max_task_calls=settings.max_task_calls,
                        max_task_output_tokens=settings.max_task_output_tokens,
                        max_task_seconds=settings.max_task_seconds,
                    )
                    if set(self.tools).intersection(PLANNING_TOOLS):
                        book.workflow.adopt_revision()
                        book.workflow.refresh_inputs()
                        book.workflow.adopt_answer()
                parent_budget = (
                    book.budget if book is not None else child.budget if child is not None else None
                )
                request_budget = (
                    RunRequestBudget(parent_budget, settings.max_model_calls)
                    if parent_budget is not None
                    else None
                )
                if child is None and set(self.tools) & set(KNOWLEDGE_TOOLS):
                    knowledge = KnowledgeDB.open(self.project_root)
                    index = CorpusIndex(
                        knowledge,
                        self.workspace,
                        embedding_dir=self.embedding_dir,
                        stopped=lambda: (
                            self.control.cancel.is_set()
                            or (request_budget is not None and request_budget.stop_reason() == "time_budget")
                        ),
                        budget=request_budget,
                    )
                    extras = knowledge_tools(
                        knowledge,
                        self.workspace,
                        allowed=frozenset(self.tools),
                        run_id=run_id,
                        user_source=lambda: self._memory_source(run_id),
                        meter=RetrievalMeter(
                            4 if settings.max_retrievals is None else settings.max_retrievals
                        ),
                        index=index,
                    )
                if book is not None and "plan" in self.tools:
                    extras += (plan_tool(book.workflow),)
                if book is not None and "question" in self.tools:
                    extras += (question_tool(book.workflow),)
                allowed = (
                    self.tools
                    if child is not None
                    else tuple(name for name in self.tools if name != "delegate")
                )
                registry = ToolRegistry(
                    self.workspace,
                    allowed,
                    max_bytes=settings.max_bytes,
                    shell_timeout=settings.shell_timeout,
                    cancel_bash_after=settings.cancel_bash_after,
                    evidence=evidence,
                    cancel_event=self.control.cancel,
                    catalog=() if child is not None else tuple(context["skills"]),
                    extras=extras,
                )
                if child is not None:
                    registry.read_scope = child.scope
                registry.attach(extensions, mcp)
                if child is None and "delegate" in self.tools:
                    delegation = DelegationTable(
                        self.workspace,
                        self.control.cancel,
                        trace.emit,
                        run_id,
                        investigate=self._investigate_child,
                        can_read=lambda: "read" in registry.allowed,
                        parent_scope=lambda: registry.scope,
                    )
                    self._delegation = delegation
                    registry.install(delegate_tool(delegation))
                (evidence / "task.txt").write_text(task or "")
                halt: RunOutcome | None = None
                if self.journal is not None:
                    if book is None:
                        raise RuntimeError("journal run has no task book")
                    halt = self._admit_task(task, book, registry, delegation, request_budget)
                    messages = self.journal.messages() if halt is None else []
                else:
                    messages = self._messages
                    if self._unresolved_effect:
                        halt = run_halt(
                            "effect_unknown", "effect_unverified", "Previous action remains unresolved."
                        )
                    elif task is not None:
                        messages.append({"role": "user", "content": task})
                outcome = halt or run(
                    messages,
                    registry=registry,
                    trace=trace,
                    journal=self.journal,
                    max_turns=settings.max_turns,
                    max_tokens=settings.max_tokens,
                    max_model_calls=settings.max_model_calls,
                    interrupt_after_chunks=settings.interrupt_after_chunks,
                    control=self.control,
                    context=_with_memory(context, knowledge, run_id),
                    context_tokens=settings.context_tokens,
                    book=book,
                    budget=request_budget,
                )
            except BaseException as error:
                # sys.exception() in finally would also see a caller's handled exception.
                primary = error
                outcome.update({"status": "error", "reason": f"{type(error).__name__}: {error}"})
                raise
            finally:
                result = self._finish_run(
                    outcome, trace, primary, extensions, mcp, delegation, index, knowledge, book
                )
            return result

    def _admit_task(
        self,
        task: str | None,
        book: TaskBook,
        registry: ToolRegistry,
        delegation: DelegationTable | None,
        request_budget: RequestBudget | None,
    ) -> RunOutcome | None:
        journal = self.journal
        if journal is None:
            raise RuntimeError("task admission requires a journal")
        if delegation is not None:
            delegation.bind(request_budget if request_budget is not None else book.budget)
        steered = self.control.queued_steer()
        if steered is not None and steered["paths"] is not None:
            book.note_scope(steered["paths"])
        self.control.restore_scope(book.scope)
        registry.scope = book.scope
        mode = "read" if all(tool.read_only for tool in registry.allowed.values()) else "write"
        admission = book.gate.acquire(mode)
        if admission["state"] != "running":
            return run_halt(
                "queued",
                str(admission.get("reason") or "queued"),
                f"Task is {admission['state']}: {admission.get('reason')}",
            )
        tail = journal.incomplete_tail()
        halt = book.recover(registry, tail) if tail is not None else None
        if halt is None:
            halt = book.blocking()
        if halt is None and task is None:
            halt = book.cancelled_question()
        if halt is None and task is not None:
            journal.append("user", {"content": task})
        return halt

    def _finish_run(
        self,
        outcome: RunOutcome,
        trace: RunTrace,
        primary: BaseException | None,
        extensions: LoadedExtensions,
        mcp: McpHost,
        delegation: DelegationTable | None,
        index: CorpusIndex | None,
        knowledge: KnowledgeDB | None,
        book: TaskBook | None,
    ) -> JsonObject:
        def finish_book() -> None:
            if book is not None:
                outcome["tasks"] = book.finish()

        def finish_delegation() -> None:
            if delegation is not None:
                delegation.close()
                self._delegation = None

        resources: list[tuple[str, Callable[[], object]]] = [
            ("delegate", finish_delegation),
            ("extension", lambda: extensions.bind(None)),
            ("mcp", mcp.close),
        ]
        if index is not None:
            resources.append(("index", index.close))
        if knowledge is not None:
            resources.append(("knowledge", knowledge.close))
        resources.append(("task", finish_book))
        failures = close_resources(resources)
        if failures:
            outcome["status"] = "effect_unknown"
            outcome["cleanup_errors"] = [failure.diagnostic() for failure in failures]
            if primary is None:
                outcome["reason"] = outcome.get("reason") or failures[0].diagnostic()
                primary = failures[0].error
            for failure in failures:
                if failure.error is not primary:
                    primary.add_note(failure.diagnostic())
        if trace.unresolved_effect() or any(
            server.get("pending_action") for server in mcp.report()["servers"]
        ):
            outcome["status"] = "effect_unknown"
            outcome["reason"] = outcome.get("reason") or "effect_unverified"
        if self.journal is None and outcome["status"] == "effect_unknown":
            self._unresolved_effect = True
        try:
            result = trace.finish(outcome, mcp.report())
        except BaseException as evidence_error:
            if primary is None:
                raise
            primary.add_note(f"final_evidence_failed:{type(evidence_error).__name__}: {evidence_error}")
            raise primary from evidence_error
        if failures:
            raise primary if primary is not None else failures[0].error
        return result
