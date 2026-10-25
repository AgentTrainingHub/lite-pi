"""One owner for bounded child investigations, cancellation and event delivery."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import replace
from queue import Empty, SimpleQueue
from threading import Event, Lock, Thread
from typing import Protocol

from lite_pi.delegation_evidence import file_hashes, file_kind, file_proofs, unsupported_claims
from lite_pi.delegation_report import (
    ChildReport,
    Survey,
    call_count,
    cancelled_report,
    child_status,
    failure_result,
    make_report,
    now_iso,
    rejected_report,
    render_report,
    tool_refusals,
)
from lite_pi.task_budget import RequestBudget
from lite_pi.tools.contracts import JsonObject, ToolDefinition, obj
from lite_pi.tools.workspace import Workspace

MAX_SURVEYS = 2
MAX_PATHS = 4
CHILD_TURNS = 3
CHILD_TOKENS = 512
_CLAIM = re.compile(r"已完成|已修改|已经改好|已经修复|have fixed|have changed|completed the edit", re.I)


class ChildInvestigator(Protocol):
    def __call__(
        self,
        task: str,
        *,
        scope: tuple[str, ...],
        budget: RequestBudget,
        cancel: Event,
        on_event: Callable[[JsonObject], None] | None = None,
        max_turns: int = 3,
        max_tokens: int = 512,
    ) -> JsonObject: ...


class DelegationTable:
    """One parent run's surveys. A sealed generation drops late reports."""

    def __init__(
        self,
        workspace: Workspace,
        cancel: Event,
        emit: Callable[[JsonObject], None],
        parent_run_id: str,
        *,
        investigate: ChildInvestigator,
        can_read: Callable[[], bool],
        parent_scope: Callable[[], tuple[str, ...] | None],
    ) -> None:
        self.investigate = investigate
        self.can_read = can_read
        self.parent_scope = parent_scope
        self._events: SimpleQueue[JsonObject] = SimpleQueue()
        self.workspace = workspace
        self.cancel = cancel
        self.emit = emit
        self.parent_run_id = parent_run_id
        self.budget: RequestBudget | None = None
        self.generation = 0
        self.sealed: int | None = None
        self.closed = False
        self.peak = 0
        self.receipts: dict[str, ChildReport] = {}
        self.dropped: list[str] = []
        self.threads: list[Thread] = []
        self._active = 0
        self._lock = Lock()

    def bind(self, budget: RequestBudget) -> None:
        self.budget = budget

    def open_generation(self) -> int:
        with self._lock:
            if self.closed:
                raise RuntimeError("delegation_closed")
            self.generation += 1
            self.sealed = None
            self.receipts = {}
            self.dropped = []
            self.peak = 0
            return self.generation

    def seal(self, generation: int) -> None:
        with self._lock:
            if self.generation == generation:
                self.sealed = generation

    def accept(self, report: ChildReport) -> bool:
        with self._lock:
            if self.closed or report.generation != self.generation or self.sealed == report.generation:
                self.dropped.append(report.survey_id)
                return False
            self.receipts[report.survey_id] = report
            return True

    def close(self, *, timeout: float = 30) -> None:
        with self._lock:
            self.closed = True
            self.sealed = self.generation
            threads = list(self.threads)
        if any(thread.is_alive() for thread in threads):
            self.cancel.set()
        alive: list[str] = []
        for thread in threads:
            thread.join(timeout)
            if thread.is_alive():
                alive.append(thread.name)
        if alive:
            raise RuntimeError("delegate_cleanup_failed:" + ",".join(alive))

    def execute(self, args: JsonObject) -> JsonObject:
        if self.closed:
            return failure_result("Delegation is closed.", "delegation_closed")
        if self.budget is None:
            return failure_result("Delegation requires the parent task budget.", "shared_budget_required")
        if self.cancel.is_set():
            generation = self.open_generation()
            self.seal(generation)
            return cancelled_report(generation, self.parent_run_id)
        if not self.can_read():
            return failure_result("Parent has not authorized file reads.", "parent_read_required")
        raw = args.get("surveys")
        if not isinstance(raw, list) or not raw or len(raw) > MAX_SURVEYS:
            return failure_result("Delegation needs one or two surveys.", "invalid_surveys")
        generation = self.open_generation()
        try:
            prepared: list[Survey | ChildReport] = [
                self._prepare(index, generation, item) for index, item in enumerate(raw)
            ]
            launch = [(index, item) for index, item in enumerate(prepared) if isinstance(item, Survey)]
            before = {
                path: digest
                for item in prepared
                if isinstance(item, Survey)
                for path, digest in item.hashes.items()
            }
            launched = self._map(launch, generation)
            reports: list[ChildReport] = []
            for index, item in enumerate(prepared):
                reports.append(launched[index] if isinstance(item, Survey) else item)
            proofs = file_proofs(self.workspace, _paths(prepared), before)
            wrote = any(
                (proof.sha256 is not None and proof.sha256 != proof.before_sha256)
                or (not proof.exists and proof.error == "path_missing")
                for proof in proofs
            )
            self.seal(generation)
            return render_report(
                generation,
                reports,
                proofs,
                child_write_detected=wrote,
                joined=True,
                parent_run_id=self.parent_run_id,
                dropped=tuple(self.dropped),
                peak=self.peak,
                budget=self.budget.snapshot(),
            )
        except Exception:
            self.seal(generation)
            raise

    def _prepare(self, index: int, generation: int, item: object) -> Survey | ChildReport:
        survey_id = f"{generation}:{index}"
        if not isinstance(item, dict):
            return rejected_report(survey_id, generation, self.parent_run_id, (), "invalid_surveys")
        task = item.get("task")
        paths = item.get("paths")
        if not isinstance(task, str) or not task.strip():
            return rejected_report(survey_id, generation, self.parent_run_id, (), "invalid_surveys")
        if (
            not isinstance(paths, list)
            or not paths
            or len(paths) > MAX_PATHS
            or any(not isinstance(path, str) or not path for path in paths)
        ):
            return rejected_report(survey_id, generation, self.parent_run_id, (), "invalid_surveys")
        normalized: list[str] = []
        for path in paths:
            try:
                normalized.append("/".join(self.workspace.parts(path)))
            except (ValueError, OSError):
                return rejected_report(survey_id, generation, self.parent_run_id, (), "path_scope")
        scope = tuple(normalized)
        parent_scope = self.parent_scope()
        if parent_scope is not None and any(
            not any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in parent_scope)
            for path in scope
        ):
            return rejected_report(survey_id, generation, self.parent_run_id, (), "path_scope")
        missing = [path for path in scope if file_kind(self.workspace, path) == "missing"]
        if missing:
            return rejected_report(survey_id, generation, self.parent_run_id, scope, "path_missing")
        if any(file_kind(self.workspace, path) != "file" for path in scope):
            return rejected_report(survey_id, generation, self.parent_run_id, scope, "file_paths_required")
        try:
            hashes = file_hashes(self.workspace, scope)
        except (OSError, ValueError) as error:
            return rejected_report(survey_id, generation, self.parent_run_id, scope, f"unreadable:{error}")
        return Survey(task.strip(), scope, hashes)

    def _map(self, launch: list[tuple[int, Survey]], generation: int) -> dict[int, ChildReport]:
        if not launch:
            return {}
        results: dict[int, ChildReport] = {}
        next_index = 0
        index_lock = Lock()

        def worker() -> None:
            nonlocal next_index
            while True:
                with index_lock:
                    current = next_index
                    next_index += 1
                if current >= len(launch):
                    return
                index, survey = launch[current]
                results[index] = self._run_survey(index, survey, generation)

        count = min(MAX_SURVEYS, len(launch))
        threads = [Thread(target=worker, name=f"delegate-{generation}-{slot}") for slot in range(count)]
        with self._lock:
            self.threads.extend(threads)
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads):
            for thread in threads:
                thread.join(0.05)
            self._drain_events()
        self._drain_events()
        alive = [thread.name for thread in threads if thread.is_alive()]
        if alive:
            raise RuntimeError("delegate_cleanup_failed:" + ",".join(alive))
        if len(results) != len(launch):
            raise RuntimeError("delegate_missing_report")
        return results

    def _run_survey(self, index: int, survey: Survey, generation: int) -> ChildReport:
        survey_id = f"{generation}:{index}"
        started = time.monotonic()
        started_at = now_iso()
        self._enter()
        try:
            if self.cancel.is_set() or self.closed:
                report = make_report(
                    survey_id=survey_id,
                    generation=generation,
                    parent_run_id=self.parent_run_id,
                    paths=survey.paths,
                    status="aborted",
                    reason="cancelled",
                    narrative="",
                    run_id=None,
                    tools=(),
                    refusals=(),
                    started=started,
                    started_at=started_at,
                )
            else:
                report = self._investigate(survey_id, survey, generation, started, started_at)
            if self.cancel.is_set() and report.status != "rejected":
                report = replace(report, status="aborted", reason="cancelled")
            report = replace(report, kept=self.accept(report))
            return report
        except Exception as error:
            report = make_report(
                survey_id=survey_id,
                generation=generation,
                parent_run_id=self.parent_run_id,
                paths=survey.paths,
                status="error",
                reason=f"{type(error).__name__}: {error}",
                narrative="",
                run_id=None,
                tools=(),
                refusals=(),
                started=started,
                started_at=started_at,
            )
            report = replace(report, kept=self.accept(report))
            return report
        finally:
            self._leave()

    def _investigate(
        self,
        survey_id: str,
        survey: Survey,
        generation: int,
        started: float,
        started_at: str,
    ) -> ChildReport:
        budget = self.budget
        if budget is None:
            raise RuntimeError("shared_budget_required")

        def on_event(event: JsonObject) -> None:
            child_run = event.get("run_id")
            child_type = event.get("type")
            self._events.put(
                {
                    "type": "delegate_child",
                    "survey_id": survey_id,
                    "child_run_id": child_run if isinstance(child_run, str) else None,
                    "child_event": child_type if isinstance(child_type, str) else "",
                }
            )

        outcome = self.investigate(
            _child_task(self.parent_run_id, survey),
            scope=survey.paths,
            budget=budget,
            cancel=self.cancel,
            on_event=on_event,
            max_turns=CHILD_TURNS,
            max_tokens=CHILD_TOKENS,
        )
        status = child_status(outcome.get("status"))
        raw_reason = outcome.get("reason")
        raw_text = outcome.get("text")
        narrative = raw_text if isinstance(raw_text, str) else ""
        raw_run = outcome.get("run_id")
        run_id = raw_run if isinstance(raw_run, str) else None
        raw_tools = outcome.get("tools")
        tools = (
            tuple(item for item in raw_tools if isinstance(item, str)) if isinstance(raw_tools, list) else ()
        )
        return make_report(
            survey_id=survey_id,
            generation=generation,
            parent_run_id=self.parent_run_id,
            paths=survey.paths,
            status=status,
            reason=raw_reason if isinstance(raw_reason, str) else None,
            narrative=narrative,
            run_id=run_id,
            tools=tools,
            refusals=tool_refusals(outcome),
            started=started,
            started_at=started_at,
            model_calls=call_count(outcome.get("request_count")),
            unsupported=unsupported_claims(narrative, self.workspace, survey.paths),
            completion_claim="unsupported" if _CLAIM.search(narrative) else "none",
        )

    def _drain_events(self) -> None:
        while True:
            try:
                event = self._events.get_nowait()
            except Empty:
                return
            self.emit(event)

    def _enter(self) -> None:
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)

    def _leave(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)


def delegate_tool(table: DelegationTable) -> ToolDefinition:
    survey = obj(
        {
            "task": {"type": "string", "minLength": 1},
            "paths": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_PATHS,
                "items": {"type": "string", "minLength": 1},
            },
        },
        ["task", "paths"],
    )
    schema = obj(
        {"surveys": {"type": "array", "minItems": 1, "maxItems": MAX_SURVEYS, "items": survey}},
        ["surveys"],
    )
    return ToolDefinition(
        "delegate",
        "Run one or two read-only investigations. Host re-reads the paths and returns sha256 excerpts. "
        "Child text is not evidence and cannot prove a file changed. Children share this session's task "
        "budget and cannot write, use shell, MCP, or extensions.",
        schema,
        table.execute,
        "unsafe",
        read_only=False,
    )


def _child_task(parent_run_id: str, survey: Survey) -> str:
    paths = "、".join(survey.paths)
    return (
        f"父任务 {parent_run_id} 委托一次只读调查。允许路径：{paths}。"
        f"任务：{survey.task} "
        "请阅读文件，每个文件最多报告两条事实，用 FACT 路径: 原文片段 标注。报告总长不超过 150 字。不要声称文件已经被修改。"
    )


def _paths(prepared: list[Survey | ChildReport]) -> tuple[str, ...]:
    found: list[str] = []
    for item in prepared:
        paths = item.paths if isinstance(item, Survey) else ()
        for path in paths:
            if path not in found:
                found.append(path)
    return tuple(found)
