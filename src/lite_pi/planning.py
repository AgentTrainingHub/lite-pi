"""Branch-local, versioned work descriptions in the task database."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from graphlib import TopologicalSorter
from uuid import uuid4

from jsonschema import Draft202012Validator

from lite_pi.clarification import Question
from lite_pi.journal import Journal
from lite_pi.recovery import canonical
from lite_pi.task_storage import assert_owner, transaction
from lite_pi.tools.contracts import JsonObject, ToolDefinition, obj
from lite_pi.tools.workspace import Workspace

PLANNING_TOOLS = ("plan", "question")
STRING = {"type": "string", "minLength": 1, "maxLength": 500}
STEP_SCHEMA = obj(
    {
        "id": {**STRING, "maxLength": 64},
        "title": STRING,
        "depends_on": {"type": "array", "items": {**STRING, "maxLength": 64}, "uniqueItems": True},
        "deliverable": STRING,
        "inputs": {
            "type": "array",
            "items": STRING,
            "uniqueItems": True,
            "description": "Project-relative file paths only. Put step IDs in depends_on, not inputs.",
        },
    },
    ["id", "title", "depends_on", "deliverable", "inputs"],
)


@dataclass
class EvidenceCheck:
    intent_id: str
    name: str
    passed: bool
    reason: str


@dataclass(frozen=True)
class EvidenceReference:
    intent_id: str
    name: str
    target: str | None
    requirement_version: int
    passed: bool
    input_paths: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceCatalog:
    entries: tuple[EvidenceReference, ...]
    truncated: bool


@dataclass
class PlanStep:
    id: str
    title: str
    depends_on: list[str]
    deliverable: str
    inputs: list[str]
    input_versions: dict[str, str | None] = field(default_factory=dict)
    claim: str | None = None
    checks: list[EvidenceCheck] = field(default_factory=list)
    valid: bool = True
    invalid_reason: str | None = None
    evidence_epoch: int = 0

    @property
    def verified(self) -> bool:
        return self.valid and bool(self.checks) and all(check.passed for check in self.checks)


@dataclass
class Plan:
    id: str
    revision: int
    reason: str
    steps: list[PlanStep]


@dataclass
class WorkflowState:
    version: int = 0
    requirement_version: int = 0
    plan: Plan | None = None
    question: Question | None = None

    def payload(self) -> JsonObject:
        return asdict(self)

    @classmethod
    def decode(cls, payload: JsonObject) -> WorkflowState:
        plan = payload["plan"]
        return cls(
            version=payload["version"],
            requirement_version=payload["requirement_version"],
            plan=Plan(
                **{
                    **plan,
                    "steps": [
                        PlanStep(**{**step, "checks": [EvidenceCheck(**c) for c in step["checks"]]})
                        for step in plan["steps"]
                    ],
                }
            )
            if plan is not None
            else None,
            question=Question(**payload["question"]) if payload["question"] is not None else None,
        )


class WorkflowStore:
    def __init__(
        self,
        db: sqlite3.Connection,
        journal: Journal,
        workspace: Workspace,
        owner_version: int,
        on_requirements_changed: Callable[[], None],
    ) -> None:
        self.db, self.journal, self.workspace = db, journal, workspace
        self.session_id = str(journal.records[0]["id"])
        self.owner_version = owner_version
        self._on_requirements_changed = on_requirements_changed

    @staticmethod
    def inspect(db: sqlite3.Connection, journal: Journal) -> WorkflowState:
        if (
            db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='workflows'").fetchone()
            is None
        ):
            return WorkflowState()
        ancestors = {r["id"] for r in journal.chain()}
        for row in db.execute(
            "SELECT anchor_id, state_json FROM workflows WHERE session_id=? ORDER BY rowid DESC",
            (str(journal.records[0]["id"]),),
        ):
            if row["anchor_id"] in ancestors:
                payload = json.loads(row["state_json"])
                if not isinstance(payload, dict):
                    raise ValueError("invalid_workflow_state")
                state = WorkflowState.decode(payload)
                if state.question and state.question.branch_id != _branch_id(journal):
                    state.question = None
                return state
        return WorkflowState()

    def snapshot(self) -> WorkflowState:
        return self.inspect(self.db, self.journal)

    def available_evidence(self) -> EvidenceCatalog:
        ancestors = {record["id"] for record in self.journal.chain()}
        references: list[EvidenceReference] = []
        seen: set[tuple[str, str | None]] = set()
        state = self.snapshot()
        required = (
            min(
                (s.evidence_epoch for s in state.plan.steps if not s.verified),
                default=state.requirement_version,
            )
            if state.plan
            else 0
        )
        for row in self.db.execute(
            "SELECT * FROM intents WHERE session_id=? AND status='settled' AND result_json IS NOT NULL ORDER BY rowid DESC",
            (self.session_id,),
        ):
            if row["assistant_id"] not in ancestors or row["name"] in PLANNING_TOOLS:
                continue
            if int(row["requirement_version"]) < required:
                continue
            result = json.loads(row["result_json"])
            arguments = json.loads(row["arguments"])
            target = arguments.get("path") or arguments.get("command")
            key = (str(row["name"]), target if isinstance(target, str) else None)
            if key in seen:
                continue
            seen.add(key)
            if len(references) == 12:
                return EvidenceCatalog(tuple(references), True)
            passed = result.get("executed") is True and not result.get("is_error")
            if row["name"] == "bash":
                passed = passed and result.get("exit_code") == 0
            references.append(
                EvidenceReference(
                    row["intent_id"],
                    row["name"],
                    target[:160] if isinstance(target, str) else None,
                    int(row["requirement_version"]),
                    passed,
                    tuple(sorted((result.get("file_versions") or {}).keys())),
                )
            )
        return EvidenceCatalog(tuple(references), False)

    def pending_question(self) -> Question | None:
        state = self.snapshot()
        question = state.question
        return question if question and question.status == "pending" else None

    def question_for_answer(self, identity: str, version: int) -> Question:
        state = self.snapshot()
        question = self.pending_question()
        if question is None or question.id != identity:
            raise ValueError("question_not_pending_on_branch")
        if state.version != version or state.requirement_version != question.requirement_version:
            raise ValueError("stale_question_version")
        return question

    def adopt_answer(self) -> None:
        question = self.pending_question()
        if question is None:
            return
        for record in reversed(self.journal.chain()):
            body = record["body"]
            if record["kind"] == "user" and body.get("question_id") == question.id:
                self.execute(
                    {
                        "action": "answer",
                        "expected_version": body["question_version"],
                        "question_id": question.id,
                        "answer": body["content"],
                        "cancel": body["question_cancelled"],
                        "answer_record_id": record["id"],
                    }
                )
                return

    def adopt_revision(self) -> None:
        for record in reversed(self.journal.chain()):
            if record["kind"] == "user" and "plan_revision" in record["body"]:
                args = record["body"]["plan_revision"]
                request = canonical(args)
                adopted = self.db.execute(
                    "SELECT 1 FROM workflows WHERE session_id=? AND anchor_id=? AND request_json=?",
                    (self.session_id, record["id"], request),
                ).fetchone()
                if adopted is None:
                    if self.journal.leaf_id() != record["id"]:
                        raise ValueError("unadopted_plan_revision_before_later_events")
                    self.execute(args)
                return

    def validate_revision(self, args: JsonObject) -> None:
        errors = list(Draft202012Validator(plan_tool(self).schema).iter_errors(args))
        if errors or args.get("action") != "revise":
            raise ValueError("invalid_plan_revision:" + (errors[0].message if errors else "action"))
        state = self.snapshot()
        if state.version != args["expected_version"]:
            raise ValueError(
                f"stale_plan_version:expected={state.version},received={args['expected_version']}"
            )
        old = _plan(state)
        proposed = self._steps(args["steps"])
        if set(args["invalidated_steps"]) - {s.id for s in old.steps} - {s.id for s in proposed}:
            raise ValueError("unknown_plan_step")

    def refresh_inputs(self) -> None:
        state = self.snapshot()
        if state.plan is None:
            return
        plan = _plan(state)
        affected = self._invalid_inputs(plan)
        if affected:
            self.execute(
                {
                    "action": "revise",
                    "expected_version": state.version,
                    "reason": "Input evidence changed: " + canonical(affected),
                    "steps": [_step_definition(s) for s in plan.steps],
                    "invalidated_steps": list(affected),
                }
            )

    def _invalid_inputs(self, plan: Plan) -> dict[str, str]:
        observed: dict[str, tuple[str | None, str | None]] = {}
        affected = {}
        for step in plan.steps:
            if not step.valid:
                continue
            for path, version in step.input_versions.items():
                if version is None:
                    if step.claim is not None:
                        affected[step.id] = f"input_not_observed:{path}"
                    continue
                if path not in observed:
                    try:
                        data = self.workspace.read(path, remember=False)
                        observed[path] = (hashlib.sha256(data).hexdigest(), None)
                    except (OSError, ValueError) as error:
                        observed[path] = (None, f"{type(error).__name__}:{error}")
                actual, diagnostic = observed[path]
                if actual != version:
                    affected[step.id] = f"{path}:{diagnostic or 'version_changed'}"
        return affected

    def execute(self, args: JsonObject) -> JsonObject:
        anchor = self.journal.leaf_id()
        request = canonical(args)
        with transaction(self.db):
            assert_owner(self.db, self.session_id, self.owner_version)
            previous = self.db.execute(
                "SELECT state_json FROM workflows WHERE session_id=? AND anchor_id=? AND request_json=?",
                (self.session_id, anchor, request),
            ).fetchone()
            if previous is not None:
                state = WorkflowState.decode(json.loads(previous["state_json"]))
            else:
                state = self.snapshot()
                if args["expected_version"] != state.version:
                    raise ValueError(
                        f"stale_plan_version:expected={state.version},received={args['expected_version']}"
                    )
                action = args["action"]
                if action == "create":
                    self._create(state, args)
                elif action == "revise":
                    self._revise(state, args)
                elif action == "record":
                    self._record(state, args)
                elif action == "ask":
                    self._ask(state, args)
                elif action == "answer":
                    self._answer(state, args)
                else:
                    raise ValueError("unknown_plan_action")
                state.version += 1
                self.db.execute(
                    "INSERT INTO workflows(event_id,session_id,anchor_id,request_json,state_json) VALUES(?,?,?,?,?)",
                    (uuid4().hex, self.session_id, anchor, request, canonical(state.payload())),
                )
        return {"text": canonical(state.payload()), "workflow": state.payload()}

    def _steps(self, definitions: list[JsonObject]) -> list[PlanStep]:
        steps = [PlanStep(**item) for item in definitions]
        identities = {step.id for step in steps}
        if len(identities) != len(steps):
            raise ValueError("duplicate_plan_step")
        if any(dep not in identities for step in steps for dep in step.depends_on):
            raise ValueError("unknown_plan_dependency")
        tuple(TopologicalSorter({s.id: s.depends_on for s in steps}).static_order())
        for step in steps:
            for path in step.inputs:
                normalized = "/".join(self.workspace.parts(path))
                step.input_versions[path] = self.workspace.observed.get(normalized.casefold())
        return steps

    def _create(self, state: WorkflowState, args: JsonObject) -> None:
        if state.plan is not None:
            raise ValueError("plan_already_exists")
        state.requirement_version += 1
        state.plan = Plan(uuid4().hex, 1, args["reason"], self._steps(args["steps"]))

    def _revise(self, state: WorkflowState, args: JsonObject) -> None:
        old = _plan(state)
        proposed = self._steps(args["steps"])
        previous = {s.id: s for s in old.steps}
        current = {s.id: s for s in proposed}
        affected = set(args["invalidated_steps"])
        if affected - previous.keys() - current.keys():
            raise ValueError("unknown_plan_step")
        for identity in previous.keys() | current.keys():
            if (
                identity not in previous
                or identity not in current
                or _definition(previous[identity]) != _definition(current[identity])
            ):
                affected.add(identity)
        affected = _dependants([*old.steps, *proposed], affected)
        for step in proposed:
            if step.id in previous:
                before = previous[step.id]
                step.claim, step.checks = before.claim, before.checks
                step.input_versions = {p: before.input_versions.get(p) for p in step.inputs}
                step.valid, step.invalid_reason = before.valid, before.invalid_reason
                step.evidence_epoch = before.evidence_epoch
            if step.id in affected:
                step.valid, step.invalid_reason = False, args["reason"]
                step.evidence_epoch = state.requirement_version + 1
        state.requirement_version += 1
        state.plan = Plan(old.id, old.revision + 1, args["reason"], proposed)
        if state.question is not None:
            state.question.status = "superseded"
        self._on_requirements_changed()

    def _ask(self, state: WorkflowState, args: JsonObject) -> None:
        if state.question and state.question.status == "pending":
            raise ValueError("question_already_pending")
        paths = ["/".join(self.workspace.parts(p)) for p in args["paths"]]
        if args["step_ids"]:
            plan = _plan(state)
            if set(args["step_ids"]) - {s.id for s in plan.steps}:
                raise ValueError("unknown_question_step")
        state.question = Question(
            uuid4().hex,
            args["text"],
            args["options"],
            paths,
            args["step_ids"],
            state.requirement_version,
            _branch_id(self.journal),
            self.journal.leaf_id(),
        )

    def _answer(self, state: WorkflowState, args: JsonObject) -> None:
        question = self.question_for_answer(args["question_id"], args["expected_version"])
        record = next((r for r in self.journal.chain() if r["id"] == args["answer_record_id"]), None)
        if record is None or record["kind"] != "user" or record["body"].get("question_id") != question.id:
            raise ValueError("answer_user_source_required")
        if (
            record["body"].get("question_version") != args["expected_version"]
            or record["body"].get("content") != args["answer"]
        ):
            raise ValueError("answer_source_mismatch")
        question.status = "cancelled" if args["cancel"] else "answered"
        question.answer = None if args["cancel"] else args["answer"]
        question.answer_record_id = record["id"]
        state.question = question

    def allows_read_while_waiting(self, path: str) -> bool:
        question = self.pending_question()
        state = self.snapshot()
        if question is None:
            return True
        if state.plan is None or not question.step_ids or not question.paths:
            return False
        normalized = "/".join(self.workspace.parts(path)).casefold()
        if any(
            normalized == p.casefold() or normalized.startswith(p.casefold() + "/") for p in question.paths
        ):
            return False
        plan = _plan(state)
        affected = _dependants(plan.steps, set(question.step_ids))
        return any(
            s.valid and s.id not in affected and normalized in {p.casefold() for p in s.inputs}
            for s in plan.steps
        )

    def _record(self, state: WorkflowState, args: JsonObject) -> None:
        plan = _plan(state)
        step = next((s for s in plan.steps if s.id == args["step_id"]), None)
        if step is None:
            raise ValueError("unknown_plan_step")
        ancestors = {r["id"] for r in self.journal.chain()}
        checks = []
        versions: dict[str, str] = {}
        for identity in args["evidence"]:
            row = self.db.execute(
                "SELECT * FROM intents WHERE intent_id=? AND session_id=?", (identity, self.session_id)
            ).fetchone()
            if (
                row is None
                or row["assistant_id"] not in ancestors
                or row["status"] != "settled"
                or row["result_json"] is None
            ):
                raise ValueError("evidence_not_settled_on_branch")
            if row["name"] in PLANNING_TOOLS:
                raise ValueError("workflow_metadata_is_not_execution_evidence")
            result = json.loads(row["result_json"])
            if result.get("requirement_version", 0) < step.evidence_epoch:
                raise ValueError(
                    f"evidence_predates_requirement:intent={identity},"
                    f"observed={result.get('requirement_version', 0)},required={step.evidence_epoch}; "
                    "read declared inputs again and use current read/check intent IDs"
                )
            passed = result.get("executed") is True and not result.get("is_error")
            if row["name"] == "bash":
                passed = passed and result.get("exit_code") == 0
            checks.append(
                EvidenceCheck(
                    identity, str(row["name"]), passed, "observed_success" if passed else "observed_failure"
                )
            )
            for path, digest in (result.get("file_versions") or {}).items():
                versions["/".join(self.workspace.parts(path)).casefold()] = digest
        step.claim = args["claim"]
        step.checks = checks
        for path in step.inputs:
            normalized = "/".join(self.workspace.parts(path)).casefold()
            step.input_versions[path] = versions.get(normalized)
        dependencies = [s for s in plan.steps if s.id in step.depends_on]
        step.valid = bool(checks) and all(c.passed for c in checks) and all(s.verified for s in dependencies)
        step.invalid_reason = None if step.valid else "failed_checks_or_unverified_dependency"
        invalid_inputs = self._invalid_inputs(plan)
        affected = _dependants(plan.steps, set(invalid_inputs) | ({step.id} if not step.valid else set()))
        for item in plan.steps:
            if item.id in affected:
                item.valid = False
                item.invalid_reason = invalid_inputs.get(item.id, "failed_checks_or_unverified_dependency")
        state.plan = plan


def _plan(state: WorkflowState) -> Plan:
    if state.plan is None:
        raise ValueError("plan_required")
    return state.plan


def _branch_id(journal: Journal) -> str:
    for record in reversed(journal.chain()):
        if record["kind"] == "user" and "branch_from" in record["body"]:
            return str(record["id"])
    return str(journal.records[0]["id"])


def _definition(step: PlanStep) -> tuple[str, str, list[str], list[str]]:
    return step.title, step.deliverable, step.depends_on, step.inputs


def _step_definition(step: PlanStep) -> JsonObject:
    return {
        "id": step.id,
        "title": step.title,
        "depends_on": step.depends_on,
        "deliverable": step.deliverable,
        "inputs": step.inputs,
    }


def _dependants(steps: list[PlanStep], affected: set[str]) -> set[str]:
    while True:
        expanded = affected | {s.id for s in steps if affected.intersection(s.depends_on)}
        if expanded == affected:
            return affected
        affected = expanded


def plan_tool(store: WorkflowStore) -> ToolDefinition:
    return ToolDefinition(
        "plan",
        "Maintain an optional persistent 3–5 step plan. create starts at expected_version=0; "
        "revise keeps stable IDs and invalidates the named steps and dependants; record stores a "
        "completion claim separately from host checks of existing intent_id evidence. "
        "Use exact intent_id values from host available_evidence; never invent them. Evidence must "
        "match the step evidence_epoch and explicitly cover EVERY declared input path, using "
        "host_receipt.observed_inputs or the candidate target. Include source reads as well as output "
        "receipts; a receipt for one file cannot prove another. If record is invalid, inspect the "
        "named missing/changed input, read it, then record with the new receipt ID. "
        "The available_evidence input_paths list says which files a receipt actually covers; "
        "a bash command mentioning a file does not establish a versioned file read. "
        "Use the current host workflow version. This grants no tools. "
        "Simple questions need no plan.",
        obj(
            {
                "action": {"type": "string", "enum": ["create", "revise", "record"]},
                "expected_version": {"type": "integer", "minimum": 0},
                "reason": STRING,
                "steps": {"type": "array", "items": STEP_SCHEMA, "minItems": 3, "maxItems": 5},
                "invalidated_steps": {"type": "array", "items": STRING, "uniqueItems": True},
                "step_id": STRING,
                "claim": STRING,
                "evidence": {"type": "array", "items": STRING, "minItems": 1, "uniqueItems": True},
            },
            ["action", "expected_version"],
            oneOf=[
                {"properties": {"action": {"const": "create"}}, "required": ["reason", "steps"]},
                {
                    "properties": {"action": {"const": "revise"}},
                    "required": ["reason", "steps", "invalidated_steps"],
                },
                {"properties": {"action": {"const": "record"}}, "required": ["step_id", "claim", "evidence"]},
            ],
        ),
        store.execute,
        "safe",
        read_only=False,
    )


def question_tool(store: WorkflowStore) -> ToolDefinition:
    return ToolDefinition(
        "question",
        "Ask one persistent clarification when the answer changes the next action. "
        "It pauses model requests until the user answers. Never ask for execution permission here: "
        "invoke the actual write/command tool and let host approval stop it before execution. "
        "An answer is not an approval. "
        "Provide affected project paths and plan step IDs (or empty lists without a plan). "
        "Options can be empty for free text; always use the current host workflow version.",
        obj(
            {
                "expected_version": {"type": "integer", "minimum": 0},
                "text": STRING,
                "options": {"type": "array", "items": STRING, "maxItems": 3, "uniqueItems": True},
                "paths": {"type": "array", "items": STRING, "uniqueItems": True},
                "step_ids": {"type": "array", "items": STRING, "uniqueItems": True},
            },
            ["expected_version", "text", "options", "paths", "step_ids"],
        ),
        lambda args: store.execute({"action": "ask", **args}),
        "safe",
        read_only=False,
    )
