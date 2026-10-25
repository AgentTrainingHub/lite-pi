"""SQLite task state for one session. The journal remains the session log."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import AbstractContextManager, ExitStack, closing
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

from lite_pi.approvals import ApprovalStore
from lite_pi.journal import Journal
from lite_pi.outcomes import run_status, tool_outcome
from lite_pi.planning import PLANNING_TOOLS, WorkflowStore
from lite_pi.project_gate import ProjectGate
from lite_pi.recovery import (
    EffectImage,
    EffectPreviewError,
    candidate_version,
    canonical,
    classify,
    constraint_token,
    effect_image,
    observe,
)
from lite_pi.run_trace import RunOutcome
from lite_pi.task_budget import RequestBudget, TaskBudget
from lite_pi.task_schema import migrate
from lite_pi.task_storage import StaleOwner as StaleOwner
from lite_pi.task_storage import assert_owner, connect, optional_row, transaction
from lite_pi.tools.contracts import (
    FileVersion,
    HaltAdmission,
    HaltOutcome,
    JsonObject,
    PreparedCall,
    RunAdmission,
)
from lite_pi.tools.workspace import Workspace

_TERMINAL = frozenset({"rejected", "revoked", "blocked", "superseded", "expired"})
_STICKY = frozenset({"effect_unknown", "awaiting_decision"})


class StoredTool(Protocol):
    def execute_stored(
        self,
        name: str,
        arguments: JsonObject,
        call_id: str,
        *,
        require_safe: bool,
        expected_file: FileVersion | None = None,
    ) -> JsonObject: ...

    def replay_policy(self, name: str) -> str | None: ...


class TaskPort(Protocol):
    @property
    def budget(self) -> RequestBudget: ...

    def note_scope(self, scope: tuple[str, ...] | None) -> None: ...

    def admit(self, prepared: PreparedCall, assistant_id: str) -> RunAdmission | HaltAdmission: ...

    def commit(self, intent_id: str, outcome: JsonObject) -> JsonObject: ...

    def context_messages(self) -> list[JsonObject]: ...

    def waiting_question(self) -> RunOutcome | None: ...

    def question_blocks(self, prepared: PreparedCall) -> bool: ...


def task_db_path(journal: Journal) -> Path:
    return journal.path.with_name(journal.path.name + ".tasks.sqlite")


def run_halt(status: str, reason: str, text: str) -> RunOutcome:
    """A halt has no model stop reason, but it still has the public result fields."""
    return {
        "status": run_status(status),
        "text": text,
        "reason": reason,
        "request_count": 0,
        "raw_stop_reason": None,
        "model": None,
        "usage": None,
        "usage_unknown": True,
        "checked_files": False,
    }


class TaskBook:
    """Owns the task database and the project concurrency gate for one run."""

    @staticmethod
    def inspect(journal: Journal) -> JsonObject:
        """Read durable state without claiming ownership or changing constraints."""
        path = task_db_path(journal)
        if not path.exists():
            return {}
        session_id = str(journal.records[0]["id"])
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN")
            approvals = [
                _public_approval(row)
                for row in db.execute("SELECT * FROM approvals WHERE session_id=?", (session_id,))
            ]
            intents = [
                _public_intent(row)
                for row in db.execute("SELECT * FROM intents WHERE session_id=?", (session_id,))
            ]
            return {
                "approvals": approvals,
                "intents": intents,
                "budget": TaskBudget(db, session_id).snapshot(),
                "workflow": WorkflowStore.inspect(db, journal).payload(),
            }

    def __init__(
        self,
        journal: Journal,
        workspace: Workspace,
        *,
        tools: tuple[str, ...],
        approvals: bool,
        max_calls: int | None,
        max_output: int | None,
        max_seconds: float | None,
        approval_ttl: float | None,
        claim: bool,
    ) -> None:
        self.journal = journal
        self.workspace = workspace
        self.tools = tools
        self.session_id = str(journal.records[0]["id"])
        self.approval_ttl = approval_ttl
        self.owner_version = 0
        self._closed = False
        self._resources = ExitStack()
        try:
            self._db = connect(task_db_path(journal))
            self._resources.callback(self._db.close)
            self.gate = ProjectGate(journal.project_root, self.session_id)
            self._resources.callback(self.gate.close)
            migrate(self._db, journal)
            self.decisions = ApprovalStore(self._db, self.session_id)
            self.scope = self._load_scope()
            self.approvals = False
            self.token = ""
            self.budget = TaskBudget(self._db, self.session_id)
            self._apply_limits(approvals, max_calls, max_output, max_seconds)
            if claim:
                self.owner_version = self._claim()
            self.budget.owner_version = self.owner_version
            self.workflow = WorkflowStore(
                self._db, journal, workspace, self.owner_version, self._supersede_plan_decisions
            )
            self._rewrite_token(supersede=False)
        except BaseException as primary:
            try:
                self._resources.close()
            except BaseException as cleanup:
                primary.add_note(f"task_initialization_cleanup_failed:{cleanup!r}")
            raise

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._resources.close()
        finally:
            self._closed = True

    def finish(self) -> JsonObject:
        try:
            self.gate.release()
            result = self.snapshot()
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                primary.add_note(f"task_cleanup_failed:{cleanup!r}")
            raise
        self.close()
        return result

    def note_scope(self, scope: tuple[str, ...] | None) -> None:
        if scope == self.scope:
            return
        self.scope = scope
        self._write_scope()
        self._rewrite_token(supersede=True)

    def admit(self, prepared: PreparedCall, assistant_id: str) -> RunAdmission | HaltAdmission:
        image = effect_image(self.workspace, prepared.name, prepared.arguments)
        version = candidate_version(prepared.name, prepared.arguments, self.scope, self.token)
        if self.approvals and not prepared.read_only and prepared.name not in PLANNING_TOOLS:
            return self._await_approval(prepared, assistant_id, image, version)
        intent_id = self._insert_intent(prepared, assistant_id, status="intent", version=version, image=image)
        return RunAdmission(intent_id)

    def commit(self, intent_id: str, outcome: JsonObject) -> JsonObject:
        row = self._db.execute(
            "SELECT arguments, requirement_version FROM intents WHERE intent_id=?", (intent_id,)
        ).fetchone()
        outcome = {
            **outcome,
            "intent_id": intent_id,
            "requirement_version": int(row["requirement_version"]) if row is not None else 0,
        }
        if (
            row is not None
            and outcome.get("executed") is True
            and not outcome.get("is_error")
            and outcome.get("name") in {"read", "edit", "write"}
        ):
            path = json.loads(row["arguments"]).get("path")
            if isinstance(path, str):
                digest = self.workspace.observed.get("/".join(self.workspace.parts(path)).casefold())
                if digest is not None:
                    outcome["file_versions"] = {path: digest}
        if set(self.tools).intersection(PLANNING_TOOLS) and outcome.get("name") not in PLANNING_TOOLS:
            outcome["model_content"] = canonical(
                {
                    "host_receipt": {
                        "intent_id": intent_id,
                        "requirement_version": outcome["requirement_version"],
                        "observed_inputs": list(outcome.get("file_versions") or {}),
                    },
                    "tool_output": outcome.get("text") or "",
                }
            )
        self._save_result(intent_id, outcome)
        self._backfill(intent_id)
        return outcome

    def recover(self, tools: StoredTool, assistant: JsonObject) -> RunOutcome | None:
        calls = assistant["body"].get("tool_calls") if isinstance(assistant.get("body"), dict) else None
        if not isinstance(calls, list):
            raise ValueError("invalid_tool_calls")
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("id"), str):
                raise ValueError("invalid_tool_calls")
            call_id = call["id"]
            if self.journal.has_tool(call_id, assistant["id"]):
                continue
            intent = self._intent_by_call(call_id, assistant["id"])
            if intent is None:
                self._append_outcome(call_id, _abandoned(call), intent_id=None, assistant_id=assistant["id"])
                continue
            halt = self._recover_intent(tools, intent)
            if halt is not None:
                return halt
        return None

    def blocking(self) -> RunOutcome | None:
        row = self._sticky_intent()
        if row is None:
            return self.waiting_question()
        reason = str(row["reason"] or row["status"])
        return run_halt(str(row["status"]), reason, str(row["preview"]))

    def waiting_question(self) -> RunOutcome | None:
        question = self.workflow.pending_question()
        if question is None:
            return None
        return run_halt("awaiting_question", question.id, question.text)

    def cancelled_question(self) -> RunOutcome | None:
        question = self.workflow.snapshot().question
        if question and question.status in {"cancelled", "superseded"}:
            chain = self.journal.chain()
            latest_user = next((r for r in reversed(chain) if r["kind"] == "user"), None)
            if latest_user is not None:
                if question.status == "cancelled" and latest_user["id"] != question.answer_record_id:
                    return None
                order = {r["id"]: i for i, r in enumerate(chain)}
                if (
                    question.status == "superseded"
                    and order[latest_user["id"]] > order[question.asked_record_id]
                ):
                    return None
            return run_halt(
                "awaiting_decision",
                "clarification_cancelled",
                "A new user requirement is needed; no clarification answer was adopted.",
            )
        return None

    def prepare_plan_revision(self) -> None:
        tail = self.journal.incomplete_tail()
        if tail is None:
            return
        pending = []
        for call in tail["body"]["tool_calls"]:
            if self.journal.has_tool(call["id"], tail["id"]):
                continue
            intent = self._intent_by_call(call["id"], tail["id"])
            if intent is not None and intent["status"] not in {
                "awaiting_approval",
                "result_saved",
                "settled",
                "superseded",
            }:
                raise ValueError("unresolved_effect_prevents_plan_revision")
            pending.append((call, intent))
        for call, intent in pending:
            if intent is None:
                self._append_outcome(call["id"], _abandoned(call), None, assistant_id=tail["id"])
            elif intent["status"] in {"result_saved", "settled"}:
                self._backfill(str(intent["intent_id"]))
            else:
                approval = self.decisions.for_intent(str(intent["intent_id"]))
                if approval is not None:
                    with self._tx():
                        self._assert_owner()
                        self.decisions.record_decision(approval, intent, "superseded", executed=False)
                self._finish_terminal(intent, _refusal(intent, "superseded"), "superseded")

    def _supersede_plan_decisions(self) -> None:
        """The workflow owns the current transaction; only branch-local candidates expire."""
        self._assert_owner()
        ancestors = {r["id"] for r in self.journal.chain()}
        for approval in self._db.execute("SELECT * FROM approvals WHERE session_id=?", (self.session_id,)):
            intent = self._locked_intent(str(approval["intent_id"]))
            if (
                intent["assistant_id"] in ancestors
                and intent["status"] == "awaiting_approval"
                and approval["decision"] in {"pending", "approved"}
            ):
                self.decisions.record_decision(approval, intent, "superseded", executed=False)

    def question_blocks(self, prepared: PreparedCall) -> bool:
        if self.workflow.pending_question() is None:
            return False
        return prepared.name != "read" or not self.workflow.allows_read_while_waiting(
            prepared.arguments["path"]
        )

    def decide(self, approval_id: str, decision: str) -> JsonObject:
        return self.decisions.decide(approval_id, decision)

    def snapshot(self) -> JsonObject:
        intents = [
            _public_intent(row)
            for row in self._db.execute(
                "SELECT * FROM intents WHERE session_id=? ORDER BY rowid", (self.session_id,)
            )
        ]
        approvals = [
            _public_approval(row)
            for row in self._db.execute(
                "SELECT * FROM approvals WHERE session_id=? ORDER BY rowid", (self.session_id,)
            )
        ]
        budget = self.budget.snapshot()
        workers = self.gate.snapshot()
        return {
            "session_id": self.session_id,
            "owner_version": self.owner_version,
            "approvals_enabled": self.approvals,
            "constraint": self.token,
            "scope": list(self.scope) if self.scope is not None else None,
            "intents": intents,
            "approvals": approvals,
            "budget": budget,
            "workers": workers,
            "workflow": self.workflow.snapshot().payload(),
        }

    def context_messages(self) -> list[JsonObject]:
        self.workflow.refresh_inputs()
        state = self.workflow.snapshot()
        if not set(self.tools).intersection(PLANNING_TOOLS) or state.version == 0:
            return []
        catalog = self.workflow.available_evidence()
        return [
            {
                "role": "system",
                "content": (
                    f"CURRENT WORKFLOW VERSION = {state.version}; use expected_version={state.version} "
                    "for the next plan/question mutation. This overrides older snapshots in history.\n"
                    "Current host workflow state (not extra authority):\n"
                    "Clarification asks for missing task requirements, never execution permission. "
                    "To propose a write or command, invoke its actual tool; when approvals are enabled "
                    "the host will stop BEFORE its effect and create the approval candidate. "
                    "Do not use question to ask whether a tool may run.\n"
                    "After input changes invalidate evidence, read the declared inputs again and "
                    "record current read/check intent IDs in dependency order. Current reads and "
                    "checks can verify already-applied work; do not repeat writes just to renew "
                    "evidence or ask permission to maintain the existing plan.\n"
                )
                + canonical(
                    {
                        **state.payload(),
                        "approvals_enabled": self.approvals,
                        "available_evidence": [asdict(item) for item in catalog.entries],
                        "evidence_catalog": {
                            "truncated": catalog.truncated,
                            "limit": 12,
                            "target_preview_chars": 160,
                            "note": "Recent eligible candidates; original receipts and current step checks remain complete in the task log.",
                        },
                    }
                ),
            }
        ]

    def _recover_intent(self, tools: StoredTool, intent: sqlite3.Row) -> RunOutcome | None:
        status = str(intent["status"])
        call_id = str(intent["call_id"])
        if status in {"result_saved", "settled"}:
            self._backfill(str(intent["intent_id"]))
            return None
        if status == "awaiting_approval":
            return self._recover_approval(tools, intent)
        if status != "intent":
            if status in _STICKY:
                return run_halt(status, str(intent["reason"] or status), str(intent["preview"]))
            if status in _TERMINAL and not self.journal.has_tool(call_id, str(intent["assistant_id"])):
                self._append_outcome(call_id, _refusal(intent, status), str(intent["intent_id"]))
                self._mark(str(intent["intent_id"]), "settled")
            return run_halt(status, str(intent["reason"] or status), str(intent["preview"]))
        image = EffectImage(
            str(intent["object_key"]),
            None if intent["before_sha256"] is None else str(intent["before_sha256"]),
            None if intent["after_sha256"] is None else str(intent["after_sha256"]),
            str(intent["preview"]),
        )
        observation = observe(self.workspace, image)
        decision = classify(str(intent["replay"]), tools.replay_policy(str(intent["name"])), observation)
        if decision == "retry":
            outcome = tools.execute_stored(
                str(intent["name"]), _arguments(intent), call_id, require_safe=True
            )
            if outcome.get("reason") in {"replay_blocked", "tool_not_authorized", "tool_revoked"}:
                self._finish_terminal(intent, outcome, "blocked")
                return run_halt("blocked", str(outcome.get("reason")), str(outcome.get("text") or ""))
            self.commit(str(intent["intent_id"]), outcome)
            return None
        if decision == "match":
            self.commit(str(intent["intent_id"]), _reconciled(intent))
            return None
        if decision == "blocked":
            self._finish_terminal(intent, _refusal(intent, "blocked"), "blocked")
            return run_halt("blocked", "replay_blocked", str(intent["preview"]))
        if decision == "absent":
            self._mark(str(intent["intent_id"]), "awaiting_decision", "effect_not_observed")
            text = f"Effect for {intent['object_key']} was not observed. It was not run again."
            self._append_outcome(call_id, _note(intent, text, "awaiting_decision"), str(intent["intent_id"]))
            return run_halt("awaiting_decision", "effect_not_observed", text)
        self._mark(str(intent["intent_id"]), "effect_unknown", "effect_unverified")
        text = f"Effect for {intent['object_key']} could not be verified and was not run again."
        self._append_outcome(call_id, _note(intent, text, "effect_unknown"), str(intent["intent_id"]))
        return run_halt("effect_unknown", "effect_unverified", text)

    def _recover_approval(self, tools: StoredTool, intent: sqlite3.Row) -> RunOutcome | None:
        approval = self.decisions.for_intent(str(intent["intent_id"]))
        if approval is None:
            return run_halt("awaiting_approval", "awaiting_approval", str(intent["preview"]))
        decision = str(approval["decision"])
        if (
            approval["expires_at"] is not None
            and time.time() >= float(approval["expires_at"])
            and decision == "approved"
        ):
            decision = "expired"
            with self._tx():
                self._db.execute(
                    "UPDATE approvals SET decision='expired' WHERE approval_id=?", (approval["approval_id"],)
                )
        if decision == "approved" and approval["candidate_version"] == intent["candidate_version"]:
            try:
                image = effect_image(self.workspace, str(intent["name"]), _arguments(intent))
            except EffectPreviewError as error:
                self._finish_terminal(intent, _refusal(intent, "candidate_changed"), "superseded")
                return run_halt("superseded", "candidate_changed", str(error))
            if image.before_sha256 != intent["before_sha256"]:
                self._finish_terminal(intent, _refusal(intent, "candidate_changed"), "superseded")
                return run_halt("superseded", "candidate_changed", "Target changed after approval preview.")
            self._mark(str(intent["intent_id"]), "intent")
            outcome = tools.execute_stored(
                str(intent["name"]),
                _arguments(intent),
                str(intent["call_id"]),
                require_safe=False,
                expected_file=FileVersion(image.before_sha256)
                if intent["name"] in {"write", "edit"}
                else None,
            )
            self.commit(str(intent["intent_id"]), outcome)
            return None
        if decision == "approved":
            self._finish_terminal(intent, _refusal(intent, "blocked"), "blocked")
            return run_halt("blocked", "approval_mismatch", str(intent["preview"]))
        if decision == "pending":
            return run_halt("awaiting_approval", "awaiting_approval", str(intent["preview"]))
        self._finish_terminal(intent, _refusal(intent, decision), decision)
        return run_halt(decision, decision, str(intent["preview"]))

    def _await_approval(
        self, prepared: PreparedCall, assistant_id: str, image: EffectImage, version: str
    ) -> HaltAdmission:
        intent_id = self._insert_intent(
            prepared, assistant_id, status="awaiting_approval", version=version, image=image
        )
        approval = self.decisions.for_intent(intent_id)
        if approval is None:
            raise ValueError("missing approval for persisted candidate")
        approval_id = str(approval["approval_id"])
        text = f"Approval {approval_id} is required before changing {image.object_key}.\n{image.preview}"
        return HaltAdmission(
            HaltOutcome(
                name=prepared.name,
                tool_call_id=prepared.tool_call_id,
                executed=False,
                is_error=True,
                text=text,
                reason="awaiting_approval",
                halt="awaiting_approval",
                approval_id=approval_id,
                preview=image.preview,
            )
        )

    def _insert_intent(
        self, prepared: PreparedCall, assistant_id: str, *, status: str, version: str, image: EffectImage
    ) -> str:
        intent_id = uuid4().hex
        with self._tx():
            self._assert_owner()
            self._db.execute(
                """
                INSERT INTO intents(
                    intent_id, session_id, assistant_id, call_id, name, arguments, replay, status, candidate_version,
                    object_key, before_sha256, after_sha256, preview, owner_version, requirement_version
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    intent_id,
                    self.session_id,
                    assistant_id,
                    prepared.tool_call_id,
                    prepared.name,
                    canonical(prepared.arguments),
                    prepared.replay,
                    status,
                    version,
                    image.object_key,
                    image.before_sha256,
                    image.after_sha256,
                    image.preview,
                    self.owner_version,
                    self.workflow.snapshot().requirement_version,
                ),
            )
            if status == "awaiting_approval":
                approval_id = uuid4().hex
                expires = None if self.approval_ttl is None else time.time() + self.approval_ttl
                scope = json.dumps(list(self.scope) if self.scope is not None else None)
                self._db.execute(
                    """
                    INSERT INTO approvals(
                            approval_id, intent_id, session_id, candidate_version, decision, preview, scope,
                        constraint_token, expires_at, receipt_json
                    ) VALUES(?,?,?,?, 'pending', ?,?,?,?, NULL)
                    """,
                    (
                        approval_id,
                        intent_id,
                        self.session_id,
                        version,
                        image.preview,
                        scope,
                        self.token,
                        expires,
                    ),
                )
        return intent_id

    def _save_result(self, intent_id: str, outcome: JsonObject) -> None:
        payload = json.dumps(outcome, ensure_ascii=False)
        with self._tx():
            self._assert_owner()
            cursor = self._db.execute(
                """
                UPDATE intents SET status='result_saved', result_json=?
                WHERE intent_id=? AND owner_version=? AND status IN ('intent', 'awaiting_approval')
                """,
                (payload, intent_id, self.owner_version),
            )
            resolved = outcome.get("resolves_effect")
            if isinstance(resolved, str):
                self._db.execute(
                    """INSERT OR IGNORE INTO reconciliations(intent_id, query_intent_id)
                    SELECT intent_id, ? FROM intents WHERE session_id=? AND status='effect_unknown'
                    AND json_extract(result_json, '$.effect_key')=?
                    AND json_extract(result_json, '$.reconcile_tool')=?""",
                    (intent_id, self.session_id, resolved, outcome.get("name")),
                )
                self._db.execute(
                    """UPDATE intents SET status='settled', reason='effect_reconciled'
                    WHERE session_id=? AND intent_id IN (SELECT intent_id FROM reconciliations)""",
                    (self.session_id,),
                )
            if cursor.rowcount != 1:
                current = self._db.execute(
                    "SELECT status FROM intents WHERE intent_id=?", (intent_id,)
                ).fetchone()
                if current is not None and current["status"] in {"result_saved", "settled"}:
                    return
                raise StaleOwner(intent_id)

    def _backfill(self, intent_id: str) -> None:
        intent = self._get(intent_id)
        if intent is None or intent["result_json"] is None:
            raise ValueError(f"missing_result:{intent_id}")
        call_id = str(intent["call_id"])
        if not self.journal.has_tool(call_id, str(intent["assistant_id"])):
            self._append_outcome(call_id, cast(JsonObject, json.loads(str(intent["result_json"]))), intent_id)
        result = json.loads(str(intent["result_json"]))
        unknown = result.get("effect") == "unknown" or result.get("reason") in {
            "effect_unknown",
            "cleanup_failed",
            "workspace_sync_failed",
        }
        reconciled = self._db.execute(
            "SELECT 1 FROM reconciliations WHERE intent_id=?", (intent_id,)
        ).fetchone()
        unknown = unknown and reconciled is None
        self._mark(
            intent_id, "effect_unknown" if unknown else "settled", result.get("reason") if unknown else None
        )

    def _finish_terminal(self, intent: sqlite3.Row, outcome: JsonObject, status: str) -> None:
        call_id = str(intent["call_id"])
        if not self.journal.has_tool(call_id, str(intent["assistant_id"])):
            self._append_outcome(call_id, outcome, str(intent["intent_id"]))
        if status in _STICKY:
            self._mark(str(intent["intent_id"]), status, str(outcome.get("reason") or status))
        else:
            self._mark(str(intent["intent_id"]), "settled", status)

    def _append_outcome(
        self, call_id: str, outcome: JsonObject, intent_id: str | None, *, assistant_id: str | None = None
    ) -> None:
        if intent_id is not None:
            assistant_id = str(self._locked_intent(intent_id)["assistant_id"])
        if assistant_id is None:
            raise ValueError("missing assistant identity")
        if self.journal.has_tool(call_id, assistant_id):
            return
        body: JsonObject = {
            "assistant_id": assistant_id,
            "tool_call_id": call_id,
            "content": str(outcome.get("model_content", outcome.get("text")) or ""),
            "outcome": outcome,
        }
        if intent_id is not None:
            body["intent_id"] = intent_id
        self.journal.append("tool", body)

    def _mark(self, intent_id: str, status: str, reason: str | None = None) -> None:
        with self._tx():
            self._assert_owner()
            current = self._locked_intent(intent_id)
            if current["status"] == status:
                return
            cursor = self._db.execute(
                """
                UPDATE intents SET status=?, reason=COALESCE(?, reason)
                WHERE intent_id=? AND owner_version=?
                """,
                (status, reason, intent_id, self.owner_version),
            )
            if cursor.rowcount != 1:
                raise StaleOwner(intent_id)

    def _claim(self) -> int:
        worker_id = uuid4().hex
        with self._tx():
            self._db.execute(
                """
                INSERT INTO ownership(session_id, version, worker_id, pid) VALUES(?,1,?,?)
                ON CONFLICT(session_id) DO UPDATE SET
                    version=version+1, worker_id=excluded.worker_id, pid=excluded.pid
                """,
                (self.session_id, worker_id, os.getpid()),
            )
            row = self._db.execute(
                "SELECT version FROM ownership WHERE session_id=?", (self.session_id,)
            ).fetchone()
            version = int(row["version"])
            self._db.execute(
                """
                UPDATE intents SET owner_version=?
                WHERE session_id=? AND status!='settled'
                """,
                (version, self.session_id),
            )
            self.budget.reap(version)
        return version

    def _apply_limits(
        self, approvals: bool, max_calls: int | None, max_output: int | None, max_seconds: float | None
    ) -> None:
        self.budget.configure(max_calls, max_output, max_seconds)
        self.approvals = approvals or self._meta("approvals") == "1"
        if self.approvals:
            with self._tx():
                self._meta_set("approvals", "1")

    def _rewrite_token(self, *, supersede: bool) -> None:
        limits = self.budget.limits()
        token = constraint_token(
            self.tools,
            self.scope,
            approvals=self.approvals,
            max_calls=limits.max_calls,
            max_output=limits.max_output,
            max_seconds=limits.max_seconds,
        )
        previous = self._meta("constraint")
        with self._tx():
            self._meta_set("constraint", token)
            if supersede or (previous is not None and previous != token):
                self._supersede()
        self.token = token

    def _supersede(self) -> None:
        rows = list(
            self._db.execute(
                """
                SELECT approval_id, intent_id FROM approvals
                WHERE session_id=? AND decision IN ('pending', 'approved')
                """,
                (self.session_id,),
            )
        )
        for row in rows:
            intent = self._db.execute(
                "SELECT status FROM intents WHERE intent_id=?", (row["intent_id"],)
            ).fetchone()
            if intent is None or intent["status"] != "awaiting_approval":
                continue
            self._db.execute(
                "UPDATE approvals SET decision='superseded' WHERE approval_id=? AND decision IN ('pending','approved')",
                (row["approval_id"],),
            )
            self._db.execute(
                "UPDATE intents SET status='superseded', reason='limits_changed' WHERE intent_id=? AND status='awaiting_approval'",
                (row["intent_id"],),
            )

    def _get(self, intent_id: str) -> sqlite3.Row | None:
        return optional_row(
            self._db.execute("SELECT * FROM intents WHERE intent_id=?", (intent_id,)).fetchone()
        )

    def _intent_by_call(self, call_id: str, assistant_id: str) -> sqlite3.Row | None:
        return optional_row(
            self._db.execute(
                "SELECT * FROM intents WHERE session_id=? AND assistant_id=? AND call_id=?",
                (self.session_id, assistant_id, call_id),
            ).fetchone()
        )

    def _locked_intent(self, intent_id: str) -> sqlite3.Row:
        row = self._get(intent_id)
        if row is None:
            raise ValueError(f"unknown_intent:{intent_id}")
        return row

    def _sticky_intent(self) -> sqlite3.Row | None:
        return optional_row(
            self._db.execute(
                """
                SELECT * FROM intents WHERE session_id=? AND status IN ('effect_unknown', 'awaiting_decision')
                ORDER BY rowid DESC LIMIT 1
                """,
                (self.session_id,),
            ).fetchone()
        )

    def _load_scope(self) -> tuple[str, ...] | None:
        raw = self._meta("scope")
        if raw is None:
            return None
        value = json.loads(raw)
        if value is None:
            return None
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError("stored scope is invalid")
        return tuple(value)

    def _write_scope(self) -> None:
        payload = json.dumps(list(self.scope) if self.scope is not None else None)
        with self._tx():
            self._meta_set("scope", payload)

    def _meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _meta_set(self, key: str, value: str) -> None:
        self._db.execute(
            "INSERT INTO meta(key, value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def _assert_owner(self) -> None:
        assert_owner(self._db, self.session_id, self.owner_version)

    def _tx(self) -> AbstractContextManager[None]:
        return transaction(self._db)


def _arguments(intent: sqlite3.Row) -> JsonObject:
    value = json.loads(str(intent["arguments"]))
    if not isinstance(value, dict):
        raise ValueError("stored arguments are not an object")
    return cast(JsonObject, value)


def _abandoned(call: JsonObject) -> JsonObject:
    function = call.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    return tool_outcome(
        {
            "name": name,
            "tool_call_id": call["id"],
            "text": "Tool call ended before its intent was stored. It was not executed.",
        },
        executed=False,
        is_error=True,
        reason="abandoned",
    )


def _refusal(intent: sqlite3.Row, status: str) -> JsonObject:
    return _note(intent, f"Tool {intent['name']} was not executed ({status}).", status)


def _note(intent: sqlite3.Row, text: str, status: str) -> JsonObject:
    return tool_outcome(
        {
            "name": str(intent["name"]),
            "tool_call_id": str(intent["call_id"]),
            "text": text,
            "path": str(intent["object_key"]),
        },
        executed=False,
        is_error=True,
        reason=status,
    )


def _reconciled(intent: sqlite3.Row) -> JsonObject:
    return tool_outcome(
        {
            "name": str(intent["name"]),
            "tool_call_id": str(intent["call_id"]),
            "path": str(intent["object_key"]),
            "text": f"Reconciled {intent['object_key']}; content matches the stored intent and was not written again.",
        },
        executed=True,
        is_error=False,
        effect="reconciled",
        reason="reconciled",
    )


def _public_intent(row: sqlite3.Row) -> JsonObject:
    result = json.loads(str(row["result_json"])) if row["result_json"] else None
    return {
        "intent_id": str(row["intent_id"]),
        "assistant_id": str(row["assistant_id"]),
        "call_id": str(row["call_id"]),
        "name": str(row["name"]),
        "arguments": json.loads(str(row["arguments"])),
        "replay": str(row["replay"]),
        "status": str(row["status"]),
        "candidate_version": str(row["candidate_version"]),
        "object_key": str(row["object_key"]),
        "before_sha256": row["before_sha256"],
        "after_sha256": row["after_sha256"],
        "preview": str(row["preview"]),
        "owner_version": int(row["owner_version"]),
        "reason": row["reason"],
        "result": result,
    }


def _public_approval(row: sqlite3.Row) -> JsonObject:
    return {
        "approval_id": str(row["approval_id"]),
        "intent_id": str(row["intent_id"]),
        "decision": str(row["decision"]),
        "candidate_version": str(row["candidate_version"]),
        "preview": str(row["preview"]),
        "scope": json.loads(str(row["scope"])),
        "expires_at": row["expires_at"],
        "receipt": json.loads(str(row["receipt_json"])) if row["receipt_json"] else None,
    }
