"""Durable approval decisions; a decision never runs the proposed tool."""

import json
import sqlite3
import time
from typing import cast

from lite_pi.task_storage import optional_row, transaction
from lite_pi.tools.contracts import JsonObject


class ApprovalStore:
    def __init__(self, db: sqlite3.Connection, session_id: str) -> None:
        self._db = db
        self.session_id = session_id

    def _locked_intent(self, identity: str) -> sqlite3.Row:
        row = self._db.execute(
            "SELECT * FROM intents WHERE intent_id=? AND session_id=?", (identity, self.session_id)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown_intent:{identity}")
        return cast(sqlite3.Row, row)

    def decide(self, approval_id: str, decision: str) -> JsonObject:
        if decision not in {"approve", "reject", "revoke"}:
            raise ValueError("decision must be approve, reject, or revoke")
        with transaction(self._db):
            approval = self.get(approval_id)
            if approval is None:
                raise ValueError(f"unknown_approval:{approval_id}")
            intent = self._locked_intent(str(approval["intent_id"]))
            if (
                approval["expires_at"] is not None
                and time.time() >= float(approval["expires_at"])
                and intent["status"] == "awaiting_approval"
            ):
                return self.record_decision(approval, intent, "expired", executed=False)
            if decision == "approve" and approval["decision"] == "approved" and approval["receipt_json"]:
                return cast(JsonObject, json.loads(str(approval["receipt_json"])))
            if approval["decision"] in {"rejected", "revoked", "expired", "superseded"}:
                return _stored_receipt(approval)
            if decision in {"reject", "revoke"} and intent["status"] != "awaiting_approval":
                result = json.loads(str(intent["result_json"])) if intent["result_json"] else None
                executed: bool | None = False
                if result is not None and "executed" in result:
                    executed = bool(result["executed"])
                elif intent["status"] in {"intent", "effect_unknown", "awaiting_decision"}:
                    executed = None
                return self.record_decision(
                    approval,
                    intent,
                    "rejected" if decision == "reject" else "revoked",
                    executed=executed,
                    keep_intent=True,
                )
            if decision == "approve":
                if approval["decision"] != "pending":
                    return _stored_receipt(approval)
                return self.record_decision(approval, intent, "approved", executed=False, keep_status=True)
            status = "rejected" if decision == "reject" else "revoked"
            return self.record_decision(approval, intent, status, executed=False)

    def record_decision(
        self,
        approval: sqlite3.Row,
        intent: sqlite3.Row,
        decision: str,
        *,
        executed: bool | None,
        keep_intent: bool = False,
        keep_status: bool = False,
    ) -> JsonObject:
        if approval["decision"] == decision and approval["receipt_json"]:
            return cast(JsonObject, json.loads(str(approval["receipt_json"])))
        receipt = {
            "approval_id": str(approval["approval_id"]),
            "intent_id": str(intent["intent_id"]),
            "decision": decision,
            "candidate_version": str(approval["candidate_version"]),
            "object": str(intent["object_key"]),
            "preview": str(approval["preview"]),
            "scope": json.loads(str(approval["scope"])),
            "executed": executed,
        }
        if decision == "superseded" and approval["receipt_json"]:
            receipt["previous_receipt"] = json.loads(str(approval["receipt_json"]))
        self._db.execute(
            "UPDATE approvals SET decision=?, receipt_json=? WHERE approval_id=?",
            (decision, json.dumps(receipt, ensure_ascii=False), approval["approval_id"]),
        )
        if not keep_intent and not keep_status and intent["status"] == "awaiting_approval":
            self._db.execute(
                "UPDATE intents SET status=?, reason=? WHERE intent_id=?",
                (decision, decision, intent["intent_id"]),
            )
        return receipt

    def get(self, approval_id: str) -> sqlite3.Row | None:
        return optional_row(
            self._db.execute(
                "SELECT * FROM approvals WHERE approval_id=? AND session_id=?", (approval_id, self.session_id)
            ).fetchone()
        )

    def for_intent(self, intent_id: str) -> sqlite3.Row | None:
        return optional_row(
            self._db.execute(
                "SELECT * FROM approvals WHERE intent_id=? AND session_id=? ORDER BY rowid DESC LIMIT 1",
                (intent_id, self.session_id),
            ).fetchone()
        )


def _stored_receipt(approval: sqlite3.Row) -> JsonObject:
    if approval["receipt_json"]:
        return cast(JsonObject, json.loads(str(approval["receipt_json"])))
    return {
        "approval_id": str(approval["approval_id"]),
        "intent_id": str(approval["intent_id"]),
        "decision": str(approval["decision"]),
        "candidate_version": str(approval["candidate_version"]),
        "preview": str(approval["preview"]),
    }
