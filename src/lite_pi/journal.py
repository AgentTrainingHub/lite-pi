"""Append-only history and validated branch projections; disk files never roll back."""

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from lite_pi.storage import atomic_write
from lite_pi.tools.contracts import JsonObject

FORMAT_VERSION = 1
SUMMARY_PREFIX = "Conversation summary:\n"


class JournalError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class Journal:
    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.records = _read(self.path)
        self.chain()  # A missing leaf is corruption, not an invitation to guess.

    @classmethod
    def create(cls, path: Path, project_root: Path, *, tools: tuple[str, ...] = ()) -> "Journal":
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        header: JsonObject = {
            "version": FORMAT_VERSION,
            "id": uuid4().hex,
            "parent_id": None,
            "kind": "header",
            "body": {
                "project_root": str(project_root.resolve()),
                "format": FORMAT_VERSION,
                "tools": list(tools),
            },
        }
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(header) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as error:
            raise JournalError("exists") from error
        atomic_write(path.with_suffix(path.suffix + ".leaf"), (header["id"] + "\n").encode())
        return cls(path)

    @property
    def lock_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".lock")

    @property
    def leaf_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".leaf")

    @property
    def project_root(self) -> Path:
        return Path(self.records[0]["body"]["project_root"])

    @property
    def authorized_tools(self) -> tuple[str, ...]:
        tools = self.records[0]["body"].get("tools")
        if not isinstance(tools, list) or any(not isinstance(name, str) for name in tools):
            raise JournalError("missing_tool_authority")
        return tuple(tools)

    def leaf_id(self) -> str:
        try:
            return self.leaf_path.read_text().strip()
        except FileNotFoundError as error:
            raise JournalError("missing_leaf") from error

    def reload(self) -> None:
        self.records = _read(self.path)
        self.chain()

    def append(self, kind: str, body: JsonObject, *, parent_id: str | None = None) -> JsonObject:
        parent = parent_id or self.leaf_id()
        if parent not in {record["id"] for record in self.records}:
            raise JournalError("missing_parent")
        record: JsonObject = {
            "version": FORMAT_VERSION,
            "id": uuid4().hex,
            "parent_id": parent,
            "kind": kind,
            "body": body,
        }
        _validate_record(record)
        if kind == "compaction":
            parent_chain = self.chain(parent)
            _project(parent_chain)
            _check_boundary(parent_chain, body["first_kept_id"], body.get("turn_user_id"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(record)
        atomic_write(self.leaf_path, (record["id"] + "\n").encode())
        return record

    def select(self, leaf_id: str, *, allow_incomplete: bool = False) -> None:
        tail = self.incomplete_tail(leaf_id) if allow_incomplete else None
        self.messages(tail["parent_id"] if tail is not None else leaf_id)
        atomic_write(self.leaf_path, (leaf_id + "\n").encode())

    def branch(self, parent_id: str, task: str) -> JsonObject:
        self.messages(parent_id)
        return self.append("user", {"content": task, "branch_from": parent_id}, parent_id=parent_id)

    def messages(self, leaf_id: str | None = None) -> list[JsonObject]:
        return self.projection(leaf_id)[0]

    def projection(self, leaf_id: str | None = None) -> tuple[list[JsonObject], list[str]]:
        """Return aligned messages and source IDs, including a summary's record ID."""
        chain = self.chain(leaf_id)
        original = _project(chain)  # Compaction must not hide an incomplete tool batch.
        latest = None
        start = 0
        for index, record in enumerate(chain):
            if record["kind"] == "compaction":
                kept = _check_boundary(
                    chain[:index], record["body"]["first_kept_id"], record["body"].get("turn_user_id")
                )
                start = next(i for i, item in enumerate(chain) if item["id"] == kept["id"])
                latest = record
        visible = [record for record in chain[start:] if record["kind"] in {"user", "assistant", "tool"}]
        ids = [record["id"] for record in visible]
        if latest is None:
            return original, ids
        if turn_user_id := latest["body"].get("turn_user_id"):
            user = next(record for record in chain if record["id"] == turn_user_id)
            visible.insert(0, user)
            ids.insert(0, user["id"])
        summary = {"role": "user", "content": SUMMARY_PREFIX + latest["body"]["summary"]}
        return [summary, *_project(visible)], [latest["id"], *ids]

    def has_tool(self, call_id: str, assistant_id: str) -> bool:
        active = False
        for record in self.chain():
            if record["id"] == assistant_id:
                active = True
            elif active and record["kind"] == "assistant":
                return False
            if active and record["kind"] == "tool" and record["body"].get("tool_call_id") == call_id:
                return True
        return False

    def incomplete_tail(self, leaf_id: str | None = None) -> dict[str, Any] | None:
        """Return a trailing assistant whose tool calls are still missing results."""
        pending: list[str] = []
        assistant: dict[str, Any] | None = None
        for record in self.chain(leaf_id):
            kind, body = record["kind"], record["body"]
            if kind in {"header", "compaction"}:
                continue
            if kind == "tool":
                call_id = body["tool_call_id"]
                if call_id not in pending:
                    raise JournalError("unpaired_tool_result")
                pending.remove(call_id)
                continue
            if pending:
                raise JournalError("unpaired_tool_call")
            assistant = None
            if kind == "assistant":
                calls = body.get("tool_calls") or []
                if calls:
                    assistant = record
                    pending = [call["id"] for call in calls]
                    if len(pending) != len(set(pending)):
                        raise JournalError("duplicate_call_id")
        if not pending:
            return None
        if assistant is None:
            raise JournalError("unpaired_tool_call")
        return assistant

    def source_ids(self, leaf_id: str | None = None) -> list[str]:
        return [record["id"] for record in self.chain(leaf_id) if record["kind"] != "header"]

    def chain(self, leaf_id: str | None = None) -> list[JsonObject]:
        by_id = {record["id"]: record for record in self.records}
        current = leaf_id or self.leaf_id()
        ordered = []
        seen: set[str] = set()
        while current is not None:
            if current in seen or current not in by_id:
                raise JournalError("missing_parent")
            seen.add(current)
            ordered.append(by_id[current])
            current = by_id[current]["parent_id"]
        return list(reversed(ordered))


def _validate_record(record: JsonObject) -> None:
    if not isinstance(record, dict) or record.get("version") != FORMAT_VERSION:
        raise JournalError("unknown_version")
    kind, body = record.get("kind"), record.get("body")
    if not isinstance(kind, str):
        raise JournalError("unknown_record_kind")
    if "parent_id" not in record or (
        record["parent_id"] is not None and not isinstance(record["parent_id"], str)
    ):
        raise JournalError("invalid_parent")
    if not isinstance(record.get("id"), str) or not record["id"] or not isinstance(body, dict):
        raise JournalError("invalid_record")
    if kind == "header":
        if not isinstance(body.get("project_root"), str) or not Path(body["project_root"]).is_absolute():
            raise JournalError("invalid_header")
    elif kind == "compaction":
        summary = body.get("summary")
        kept = body.get("first_kept_id")
        if not isinstance(summary, str) or not summary.strip():
            raise JournalError("invalid_compaction")
        if not isinstance(kept, str) or not kept:
            raise JournalError("invalid_compaction")
        if "turn_user_id" in body and (not isinstance(body["turn_user_id"], str) or not body["turn_user_id"]):
            raise JournalError("invalid_compaction")
    elif kind in {"user", "assistant", "tool"}:
        if not isinstance(body.get("content"), str):
            raise JournalError("invalid_content")
        if kind == "tool" and (not isinstance(body.get("tool_call_id"), str) or not body["tool_call_id"]):
            raise JournalError("invalid_call_id")
        calls = body.get("tool_calls") or []
        if not isinstance(calls, list):
            raise JournalError("invalid_tool_calls")
        for call in calls:
            if (
                not isinstance(call, dict)
                or not isinstance(call.get("id"), str)
                or not call["id"]
                or not isinstance(call.get("function"), dict)
                or not isinstance(call["function"].get("name"), str)
                or not call["function"]["name"]
                or not isinstance(call["function"].get("arguments"), str)
            ):
                raise JournalError("invalid_tool_calls")
    else:
        raise JournalError("unknown_record_kind")


def _project(chain: list[JsonObject]) -> list[JsonObject]:
    projected = []
    pending: set[str] = set()
    for record in chain:
        kind, body = record["kind"], record["body"]
        if kind in {"header", "compaction"}:
            continue
        if kind == "tool":
            call_id = body["tool_call_id"]
            if call_id not in pending:
                raise JournalError("unpaired_tool_result")
            pending.remove(call_id)
            projected.append({"role": "tool", "tool_call_id": call_id, "content": body["content"]})
            continue
        if pending:
            raise JournalError("unpaired_tool_call")
        message = {"role": kind, "content": body.get("content") or ""}
        if kind == "assistant":
            for key in ("tool_calls", "reasoning_content"):
                if body.get(key):
                    message[key] = body[key]
            calls = body.get("tool_calls") or []
            for call in calls:
                call_id = call["id"]
                if call_id in pending:
                    raise JournalError("duplicate_call_id")
                pending.add(call_id)
        projected.append(message)
    if pending:
        raise JournalError("unpaired_tool_call")
    return projected


def _check_boundary(records: list[JsonObject], kept_id: str, turn_user_id: str | None = None) -> JsonObject:
    kept = next((record for record in records if record["id"] == kept_id), None)
    if kept is None:
        raise JournalError("missing_compaction_boundary")
    if turn_user_id is None:
        if kept["kind"] != "user":
            raise JournalError("invalid_compaction_boundary")
        return kept
    if kept["kind"] != "assistant":
        raise JournalError("invalid_compaction_boundary")
    earlier = records[: records.index(kept)]
    user = next((record for record in reversed(earlier) if record["kind"] == "user"), None)
    if user is None or user["id"] != turn_user_id:
        raise JournalError("invalid_compaction_boundary")
    return kept


def _read(path: Path) -> list[JsonObject]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as error:
        raise JournalError("missing_journal") from error
    if not raw or not raw.endswith(b"\n"):
        raise JournalError("incomplete_tail")
    try:
        records = [json.loads(line) for line in raw.decode().splitlines()]
    except (json.JSONDecodeError, UnicodeError) as error:
        raise JournalError("incomplete_tail") from error
    ids: set[str] = set()
    for index, record in enumerate(records):
        _validate_record(record)
        if record["id"] in ids:
            raise JournalError("duplicate_id")
        parent = record.get("parent_id")
        if index == 0:
            if record["kind"] != "header" or parent is not None:
                raise JournalError("invalid_header")
        elif record["kind"] == "header" or not isinstance(parent, str) or parent not in ids:
            raise JournalError("missing_parent")
        ids.add(record["id"])
    return records
