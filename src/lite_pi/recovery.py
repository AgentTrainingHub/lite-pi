"""Decide whether a stored tool intent may run again, or must be reconciled."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from lite_pi.tools.contracts import JsonObject
from lite_pi.tools.edit import apply_edits, file_diff
from lite_pi.tools.workspace import Workspace

Decision = Literal["retry", "match", "absent", "unknown", "blocked"]


def canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def constraint_token(
    tools: tuple[str, ...],
    scope: tuple[str, ...] | None,
    *,
    approvals: bool,
    max_calls: int | None,
    max_output: int | None,
    max_seconds: float | None,
) -> str:
    payload = {
        "tools": list(tools),
        "scope": list(scope) if scope is not None else None,
        "approvals": approvals,
        "max_calls": max_calls,
        "max_output": max_output,
        "max_seconds": max_seconds,
    }
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


def candidate_version(name: str, arguments: JsonObject, scope: tuple[str, ...] | None, token: str) -> str:
    payload = {
        "name": name,
        "arguments": arguments,
        "scope": list(scope) if scope is not None else None,
        "constraint": token,
    }
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


@dataclass(frozen=True)
class EffectImage:
    object_key: str
    before_sha256: str | None
    after_sha256: str | None
    preview: str


class EffectPreviewError(ValueError):
    """The tool target could not be inspected before any effect or intent was created."""


def effect_image(workspace: Workspace, name: str, arguments: JsonObject) -> EffectImage:
    path = arguments.get("path")
    key = path if isinstance(path, str) else name
    if not isinstance(path, str):
        return EffectImage(key, None, None, canonical({"name": name, "arguments": arguments}))
    if name == "read":
        return EffectImage(path, None, None, f"read {path}")
    if name not in {"write", "edit"}:
        return EffectImage(key, None, None, canonical({"name": name, "arguments": arguments}))
    try:
        original = _read_optional(workspace, path)
    except (OSError, ValueError) as error:
        raise EffectPreviewError(f"{type(error).__name__}: {error}") from error
    before = None if original is None else hashlib.sha256(original).hexdigest()
    if name == "write" and isinstance(arguments.get("content"), str):
        updated = arguments["content"].encode()
        preview = _diff(path, original, updated)
        return EffectImage(path, before, hashlib.sha256(updated).hexdigest(), preview)
    if name == "edit":
        projected = _project_edit(original, arguments) if original is not None else None
        if projected is None:
            return EffectImage(path, before, None, f"Edit of {path} cannot be previewed without writing.")
        return EffectImage(
            path, before, hashlib.sha256(projected).hexdigest(), _diff(path, original, projected)
        )
    return EffectImage(path, before, None, f"{name} {path}")


def observe(workspace: Workspace, image: EffectImage) -> str:
    """Return match, absent, mismatch, or unqueryable from the current file bytes."""
    if image.after_sha256 is None or image.object_key == "":
        return "unqueryable"
    try:
        current = _hash_file(workspace, image.object_key)
    except (OSError, ValueError):
        return "unqueryable"
    if current == image.after_sha256:
        return "match"
    if current is None and image.before_sha256 is None:
        return "absent"
    if image.before_sha256 is not None and current == image.before_sha256:
        return "absent"
    if current is None:
        return "mismatch"
    return "mismatch"


def classify(stored_replay: str, current_replay: str | None, observation: str) -> Decision:
    """Retry only when the stored policy and the current policy are both safe."""
    if observation == "match":
        return "match"
    if current_replay is None:
        return "blocked"
    if stored_replay == "safe" and current_replay == "safe":
        return "retry"
    if stored_replay == "safe":
        return "blocked"
    if observation == "absent":
        return "absent"
    return "unknown"


def _hash_file(workspace: Workspace, path: str) -> str | None:
    data = _read_optional(workspace, path)
    if data is None:
        return None
    return hashlib.sha256(data).hexdigest()


def _read_optional(workspace: Workspace, path: str) -> bytes | None:
    try:
        return workspace.read(path, remember=False)
    except FileNotFoundError:
        return None


def _diff(path: str, original: bytes | None, updated: bytes) -> str:
    try:
        return file_diff(path, original or b"", updated)
    except UnicodeError:
        return f"{path} changes {len(original or b'')} bytes to {len(updated)} bytes."


def _project_edit(original: bytes, arguments: JsonObject) -> bytes | None:
    edits = arguments.get("edits")
    if not isinstance(edits, list):
        old, new = arguments.get("old"), arguments.get("new")
        if not isinstance(old, str) or not isinstance(new, str):
            return None
        edits = [{"old": old, "new": new}]
    if any(
        not isinstance(edit, dict)
        or not isinstance(edit.get("old"), str)
        or not edit["old"]
        or not isinstance(edit.get("new"), str)
        for edit in edits
    ):
        return None
    try:
        return apply_edits(original, edits)
    except (ValueError, UnicodeError):
        return None
