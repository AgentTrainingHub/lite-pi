"""Unique, non-overlapping replacements preserve untouched bytes and file modes."""

import difflib
from itertools import pairwise

from lite_pi.tools.contracts import FileVersion, JsonObject
from lite_pi.tools.workspace import Workspace


def edit_many(
    workspace: Workspace, path: str, edits: list[dict[str, str]], *, expected_file: FileVersion | None = None
) -> JsonObject:
    with workspace.mutation(path):
        original = workspace.read(path, remember=False)
        updated = apply_edits(original, edits)
        diff = file_diff(path, original, updated)
        workspace.replace(path, updated, expected=original, overwrite=True, expected_file=expected_file)
    return {"text": diff, "path": path}


def write_file(
    workspace: Workspace,
    path: str,
    content: str,
    *,
    overwrite: bool = False,
    expected_file: FileVersion | None = None,
) -> JsonObject:
    with workspace.mutation(path):
        try:
            original = workspace.read(path, remember=False)
        except FileNotFoundError:
            original = None
        updated = content.encode("utf-8")
        diff = file_diff(path, original or b"", updated)
        workspace.replace(path, updated, expected=original, overwrite=overwrite, expected_file=expected_file)
    return {"text": diff, "path": path}


def file_diff(path: str, original: bytes, updated: bytes) -> str:
    return (
        "".join(
            difflib.unified_diff(
                original.decode("utf-8").splitlines(keepends=True),
                updated.decode("utf-8").splitlines(keepends=True),
                fromfile=path,
                tofile=path,
            )
        )
        or "(no textual change)"
    )


def apply_edits(original: bytes, edits: list[dict[str, str]]) -> bytes:
    original.decode("utf-8")
    spans = []
    for edit in edits:
        old, new = edit["old"].encode(), edit["new"].encode()
        count = original.count(old)
        if count != 1:
            raise ValueError(f"edit refused: old matched {count} times; file unchanged")
        start = original.index(old)
        spans.append((start, start + len(old), new))
    spans.sort()
    if any(right[0] < left[1] for left, right in pairwise(spans)):
        raise ValueError("edit refused: replacements overlap; file unchanged")
    updated = original
    for start, end, new in reversed(spans):
        updated = updated[:start] + new + updated[end:]
    return updated
