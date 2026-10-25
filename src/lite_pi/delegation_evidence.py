"""Read bounded file evidence through the same workspace boundary as tools."""

from __future__ import annotations

import hashlib
import os
import re
import stat

from lite_pi.delegation_report import FileProof
from lite_pi.tools.workspace import Workspace

EXCERPT_LIMIT = 1500
_FACT = re.compile(r"^FACT\s+(\S+):\s+(.+)$", re.MULTILINE)


def file_kind(workspace: Workspace, path: str) -> str:
    try:
        with workspace.parent(path) as (directory, name):
            info = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "error"
    if stat.S_ISDIR(info.st_mode):
        return "dir"
    if stat.S_ISREG(info.st_mode):
        return "file"
    return "other"


def _read_file(workspace: Workspace, path: str) -> bytes:
    return workspace.read(path, remember=False, max_bytes=50 * 1024)


def file_hashes(workspace: Workspace, paths: tuple[str, ...]) -> dict[str, str]:
    found: dict[str, str] = {}
    for path in paths:
        if file_kind(workspace, path) != "file":
            continue
        found[path] = hashlib.sha256(_read_file(workspace, path)).hexdigest()
    return found


def file_proofs(
    workspace: Workspace, paths: tuple[str, ...], before: dict[str, str]
) -> tuple[FileProof, ...]:
    proofs: list[FileProof] = []
    for path in paths:
        kind = file_kind(workspace, path)
        if kind == "missing":
            proofs.append(FileProof(path, False, None, before.get(path), "", "path_missing"))
            continue
        if kind != "file":
            proofs.append(FileProof(path, False, None, before.get(path), "", kind))
            continue
        try:
            data = _read_file(workspace, path)
        except (OSError, UnicodeError, ValueError) as error:
            proofs.append(
                FileProof(path, True, None, before.get(path), "", f"{type(error).__name__}: {error}")
            )
            continue
        digest = hashlib.sha256(data).hexdigest()
        try:
            text = data.decode()
        except UnicodeError as error:
            proofs.append(FileProof(path, True, digest, before.get(path), "", f"decode:{error}"))
            continue
        excerpt = text if len(text) <= EXCERPT_LIMIT else text[:EXCERPT_LIMIT] + "\n…截断"
        proofs.append(FileProof(path, True, digest, before.get(path), excerpt, None))
    return tuple(proofs)


def unsupported_claims(narrative: str, workspace: Workspace, scope: tuple[str, ...]) -> tuple[str, ...]:
    found: list[str] = []
    for match in _FACT.finditer(narrative):
        path, quote = match.group(1), match.group(2).strip()
        if len(found) >= 8:
            break
        try:
            relative = "/".join(workspace.parts(path))
        except (ValueError, OSError):
            found.append(f"{path}: path_scope")
            continue
        if not any(relative == item or relative.startswith(item.rstrip("/") + "/") for item in scope):
            found.append(f"{relative}: outside_survey")
            continue
        if not quote or len(quote) > 500 or file_kind(workspace, relative) != "file":
            found.append(f"{relative}: unsupported_quote")
            continue
        try:
            text = _read_file(workspace, relative).decode()
        except (OSError, UnicodeError, ValueError):
            found.append(f"{relative}: unreadable")
            continue
        if quote not in text:
            found.append(f"{relative}: quote_not_in_file")
    return tuple(found)
