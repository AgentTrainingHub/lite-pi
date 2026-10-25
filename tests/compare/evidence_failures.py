"""Diagnostic writes must preserve the exception that caused the failed run."""

from pathlib import Path

from support import save

from lite_pi.tools.contracts import JsonObject


def record_failure(
    path: Path,
    error: BaseException,
    *,
    primary: BaseException | None = None,
    details: JsonObject | None = None,
) -> None:
    original = primary if primary is not None else error
    if primary is not None:
        primary.add_note(f"Evidence capture failed: {type(error).__name__}: {error}")
    try:
        save(path, {"type": type(error).__name__, "reason": str(error), **(details or {})})
    except Exception as diagnostic_error:
        original.add_note(
            f"Diagnostic write {path.name} failed: {type(diagnostic_error).__name__}: {diagnostic_error}"
        )
