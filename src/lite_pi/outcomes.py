"""Host control fields decoded separately from extensible tool content."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypedDict, Unpack, cast, get_args

type RunStatus = Literal[
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "aborted",
    "error",
    "turn_limit",
    "budget",
    "blocked",
    "compaction_failed",
    "effect_unknown",
    "awaiting_approval",
    "awaiting_question",
    "awaiting_decision",
    "superseded",
    "rejected",
    "revoked",
    "expired",
    "cancelled",
    "queued",
]
type Effect = Literal["none", "applied", "unknown", "reconciled"]


def run_status(value: object) -> RunStatus:
    if value not in get_args(RunStatus.__value__):
        raise ValueError(f"invalid run status: {value!r}")
    return cast(RunStatus, value)


@dataclass(frozen=True)
class ToolControl:
    executed: bool
    is_error: bool
    reason: str | None
    effect: Effect
    blocks_run: bool
    halt: RunStatus | None
    effect_key: str | None
    reconcile_tool: str | None
    resolves_effect: str | None

    @classmethod
    def read(cls, value: Mapping[str, object]) -> "ToolControl":
        def flag(key: str, default: bool) -> bool:
            item = value.get(key, default)
            if not isinstance(item, bool):
                raise ValueError(f"invalid tool control: {key}")
            return item

        def optional_text(key: str) -> str | None:
            item = value.get(key)
            if item is not None and not isinstance(item, str):
                raise ValueError(f"invalid tool control: {key}")
            return item

        effect = value.get("effect", "none")
        if effect not in get_args(Effect.__value__):
            raise ValueError("invalid tool control: effect")
        halt = value.get("halt")
        return cls(
            flag("executed", True),
            flag("is_error", False),
            optional_text("reason"),
            cast(Effect, effect),
            flag("blocks_run", False),
            run_status(halt) if halt is not None else None,
            optional_text("effect_key"),
            optional_text("reconcile_tool"),
            optional_text("resolves_effect"),
        )


class ToolFields(TypedDict, total=False):
    """Fixed host controls; native producers must use typed constructor keywords."""

    executed: bool
    is_error: bool
    reason: str | None
    effect: Effect
    blocks_run: bool
    halt: RunStatus
    effect_key: str | None
    reconcile_tool: str | None
    resolves_effect: str | None


def tool_outcome(
    content: Mapping[str, object] | None = None, **control: Unpack[ToolFields]
) -> dict[str, object]:
    """Build host controls separately from tool-specific payload and external JSON."""
    result = dict(content or {})
    result.update(control)
    ToolControl.read(result)
    return result
