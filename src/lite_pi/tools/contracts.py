"""Shared tool registration contracts; independent of registry and transports."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from threading import Event
from typing import Any, Literal, Protocol, TypedDict

from lite_pi.outcomes import RunStatus

# JSON Schema and third-party payloads are open-ended; internal registrations are typed below.
type JsonObject = dict[str, Any]
type ToolHandler = Callable[[JsonObject], JsonObject]

SESSION_TOOLS = frozenset({"plan", "question"})
HOST_TOOLS = frozenset({"read", "edit", "write", "bash", "skill"}) | SESSION_TOOLS


def obj(properties: JsonObject, required: list[str], **constraints: Any) -> JsonObject:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
        **constraints,
    }


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    schema: JsonObject
    execute: ToolHandler
    replay: Literal["safe", "unsafe"] = "unsafe"
    read_only: bool = False

    def advertised(self) -> JsonObject:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": deepcopy(self.schema),
            },
        }


class ToolCatalog(Protocol):
    """Operations available to a remote tool source, without execution internals."""

    @property
    def cancel_event(self) -> Event | None: ...

    def install(self, tool: ToolDefinition, owner: str | None = None) -> None: ...

    def replace(
        self, name: str, schema: JsonObject, execute: ToolHandler, description: str | None = None
    ) -> None: ...

    def revoke(self, name: str) -> None: ...


class ToolSource(Protocol):
    def install(self, registry: ToolCatalog) -> None: ...
    def prepare(self, registry: ToolCatalog) -> None: ...
    def refresh_one(self, name: str, registry: ToolCatalog) -> None: ...


@dataclass(frozen=True)
class PreparedCall:
    name: str
    tool_call_id: str
    arguments: JsonObject
    replay: Literal["safe", "unsafe"]
    read_only: bool = False


@dataclass(frozen=True)
class RunAdmission:
    intent_id: str


class HaltOutcome(TypedDict):
    name: str
    tool_call_id: str
    executed: Literal[False]
    is_error: Literal[True]
    text: str
    reason: str
    halt: RunStatus
    approval_id: str
    preview: str


@dataclass(frozen=True)
class HaltAdmission:
    outcome: HaltOutcome


@dataclass(frozen=True)
class FileVersion:
    sha256: str | None
