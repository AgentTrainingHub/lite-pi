"""Project extensions register tools and hooks. A failed load commits nothing."""

from __future__ import annotations

import copy
import hashlib
import re
import sys
import types
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict, cast, get_args
from uuid import uuid4

from jsonschema import Draft202012Validator

from lite_pi.tools.contracts import HOST_TOOLS, JsonObject, ToolDefinition, ToolHandler
from lite_pi.tools.workspace import Workspace

HOOK_INPUT_LIMIT = 8000
HOOK_OUTPUT_LIMIT = 4000
EXTENSION_ROOT = Path(".lite-pi") / "extensions"


class ExtensionError(ValueError):
    pass


type HookName = Literal["tool_call", "tool_result", "context"]
HOOKS = frozenset(get_args(HookName.__value__))


class HookInput(TypedDict, total=False):
    source: str
    name: str
    tool_call_id: str
    arguments: JsonObject
    result: JsonObject
    messages: list[JsonObject]


class HookDecision(TypedDict, total=False):
    block: bool
    reason: str
    text: str
    arguments: JsonObject
    revoke: str
    structured: JsonObject


type HookHandler = Callable[[HookInput], HookDecision | None]


@dataclass(frozen=True)
class ExtensionHost:
    write: Callable[[str, str], JsonObject]
    revoke: Callable[[str], None]


@dataclass(frozen=True)
class HookRegistration:
    event: HookName
    handler: HookHandler
    source: str


@dataclass(frozen=True)
class ExtensionInfo:
    name: str
    path: str
    sha256: str
    tools: tuple[str, ...]


@dataclass(frozen=True)
class LoadedExtensions:
    tools: tuple[ToolDefinition, ...] = ()
    hooks: tuple[HookRegistration, ...] = ()
    loaded: tuple[ExtensionInfo, ...] = ()
    apis: tuple[ExtensionAPI, ...] = ()
    revocations: tuple[str, ...] = ()

    def bind(self, host: ExtensionHost | None) -> None:
        for api in self.apis:
            api.bind(host)

    def report(self) -> list[JsonObject]:
        return [{**asdict(info), "tools": list(info.tools)} for info in self.loaded]


@dataclass
class _Pending:
    tools: list[ToolDefinition] = field(default_factory=list)
    hooks: list[tuple[HookName, HookHandler]] = field(default_factory=list)
    revocations: list[str] = field(default_factory=list)


class ExtensionAPI:
    def __init__(self, workspace: Workspace, pending: _Pending) -> None:
        self._workspace = workspace
        self._pending = pending
        self._host: ExtensionHost | None = None

    def register_tool(
        self,
        name: str,
        description: str,
        schema: JsonObject,
        execute: ToolHandler,
        replay: Literal["safe", "unsafe"] = "unsafe",
    ) -> None:
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", name) is None
            or name in HOST_TOOLS
            or name.startswith("mcp__")
        ):
            raise ExtensionError(f"invalid_tool_name:{name}")
        if any(item.name == name for item in self._pending.tools):
            raise ExtensionError(f"duplicate_tool:{name}")
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ExtensionError(f"invalid_tool_schema:{name}")
        Draft202012Validator.check_schema(schema)
        if not callable(execute) or not isinstance(description, str):
            raise ExtensionError(f"invalid_tool_handler:{name}")
        if replay not in {"safe", "unsafe"}:
            raise ExtensionError(f"invalid_replay:{name}")
        self._pending.tools.append(ToolDefinition(name, description, copy.deepcopy(schema), execute, replay))

    def on(self, event: HookName, handler: HookHandler) -> None:
        if event not in HOOKS or not callable(handler):
            raise ExtensionError(f"invalid_hook:{event}")
        self._pending.hooks.append((event, handler))

    def revoke_tool(self, name: str) -> None:
        if not isinstance(name, str) or name in HOST_TOOLS:
            raise ExtensionError(f"invalid_tool_name:{name}")
        if self._host is None:
            self._pending.tools = [tool for tool in self._pending.tools if tool.name != name]
            if name not in self._pending.revocations:
                self._pending.revocations.append(name)
            return
        self._host.revoke(name)

    def bind(self, host: ExtensionHost | None) -> None:
        self._host = host

    def write(self, path: str, content: str) -> JsonObject:
        if self._host is None:
            raise ExtensionError("extension_api_inactive")
        return self._host.write(path, content)

    def read(self, path: str) -> str:
        return self._workspace.read(path, remember=False).decode()


def load_extensions(root: Path, workspace: Workspace) -> LoadedExtensions:
    try:
        directories = workspace.directories(str(EXTENSION_ROOT))
    except FileNotFoundError:
        return LoadedExtensions()
    except (OSError, ValueError) as error:
        raise ExtensionError(f"invalid_extensions_dir:{error}") from error
    loaded: list[ExtensionInfo] = []
    tools: list[ToolDefinition] = []
    hooks: list[HookRegistration] = []
    apis: list[ExtensionAPI] = []
    revocations: list[str] = []
    names: set[str] = set()
    for directory in directories:
        child = root / EXTENSION_ROOT / directory
        script = child / "extension.py"
        relative = str(script.relative_to(root))
        try:
            data = workspace.read(relative, remember=False, max_bytes=256 * 1024)
        except (OSError, ValueError) as error:
            raise ExtensionError(f"invalid_extension_file:{directory}:{error}") from error
        pending = _Pending()
        api = ExtensionAPI(workspace, pending)
        module_name = f"lite_pi_extension_{child.name}_{uuid4().hex}"
        module = types.ModuleType(module_name)
        module.__file__ = str(script)
        sys.modules[module_name] = module
        try:
            exec(compile(data, str(script), "exec"), module.__dict__)
            register = getattr(module, "register", None)
            if not callable(register):
                raise ExtensionError(f"missing_register:{child.name}")
            register(api)
        except ExtensionError:
            raise
        except Exception as error:
            raise ExtensionError(
                f"extension_init_failed:{child.name}:{type(error).__name__}: {error}"
            ) from error
        finally:
            sys.modules.pop(module_name, None)
        for tool in pending.tools:
            if tool.name in names:
                raise ExtensionError(f"duplicate_tool:{tool.name}")
            names.add(tool.name)
            tools.append(tool)
        for event, handler in pending.hooks:
            hooks.append(HookRegistration(event, handler, relative))
        apis.append(api)
        revocations.extend(pending.revocations)
        loaded.append(
            ExtensionInfo(
                child.name,
                relative,
                hashlib.sha256(data).hexdigest(),
                tuple(tool.name for tool in pending.tools),
            )
        )
    return LoadedExtensions(tuple(tools), tuple(hooks), tuple(loaded), tuple(apis), tuple(revocations))


def bound_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit]


def hook_view[T](value: T, limit: int = HOOK_INPUT_LIMIT) -> T:
    """Preserve the event shape while bounding strings at the plugin boundary."""
    return cast(T, _bounded(value, limit))


def _bounded(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return bound_text(value, limit)
    if isinstance(value, list):
        return [_bounded(item, limit) for item in value]
    if isinstance(value, dict):
        return {str(key): _bounded(item, limit) for key, item in value.items()}
    return value
