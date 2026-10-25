"""Stdio MCP client. Tool calls are not repeated after a transport failure."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import math
import os
import re
import threading
import time
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

import anyio
import mcp.types as types
from jsonschema import Draft202012Validator
from mcp import ClientSession
from mcp.types import LATEST_PROTOCOL_VERSION
from pydantic import AnyUrl

from lite_pi.browser.scope import Origin, host_accepts, url_allowed
from lite_pi.lifecycle import close_resources
from lite_pi.mcp_transport import OwnedStdio
from lite_pi.outcomes import tool_outcome
from lite_pi.tools.contracts import JsonObject, ToolCatalog, ToolDefinition, ToolHandler, obj
from lite_pi.tools.workspace import Workspace

SDK_VERSION = importlib.metadata.version("mcp")
VISIBLE_LIMIT = 8000
NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,32}$")
RESERVED = {"list_resources", "read_resource"}
UNSUPPORTED = (
    "http",
    "sse",
    "streamable-http",
    "oauth",
    "sampling",
    "prompts",
    "roots",
    "elicitation",
)


class McpError(ValueError):
    pass


@dataclass(frozen=True)
class McpConfig:
    name: str
    command: str
    args: tuple[str, ...]
    tools: tuple[str, ...]
    timeout_seconds: float
    origins: tuple[str, ...] = ()
    read_only_tools: tuple[str, ...] = ()


def mcp_name(server: str, tool: str) -> str:
    if (
        not NAME.fullmatch(server)
        or "__" in server
        or not NAME.fullmatch(tool)
        or len(f"mcp__{server}__{tool}") > 64
    ):
        raise McpError(f"invalid_mcp_name:{server}:{tool}")
    return f"mcp__{server}__{tool}"


def load_mcp_config(root: Path) -> tuple[McpConfig, ...]:
    try:
        data = Workspace(root).read(".lite-pi/mcp.json", remember=False, max_bytes=50 * 1024)
    except FileNotFoundError:
        return ()
    except (OSError, ValueError) as error:
        raise McpError(f"invalid_mcp_config:{error}") from error
    try:
        body = json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise McpError(f"invalid_mcp_config:{error}") from error
    servers = body.get("servers") if isinstance(body, dict) else None
    if not isinstance(servers, list):
        raise McpError("invalid_mcp_config")
    configs = tuple(_checked(spec) for spec in servers)
    names: set[str] = set()
    for config in configs:
        if config.name in names:
            raise McpError(f"duplicate_mcp_server:{config.name}")
        names.add(config.name)
    return configs


def _checked(spec: object) -> McpConfig:
    if not isinstance(spec, dict):
        raise McpError("invalid_mcp_server")
    if set(spec) - {"name", "tools", "timeout_seconds", "command", "args", "origins", "read_only_tools"}:
        raise McpError("unsupported_mcp_setting")
    name = spec.get("name")
    if not isinstance(name, str) or NAME.fullmatch(name) is None or "__" in name:
        raise McpError(f"invalid_mcp_server:{name}")
    tools = spec.get("tools", [])
    if (
        not isinstance(tools, list)
        or any(
            not isinstance(item, str) or NAME.fullmatch(item) is None or item in RESERVED for item in tools
        )
        or len(set(tools)) != len(tools)
    ):
        raise McpError(f"invalid_mcp_tools:{name}")
    for tool in [*tools, *RESERVED]:
        mcp_name(name, tool)
    timeout = spec.get("timeout_seconds", 8)
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
        raise McpError(f"invalid_mcp_timeout:{name}")
    command = spec.get("command")
    args = spec.get("args", [])
    if not isinstance(command, str) or not command or not isinstance(args, list):
        raise McpError(f"invalid_mcp_command:{name}")
    if any(not isinstance(item, str) for item in args):
        raise McpError(f"invalid_mcp_args:{name}")
    origins = spec.get("origins", [])
    if not isinstance(origins, list) or any(not isinstance(item, str) for item in origins):
        raise McpError(f"invalid_mcp_origins:{name}")
    try:
        parsed = tuple(Origin.parse(item).origin() for item in origins)
    except ValueError as error:
        raise McpError(f"invalid_mcp_origins:{name}") from error
    readonly = spec.get("read_only_tools", [])
    if not isinstance(readonly, list) or any(
        not isinstance(item, str) or item not in tools for item in readonly
    ):
        raise McpError(f"invalid_mcp_read_only_tools:{name}")
    return McpConfig(name, command, tuple(args), tuple(tools), float(timeout), parsed, tuple(readonly))


class McpServer:
    def __init__(self, spec: McpConfig, *, cwd: Path, state: Path) -> None:
        self.name = spec.name
        self.selected = spec.tools
        self.timeout = spec.timeout_seconds
        self.command = spec.command
        self.args = list(spec.args)
        self.origins = spec.origins
        self.read_only_tools = spec.read_only_tools
        self._pending_action: str | None = None
        self.cwd = cwd
        self.state = state
        self.state.mkdir(parents=True, exist_ok=True)
        self.err_path = state / "stderr.txt"
        self.session: ClientSession | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.stop: asyncio.Event | None = None
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.dead = False
        self.protocol: str | None = None
        self.server_info: JsonObject | None = None
        self.failures: list[str] = []
        self.calls = 0
        self._closed = False
        self._transport = OwnedStdio()
        self._closing = threading.Event()
        self._scope: anyio.CancelScope | None = None
        self.cancel_event: threading.Event | None = None
        self.has_resources = False
        self._err = self.err_path.open("w", encoding="utf-8")

    def start(self) -> None:
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=self._thread_main, name=f"mcp-{self.name}", daemon=True)
        self.loop = loop
        self.thread = thread
        thread.start()
        deadline = time.monotonic() + self.timeout
        while not self.ready.wait(0.05):
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise McpError(f"mcp_start_cancelled:{self.name}")
            if time.monotonic() >= deadline:
                raise McpError(f"mcp_timeout:{self.name}:{self._stderr()}")
        if self.error is not None:
            raise McpError(
                f"mcp_init_failed:{self.name}:{type(self.error).__name__}: {self.error}:{self._stderr()}"
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closing.set()
        loop = self.loop
        schedule_error: RuntimeError | None = None
        try:
            if loop is not None and loop.is_running():

                def shutdown() -> None:
                    if self.stop is not None:
                        self.stop.set()
                    if self._scope is not None:
                        self._scope.cancel()

                try:
                    loop.call_soon_threadsafe(shutdown)
                except RuntimeError as error:
                    # The owning thread can close its loop between the two calls.
                    schedule_error = error
                    self._transport.kill()
            if self.thread is not None:
                self.thread.join(self.timeout + 6)
                if self.thread.is_alive():
                    self._transport.kill()
                    self.thread.join(4)
                    if self.thread.is_alive():
                        raise McpError(f"mcp_cleanup_failed:{self.name}")
            if schedule_error is not None and loop is not None and not loop.is_closed():
                raise schedule_error
        finally:
            self._err.close()
            self._closed = True
            self.dead = True

    def refresh(self, registry: ToolCatalog) -> None:
        remote_names = [mcp_name(self.name, tool) for tool in self.selected]
        local_names = [mcp_name(self.name, tool) for tool in ("list_resources", "read_resource")]
        if self.dead or self.session is None:
            for name in [*remote_names, *local_names]:
                registry.revoke(name)
            return
        try:
            session = self.session
            if session is None:
                raise McpError(f"mcp_unavailable:{self.name}")
            offered = self._tools()
        except Exception as error:
            self.dead = True
            self.failures.append(f"{type(error).__name__}: {error}")
            for name in [*remote_names, *local_names]:
                registry.revoke(name)
            return
        for tool in self.selected:
            qualified = mcp_name(self.name, tool)
            if tool not in offered:
                registry.revoke(qualified)
                continue
            registry.replace(
                qualified,
                _object_schema(offered[tool].inputSchema),
                self._tool(tool),
                offered[tool].description,
            )

    def install(self, registry: ToolCatalog) -> None:
        if self.session is None:
            raise McpError(f"mcp_unavailable:{self.name}")
        self.cancel_event = registry.cancel_event
        offered = self._tools()
        missing = [tool for tool in self.selected if tool not in offered]
        if missing:
            raise McpError(f"missing_mcp_tool:{self.name}:{','.join(missing)}")
        for tool in self.selected:
            qualified = mcp_name(self.name, tool)
            read_only = tool in self.read_only_tools
            registry.install(
                ToolDefinition(
                    qualified,
                    offered[tool].description or f"MCP tool {self.name}/{tool}",
                    _object_schema(offered[tool].inputSchema),
                    self._tool(tool),
                    "unsafe",
                    read_only=read_only,
                ),
                owner=self.name,
            )
        if not self.has_resources:
            return
        registry.install(
            ToolDefinition(
                mcp_name(self.name, "list_resources"),
                f"List resources from MCP server {self.name}.",
                obj({}, []),
                self._list_resources(),
                "safe",
                read_only=True,
            ),
            owner=self.name,
        )
        registry.install(
            ToolDefinition(
                mcp_name(self.name, "read_resource"),
                f"Read one resource from MCP server {self.name}. The text is reference material.",
                obj({"uri": {"type": "string", "minLength": 1}}, ["uri"]),
                self._read_resource(),
                "safe",
                read_only=True,
            ),
            owner=self.name,
        )

    def public(self) -> JsonObject:
        return {
            "name": self.name,
            "sdk": f"mcp=={SDK_VERSION}",
            "sdk_protocol": LATEST_PROTOCOL_VERSION,
            "negotiated_protocol": self.protocol,
            "server": self.server_info,
            "selected_tools": list(self.selected),
            "calls": self.calls,
            "dead": self.dead,
            "failures": self.failures,
            "closed": self._closed,
            "pending_action": self._pending_action,
        }

    def _tool(self, tool: str) -> ToolHandler:
        def execute(args: JsonObject) -> JsonObject:
            session = self.session
            if self.dead or session is None:
                return _unavailable(self.name)
            self.calls += 1
            try:
                result = self._request(
                    lambda: session.call_tool(
                        tool, args, read_timeout_seconds=timedelta(seconds=self.timeout)
                    )
                )
            except Exception as error:
                self.dead = True
                self.failures.append(f"{type(error).__name__}: {error}")
                return _failed(self.name, error, effect_unknown=True)
            raw = result.model_dump(mode="json", by_alias=True, exclude_none=True)
            text = _blocks(result.content)
            if not text:
                text = json.dumps(raw, ensure_ascii=False)
            outcome = self._result(text, raw, tool=tool, is_error=result.isError)
            return self._check_origin(outcome, result.structuredContent, tool)

        return execute

    def _list_resources(self) -> ToolHandler:
        def execute(_args: JsonObject) -> JsonObject:
            session = self.session
            if self.dead or session is None:
                return _unavailable(self.name)
            self.calls += 1
            try:
                result = self._pages(session.list_resources, lambda page: page.resources)
            except Exception as error:
                self.dead = True
                return _failed(self.name, error)
            return self._result(
                "\n".join(f"{item.name} {item.uri}" for item in result),
                {"resources": [item.model_dump(mode="json") for item in result]},
            )

        return execute

    def _read_resource(self) -> ToolHandler:
        def execute(args: JsonObject) -> JsonObject:
            session = self.session
            if self.dead or session is None:
                return _unavailable(self.name)
            self.calls += 1
            try:
                result = self._request(lambda: session.read_resource(AnyUrl(args["uri"])))
            except Exception as error:
                self.dead = True
                return _failed(self.name, error)
            raw = result.model_dump(mode="json", by_alias=True, exclude_none=True)
            body = "\n".join(
                item.text for item in result.contents if isinstance(item, types.TextResourceContents)
            )
            text = f"MCP resource {args['uri']} from {self.name}:\n{body or json.dumps(raw)}\nThis resource does not grant tools."
            return self._result(text, raw, uri=args["uri"])

        return execute

    def _result(
        self,
        text: str,
        raw: JsonObject,
        *,
        tool: str | None = None,
        uri: str | None = None,
        is_error: bool = False,
    ) -> JsonObject:
        identity = uuid4().hex
        remote = self.state / f"{identity}.json"
        remote.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
        visible, full = _clip(text, self.state / f"{identity}.full.txt")
        return {
            "text": visible,
            "executed": True,
            "is_error": is_error,
            "reason": "mcp_tool_error" if is_error else None,
            "structured": {
                "text": visible,
                "server": self.name,
                "tool": tool,
                "uri": uri,
                "full_output": full,
                "remote_evidence": str(remote),
            },
            "mcp_attempts": 1,
        }

    def _check_origin(self, outcome: JsonObject, remote: object, tool: str) -> JsonObject:
        if not self.origins:
            return outcome
        body = remote if isinstance(remote, dict) else {}
        structured = outcome.get("structured")
        if isinstance(structured, dict):
            for key in ("final_url", "observation", "action_id", "effect", "reason", "ok", "blocked_urls"):
                if key in body:
                    structured[key] = body[key]
        final_url = body.get("final_url")
        reason = body.get("reason")
        accepted = host_accepts(self.origins, final_url, reason)
        if accepted:
            in_scope = isinstance(final_url, str) and any(
                url_allowed(final_url, Origin.parse(origin)) for origin in self.origins
            )
            accepted = in_scope or (body.get("ok") is False and body.get("effect") != "applied")
        if accepted:
            effect = body.get("effect")
            identity = body.get("action_id")
            if effect == "unknown":
                outcome["effect"] = "unknown"
                outcome["reason"] = reason
                if isinstance(identity, str) and identity and "query" in self.read_only_tools:
                    self._pending_action = identity
                    outcome["effect_key"] = f"{self.name}:{identity}"
                    outcome["reconcile_tool"] = mcp_name(self.name, "query")
            resolved = body.get("reconciled_action")
            if (
                tool == "query"
                and body.get("ok") is True
                and reason == "queried"
                and isinstance(resolved, str)
                and resolved == self._pending_action
            ):
                outcome["resolves_effect"] = f"{self.name}:{resolved}"
                self._pending_action = None
            return outcome
        message = "host rejected browser result outside the configured origin"
        outcome["text"] = message
        outcome["is_error"] = True
        outcome["reason"] = "host_origin_rejected"
        if tool not in self.read_only_tools:
            outcome["effect"] = "unknown"
        if isinstance(structured, dict):
            structured["text"] = message
            structured["reason"] = "host_origin_rejected"
        return outcome

    def _pages[P: types.PaginatedResult, T](
        self,
        fetch: Callable[[str | None], Coroutine[Any, Any, P]],
        select: Callable[[P], list[T]],
    ) -> list[T]:
        cursor: str | None = None
        seen: set[str] = set()
        items: list[T] = []
        for _ in range(100):
            page = self._request(partial(fetch, cursor))
            items.extend(select(page))
            cursor = page.nextCursor
            if cursor is None:
                return items
            if cursor in seen:
                raise McpError(f"mcp_pagination_cycle:{self.name}")
            seen.add(cursor)
        raise McpError(f"mcp_page_limit:{self.name}")

    def _tools(self) -> dict[str, types.Tool]:
        session = self.session
        if session is None:
            raise McpError(f"mcp_unavailable:{self.name}")
        offered: dict[str, types.Tool] = {}
        for tool in self._pages(session.list_tools, lambda page: page.tools):
            if tool.name in offered:
                raise McpError(f"duplicate_remote_tool:{self.name}:{tool.name}")
            _object_schema(tool.inputSchema)
            offered[tool.name] = tool
        return offered

    def _request[T](self, factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
        loop = self.loop
        if loop is None or self.session is None or not loop.is_running():
            raise McpError(f"mcp_unavailable:{self.name}")
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise McpError("cancelled")
        future = asyncio.run_coroutine_threadsafe(factory(), loop)
        deadline = time.monotonic() + self.timeout + 1
        while True:
            if self.cancel_event is not None and self.cancel_event.is_set():
                future.cancel()
                raise McpError("cancelled")
            try:
                return future.result(timeout=min(0.05, max(0, deadline - time.monotonic())))
            except TimeoutError:
                if future.done():
                    raise
                if time.monotonic() >= deadline:
                    future.cancel()
                    raise

    def _thread_main(self) -> None:
        loop = self.loop
        if loop is None:
            return
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._main())
        except asyncio.CancelledError:
            pass
        except Exception as error:
            self.error = error
            self.ready.set()
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _main(self) -> None:
        stop = asyncio.Event()
        self.stop = stop
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        for key in ("HOME", "PLAYWRIGHT_BROWSERS_PATH", "TMPDIR", "LANG"):
            value = os.environ.get(key)
            if value:
                env[key] = value
        try:
            async with (
                self._transport.connect(self.command, self.args, env, self.cwd, self._err) as (read, write),
                ClientSession(
                    read,
                    write,
                    read_timeout_seconds=timedelta(seconds=self.timeout),
                    client_info=types.Implementation(name="lite-pi", version="0.1.0"),
                ) as session,
            ):
                # Cancel only the request scope, so SDK context managers exit normally
                # and can finish their stdio EOF / TERM / KILL shutdown sequence.
                with anyio.CancelScope() as scope:
                    self._scope = scope
                    self.session = session
                    if self._closing.is_set():
                        return
                    init = await session.initialize()
                    self.has_resources = init.capabilities.resources is not None
                    self.protocol = str(init.protocolVersion)
                    self.server_info = {"name": init.serverInfo.name, "version": init.serverInfo.version}
                    self.ready.set()
                    await stop.wait()
                self._scope = None
        except Exception as error:
            self.error = error
            self.ready.set()

    def _stderr(self) -> str:
        try:
            self._err.flush()
            return self.err_path.read_text()[-1000:]
        except OSError:
            return ""


class McpHost:
    def __init__(self, servers: list[McpServer]) -> None:
        self.servers = servers

    @classmethod
    def open(cls, root: Path, evidence: Path, *, cancel_event: threading.Event | None = None) -> McpHost:
        specs = load_mcp_config(root)
        started: list[McpServer] = []
        try:
            for spec in specs:
                server = McpServer(
                    spec,
                    cwd=root,
                    state=evidence / "mcp" / spec.name,
                )
                started.append(server)
                server.cancel_event = cancel_event
                server.start()
        except BaseException as error:
            try:
                cls(started).close()
            except BaseException as cleanup:
                error.add_note(f"MCP startup cleanup failed: {cleanup}")
            raise
        return cls(started)

    def install(self, registry: ToolCatalog) -> None:
        for server in self.servers:
            server.install(registry)

    def prepare(self, registry: ToolCatalog) -> None:
        for server in self.servers:
            server.refresh(registry)

    def refresh_one(self, name: str, registry: ToolCatalog) -> None:
        for server in self.servers:
            if server.name == name:
                server.refresh(registry)
                return

    def close(self) -> None:
        failures = close_resources([(server.name, server.close) for server in self.servers])
        if failures:
            primary = failures[0].error
            for failure in failures:
                primary.add_note(failure.diagnostic())
            raise primary

    def report(self) -> JsonObject:
        return {
            "sdk": f"mcp=={SDK_VERSION}",
            "sdk_protocol": LATEST_PROTOCOL_VERSION,
            "unsupported": list(UNSUPPORTED),
            "servers": [server.public() for server in self.servers],
        }


def _object_schema(schema: object) -> JsonObject:
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise McpError("invalid_mcp_schema")
    Draft202012Validator.check_schema(schema)
    return dict(schema)


def _blocks(content: Sequence[types.ContentBlock]) -> str:
    return "\n".join(block.text for block in content if isinstance(block, types.TextContent))


def _clip(text: str, path: Path) -> tuple[str, str | None]:
    if len(text) <= VISIBLE_LIMIT:
        return text, None
    path.write_text(text)
    return text[:VISIBLE_LIMIT] + f"\n[truncated; full output: {path.name}]", str(path)


def _unavailable(name: str) -> JsonObject:
    text = f"MCP server {name} is unavailable"
    return tool_outcome(
        {"text": text, "structured": {"text": text, "server": name}, "mcp_attempts": 0},
        executed=False,
        is_error=True,
        reason="mcp_unavailable",
    )


def _failed(name: str, error: BaseException, *, effect_unknown: bool = False) -> JsonObject:
    text = f"{type(error).__name__}: {error}"
    return tool_outcome(
        {"text": text, "structured": {"text": text, "server": name}, "mcp_attempts": 1},
        is_error=True,
        reason="mcp_call_failed",
        effect="unknown" if effect_unknown else "none",
        executed=effect_unknown,
    )
