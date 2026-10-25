"""Local stdio MCP server used to exercise the host. It is a real protocol peer."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server
from pydantic import AnyUrl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", default="serve")
    parser.add_argument("--state", required=True, type=Path)
    args = parser.parse_args()
    name, mode, state = args.name, args.mode, args.state
    state.mkdir(parents=True, exist_ok=True)
    (state / "pid").write_text(str(os.getpid()))
    if mode == "fail-init":
        print(f"fixture {name} refused to initialize", file=sys.stderr)
        raise SystemExit(2)
    if mode == "hang-init":
        import time

        time.sleep(60)
    server = Server(name, version="0.1.0")
    flags = {"shifted": False, "dropped": False}
    large_calls = 0

    def record(action: str) -> None:
        with (state / "calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(action + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @server.list_tools()
    async def list_tools(request: types.ListToolsRequest) -> types.ListToolsResult:
        tools: list[types.Tool] = []
        if not flags["dropped"] and mode in {
            "serve",
            "control",
            "crash",
            "pages",
            "no-resources",
            "structured",
        }:
            required = ["label", "generation"] if flags["shifted"] else ["label"]
            properties: dict = {"label": {"type": "string"}}
            if flags["shifted"]:
                properties["generation"] = {"type": "string"}
            tools.append(
                types.Tool(
                    name="mark",
                    description=f"Record a mark for {name}.",
                    inputSchema={
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                )
            )
        if mode == "control":
            for extra in ("shift", "drop"):
                tools.append(
                    types.Tool(
                        name=extra,
                        description=f"Test control {extra}.",
                        inputSchema={
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    )
                )
        if mode == "slow":
            tools.append(
                types.Tool(
                    name="slow",
                    description="Sleep inside the server.",
                    inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
                )
            )
        if mode == "large":
            tools.append(
                types.Tool(
                    name="large",
                    description="Return a large result.",
                    inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
                )
            )
        if mode == "pages":
            for tool in tools:
                tool.inputSchema.pop("additionalProperties", None)
        if mode == "pages" and (request is None or request.params is None or not request.params.cursor):
            return types.ListToolsResult(tools=[], nextCursor="second")
        return types.ListToolsResult(tools=tools)

    @server.call_tool()
    async def call_tool(tool_name: str, arguments: dict | None):
        nonlocal large_calls
        record(tool_name)
        if tool_name == "shift":
            flags["shifted"] = True
            return [types.TextContent(type="text", text="shifted")]
        if tool_name == "drop":
            flags["dropped"] = True
            return [types.TextContent(type="text", text="dropped")]
        if tool_name == "slow":
            path = Path("notes")
            path.mkdir(exist_ok=True)
            (path / "slow-partial.txt").write_text("PARTIAL\n")
            await asyncio.sleep(30)
            return [types.TextContent(type="text", text="done")]
        if tool_name == "large":
            large_calls += 1
            return [types.TextContent(type="text", text=("L" if large_calls == 1 else "R") * 100_000)]
        if tool_name == "mark" and mode == "crash":
            path = Path("notes")
            path.mkdir(exist_ok=True)
            (path / "crash.txt").write_text("crashed\n")
            os._exit(1)
        if tool_name != "mark":
            raise ValueError(f"unknown tool {tool_name}")
        if mode == "structured":
            return types.CallToolResult(
                isError=True,
                content=[
                    types.TextContent(type="text", text="partial"),
                    types.ImageContent(type="image", data="aW1hZ2U=", mimeType="image/png"),
                ],
                structuredContent={"written": 1, "status": "partial"},
            )
        token = f"{name.upper()}-MARK"
        notes = Path("notes")
        notes.mkdir(exist_ok=True)
        (notes / f"{name}.txt").write_text(token + "\n")
        return [types.TextContent(type="text", text=token)]

    @server.list_resources()
    async def list_resources(request: types.ListResourcesRequest) -> types.ListResourcesResult:
        record("list_resources")
        resources = [
            types.Resource(
                uri=AnyUrl("fixture://guide"),
                name="guide",
                mimeType="text/plain",
                description="Reference text for the fixture.",
            )
        ]

        if mode == "pages" and (request.params is None or not request.params.cursor):
            return types.ListResourcesResult(resources=[], nextCursor="second")
        return types.ListResourcesResult(resources=resources)

    @server.read_resource()
    async def read_resource(uri: AnyUrl) -> list[ReadResourceContents]:
        record("read_resource")
        if str(uri) != "fixture://guide":
            return [ReadResourceContents(content=f"unknown resource {uri}", mime_type="text/plain")]
        body = f"The guide token is {name.upper()}-GUIDE. This resource does not grant tools."
        return [ReadResourceContents(content=body, mime_type="text/plain")]

    if mode == "no-resources":
        del server.request_handlers[types.ListResourcesRequest]
        del server.request_handlers[types.ReadResourceRequest]

    async def run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(run())


if __name__ == "__main__":
    main()
