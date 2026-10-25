"""Stdio peer that reports a foreign final URL. The host must reject it."""

from __future__ import annotations

import asyncio

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server


def main() -> None:
    server = Server("gate", version="0.1.0")

    @server.list_tools()
    async def list_tools(_request: types.ListToolsRequest) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name="open",
                    description="Report a page.",
                    inputSchema={
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                        "additionalProperties": False,
                    },
                )
            ]
        )

    @server.call_tool()
    async def call_tool(_name: str, _arguments: dict[str, object] | None) -> types.CallToolResult:
        return types.CallToolResult(
            isError=False,
            content=[types.TextContent(type="text", text="opened http://example.com/")],
            structuredContent={
                "ok": True,
                "reason": "opened",
                "effect": "none",
                "final_url": "http://example.com/",
            },
        )

    async def run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(run())


if __name__ == "__main__":
    main()
