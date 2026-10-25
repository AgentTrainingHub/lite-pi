"""Owned stdio transport with bounded shutdown, using public MCP message types."""

import os
import signal
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TextIO

import anyio
from anyio.abc import Process
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from anyio.streams.text import TextReceiveStream
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage
from pydantic import ValidationError

type TransportStreams = tuple[
    MemoryObjectReceiveStream[SessionMessage | Exception], MemoryObjectSendStream[SessionMessage]
]


class OwnedStdio:
    """Only this instance may terminate the process group it started."""

    def __init__(self) -> None:
        self._process: Process | None = None
        self._lock = threading.Lock()

    def kill(self) -> None:
        with self._lock:
            process = self._process
            if process is not None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)

    @asynccontextmanager
    async def connect(
        self, command: str, args: list[str], env: dict[str, str], cwd: Path, stderr: TextIO
    ) -> AsyncIterator[TransportStreams]:
        process = await anyio.open_process(
            [command, *args], env=env, cwd=cwd, stderr=stderr, start_new_session=True
        )
        with self._lock:
            self._process = process
        incoming, read = anyio.create_memory_object_stream[SessionMessage | Exception](0)
        write, outgoing = anyio.create_memory_object_stream[SessionMessage](0)
        try:
            async with anyio.create_task_group() as group:
                group.start_soon(self._read, process, incoming)
                group.start_soon(self._write, process, outgoing)
                try:
                    yield read, write
                finally:
                    # Shield cleanup from the request's cancellation scope. Descendants
                    # can keep stdout open after the leader exits, so kill the group too.
                    with anyio.CancelScope(shield=True):
                        try:
                            if process.stdin is not None:
                                await process.stdin.aclose()
                            with anyio.move_on_after(1):
                                await process.wait()
                        finally:
                            self.kill()
                            with anyio.fail_after(3):
                                await process.wait()
                            group.cancel_scope.cancel()
        finally:
            with anyio.CancelScope(shield=True):
                self.kill()
                await read.aclose()
                await write.aclose()
                await incoming.aclose()
                await outgoing.aclose()
                await process.aclose()
            with self._lock:
                self._process = None

    @staticmethod
    async def _read(process: Process, incoming: MemoryObjectSendStream[SessionMessage | Exception]) -> None:
        if process.stdout is None:
            raise RuntimeError("MCP stdout was not opened")
        pending = ""
        async with incoming:
            async for chunk in TextReceiveStream(process.stdout):
                lines = (pending + chunk).split("\n")
                pending = lines.pop()
                for line in lines:
                    try:
                        message = JSONRPCMessage.model_validate_json(line)
                    except ValidationError as error:
                        await incoming.send(error)
                    else:
                        await incoming.send(SessionMessage(message))

    @staticmethod
    async def _write(process: Process, outgoing: MemoryObjectReceiveStream[SessionMessage]) -> None:
        if process.stdin is None:
            raise RuntimeError("MCP stdin was not opened")
        async with outgoing:
            async for message in outgoing:
                body = message.message.model_dump_json(by_alias=True, exclude_none=True)
                await process.stdin.send((body + "\n").encode())
