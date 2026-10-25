"""Partial response evidence and bounded HTTP body reading."""

import http.client
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class StreamCapture:
    key: str
    path: Path | None
    chunks: list[bytes] = field(default_factory=list)
    pending: bytes = b""

    def persist(self, chunk: bytes, *, final: bool = False) -> None:
        if self.path is None:
            return
        # Retain the key-length suffix so a secret split across chunks is redacted.
        self.pending = (self.pending + chunk).replace(self.key.encode(), b"[redacted]")
        count = len(self.pending) if final else max(0, len(self.pending) - len(self.key.encode()) + 1)
        if count:
            with self.path.open("ab") as handle:
                handle.write(self.pending[:count])
                handle.flush()
                os.fsync(handle.fileno())
            self.pending = self.pending[count:]

    def mark(self, kind: str, size: int = 0) -> None:
        if self.path is None:
            return
        with self.path.with_name("stream.jsonl").open("a") as handle:
            handle.write(
                json.dumps({"kind": kind, "at": datetime.now(UTC).isoformat(), "bytes": size}) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())

    def read(
        self,
        response: http.client.HTTPResponse,
        on_chunk: Callable[[bytes], None] | None,
        stop: Callable[[], bool] | None,
        stop_after_chunks: int | None,
    ) -> str | None:
        error_code = None
        try:
            if stop is not None and stop():
                return "aborted"
            while True:
                chunk = response.read(256 if stop_after_chunks else 8192)
                if not chunk:
                    return "aborted" if stop is not None and stop() else None
                self.chunks.append(chunk)
                self.persist(chunk)
                self.mark("chunk", len(chunk))
                if on_chunk is not None:
                    on_chunk(chunk)
                if (stop is not None and stop()) or (
                    stop_after_chunks is not None and len(self.chunks) >= stop_after_chunks
                ):
                    return "aborted"
        except (http.client.IncompleteRead, OSError, TimeoutError) as error:
            partial = getattr(error, "partial", b"")
            if partial:
                self.chunks.append(partial)
                self.persist(partial)
            error_code = "aborted" if stop is not None and stop() else type(error).__name__
            return error_code
        finally:
            self.persist(b"", final=True)
