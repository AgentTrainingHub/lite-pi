"""Incremental provider SSE decoding; incomplete trailing lines remain buffered."""

import json
from dataclasses import dataclass, field

from lite_pi.tools.contracts import JsonObject


@dataclass
class StreamAssembler:
    pending: bytearray = field(default_factory=bytearray)
    content: list[str] = field(default_factory=list)
    calls: dict[int, JsonObject] = field(default_factory=dict)
    finish_reason: str | None = None
    model: str | None = None
    errors: list[str] = field(default_factory=list)

    _scanned: int = 0

    def feed(self, chunk: bytes) -> str:
        self.pending.extend(chunk)
        pieces: list[str] = []
        start = 0
        while (end := self.pending.find(b"\n", max(start, self._scanned))) >= 0:
            line = bytes(self.pending[start:end]).rstrip(b"\r")
            start = end + 1
            pieces.extend(self._line(line))
        if start:
            del self.pending[:start]
        self._scanned = len(self.pending)
        return "".join(pieces)

    def finish(self, *, partial: bool = False) -> None:
        if not partial and self.pending:
            self._line(bytes(self.pending))
            self.pending.clear()
        if not partial and self.errors:
            raise ValueError("invalid SSE: " + "; ".join(self.errors))

    def message(self) -> JsonObject:
        result: JsonObject = {"role": "assistant", "content": "".join(self.content)}
        if self.calls:
            result["tool_calls"] = [self.calls[index] for index in sorted(self.calls)]
        return result

    def _line(self, line: bytes) -> list[str]:
        if not line.startswith(b"data:"):
            return []
        raw = line[5:].strip()
        if raw in (b"", b"[DONE]"):
            return []
        try:
            event = json.loads(raw)
            if not isinstance(event, dict):
                raise ValueError("event must be an object")
            return self._event(event)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            # Keep reading so the spend gate can settle the actual response usage.
            self.errors.append(type(error).__name__ + ": malformed stream event")
            return []

    def _event(self, event: JsonObject) -> list[str]:
        if isinstance(event.get("model"), str):
            self.model = event["model"]
        pieces: list[str] = []
        for choice in event.get("choices") or []:
            if isinstance(choice.get("finish_reason"), str):
                self.finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                pieces.append(delta["content"])
                self.content.append(delta["content"])
            for tool in delta.get("tool_calls") or []:
                self._tool(tool)
        return pieces

    def _tool(self, tool: JsonObject) -> None:
        index = tool.get("index", 0)
        if not isinstance(index, int) or index < 0:
            raise ValueError("invalid tool index")
        current = self.calls.setdefault(
            index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
        )
        if isinstance(tool.get("id"), str):
            current["id"] = tool["id"]
        function = tool.get("function") or {}
        for key in ("name", "arguments"):
            if isinstance(function.get(key), str):
                current["function"][key] += function[key]
