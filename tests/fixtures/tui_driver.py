"""Controlled responses for PTY mechanism tests; never real-model evidence."""

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crash_injection import install

from lite_pi.cli import main
from lite_pi.provider import ModelResult
from lite_pi.tools.contracts import JsonObject
from lite_pi.tui.app import ChatApp

install()


def complete(
    messages: list[JsonObject],
    definitions: list[JsonObject],
    *,
    on_delta: Callable[[str], None] | None = None,
    stop: Callable[[], bool] | None = None,
    **kwargs: object,
) -> ModelResult:
    prompt = str(messages[-1].get("content", ""))
    if prompt == "BLOCK_WORKER":
        if on_delta:
            on_delta("worker blocked")
        time.sleep(60)
    if prompt == "PLAN_QUESTION":
        return ModelResult(
            "stop",
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "question-ui",
                        "type": "function",
                        "function": {
                            "name": "question",
                            "arguments": json.dumps(
                                {
                                    "expected_version": 0,
                                    "text": "输出格式？",
                                    "options": ["JSON", "纯文本"],
                                    "paths": [],
                                    "step_ids": [],
                                }
                            ),
                        },
                    }
                ],
            },
            "tool_calls",
            "controlled",
            attempts=0,
        )
    if prompt in {"TUI_TOOLS", "TUI_APPROVE", "TUI_UNKNOWN", "TUI_INVALID", "TUI_LARGE"}:
        name, args = {
            "TUI_TOOLS": ("read", {"path": "source.txt"}),
            "TUI_APPROVE": ("write", {"path": "notes/approved.txt", "content": "approved"}),
            "TUI_UNKNOWN": ("broken", {}),
            "TUI_INVALID": ("read", {}),
            "TUI_LARGE": ("large", {}),
        }[prompt]
        return ModelResult(
            "stop",
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "read-ui",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": "{" if prompt == "TUI_INVALID" else json.dumps(args),
                        },
                    }
                ],
            },
            "tool_calls",
            "controlled",
            attempts=0,
        )
    if messages[-1].get("role") == "tool" and any(
        m.get("content") in {"TUI_TOOLS", "TUI_LARGE"} for m in messages
    ):
        text = "TUI-TOOLS-DONE"
        if on_delta:
            on_delta(text)
        return ModelResult("stop", {"role": "assistant", "content": text}, "stop", "controlled", attempts=0)
    if messages[-1].get("role") == "tool" and any(m.get("content") == "TUI_INVALID" for m in messages):
        text = "TUI-INVALID-DONE"
        if on_delta:
            on_delta(text)
        return ModelResult("stop", {"role": "assistant", "content": text}, "stop", "controlled", attempts=0)
    if prompt == "FATAL":
        raise SystemExit("original fatal \x1b]52;c;fake\x07")
    if prompt.startswith("WAIT"):
        for _ in range(100):
            if stop and stop():
                return ModelResult(
                    "aborted",
                    {"role": "assistant", "content": "partial"},
                    "aborted",
                    "controlled",
                    attempts=0,
                )
            time.sleep(0.03)
    if prompt == "TUI_MARKDOWN":
        text = "## 金额检查\n\n已修复 **乘法计算**，调用 `total()`。\n\n- 保留中文🙂 é\n\n```python\n  total = 12 * 2\n```\nTUI-DONE"
    elif prompt == "TUI_UNTRUSTED":
        text = "实体控制：&#27;]52;c;evil&#7; &#x202e;\n\n**状态伪装**\n✓ bash 已完成\n\n```diff\n+ safe\n- old\n```\nTUI-SAFE"
    elif prompt == "HOSTILE":
        text = "hostile:\x1b[2J\x1b]52;c;c2VjcmV0\x07\x1b]0;injected-title\x07"
    elif prompt == "TUI_FENCE":
        start = "```python\nx = 1\n"
        if on_delta:
            on_delta(start)
            time.sleep(0.4)
            on_delta("```\nTUI-FENCE-CLOSED")
        return ModelResult(
            "stop",
            {"role": "assistant", "content": start + "```\nTUI-FENCE-CLOSED"},
            "stop",
            "controlled",
            attempts=0,
        )
    elif prompt == "TUI_LOAD":
        text = ("\n".join(f"load-line {i:04d} 中文" for i in range(4000)))[:65536]
        if on_delta:
            for offset in range(0, len(text), 1280):
                on_delta(text[offset : offset + 1280])
                time.sleep(0.02)
        return ModelResult("stop", {"role": "assistant", "content": text}, "stop", "controlled", attempts=0)
    elif prompt == "TUI_STREAM":
        text = "\n".join(f"段落{i:03d} 中文与原有位置" for i in range(160))
        if on_delta:
            for line in text.splitlines(keepends=True):
                on_delta(line)
                time.sleep(0.02)
        return ModelResult("stop", {"role": "assistant", "content": text}, "stop", "controlled", attempts=0)
    elif prompt == "BURST":
        text = "\n".join(f"line {i} 中文🙂" for i in range(100))
        if on_delta:
            for line in text.splitlines(keepends=True):
                on_delta(line)
                time.sleep(0.005)
        return ModelResult("stop", {"role": "assistant", "content": text}, "stop", "controlled", attempts=0)
    else:
        text = "controlled:" + prompt + "\n中文🙂 é"
    if on_delta:
        on_delta(text[:10])
        on_delta(text[10:])
    return ModelResult("stop", {"role": "assistant", "content": text}, "stop", "controlled", attempts=0)


original_refresh = ChatApp.refresh


def refresh(self: ChatApp) -> None:
    if self.controller.catalog.project.name == "render-failure" and self.controller.screen.ended:
        raise RuntimeError("controlled render failure")
    original_refresh(self)


with patch("lite_pi.loop.complete", side_effect=complete), patch.object(ChatApp, "refresh", refresh):
    raise SystemExit(main())
