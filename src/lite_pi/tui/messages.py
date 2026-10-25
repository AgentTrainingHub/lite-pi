"""Typed display content and its cached, read-only presentation."""

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path
from uuid import uuid4

from prompt_toolkit.formatted_text import StyleAndTextTuples

from lite_pi.tui.markup import TextBlock, TextLine, markdown
from lite_pi.tui.text import plain


def detail_preview(text: str, path: str) -> str:
    # Keep expansion bounded; the protected receipt remains the complete source.
    limit = 4096
    if len(text.encode("utf-8")) <= 64 * 1024:
        return text
    receipt = str(Path(path).parent.parent / "result.json") if path else "见运行 evidence"
    return text[:limit] + f"\n显示预览已截断；运行回执位置：{receipt}\n原始工具证据：{path or '见工具回执'}"


class SourceRegion(IntEnum):
    # Separate optional detail sections from the source line numbers in message bodies.
    DETAILS = 1000000
    ORIGINAL = 1500000
    SPACING = 2000000


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    NOTICE = "notice"
    DIAGNOSTIC = "diagnostic"
    RECEIPT = "receipt"
    APPROVAL = "approval"
    CONTROL = "control"


class ToolState(StrEnum):
    RUNNING = "执行中"
    DONE = "已完成"
    FAILED = "失败"
    NOT_EXECUTED = "未执行"
    UNKNOWN = "效果未知"


@dataclass(frozen=True)
class TextContent:
    role: Role
    text: str
    details: str = ""

    def lines(self, expanded: bool) -> list[TextLine]:
        if self.role == Role.ASSISTANT:
            if not self.text:
                return []
            return [TextLine([("class:assistant", "● lite-pi")], -1), *markdown(plain(self.text))]
        if self.role == Role.USER:
            title, style = "› 你", "class:user"
        elif self.role in {Role.RECEIPT, Role.APPROVAL, Role.CONTROL}:
            values = plain(self.text).splitlines()
            return [
                TextLine([("bold", values[0])], -1),
                *[TextLine([("", value)], i) for i, value in enumerate(values[1:])],
                *self.detail_lines(expanded),
            ]
        else:
            title, style = (
                ("! 诊断", "class:failure") if self.role == Role.DIAGNOSTIC else ("· 系统", "class:meta")
            )
        values = plain(self.text).splitlines()
        return [
            TextLine([(style, title)], -1),
            *[TextLine([("", value)], i) for i, value in enumerate(values)],
            *self.detail_lines(expanded),
        ]

    def detail_lines(self, expanded: bool) -> list[TextLine]:
        if not expanded or not self.details:
            return []
        return [
            TextLine([("class:meta", value)], SourceRegion.DETAILS + i)
            for i, value in enumerate(plain(self.details).splitlines())
        ]

    def revision(self) -> tuple[str, ...]:
        return (self.role, self.text, self.details)


@dataclass(frozen=True)
class ToolContent:
    name: str
    target: str
    state: ToolState
    output: str = ""
    details: str = ""
    reason: str = ""
    truncated: bool = False
    evidence_path: str = ""
    original: str | None = None

    def lines(self, expanded: bool) -> list[TextLine]:
        icon, style = {
            ToolState.RUNNING: ("⠋", "class:tool"),
            ToolState.DONE: ("✓", "class:success"),
            ToolState.FAILED: ("!", "class:failure"),
            ToolState.NOT_EXECUTED: ("!", "class:failure"),
            ToolState.UNKNOWN: ("?", "class:attention"),
        }[self.state]
        title: StyleAndTextTuples = [
            (style, icon + " "),
            ("bold", self.name),
            ("", " · " + self.target if self.target else ""),
            ("bold", "  " + self.state.value),
        ]
        lines = [TextLine(title, -1)]
        if self.reason:
            lines.append(TextLine([("", "原因：" + self.reason)], 0))
        output = plain(
            detail_preview(self.output, self.evidence_path) if expanded else self.output[:4096]
        ).splitlines()
        visible = output if expanded else output[:3]
        lines.extend(
            TextLine([("", "  " + value)], i + 1, clip=not expanded) for i, value in enumerate(visible)
        )
        if not expanded and (self.output.count("\n") + 1) > 3:
            lines.append(
                TextLine(
                    [
                        (
                            "class:meta",
                            f"  … 另有 {max(0, len(self.output.splitlines()) - 3)} 行 · Ctrl+O 工具详情",
                        )
                    ],
                    len(output) + 1,
                )
            )
        if not expanded and len(self.output) > 4096:
            lines.append(TextLine([("bold", "显示摘要已截断 · Ctrl+O 查看回执")], len(output) + 2))
        if self.truncated:
            lines.append(TextLine([("bold", "结果已截断；Ctrl+O 读取原始证据")], len(output) + 2))
        if expanded:
            lines.extend(
                TextLine([("class:meta", value)], SourceRegion.DETAILS + i)
                for i, value in enumerate(
                    plain(detail_preview(self.details, self.evidence_path)).splitlines()
                )
            )
            if self.original is not None:
                lines.extend(
                    TextLine([("class:meta", value)], SourceRegion.ORIGINAL + i)
                    for i, value in enumerate(plain(self.original).splitlines())
                )
        return lines

    def revision(self) -> tuple[str, ...]:
        return (
            self.name,
            self.target,
            self.state,
            self.output,
            self.details,
            self.reason,
            str(self.truncated),
            self.evidence_path,
        )


@dataclass
class Entry:
    content: TextContent | ToolContent
    key: str = ""
    identity: str = field(default_factory=lambda: uuid4().hex)
    _cache_key: tuple[TextContent | ToolContent, bool] | None = field(default=None, repr=False)
    _lines: list[TextLine] = field(default_factory=list, repr=False)

    def block(self, expanded: bool) -> TextBlock:
        key = (self.content, expanded)
        if self._cache_key != key:
            lines = self.content.lines(expanded)
            self._lines = [*lines, TextLine([], SourceRegion.SPACING)] if lines else []
            self._cache_key = key
        return TextBlock(self.identity, self.content.revision(), self._lines)
