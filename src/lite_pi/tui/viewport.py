"""Read-only styled rows with an independent reading anchor and editable focus elsewhere."""

from collections.abc import Callable
from dataclasses import dataclass

from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout.controls import GetLinePrefixCallable, UIContent, UIControl
from wcwidth import iter_graphemes, wcswidth

from lite_pi.tui.markup import TextBlock, TextLine
from lite_pi.tui.text import fit, plain


@dataclass(frozen=True)
class Anchor:
    block: str
    line: int
    column: int


@dataclass(frozen=True)
class Row:
    anchor: Anchor
    fragments: StyleAndTextTuples
    visible_end: int | None = None


def wrap(block: str, line: TextLine, width: int) -> list[Row]:
    if line.clip:
        text = "".join(fragment[1] for fragment in line.fragments)
        visible = fit(text, width)
        return [
            Row(Anchor(block, line.source, 0), [("", visible)], len(visible) - 1 if visible != text else None)
        ]
    rows: list[Row] = []
    parts: StyleAndTextTuples = []
    cells = column = start = 0
    for style, text, *_ in line.fragments:
        # Markdown entity decoding can reintroduce controls after the source was sanitized.
        for grapheme in iter_graphemes(plain(text).expandtabs(4)):
            size = max(0, wcswidth(grapheme))
            if parts and cells + size > width:
                rows.append(Row(Anchor(block, line.source, start), parts))
                start = column
                parts = [("class:meta", "↪ ")]
                cells = 2
            if parts and parts[-1][0] == style:
                parts[-1] = (style, parts[-1][1] + grapheme)
            else:
                parts.append((style, grapheme))
            cells += size
            column += len(grapheme)
    rows.append(Row(Anchor(block, line.source, start), parts))
    return rows


class TranscriptControl(UIControl):
    def __init__(self, blocks: Callable[[], list[TextBlock]]) -> None:
        self.blocks = blocks
        self.following = True
        self.unread = False
        self.top = 0
        self.height = 1
        self.rows: list[Row] = []
        self.anchor: Anchor | None = None
        self._revisions: list[tuple[str, tuple[str, ...]]] = [
            (block.identity, block.revision) for block in blocks()
        ]
        self._cache: dict[str, tuple[list[TextLine], int, list[Row]]] = {}

    def preferred_height(
        self,
        width: int,
        max_available_height: int,
        wrap_lines: bool,
        get_line_prefix: GetLinePrefixCallable | None,
    ) -> int:
        return max_available_height

    def page(self, direction: int) -> None:
        self.top = max(
            0, min(max(0, len(self.rows) - self.height), self.top + direction * max(1, self.height - 1))
        )
        self.following = direction > 0 and self.top == max(0, len(self.rows) - self.height)
        if self.following:
            self.unread = False
        self.anchor = self.rows[self.top].anchor if self.rows else None

    def latest(self) -> None:
        self.following = True
        self.unread = False
        self.anchor = None

    def observe(self) -> None:
        revisions = [(block.identity, block.revision) for block in self.blocks()]
        if not self.following and revisions != self._revisions:
            self.unread = True
        self._revisions = revisions

    def create_content(self, width: int, height: int) -> UIContent:
        blocks = self.blocks()
        self.height = max(1, height)
        cache = {}
        rows = []
        for block in blocks:
            cached = self._cache.get(block.identity)
            if cached is None or cached[0] is not block.lines or cached[1] != width:
                rendered = [row for line in block.lines for row in wrap(block.identity, line, max(4, width))]
                cached = (block.lines, width, rendered)
            cache[block.identity] = cached
            rows.extend(cached[2])
        self._cache = cache
        self.rows = rows
        floor = max(0, len(rows) - height)
        if self.following:
            self.top = floor
        elif self.anchor is not None:
            choices = [(i, row) for i, row in enumerate(rows) if row.anchor.block == self.anchor.block]
            matching_line = [(i, row) for i, row in choices if row.anchor.line == self.anchor.line]
            candidates = [
                i
                for i, row in matching_line
                if row.anchor.column <= self.anchor.column
                and (row.visible_end is None or self.anchor.column < row.visible_end)
            ]
            # A collapsed detail no longer exists: retain its block, at the title.
            self.top = min(candidates[-1] if candidates else choices[0][0] if choices else self.top, floor)
        self.top = max(0, self.top)
        self.anchor = rows[self.top].anchor if rows else None
        return UIContent(
            get_line=lambda number: rows[number].fragments,
            line_count=len(rows),
            cursor_position=Point(x=0, y=max(0, len(rows) - 1) if self.following else self.top),
            show_cursor=False,
        )
