"""Markdown tokens become styled text; terminal control remains with prompt_toolkit."""

import re
from dataclasses import dataclass

from markdown_it import MarkdownIt
from markdown_it.token import Token
from prompt_toolkit.formatted_text import StyleAndTextTuples


@dataclass(frozen=True)
class TextLine:
    fragments: StyleAndTextTuples
    source: int
    clip: bool = False


@dataclass(frozen=True)
class TextBlock:
    identity: str
    revision: tuple[str, ...]
    lines: list[TextLine]


PARSER = MarkdownIt("commonmark", {"html": False})


def inline(tokens: list[Token], base: str = "") -> StyleAndTextTuples:
    parts: StyleAndTextTuples = []
    styles = [base]
    links: list[str] = []
    for token in tokens:
        kind = token.type
        if kind in {"strong_open", "em_open"}:
            styles.append(styles[-1] + (" bold" if kind == "strong_open" else " italic"))
        elif kind in {"strong_close", "em_close"}:
            styles.pop()
        elif kind == "code_inline":
            parts.append((styles[-1] + " class:code", token.content))
        elif kind in {"softbreak", "hardbreak"}:
            parts.append((styles[-1], "\n"))
        elif kind == "link_open":
            links.append(str(token.attrGet("href") or ""))
        elif kind == "link_close":
            parts.append(("class:meta", f" ({links.pop()})"))
        elif kind == "image":
            parts.append((styles[-1], f"![{token.content}]({token.attrGet('src') or ''})"))
        else:
            parts.append((styles[-1], token.content))
    return parts


def markdown(text: str) -> list[TextLine]:
    lines: list[TextLine] = []
    source_lines = text.splitlines()
    heading = False
    lists: list[int | None] = []
    prefix = ""
    last_end = 0
    for token in PARSER.parse(text):
        source = token.map[0] if token.map else last_end
        if token.map:
            last_end = token.map[1]
        kind = token.type
        if kind == "heading_open":
            heading = True
        elif kind == "heading_close":
            heading = False
        elif kind in {"bullet_list_open", "ordered_list_open"}:
            lists.append(None if kind == "bullet_list_open" else int(token.attrGet("start") or 1))
        elif kind in {"bullet_list_close", "ordered_list_close"}:
            lists.pop()
        elif kind == "list_item_open":
            number = lists[-1]
            prefix = "  " * (len(lists) - 1) + ("• " if number is None else f"{number}. ")
            if number is not None:
                lists[-1] = number + 1
        elif kind == "inline":
            parts = inline(token.children or [], "bold" if heading else "")
            current: StyleAndTextTuples = [("class:meta", prefix)] if prefix else []
            offset = 0
            for style, value, *_ in parts:
                for index, segment in enumerate(value.split("\n")):
                    if index:
                        lines.append(TextLine(current, source + offset))
                        offset += 1
                        current = []
                    current.append((style, segment))
            lines.append(TextLine(current, source + offset))
            prefix = ""
        elif kind in {"fence", "code_block"}:
            if kind == "fence" and token.map:
                end = token.map[1]
                closing = source_lines[end - 1].strip() if end else ""
                if (
                    re.fullmatch(re.escape(token.markup[0]) + "{" + str(len(token.markup)) + ",}", closing)
                    is None
                    or end <= source + 1
                ):
                    lines.extend(
                        TextLine([("", value)], source + i)
                        for i, value in enumerate(source_lines[source:end])
                    )
                    continue
            lines.append(TextLine([("class:meta", f"┌ {token.info or 'code'}")], source))
            for index, value in enumerate(token.content.splitlines()):
                style = ""
                if token.info.strip() == "diff" and value.startswith(("+", "-")):
                    style = "class:success" if value.startswith("+") else "class:failure"
                lines.append(TextLine([(style, value)], source + index + 1))
            lines.append(TextLine([("class:meta", "└")], source + len(token.content.splitlines()) + 1))
        elif kind == "paragraph_close" and not lists:
            lines.append(TextLine([], source))
        elif kind == "hr":
            lines.append(TextLine([("class:meta", "────")], source))
    while lines and not lines[-1].fragments:
        lines.pop()
    return lines
