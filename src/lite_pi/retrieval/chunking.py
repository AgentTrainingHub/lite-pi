"""Split project text into locatable chunks. Headings stay with the section they open."""

from __future__ import annotations

from dataclasses import dataclass

MAX_CHARS = 800


@dataclass(frozen=True)
class TextChunk:
    start_line: int
    end_line: int
    text: str


def chunk_text(text: str, *, max_chars: int = MAX_CHARS) -> tuple[TextChunk, ...]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    lines = text.splitlines(keepends=True)
    if not lines:
        return ()
    sections: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if current and line.startswith("#"):
            sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)
    chunks: list[TextChunk] = []
    line_no = 1
    for section in sections:
        packed = _pack(section, line_no, max_chars)
        chunks.extend(packed)
        line_no += len(section)
    return tuple(chunks)


def _pack(lines: list[str], start: int, max_chars: int) -> list[TextChunk]:
    if len("".join(lines)) <= max_chars:
        return [TextChunk(start, start + len(lines) - 1, "".join(lines))]
    blocks: list[list[str]] = []
    block: list[str] = []
    for line in lines:
        if block and line.strip() == "":
            blocks.append(block)
            block = [line]
        else:
            block.append(line)
    if block:
        blocks.append(block)
    chunks: list[TextChunk] = []
    offset = 0
    pending: list[str] = []
    for block_lines in blocks:
        if len("".join(block_lines)) > max_chars:
            if pending:
                chunks.append(_emit(pending, start + offset - len(pending)))
                pending = []
            for piece in _hard(block_lines, max_chars):
                chunks.append(_emit(piece, start + offset))
                offset += len(piece)
            continue
        candidate = pending + block_lines
        if pending and len("".join(candidate)) > max_chars:
            chunks.append(_emit(pending, start + offset - len(pending)))
            pending = list(block_lines)
        else:
            pending = candidate
        offset += len(block_lines)
    if pending:
        chunks.append(_emit(pending, start + offset - len(pending)))
    return chunks


def _hard(lines: list[str], max_chars: int) -> list[list[str]]:
    pieces: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if current and len("".join(current)) + len(line) > max_chars:
            pieces.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        pieces.append(current)
    return pieces


def _emit(lines: list[str], start: int) -> TextChunk:
    return TextChunk(start, start + len(lines) - 1, "".join(lines))
