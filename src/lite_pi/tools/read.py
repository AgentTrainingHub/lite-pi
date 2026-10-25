"""Bounded UTF-8 reads with an exact cursor, including inside long lines."""

from lite_pi.tools.contracts import JsonObject
from lite_pi.tools.workspace import Workspace


def read_text(
    workspace: Workspace,
    path: str,
    *,
    offset: int = 1,
    limit: int = 2000,
    column: int = 0,
    max_bytes: int = 50 * 1024,
) -> JsonObject:
    data = workspace.read(path)
    lines = data.decode("utf-8").splitlines(keepends=True)
    if not lines and offset == 1 and column == 0:
        return {"text": "(empty file)", "truncation": None}
    if offset > len(lines) or column > len(lines[offset - 1]):
        raise ValueError("read cursor is beyond end of file")
    selected = "".join(lines[offset - 1 : offset - 1 + min(limit, 2000)])
    selected = selected[column:]
    content = selected.encode()[:max_bytes].decode("utf-8", errors="ignore")
    if not content and selected:
        raise ValueError("max_bytes cannot fit the next UTF-8 character")
    line_index, next_column = offset - 1, column
    remaining = len(content)
    while line_index < len(lines) and remaining >= len(lines[line_index]) - next_column:
        remaining -= len(lines[line_index]) - next_column
        line_index += 1
        next_column = 0
    next_column += remaining
    cursor = None if line_index == len(lines) else {"offset": line_index + 1, "column": next_column}
    text = content
    if cursor:
        text += f"\n[Continue with offset={cursor['offset']}, column={cursor['column']}.]"
    return {"text": text, "truncation": cursor, "content": content}
