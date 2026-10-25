"""Only journal identity and projection are needed for local session navigation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from lite_pi.journal import Journal
from lite_pi.session import AgentSession
from lite_pi.tui.display import tool_content
from lite_pi.tui.messages import Entry, Role, TextContent


@dataclass(frozen=True)
class SavedSession:
    path: Path
    identity: str


class Catalog:
    def __init__(self, project: Path, tools: tuple[str, ...], *, embedding_dir: Path | None = None) -> None:
        self.project = project.resolve(strict=True)
        self.tools = tools
        self.embedding_dir = embedding_dir
        self.directory = self.project / ".artifacts" / "sessions"

    def create(self) -> AgentSession:
        self._check(self.directory)
        journal = Journal.create(self.directory / f"{uuid4().hex}.jsonl", self.project, tools=self.tools)
        return AgentSession(self.project, tools=self.tools, journal=journal, embedding_dir=self.embedding_dir)

    def open(self, path: Path) -> AgentSession:
        self._check(path)
        for suffix in (".leaf", ".tasks.sqlite", ".tasks.sqlite-wal", ".tasks.sqlite-shm"):
            self._check(path.with_name(path.name + suffix))
        session = AgentSession.open(path, embedding_dir=self.embedding_dir)
        if session.project_root != self.project:
            raise ValueError("会话属于其他项目")
        if session.journal is not None:
            tail = session.journal.incomplete_tail()
            session.journal.messages(tail["parent_id"] if tail else None)
        return session

    def list_sessions(self) -> tuple[list[SavedSession], list[str]]:
        self._check(self.directory)
        found, errors = [], []
        for path in sorted(self.directory.glob("*.jsonl")):
            try:
                session = self.open(path)
                if session.journal is None:
                    raise RuntimeError("opened session has no journal")
                found.append(SavedSession(path, str(session.journal.records[0]["id"])))
            except (OSError, ValueError, RuntimeError) as error:
                errors.append(f"{path.name}: {error}")
        return found, errors

    @staticmethod
    def history(session: AgentSession) -> list[Entry]:
        if session.journal is None:
            raise RuntimeError("session history requires a journal")
        entries = []
        calls: dict[str, tuple[str, str]] = {}
        for record in session.journal.chain():
            if record["kind"] in {"user", "assistant", "tool"}:
                body = record["body"]
                if record["kind"] == "assistant":
                    calls = {
                        call["id"]: (call["function"]["name"], call["function"]["arguments"])
                        for call in body.get("tool_calls") or []
                    }
                if record["kind"] == "tool" and body.get("outcome"):
                    outcome = body["outcome"]
                    name, arguments = calls.get(
                        body["tool_call_id"], (str(outcome.get("name") or "tool"), "")
                    )
                    entries.append(Entry(tool_content(name, arguments, outcome), identity=str(record["id"])))
                else:
                    role = (
                        Role.USER
                        if record["kind"] == "user"
                        else Role.ASSISTANT
                        if record["kind"] == "assistant"
                        else Role.NOTICE
                    )
                    entries.append(
                        Entry(TextContent(role, str(body.get("content") or "")), identity=str(record["id"]))
                    )
        return entries

    @staticmethod
    def _check(path: Path) -> None:
        for part in (path, *path.parents):
            if part.is_symlink():
                raise ValueError(f"会话路径不允许符号链接：{part}")
        if path.is_file() and path.stat().st_nlink != 1:
            raise ValueError("会话路径不允许硬链接")
