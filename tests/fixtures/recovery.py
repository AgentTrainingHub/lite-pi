"""Temporary project, persistent journal and admission fixture."""

import tempfile
import unittest
from pathlib import Path

from fixtures.responses import tool_call

from lite_pi.journal import Journal
from lite_pi.tasks import TaskBook
from lite_pi.tools.contracts import HaltAdmission, PreparedCall, RunAdmission
from lite_pi.tools.registry import ToolRegistry
from lite_pi.tools.workspace import Workspace


class RecoveryFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lite-pi-recovery-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.work = self.base / "project"
        self.work.mkdir()
        self.evidence = self.base / "evidence"
        self.evidence.mkdir()

    def journal(self, *tools: str) -> Journal:
        self.sequence = getattr(self, "sequence", 0) + 1
        return Journal.create(self.base / f"session-{self.sequence}.jsonl", self.work, tools=tools)

    def registry(self, *tools: str) -> ToolRegistry:
        return ToolRegistry(
            Workspace(self.work),
            tools,
            max_bytes=20_000,
            shell_timeout=5,
            cancel_bash_after=None,
            evidence=self.evidence,
        )

    def book(self, journal: Journal, **kwargs) -> TaskBook:
        opened = TaskBook(
            journal,
            Workspace(self.work),
            tools=journal.authorized_tools,
            approvals=kwargs.get("approvals", False),
            max_calls=kwargs.get("max_calls"),
            max_output=kwargs.get("max_output"),
            max_seconds=kwargs.get("max_seconds"),
            approval_ttl=kwargs.get("approval_ttl"),
            claim=kwargs.get("claim", True),
        )
        self.addCleanup(opened.close)
        return opened

    def assistant(self, journal: Journal, call: dict) -> None:
        journal.append("user", {"content": "continue"})
        journal.append("assistant", {"content": "", "tool_calls": [call]})

    def admit(self, book: TaskBook, prepared: PreparedCall | dict) -> RunAdmission | HaltAdmission:
        self.assertIsInstance(prepared, PreparedCall)
        tail = book.journal.incomplete_tail()
        if tail is None:
            self.assistant(book.journal, tool_call(prepared.name, prepared.arguments, prepared.tool_call_id))
            tail = book.journal.incomplete_tail()
        return book.admit(prepared, tail["id"])
