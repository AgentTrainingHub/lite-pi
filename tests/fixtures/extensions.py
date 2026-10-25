"""Extension host setup and process checks for mechanism contracts."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lite_pi.extensions import load_extensions
from lite_pi.tools.registry import ToolRegistry
from lite_pi.tools.workspace import Workspace


def model(content: str = "done"):
    from lite_pi.provider import ModelResult

    return ModelResult(
        "stop",
        {"role": "assistant", "content": content},
        raw_stop_reason="stop",
        actual_model="deterministic-fixture",
        attempts=0,
    )


def alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class ExtensionFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lite-pi-extensions-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.work = self.base / "project"
        self.work.mkdir()
        runs = patch("lite_pi.session.RUNS", self.base / "runs")
        runs.start()
        self.addCleanup(runs.stop)

    def extension(self, name: str, source: str) -> None:
        path = self.work / ".lite-pi" / "extensions" / name
        path.mkdir(parents=True)
        (path / "extension.py").write_text(source)

    def registry(self) -> ToolRegistry:
        registry = ToolRegistry(
            Workspace(self.work),
            (),
            max_bytes=1024,
            shell_timeout=1,
            cancel_bash_after=None,
            evidence=self.base / "evidence",
        )
        registry.attach(load_extensions(self.work, Workspace(self.work)), None)
        return registry
