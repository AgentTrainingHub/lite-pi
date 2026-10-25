"""Real PTY process driver. Controlled fixtures and real APIs share this outer seam."""

from __future__ import annotations

import codecs
import errno
import fcntl
import json
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import termios
import time
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

import pyte

ROOT = Path(__file__).resolve().parents[1]


class Terminal:
    def __init__(
        self,
        root: Path,
        evidence: Path,
        *,
        controlled: bool = False,
        session: Path | None = None,
        extra: tuple[str, ...] = (),
        columns: int = 80,
        rows: int = 24,
        color: bool = True,
    ) -> None:
        evidence.mkdir(parents=True)
        self.evidence = evidence
        self.master, self.slave = pty.openpty()
        self._resources = ExitStack()
        self._resources.callback(os.close, self.slave)
        self._resources.callback(os.close, self.master)
        self._closed = False
        try:
            self.before = termios.tcgetattr(self.slave)
            self.screen = pyte.Screen(columns, rows)
            self.stream = pyte.Stream(self.screen)
            self.resize(columns, rows, notify=False)
            command = [sys.executable]
            command += [str(ROOT / "tests/fixtures/tui_driver.py")] if controlled else ["-m", "lite_pi"]
            command += ["chat", "--root", str(root), *extra]
            if session is not None:
                command += ["--session", str(session)]
            env = {key: os.environ[key] for key in ("HOME", "TMPDIR") if key in os.environ}
            env.update(
                PATH="/usr/bin:/bin:/opt/homebrew/bin", TERM="xterm-256color", PYTHONDONTWRITEBYTECODE="1"
            )
            if not color:
                env["NO_COLOR"] = "1"
            self.process = subprocess.Popen(
                command,
                stdin=self.slave,
                stdout=self.slave,
                stderr=self.slave,
                cwd=ROOT,
                env=env,
                start_new_session=True,
            )
            self.raw = bytearray()
            self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            (evidence / "process.json").write_text(
                json.dumps(
                    {
                        "pid": self.process.pid,
                        "command": command,
                        "layer": "deterministic" if controlled else "real_model",
                    }
                )
                + "\n"
            )
            self._resources.callback(self._stop)
        except BaseException as error:
            if hasattr(self, "process"):
                self._resources.callback(self._stop)
            try:
                self._resources.close()
            except BaseException as cleanup:
                error.add_note(f"PTY initialization cleanup: {cleanup!r}")
            raise

    def resize(self, columns: int, rows: int, *, notify: bool = True) -> None:
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))
        self.screen.resize(rows, columns)
        if notify:
            os.kill(self.process.pid, signal.SIGWINCH)

    def read(self, timeout: float = 0.1) -> None:
        if select.select([self.master], [], [], timeout)[0]:
            try:
                chunk = os.read(self.master, 65536)
            except OSError as error:
                if error.errno == errno.EIO:
                    return
                raise
            self.raw.extend(chunk)
            self.stream.feed(self.decoder.decode(chunk))
            if b"\x1b[6n" in chunk:
                os.write(self.master, b"\x1b[1;1R")

    def send(self, data: str | bytes) -> None:
        os.write(self.master, data.encode() if isinstance(data, str) else data)

    def wait(self, predicate: Callable[[], bool], timeout: float = 20) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.read()
            if predicate():
                return
            if self.process.poll() is not None:
                raise AssertionError(f"PTY exited {self.process.returncode}: {self.text[-1500:]}")
        self.capture("timeout")
        raise AssertionError(f"PTY timeout: {self.text[-1800:]}")

    @property
    def text(self) -> str:
        # pyte.display indexes char[0] on orphan wide-character stubs during redraw.
        # A cleared stub occupies a blank cell; an intact wide character owns its next cell.
        from wcwidth import wcswidth

        lines = []
        for y in range(self.screen.lines):
            cells = []
            skip = False
            for x in range(self.screen.columns):
                if skip:
                    skip = False
                    continue
                char = self.screen.buffer[y][x].data or " "
                cells.append(char)
                skip = wcswidth(char) == 2
            lines.append("".join(cells))
        return "\n".join(lines)

    def capture(self, name: str) -> None:
        self.evidence.joinpath(name + ".txt").write_text(self.text)
        self.evidence.joinpath("raw.ansi").write_bytes(self.raw)

    def _stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(5)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        failure: BaseException | None = None
        try:
            self._finish()
        except BaseException as error:
            failure = error
            raise
        finally:
            try:
                self._resources.close()
            except BaseException as error:
                if failure is None:
                    raise
                failure.add_note(f"PTY cleanup: {error!r}")

    def _finish(self) -> None:
        if self.process.poll() is None:
            self.send(b"\x15/quit\r")
            end = time.monotonic() + 15
            while time.monotonic() < end and self.process.poll() is None:
                self.read()
        if self.process.poll() is None:
            self._stop()
        self.read(0.1)
        self.capture("closed")
        after = termios.tcgetattr(self.slave)
        # Darwin sets PENDIN on raw-to-canonical restoration; it is a pending-retype flag.
        expected, actual = list(self.before), list(after)
        expected[3] &= ~getattr(termios, "PENDIN", 0)
        actual[3] &= ~getattr(termios, "PENDIN", 0)
        self.evidence.joinpath("terminal.json").write_text(
            json.dumps(
                {
                    "returncode": self.process.returncode,
                    "restored": actual == expected,
                    "before": str(self.before),
                    "after": str(after),
                },
                indent=2,
            )
            + "\n"
        )
