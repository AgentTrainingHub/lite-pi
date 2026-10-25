"""Prompt-toolkit owns terminal I/O; Session owns all model and tool effects."""

import asyncio
import os
import shlex
import signal
import sqlite3
import sys
import traceback
from decimal import ROUND_CEILING, Decimal
from importlib.metadata import version
from pathlib import Path

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.patch_stdout import patch_stdout
from wcwidth import wcswidth

from lite_pi.session import RunLimits
from lite_pi.tui.catalog import Catalog
from lite_pi.tui.controller import Controller
from lite_pi.tui.display import welcome
from lite_pi.tui.messages import Role, TextContent
from lite_pi.tui.text import fit, plain
from lite_pi.tui.theme import STYLE
from lite_pi.tui.viewport import TranscriptControl


def resume_command(project: Path, path: Path) -> str:
    # ANSI-C quoting preserves unusual filenames without printing terminal controls.
    def quote(value: str) -> str:
        safe = plain(value)
        if safe == value and "\n" not in value and "\t" not in value:
            return shlex.quote(value)
        return (
            "$'"
            + "".join(
                "".join(f"\\{byte:03o}" for byte in c.encode("utf-8"))
                if plain(c) != c or c in "\n\t"
                else "\\" + c
                if c in "\\'"
                else c
                for c in value
            )
            + "'"
        )

    return f"lite-pi chat --root {quote(str(project))} --session {quote(str(path))}"


class ChatApp:
    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self._screen = controller.screen
        self.buffer = Buffer(multiline=True, history=InMemoryHistory())
        self.view = TranscriptControl(lambda: self.controller.screen.blocks())
        self.transcript = Window(self.view, get_vertical_scroll=lambda _window: self.view.top)
        self.keys = KeyBindings()
        self._keys()
        editor = Window(
            BufferControl(buffer=self.buffer),
            height=self._editor_height,
            wrap_lines=True,
            get_line_prefix=lambda _line, wrap: "> " if not wrap else "  ",
        )
        intro = Window(FormattedTextControl(self._welcome), dont_extend_height=True)
        small = Condition(
            lambda: self.app.output.get_size().columns < 40 or self.app.output.get_size().rows < 12
        )
        hint = Window(FormattedTextControl(self._status), height=2, wrap_lines=False)
        layout = HSplit(
            [
                ConditionalContainer(
                    intro,
                    filter=Condition(lambda: not self.controller.screen.entries) & ~small,
                ),
                ConditionalContainer(
                    self.transcript, filter=Condition(lambda: bool(self.controller.screen.entries)) & ~small
                ),
                ConditionalContainer(
                    Window(FormattedTextControl(lambda: "有新内容 · Ctrl+End 返回最新"), height=1),
                    filter=Condition(lambda: self.view.unread and not self.view.following),
                ),
                ConditionalContainer(
                    Window(
                        FormattedTextControl("窗口过小，至少 40×12；Ctrl+D 空草稿退出"),
                        height=2,
                        wrap_lines=True,
                    ),
                    filter=small,
                ),
                ConditionalContainer(
                    Window(FormattedTextControl(self._details), height=4, wrap_lines=False),
                    filter=Condition(lambda: self.controller.screen.expanded) & ~small,
                ),
                Window(height=1, char="─", style="class:border"),
                editor,
                Window(height=1, char="─", style="class:border"),
                hint,
            ]
        )
        self.app: Application[None] = Application(
            layout=Layout(layout, focused_element=editor),
            key_bindings=self.keys,
            full_screen=False,
            style=STYLE,
            color_depth=ColorDepth.DEPTH_1_BIT if "NO_COLOR" in os.environ else None,
            mouse_support=False,
        )

    def _editor_height(self) -> int:
        size = self.app.output.get_size()
        width = max(1, size.columns - 2)
        visual = sum(
            max(1, (max(0, wcswidth(line.expandtabs(4))) + width - 1) // width)
            for line in self.buffer.text.split("\n")
        )
        return min(visual, 3 if size.rows <= 12 else 4)

    def _welcome(self) -> StyleAndTextTuples:
        c = self.controller
        identity = str(c.session.journal.records[0]["id"])[:8] if c.session.journal else "新会话"
        return welcome(
            self.app.output.get_size().columns,
            c.catalog.project,
            identity,
            c.session.tools,
            version("lite-pi"),
        )

    def _status(self) -> str:
        c = self.controller
        width = self.app.output.get_size().columns
        phase = self._phase()
        model = c.screen.actual_model or "deepseek-flash · 配置"
        amount = Decimal(str(c.ledger["spent_cny"])).quantize(Decimal("0.0001"), rounding=ROUND_CEILING)
        limit = Decimal(str(c.ledger["allowance_cny"])).normalize()
        money = f"约 {amount}/{limit} CNY"
        money_width = max(0, wcswidth(money))
        left = fit(f"{phase} · {model}", max(0, width - money_width - 1))
        main = left + " " * max(1, width - max(0, wcswidth(left)) - money_width) + money
        if c.busy:
            keys = "取消中，等待收尾" if c.cancel_requested else "Enter 转向 · Ctrl+C 取消 · Ctrl+O 详情"
        elif c.screen.blocked:
            keys = "Enter 保留草稿 · Ctrl+O 详情 · /help"
        else:
            keys = "Enter 提交 · Alt+Enter/Ctrl+J 换行 · Ctrl+D 退出 · Ctrl+O 详情"
        return fit(main, width) + "\n" + fit(keys, width)

    def _phase(self) -> str:
        c = self.controller
        if c.screen.phase == "effect_unknown":
            return "效果未知，当前会话受阻"
        if c.closing:
            return "退出中"
        if c.busy:
            return "取消中，等待收尾" if c.cancel_requested else "运行中"
        return c.screen.phase_label

    def _details(self) -> str:
        c = self.controller
        width = self.app.output.get_size().columns
        usage = c.screen.result.get("usage") or {}
        tokens = (
            "未知" if c.screen.result.get("usage_unknown") or not usage else usage.get("total_tokens", "未知")
        )
        task = (c.screen.result.get("tasks") or {}).get("budget") or {}
        values = [
            f"目录 {c.catalog.project} · 会话 {c.path.name}",
            f"配置 deepseek-flash · 实际 {c.screen.actual_model or '未知'} · 本次 token {tokens}",
            f"累计 {c.ledger['spent_cny']}/{c.ledger['allowance_cny']} CNY · 预留 {c.ledger.get('reserved_cny')} · 未知 {c.ledger.get('unknown_usage_calls')}",
            "任务预算 " + str(task),
        ]
        return "\n".join(fit(value, width) for value in values)

    def _keys(self) -> None:
        @self.keys.add("enter")
        def submit(event: KeyPressEvent) -> None:
            del event
            text = self.buffer.text
            if text.startswith(("/new", "/sessions")) and "\n" in text:
                self.controller.screen.note("会话切换前请先处理多行草稿")
                self.refresh()
                return
            try:
                accepted = self.controller.submit(text)
            except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
                self.controller.screen.note(f"拒绝 · {type(error).__name__}: {error}")
            else:
                if accepted:
                    self.buffer.append_to_history()
                    self.buffer.reset()
            self.refresh()
            if self.controller.closing:
                self.app.exit()

        @self.keys.add("c-j")
        @self.keys.add("escape", "enter")
        def newline(event: KeyPressEvent) -> None:
            event.current_buffer.insert_text("\n")

        @self.keys.add("c-c")
        def cancel(event: KeyPressEvent) -> None:
            if self.controller.busy:
                self.controller.cancel()
            else:
                if self.controller.screen.phase == "awaiting_question":
                    self.controller.cancel()
                event.current_buffer.reset()
            self.refresh()

        @self.keys.add("escape")
        def escape(event: KeyPressEvent) -> None:
            del event
            self.controller.cancel()
            self.refresh()

        @self.keys.add("c-d")
        def exit_or_delete(event: KeyPressEvent) -> None:
            if event.current_buffer.text:
                event.current_buffer.delete()
            elif not self.controller.busy:
                self.controller.close()
                self.app.exit()
            else:
                self.controller.screen.note("/quit 取消并退出；Ctrl+C 仅取消")
                self.refresh()

        @self.keys.add("c-o")
        def expand(event: KeyPressEvent) -> None:
            del event
            self.controller.toggle_details()
            self.refresh()

        @self.keys.add("pageup")
        @self.keys.add("pagedown")
        def scroll(event: KeyPressEvent) -> None:
            self.view.page(-1 if event.key_sequence[0].key == "pageup" else 1)
            self.app.invalidate()

        @self.keys.add("c-end")
        def latest(event: KeyPressEvent) -> None:
            self.view.latest()
            self.app.invalidate()

    def refresh(self) -> None:
        if self._screen is not self.controller.screen:
            self._screen = self.controller.screen
            self.view.latest()
        self.view.observe()
        self.app.invalidate()

    async def run(self) -> None:
        async def pump() -> None:
            try:
                while True:
                    if self.controller.drain():
                        if self.controller.restored_draft:
                            draft = self.controller.restored_draft
                            self.controller.restored_draft = ""
                            self.buffer.text = draft + ("\n" + self.buffer.text if self.buffer.text else "")
                        self.refresh()
                    if self.controller.closing:
                        self.app.exit()
                        return
                    await asyncio.sleep(0.04)
            except Exception as error:
                self.app.exit(exception=error)

        def ready() -> None:
            self.app.create_background_task(pump())

        loop = asyncio.get_running_loop()
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
        for sig in previous:
            loop.add_signal_handler(sig, self.controller.close)
        failure: BaseException | None = None
        try:
            self.refresh()
            await self.app.run_async(pre_run=ready, set_exception_handler=False)
        except EOFError:
            self.controller.close()
        except BaseException as error:
            failure = error
            raise
        finally:
            try:
                self.controller.join()
            except BaseException as error:
                if failure is None:
                    raise
                failure.add_note(f"Session 收尾失败：{type(error).__name__}: {error}")
            finally:
                for sig, handler in previous.items():
                    loop.remove_signal_handler(sig)
                    signal.signal(sig, handler)


def run_chat(
    project: Path,
    session_path: Path | None,
    tools: tuple[str, ...],
    limits: RunLimits,
    *,
    embedding_dir: Path | None = None,
) -> int:
    catalog = Catalog(project, tools, embedding_dir=embedding_dir)
    session = catalog.open(session_path) if session_path else catalog.create()
    controller = Controller(catalog, session, limits)
    try:
        with patch_stdout(raw=False):
            asyncio.run(ChatApp(controller).run())
    except Exception as error:
        controller.failed = True
        print(plain("".join(traceback.format_exception(error))), file=sys.stderr)
    if controller.failed:
        for entry in controller.screen.entries:
            if isinstance(entry.content, TextContent) and entry.content.role == Role.DIAGNOSTIC:
                print(plain(entry.content.details or entry.content.text), file=sys.stderr)
    print(f"\n会话已保存：{plain(controller.path)}")
    print(f"继续：{resume_command(project, controller.path)}")
    return 1 if controller.failed else 0
