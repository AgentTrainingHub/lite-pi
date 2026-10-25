"""One worker owns a session operation; only snapshots cross into the UI thread."""

import json
import sqlite3
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Lock, Thread
from typing import Literal, TypedDict
from uuid import uuid4

from lite_pi.control import ControlInput
from lite_pi.journal import JournalError
from lite_pi.planning import WorkflowState
from lite_pi.session import AgentSession, RunLimits
from lite_pi.spend.gate import SpendBlocked
from lite_pi.spend.ledger import Ledger
from lite_pi.tools.contracts import JsonObject
from lite_pi.tui.catalog import Catalog, SavedSession
from lite_pi.tui.display import Screen
from lite_pi.tui.evidence import original
from lite_pi.tui.messages import Entry, Role, TextContent, ToolContent

HELP = """Enter 提交；运行中 Enter 转向；Alt+Enter / Ctrl+J 换行
Ctrl+C / Esc 取消运行；空闲 Ctrl+C 清除草稿；空草稿 Ctrl+D 退出
等待澄清时 Enter 回答或选择；Esc 取消问题；退出后问题仍保留
PageUp / PageDown 回看；Ctrl+End 返回最新；Ctrl+O 工具与技术回执详略
/help  /status  /new  /sessions [序号]  /continue  /cancel
/steer [--paths a,b --] 消息  /follow 消息  /approve ID  /reject ID  /quit
//文字 发送以 / 开头的普通文字"""


@dataclass(frozen=True)
class Update:
    operation: str
    kind: Literal["event", "result", "error"]
    value: JsonObject
    failure: "Failure | None" = None
    returned_inputs: list[ControlInput] = field(default_factory=list)


class Failure(TypedDict):
    error: str
    notes: list[str]
    traceback: str
    fatal: bool


class Controller:
    def __init__(self, catalog: Catalog, session: AgentSession, limits: RunLimits) -> None:
        self.catalog, self.session, self.limits = catalog, session, limits
        self.screen = Screen(entries=Catalog.history(session))
        self.screen.restore_tasks(session.inspect())
        self.queue: SimpleQueue[Update] = SimpleQueue()
        self.worker: Thread | None = None
        self._control_lock = Lock()
        self._accepting = False
        self.operation = ""
        self.cancel_requested = False
        self.failed = False
        self.restored_draft = ""
        self.closing = False
        self.saved: list[SavedSession] = []
        self.ledger: JsonObject = Ledger().status() or {}

    @property
    def busy(self) -> bool:
        return self.worker is not None

    @property
    def path(self) -> Path:
        if self.session.journal is None:
            raise RuntimeError("terminal session has no journal")
        return self.session.journal.path

    def start(self, text: str | None) -> None:
        if self.busy:
            raise ValueError("当前运行尚未收尾")
        if self.session.journal is None:
            raise RuntimeError("terminal session has no journal")
        if text is not None and self.session.journal.incomplete_tail() is not None:
            raise ValueError("会话有未完成工具批次，请先 /continue")
        self._accepting = True
        self.operation = uuid4().hex
        operation = self.operation
        self.cancel_requested = False
        self.screen.phase = "执行中"
        self.screen.begin()
        if text is not None:
            self.screen.entries.append(Entry(TextContent(Role.USER, text)))

        def observe(event: JsonObject) -> None:
            if self.cancel_requested and not self.session.control.cancel.is_set():
                self.session.cancel()
            self.queue.put(Update(operation, "event", event))

        def work() -> None:
            limits: RunLimits = {**self.limits, "on_event": observe}
            try:
                if self.cancel_requested:
                    result: JsonObject = {"status": "aborted", "reason": "cancelled before start"}
                elif text is None:
                    result = self.session.resume(**limits)
                else:
                    result = self.session.submit(text, **limits)
                self._publish(operation, "result", result)
            except BaseException as error:
                self._publish(
                    operation,
                    "error",
                    {},
                    failure={
                        "error": f"{type(error).__name__}: {error}",
                        "notes": getattr(error, "__notes__", []),
                        "traceback": traceback.format_exc(),
                        "fatal": not isinstance(
                            error, (OSError, ValueError, JournalError, SpendBlocked, sqlite3.Error)
                        ),
                    },
                )

        self.worker = Thread(target=work, name="lite-pi-session", daemon=True)
        self.worker.start()

    def _publish(
        self,
        operation: str,
        kind: Literal["result", "error"],
        value: JsonObject,
        *,
        failure: Failure | None = None,
    ) -> None:
        with self._control_lock:
            self._accepting = False
            returned = self.session.return_pending()
            self.queue.put(Update(operation, kind, value, failure, returned))

    def drain(self) -> bool:
        changed = False
        while True:
            try:
                update = self.queue.get_nowait()
            except Empty:
                break
            if update.operation != self.operation:
                continue
            changed = True
            if update.kind == "event":
                self.screen.event(update.value)
            else:
                if self.worker is None:
                    raise RuntimeError("run result arrived without a worker thread")
                # A result is queued just before thread exit; never block the UI
                # if a worker violates that lifecycle contract.
                self.worker.join(0.1)
                if self.worker.is_alive():
                    self.queue.put(update)
                    break
                self.worker = None
                if update.kind == "result":
                    self.screen.finish(update.value)
                else:
                    if update.failure is None:
                        raise RuntimeError("error update has no failure details")
                    if not self.screen.ended:
                        self.screen.phase = "错误"
                    self.screen.entries.append(
                        Entry(
                            TextContent(
                                Role.DIAGNOSTIC,
                                update.failure["error"] + "\n" + "\n".join(update.failure["notes"]),
                                update.failure["traceback"],
                            ),
                        )
                    )
                    if update.failure["fatal"] or self.screen.phase == "effect_unknown":
                        self.failed = self.closing = True
                returned = update.returned_inputs
                if returned:
                    self.restored_draft = "\n".join(str(item["text"]) for item in returned)
                    self.screen.note("未进入请求的输入已回填草稿，等待明确提交")
                self.ledger = Ledger().status() or {}
        if changed and self.screen.expanded:
            self.load_details()
        return changed

    def load_details(self) -> None:
        for entry in self.screen.entries:
            content = entry.content
            if isinstance(content, ToolContent) and content.evidence_path and content.original is None:
                entry.content = replace(content, original=original(content.evidence_path))

    def toggle_details(self) -> None:
        self.screen.expanded = not self.screen.expanded
        if self.screen.expanded:
            self.load_details()

    def cancel(self) -> None:
        if self.busy:
            self.cancel_requested = True
            self.session.cancel()
            self.screen.phase = "取消中"
        else:
            state = self._question_state()
            if state is not None:
                if state.question is None:
                    raise RuntimeError("pending workflow state has no question")
                self.session.answer(state.question.id, "", expected_version=state.version, cancel=True)
                self.screen.restore_tasks(self.session.inspect())
                if self.screen.phase == "awaiting_question":
                    self.screen.phase = "stop"
                self.screen.note("澄清已取消，未采用答案；等待新的要求。")

    def close(self) -> None:
        self.closing = True
        if self.busy:
            self.cancel()

    def _question_state(self) -> WorkflowState | None:
        state = (self.screen.result.get("tasks") or {}).get("workflow") or {}
        if not state:
            return None
        decoded = WorkflowState.decode(state)
        return decoded if decoded.question and decoded.question.status == "pending" else None

    def join(self, timeout: float = 15) -> None:
        if self.worker is not None:
            self.cancel()
            self.worker.join(timeout)
            if self.worker.is_alive():
                self.failed = True
                raise RuntimeError(
                    "Session 收尾超时；终端已退出，执行效果尚未核实。续开会话时先检查恢复状态。"
                )
            self.drain()

    def submit(self, text: str) -> bool:
        """True clears the editor; rejected commands retain the exact draft."""
        text = text.strip()
        if not text:
            return False
        if text.startswith("//"):
            text = text[1:]
        elif text.startswith("/"):
            command, _, arg = text.partition(" ")
            return self.command(command, arg.strip())
        if self.busy:
            self._control("steer", text)
        else:
            state = self._question_state()
            if state is not None:
                if state.question is None:
                    raise RuntimeError("pending workflow state has no question")
                self.session.answer(state.question.id, text, expected_version=state.version)
                self.screen.entries.append(Entry(TextContent(Role.USER, text)))
                self.screen.restore_tasks(self.session.inspect())
                self.start(None)
                return True
            if self.screen.result.get("reason") == "clarification_cancelled":
                self.screen.phase = "stop"
            if self.screen.blocked:
                raise ValueError("当前会话受阻；请先处理审批或核对未知效果，再 /continue；草稿已保留")
            self.start(text)
        return True

    def _control(self, kind: str, text: str) -> None:
        with self._control_lock:
            if not self.busy or self.cancel_requested or not self._accepting:
                raise ValueError("当前运行已停止接收输入，请等待收尾后再次提交")
            if kind == "steer" and text.startswith("--paths "):
                scope, separator, message = text[len("--paths ") :].partition(" -- ")
                paths = tuple(part.strip() for part in scope.split(",") if part.strip())
                if not separator or not paths or not message.strip():
                    raise ValueError("用法：/steer --paths 路径1,路径2 -- 消息")
                result = self.session.steer(message, paths=paths)
            else:
                result = self.session.steer(text) if kind == "steer" else self.session.follow_up(text)
        label = "转向" if kind == "steer" else "后续任务"
        self.screen.entries.append(
            Entry(TextContent(Role.CONTROL, f"{label}已接收，待进入请求 · {result['id'][:8]}\n{text}"))
        )

    def command(self, command: str, arg: str) -> bool:
        if command in {"/help", "/status", "/new", "/continue", "/cancel", "/quit"} and arg:
            raise ValueError(f"{command} 不接受参数")
        if command == "/help":
            self.screen.note(HELP)
        elif command == "/status":
            self.ledger = Ledger().status() or {}
            if not self.busy:
                self.screen.result["tasks"] = self.session.inspect()
            controls = self.session.control.snapshot()
            self.screen.note(
                json.dumps(
                    {
                        "project": str(self.catalog.project),
                        "session": str(self.path),
                        "tools": self.session.tools,
                        "phase": self.screen.phase,
                        "configured_model": "deepseek-flash",
                        "actual_model": self.screen.actual_model,
                        "ledger": self.ledger,
                        "tasks": self.screen.result.get("tasks"),
                        "usage": self.screen.result.get("usage"),
                        "controls": controls,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif command == "/quit":
            self.close()
        elif command == "/cancel":
            self.cancel()
        elif command in {"/steer", "/follow"}:
            if not arg:
                raise ValueError(f"用法：{command} 消息")
            self._control("steer" if command == "/steer" else "follow_up", arg)
        elif command in {"/new", "/sessions", "/continue", "/approve", "/reject"}:
            if self.busy:
                raise ValueError("请先取消或等待当前运行收尾")
            if command == "/new":
                self._switch(self.catalog.create())
            elif command == "/sessions":
                if arg:
                    index = int(arg) - 1
                    if index < 0 or index >= len(self.saved):
                        raise ValueError("请先 /sessions 查看有效序号")
                    self._switch(self.catalog.open(self.saved[index].path))
                else:
                    self.saved, errors = self.catalog.list_sessions()
                    self.screen.note(
                        "\n".join(
                            f"{i}: {s.identity[:8]} · {s.path.name}" for i, s in enumerate(self.saved, 1)
                        )
                        or "没有会话"
                    )
                    for error in errors:
                        self.screen.note(error)
            elif command == "/continue":
                self.start(None)
            else:
                if not arg:
                    raise ValueError(f"用法：{command} 审批ID")
                result = self.session.decide(arg, "approve" if command == "/approve" else "reject")
                self.screen.restore_tasks(self.session.inspect())
                self.screen.note(json.dumps(result, ensure_ascii=False, indent=2) + "\n/continue 明确继续")
        else:
            raise ValueError("未知命令；/help 查看帮助，// 可发送以 / 开头的消息")
        return True

    def _switch(self, session: AgentSession) -> None:
        screen = Screen(entries=Catalog.history(session))
        screen.restore_tasks(session.inspect())
        self.session, self.screen = session, screen
        self.operation = uuid4().hex
        self.saved = []
