"""Project trusted run events into typed display content without executing effects."""

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.utils import get_cwidth

from lite_pi.planning import WorkflowState
from lite_pi.tools.contracts import JsonObject
from lite_pi.tui.markup import TextBlock
from lite_pi.tui.messages import Entry, Role, TextContent, ToolContent, ToolState
from lite_pi.tui.text import plain


def clipped(text: object, width: int) -> str:
    value = plain(text).replace("\n", " ").expandtabs(4)
    if get_cwidth(value) <= width:
        return value
    tail = ""
    for char in reversed(value):
        if get_cwidth(tail + char) > max(0, width - 1):
            break
        tail = char + tail
    return "…" + tail


def tool_content(name: str, arguments: str, outcome: JsonObject | None = None) -> ToolContent:
    try:
        parsed = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        parsed = arguments
    if isinstance(parsed, dict):
        target = str(
            parsed.get("path")
            or parsed.get("command")
            or parsed.get("uri")
            or (json.dumps(parsed, ensure_ascii=False) if parsed else "")
        )
    else:
        target = "参数格式无效 · " + arguments
    if outcome is None:
        return ToolContent(name, target, ToolState.RUNNING)
    state = (
        ToolState.UNKNOWN
        if outcome.get("effect") == "unknown"
        else ToolState.NOT_EXECUTED
        if not outcome.get("executed")
        else ToolState.FAILED
        if outcome.get("is_error")
        else ToolState.DONE
    )
    return ToolContent(
        name,
        target,
        state,
        str(outcome.get("text") or ""),
        json.dumps({**outcome, "arguments": parsed}, ensure_ascii=False, indent=2),
        str(outcome.get("reason") or ""),
        bool(outcome.get("truncation")) or outcome.get("output_complete") is False,
        str(outcome.get("raw_evidence") or ""),
    )


@dataclass
class Screen:
    entries: list[Entry] = field(default_factory=list)
    phase: str = "空闲"
    result: JsonObject = field(default_factory=dict)
    actual_model: str | None = None
    calls: dict[str, JsonObject] = field(default_factory=dict)
    expanded: bool = False
    _assistant: Entry | None = None
    ended: bool = False
    _run: str = ""
    _turn: int = 0
    _tool_entries: dict[str, Entry] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.phase in {
            "awaiting_approval",
            "awaiting_decision",
            "awaiting_continue",
            "awaiting_question",
            "effect_unknown",
        }

    @property
    def phase_label(self) -> str:
        return {
            "空闲": "就绪",
            "stop": "本轮结束",
            "aborted": "已取消",
            "budget": "预算停止",
            "turn_limit": "轮次上限",
            "effect_unknown": "效果未知，当前会话受阻",
            "awaiting_approval": "等待审批",
            "awaiting_decision": "等待决定",
            "awaiting_continue": "等待继续",
            "awaiting_question": "等待答复",
            "blocked": "运行受阻",
            "length": "回答已截断",
            "compaction_failed": "摘要失败，历史已保留",
            "superseded": "候选已失效，请继续",
            "error": "运行失败",
        }.get(self.phase, self.phase)

    def begin(self) -> None:
        self._assistant = None
        self.calls.clear()
        self._tool_entries.clear()
        self.ended = False
        self.result = {"tasks": self.result.get("tasks") or {}}

    def note(self, text: str) -> None:
        self.entries.append(Entry(TextContent(Role.NOTICE, text)))

    def event(self, event: JsonObject) -> None:
        kind = event["type"]
        if kind == "run_start":
            self._run = str(event.get("run_id") or "")
        elif kind == "turn_start":
            self._turn = int(event.get("turn") or self._turn + 1)
            self.calls.clear()
            self._assistant = Entry(TextContent(Role.ASSISTANT, ""))
            self.entries.append(self._assistant)
        elif kind in {"message_delta", "message_end"}:
            if self._assistant is None:
                self._assistant = Entry(TextContent(Role.ASSISTANT, ""))
                self.entries.append(self._assistant)
            content = self._assistant.content
            if not isinstance(content, TextContent):
                raise RuntimeError("assistant entry does not hold text content")
            if kind == "message_delta":
                self._assistant.content = replace(content, text=content.text + str(event["text"]))
            elif not content.text:
                self._assistant.content = replace(content, text=str(event.get("text") or ""))
        elif kind == "model_response":
            for call in event["message"].get("tool_calls") or []:
                self.calls[call["id"]] = call["function"]
        elif kind in {"tool_start", "tool_end", "tool_result"}:
            outcome = None if kind == "tool_start" else event["result"]
            call_id = str(event.get("tool_call_id") if outcome is None else outcome.get("tool_call_id"))
            key = f"{self._run}:{self._turn}:{call_id}"
            function = self.calls.get(call_id) or {}
            name = str(event.get("name") if outcome is None else outcome.get("name"))
            content = tool_content(name, str(function.get("arguments") or ""), outcome)
            item = self._tool_entries.get(key)
            if item is None:
                item = Entry(content, key)
                self._tool_entries[key] = item
                self.entries.append(item)
            else:
                item.content = content
        elif kind == "control_entered":
            control = event.get("control") or event
            label = "转向" if control.get("kind") == "steer" else "后续任务"
            self.entries.append(
                Entry(TextContent(Role.CONTROL, f"{label}已进入请求 · {control.get('id', '')}"))
            )
        elif kind == "run_end":
            self.finish(event.get("result") or event)

    def finish(self, result: JsonObject) -> None:
        if self.ended:
            return
        self.ended = True
        self.result = result
        self.actual_model = result.get("model") or self.actual_model
        self.phase = str(result.get("status", "error"))
        label = self.phase_label
        if self.phase == "aborted":
            label += "，已发生的修改保留"
        reason = str(result.get("reason") or "")
        summary = label + ("\n原因：" + reason if reason else "")
        metadata = {
            key: result.get(key)
            for key in (
                "status",
                "reason",
                "raw_stop_reason",
                "evidence",
                "model",
                "usage",
                "usage_unknown",
                "request_count",
                "controls",
            )
        }
        self.entries.append(
            Entry(TextContent(Role.RECEIPT, summary, json.dumps(metadata, ensure_ascii=False, indent=2)))
        )
        self.approvals(result.get("tasks") or {})

    def approvals(self, tasks: JsonObject) -> None:
        self.workflow(tasks.get("workflow") or {})
        intents = {intent["intent_id"]: intent for intent in tasks.get("intents") or []}
        for approval in tasks.get("approvals") or []:
            key = "approval:" + str(approval["approval_id"])
            item = next((item for item in self.entries if item.key == key), None)
            if approval["decision"] == "pending":
                intent = intents.get(approval["intent_id"]) or {}
                detail = {**approval, **{k: intent.get(k) for k in ("name", "arguments", "object_key")}}
                content = TextContent(
                    Role.APPROVAL,
                    f"? 待审批 · {approval['approval_id']}\n"
                    f"{intent.get('name', '')} · {intent.get('object_key', '')}\n"
                    + json.dumps(intent.get("arguments"), ensure_ascii=False)
                    + "\n"
                    + str(approval.get("preview") or intent.get("preview") or ""),
                    json.dumps(detail, ensure_ascii=False, indent=2),
                )
                if item is None:
                    self.entries.append(Entry(content, key))
                else:
                    item.content = content
            elif item is not None:
                item.content = TextContent(
                    Role.CONTROL,
                    f"审批 {approval['approval_id']} · {approval['decision']}\n/continue 明确继续",
                )

    def workflow(self, payload: JsonObject) -> None:
        if not payload:
            return
        state = WorkflowState.decode(payload)
        if state.plan:
            self._plan(state)
        question = state.question
        if question:
            pending = question.status == "pending"
            lines = [f"澄清 · {question.status} · 状态版本 {state.version}", question.text]
            if pending:
                lines += [f"{i}. {option}" for i, option in enumerate(question.options, 1)]
                lines.append("输入选项序号或自由答复，Enter 提交；Esc 取消问题。答复不表示批准。")
            elif question.answer is not None:
                lines.append("已采用答复：" + question.answer)
            self._workflow_entry("question:" + question.id, TextContent(Role.CONTROL, "\n".join(lines)))

    def _plan(self, state: WorkflowState) -> None:
        plan = state.plan
        if not plan:
            return
        key = "plan:" + plan.id
        lines = [f"计划修订 {plan.revision} · 状态版本 {state.version} · {plan.reason}"]
        for step in plan.steps:
            checks = step.checks
            status = f"证据 {sum(c.passed for c in checks)}/{len(checks)} 通过" if checks else "未核验"
            if not step.valid:
                status = "结果失效 · " + str(step.invalid_reason)
            lines.append(f"{step.id} · {step.title} · {status}\n交付条件：{step.deliverable}")
            if step.claim:
                lines.append("模型声明：" + step.claim)
        content = TextContent(
            Role.CONTROL, "\n".join(lines), json.dumps(state.payload(), ensure_ascii=False, indent=2)
        )
        self._workflow_entry(key, content)

    def _workflow_entry(self, key: str, content: TextContent) -> None:
        item = next((entry for entry in self.entries if entry.key == key), None)
        if item is None:
            self.entries.append(Entry(content, key))
        else:
            item.content = content

    def restore_tasks(self, tasks: JsonObject) -> None:
        self.result["tasks"] = tasks
        states = {intent["status"] for intent in tasks.get("intents") or []}
        if "effect_unknown" in states:
            self.phase = "effect_unknown"
        elif any(item["decision"] == "pending" for item in tasks.get("approvals") or []):
            self.phase = "awaiting_approval"
        elif "awaiting_decision" in states:
            self.phase = "awaiting_decision"
        elif ((tasks.get("workflow") or {}).get("question") or {}).get("status") == "pending":
            self.phase = "awaiting_question"
        elif self.phase == "awaiting_approval":
            self.phase = "awaiting_continue"
        self.approvals(tasks)

    def transcript(self) -> str:
        return "\n".join(
            "".join(fragment[1] for fragment in line.fragments)
            for block in self.blocks()
            for line in block.lines
        )

    def blocks(self) -> list[TextBlock]:
        return [entry.block(self.expanded) for entry in self.entries]


def welcome(
    width: int, project: Path, session: str, tools: tuple[str, ...], version: str
) -> StyleAndTextTuples:
    if width < 40:
        return [("", "窗口过窄，请扩大窗口；Ctrl+D 可退出。")]
    fields = [
        "欢迎使用 lite-pi · /help 查看帮助",
        f"目录  {clipped(project, max(12, width - 22))}",
        "配置模型  deepseek-flash",
        f"会话  {session}",
        "工具  " + clipped(" · ".join(tools), max(10, width - 22)),
    ]
    if width < 60:
        return [("class:brand", f"lite-pi {version}\n"), ("", "\n".join(fields[1:]))]
    art = ["▄▄▄▄▄▄▄▄▄", "▀▀██▀██▀▀", "  ██ ██  ", "  ██ ██  ", "         "]
    output: StyleAndTextTuples = [("class:brand", f"lite-pi {version}\n")]
    for logo, text in zip(art, fields, strict=True):
        output.extend([("class:brand", f"  {logo}   "), ("", text + "\n")])
    return output
