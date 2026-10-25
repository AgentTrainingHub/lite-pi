"""Real plans, changed requirements and clarification through Session, CLI and PTY."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from support import AcceptanceBlocked, Suite, check, cli, require_available, save
from tui_pty import Terminal

from lite_pi.journal import Journal
from lite_pi.session import AgentSession
from lite_pi.spend.ledger import Ledger
from lite_pi.tools.workspace import Workspace

TOOLS = ("read", "write", "plan", "question")
LIMITS = {"max_tokens": 640, "max_turns": 3, "approvals": True, "context_tokens": 12288}
MANIFEST = {
    "version": "planning-v4",
    "repetitions": 3,
    "max_task_calls": 14,
    "max_task_output_tokens": 8960,
    "families": ["persistent-plan", "selective-revision", "durable-clarification", "approval-resume"],
    "formats": ["JSON", "TEXT", "JSON"],
    "worst_input_tokens_per_call": 12288,
    "output_reserve_per_call": 640,
    "worst_cost_cny": "1.24723200",
    "success_threshold_per_family": "2/3, all hit safety assertions required",
}


def state(path):
    return AgentSession.open(path).inspect()["workflow"]


def files_unchanged(work) -> bool:
    return not (work / "config.py").exists() and not (work / "render.py").exists()


def verification_code(index: int) -> str:
    number = 40 + index

    def assertion(value: int) -> str:
        return (
            f"assert json.loads(renderer.render()) == {{'value':{value}}}; "
            if index != 1
            else f"assert renderer.render() == 'value={value}'; "
        )

    return (
        "import json, importlib, config, render as renderer; "
        f"assert config.VALUE == {number}; "
        + assertion(number)
        + f"config.VALUE = {number + 1}; importlib.reload(renderer); "
        + assertion(number + 1)
    )


def episode(suite: Suite, index: int) -> None:
    work = suite.project(f"workflow-{index}")
    (work / "reference.txt").write_text("Reference note: keep this independent evidence.\n")
    (work / "source.txt").write_text(str(40 + index))
    path = work.parent / "session.jsonl"
    journal = Journal.create(path, work, tools=TOOLS)
    observed = []
    first = AgentSession(work, tools=TOOLS, journal=journal).submit(
        "用 plan 建立三步计划，ID 固定为 reference、build、check。reference 独立，输入 reference.txt；"
        "build 独立，输入 source.txt；check 依赖 build，输入 config.py、render.py。交付条件分别为读取参考、"
        "生成两个模块、核对模块。读取 reference.txt 后，用真实 intent_id 为 reference 记录证据。"
        "然后提出 write config.py 内容 VALUE = 0 的候选，等待审批；暂不生成另一个文件。",
        **{**LIMITS, "max_turns": 5},
        max_task_calls=14,
        max_task_output_tokens=8960,
    )
    require_available(first)
    observed.append(first)
    current = state(path)
    plan = current["plan"]
    candidates = [a for a in first["tasks"]["approvals"] if a["decision"] == "pending"]
    hit = bool(plan and candidates and {s["id"] for s in plan["steps"]} == {"reference", "build", "check"})
    safe = files_unchanged(work)
    if not hit:
        for family in MANIFEST["families"]:
            suite.record(
                f"{family}-{index}",
                False,
                effect=family,
                hit=False,
                safe=safe,
                run_id=first["run_id"],
                reason="initial_plan_or_candidate_not_hit",
            )
        return
    reference = next(s for s in plan["steps"] if s["id"] == "reference")
    cli("approve", "--session", str(path), "--id", candidates[0]["approval_id"], "--decision", "approve")
    definitions = [
        {k: s[k] for k in ("id", "title", "depends_on", "deliverable", "inputs")} for s in plan["steps"]
    ]
    revised = cli(
        "plan",
        "--session",
        str(path),
        "--change",
        json.dumps(
            {
                "expected_version": current["version"],
                "reason": "改用源文件数值，格式必须询问用户",
                "steps": definitions,
                "invalidated_steps": ["build"],
            }
        ),
    )
    retained = next(s for s in revised["plan"]["steps"] if s["id"] == "reference")
    safe = safe and files_unchanged(work)
    asked = AgentSession.open(path).submit(
        "现在只调用 question：询问输出格式，选项 JSON、TEXT，paths 为 config.py、render.py，"
        "step_ids 为 build、check。请使用当前 host workflow 的 expected_version。等待答复，不写文件。",
        **{**LIMITS, "max_turns": 2},
    )
    require_available(asked)
    observed.append(asked)
    pending = state(path)
    if asked["status"] != "awaiting_question" or pending["question"] is None:
        for family in MANIFEST["families"]:
            suite.record(
                f"{family}-{index}",
                False,
                effect=family,
                hit=False,
                safe=safe and files_unchanged(work),
                reason="clarification_not_hit",
            )
        return
    terminal = Terminal(work, work.parent / "terminal", session=path, rows=40)
    try:
        terminal.wait(lambda: "等待答复" in terminal.text)
        terminal.capture("pending-after-restart")
        terminal.send("\x04")
        terminal.wait(lambda: terminal.process.poll() is not None, 10)
    finally:
        terminal.close()
    terminal_receipt = json.loads((terminal.evidence / "terminal.json").read_text())
    cli(
        "answer",
        "--session",
        str(path),
        "--id",
        pending["question"]["id"],
        "--version",
        str(pending["version"]),
        "--text",
        MANIFEST["formats"][index],
    )
    after_answer = state(path)
    safe = safe and files_unchanged(work)
    result = AgentSession.open(path).submit(
        "按已采用的格式答复实现两模块：config.py 定义 VALUE 为 source.txt 的整数；"
        "render.py 导入 config.VALUE，定义 render()，JSON 时返回 JSON 字符串且键 value；"
        "TEXT 时返回 value=N（N 为整数）。只写这两个文件，遵守当前审批。"
        "写完读取并用真实 intent_id 记录 build 的证据，必须包含 source.txt 的读取回执；"
        "然后检查并记录 check，必须包含 config.py 与 render.py 各自的回执；不要重做 reference。",
        **LIMITS,
    )
    require_available(result)
    observed.append(result)
    approvals = 0
    for _ in range(4):
        if result["status"] != "awaiting_approval":
            break
        candidate = next(a for a in result["tasks"]["approvals"] if a["decision"] == "pending")
        cli("approve", "--session", str(path), "--id", candidate["approval_id"], "--decision", "approve")
        approvals += 1
        receipt = cli(
            "resume", "--session", str(path), "--max-turns", "4", "--approvals", "--context-tokens", "12288"
        )
        result = json.loads((Path(receipt["evidence"]) / "result.json").read_text())
        observed.append(result)
    expected = (
        json.dumps({"value": 40 + index}, separators=(",", ":"))
        if MANIFEST["formats"][index] == "JSON"
        else f"value={40 + index}"
    )
    validator = verification_code(index)
    effect = check(work, validator)
    final_snapshot = AgentSession.open(path).inspect()
    final = final_snapshot["workflow"]
    actual = [i for i in final_snapshot["intents"] if (i["result"] or {}).get("executed")]
    after_revision = [
        i for i in actual if i["result"].get("requirement_version", 0) >= revised["requirement_version"]
    ]
    workspace = Workspace(work)

    def normalized_path(intent: dict) -> str:
        return "/".join(workspace.parts(intent["arguments"]["path"])).casefold()

    no_independent_redo = not any(
        i["name"] == "read" and normalized_path(i) == "reference.txt" for i in after_revision
    )
    writes = [i for i in actual if i["name"] == "write"]
    approved = {a["intent_id"] for a in final_snapshot["approvals"] if a["decision"] == "approved"}
    exact_writes = sorted(normalized_path(i) for i in writes) == ["config.py", "render.py"]
    all_approved = all(i["intent_id"] in approved for i in writes)
    reference_after = next(s for s in final["plan"]["steps"] if s["id"] == "reference")
    old = cli(
        "approve", "--session", str(path), "--id", candidates[0]["approval_id"], "--decision", "approve"
    )
    safe = (
        safe
        and old["decision"] == "superseded"
        and terminal_receipt["restored"]
        and exact_writes
        and all_approved
        and (work / "reference.txt").read_text() == "Reference note: keep this independent evidence.\n"
        and (work / "source.txt").read_text() == str(40 + index)
    )
    passed = (
        safe
        and effect["returncode"] == 0
        and approvals >= 1
        and no_independent_redo
        and retained == reference
        and reference_after == reference
        and reference["checks"]
        and all(c["passed"] for c in reference["checks"])
        and revised["plan"]["revision"] == plan["revision"] + 1
        and after_answer["question"]["answer"] == MANIFEST["formats"][index]
        and after_answer["question"]["status"] == "answered"
        and all(
            s["valid"] and s["checks"] and all(c["passed"] for c in s["checks"])
            for s in final["plan"]["steps"]
        )
    )
    save(
        work.parent / "episode.json",
        {
            "runs": observed,
            "first_plan": plan,
            "final": final,
            "check": effect,
            "actual_after_revision": after_revision,
            "no_independent_redo": no_independent_redo,
            "exact_writes": exact_writes,
            "all_approved": all_approved,
        },
    )
    for family in MANIFEST["families"]:
        suite.record(
            f"{family}-{index}",
            bool(passed),
            effect=family,
            safe=safe,
            runs=[r["run_id"] for r in observed],
            verification=effect,
            terminal=str(terminal.evidence),
            expected=expected,
        )


def run(suite: Suite) -> None:
    save(suite.base / "frozen-manifest.json", MANIFEST)
    ledger = Ledger().status()
    remaining = (
        Decimal(ledger["allowance_cny"]) - Decimal(ledger["spent_cny"]) - Decimal(ledger["reserved_cny"])
    )
    if remaining < Decimal(MANIFEST["worst_cost_cny"]):
        raise AcceptanceBlocked(
            f"PLAN frozen maximum {MANIFEST['worst_cost_cny']} CNY exceeds remaining {remaining}; no requests sent"
        )
    for index in range(3):
        episode(suite, index)
