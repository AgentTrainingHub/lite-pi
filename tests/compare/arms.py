"""One handwritten session arm and one LangGraph arm over the same project copy."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from support import check

from compare.contracts import GraphObservation, HandwrittenObservation
from compare.flow import GraphRun, ProviderModel, RegistryTools, compile_flow, initial_state
from compare.judge import FIX_TASK, product_messages, sha_tree, tool_feedback
from lite_pi.journal import Journal
from lite_pi.session import CODE_TOOLS, AgentSession
from lite_pi.tasks import TaskBook
from lite_pi.tools.contracts import JsonObject
from lite_pi.tools.registry import ToolRegistry
from lite_pi.tools.workspace import Workspace

MAX_CALLS = 12
FIRST_TURNS = 1
RESUME_TURNS = 7
MAX_TOKENS = 800


def prepare(parent: Path, source: Path) -> Path:
    work = parent / "project"
    shutil.copytree(source, work)
    return work


def followup(checked: JsonObject) -> str:
    stdout = checked.get("stdout")
    text = stdout if isinstance(stdout, str) else ""
    return (
        "python3 check.py still fails. Continue from the existing edits. Do not modify check.py.\n"
        + text[:1500]
    )


def run_handwritten(parent: Path, source: Path, *, task: str = FIX_TASK) -> HandwrittenObservation:
    work = prepare(parent, source)
    before = sha_tree(work)
    journal = Journal.create(parent / "session.jsonl", work, tools=CODE_TOOLS)
    started = time.monotonic()
    session = AgentSession(work, tools=CODE_TOOLS, journal=journal)
    first = session.submit(task, max_turns=FIRST_TURNS, max_tokens=MAX_TOKENS, max_task_calls=MAX_CALLS)
    checked = check(work)
    first_check = checked["returncode"]
    results = [first]
    resumed = False
    if checked["returncode"] != 0:
        resumed = True
        opened = AgentSession.open(journal.path)
        if opened.journal is not None and opened.journal.incomplete_tail() is not None:
            second = opened.resume(max_turns=RESUME_TURNS, max_tokens=MAX_TOKENS, max_task_calls=MAX_CALLS)
        else:
            second = opened.submit(
                followup(checked), max_turns=RESUME_TURNS, max_tokens=MAX_TOKENS, max_task_calls=MAX_CALLS
            )
        results.append(second)
        checked = check(work)
    after = sha_tree(work)
    messages: list[JsonObject] = []
    for result in results:
        messages.extend(product_messages(result))
    return {
        "arm": "handwritten",
        "elapsed": round(time.monotonic() - started, 3),
        "results": results,
        "resumed": resumed,
        "checked": checked,
        "before": before,
        "after": after,
        "state_passed": resumed
        and first_check != 0
        and results[-1].get("status") == "stop"
        and tool_feedback(messages),
        "model": first.get("model"),
    }


def run_graph(parent: Path, source: Path, *, run_id: str) -> GraphObservation:
    work = prepare(parent, source)
    before = sha_tree(work)
    evidence = parent / "graph"
    evidence.mkdir()
    journal = Journal.create(parent / "graph-session.jsonl", work, tools=CODE_TOOLS)
    workspace = Workspace(work)
    book = TaskBook(
        journal,
        workspace,
        tools=CODE_TOOLS,
        approvals=False,
        max_calls=MAX_CALLS,
        max_output=None,
        max_seconds=None,
        approval_ttl=None,
        claim=True,
    )
    try:
        admission = book.gate.acquire("write")
        if admission["state"] != "running":
            raise RuntimeError(f"project gate: {admission['reason']}")
        registry = ToolRegistry(
            workspace,
            CODE_TOOLS,
            max_bytes=50 * 1024,
            shell_timeout=10,
            cancel_bash_after=None,
            evidence=evidence,
        )
        tools = RegistryTools(registry)
        run = GraphRun(
            compile_flow(
                ProviderModel(),
                tools,
                run_id=run_id,
                max_turns=FIRST_TURNS + RESUME_TURNS,
                max_tokens=MAX_TOKENS,
                budget=book.budget,
            ),
            run_id,
            evidence,
            closer=book.finish,
        )
    except BaseException as error:
        try:
            book.close()
        except Exception as cleanup:
            error.add_note(f"Graph setup cleanup also failed: {type(cleanup).__name__}: {cleanup}")
        raise
    started = time.monotonic()
    with run:
        state = run.invoke(initial_state(FIX_TASK), interrupt_after=["boundary"])
        checked = check(work)
        resumed = False
        if checked["returncode"] != 0 and state["status"] not in {"blocked", "budget"}:
            resumed = True
            run.followup(followup(checked))
            state = run.invoke(None)
            checked = check(work)
        after = sha_tree(work)
        return {
            "arm": "langgraph",
            "elapsed": round(time.monotonic() - started, 3),
            "run_id": run_id,
            "resumed": resumed,
            "checked": checked,
            "before": before,
            "after": after,
            "state_passed": resumed and state["status"] == "stop" and tool_feedback(list(state["messages"])),
            "receipt": run.receipt(),
            "model_node_runs": state["model_node_runs"],
            "tool_node_runs": state["tool_node_runs"],
        }
