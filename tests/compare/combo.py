"""One new project that asks for retrieval, memory, MCP, browser, and delegation together."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from mcp_support import fixture_server

from lite_pi.knowledge.db import KnowledgeDB
from lite_pi.knowledge.tools import KNOWLEDGE_TOOLS
from lite_pi.memory.ledger import MemoryLedger
from lite_pi.session import CODE_TOOLS, RUNS, AgentSession
from lite_pi.tools.contracts import JsonObject

FEE = "17.5"
UNIT = "重量单位使用克"
LOOKUP = 'def lookup(name: str) -> int:\n    table = {"a": 1}\n    return table[name]\n'
CHECK = (
    "import json\nfrom pathlib import Path\nfrom src.lookup import lookup\n"
    "note = json.loads(Path('note.json').read_text())\n"
    "assert note == {'shipping_fee': 17.5, 'weight_unit': '克'}\n"
    "assert lookup('missing') == 0 and lookup('a') == 1\nprint('ok')\n"
)

TOOLS = CODE_TOOLS + ("delegate",) + KNOWLEDGE_TOOLS


def prepare(work: Path) -> str:
    (work / "src").mkdir(parents=True)
    (work / "src/__init__.py").write_text("", encoding="utf-8")
    (work / "src/lookup.py").write_text(LOOKUP, encoding="utf-8")
    (work / "docs").mkdir()
    (work / "docs/fee.md").write_text(f"标准运费是 {FEE} 元。\n", encoding="utf-8")
    (work / "check.py").write_text(CHECK, encoding="utf-8")
    lite = work / ".lite-pi"
    lite.mkdir()
    (lite / "corpus.json").write_text(
        json.dumps({"sources": [{"path": "docs/fee.md", "version": "1"}]}),
        encoding="utf-8",
    )
    built = AgentSession(work).rebuild_index()
    if built.get("reason") != "rebuilt":
        raise RuntimeError(str(built.get("reason") or "index failed"))
    database = KnowledgeDB.open(work)
    try:
        ledger = MemoryLedger(database)
        reference = ledger.record_user_source("combo", UNIT)
        proposed = ledger.propose(
            object_key="unit",
            statement=UNIT,
            source_kind="user_message",
            source_ref=reference,
        )
    finally:
        database.close()
    proposed_memory = proposed.get("memory")
    memory_id = proposed_memory.get("memory_id") if isinstance(proposed_memory, dict) else None
    if not isinstance(memory_id, str):
        raise RuntimeError("memory id missing")
    confirmed = AgentSession(work).confirm_memory(memory_id)
    confirmed_memory = confirmed.get("memory")
    if not isinstance(confirmed_memory, dict) or confirmed_memory.get("status") != "active":
        raise RuntimeError(str(confirmed.get("reason") or "memory not active"))
    return memory_id


def write_servers(work: Path, origin: str, browser_evidence: Path) -> None:
    import sys

    marker = fixture_server("marker", work.parent / "observer" / "marker", timeout=20)
    shop = {
        "name": "shop",
        "command": sys.executable,
        "args": [
            "-m",
            "lite_pi.browser.mcp_server",
            "--origin",
            origin,
            "--evidence",
            str(browser_evidence),
            "--channel",
            "chrome",
        ],
        "tools": ["open", "observe", "act", "query"],
        "origins": [origin],
        "read_only_tools": ["open", "observe", "query"],
        "timeout_seconds": 45,
    }
    (work / ".lite-pi/mcp.json").write_text(
        json.dumps({"servers": [marker, shop]}),
        encoding="utf-8",
    )


def task_text(origin: str) -> str:
    return (
        f"请让 check.py 通过，不要修改 check.py。"
        f"用项目检索确认文档中的运费，已生效记忆给出重量单位，订货页在 {origin} ，"
        "本地标记服务需要留下一次 mark 调用。请在订货页购买一本笔记本，只下单一次。"
        "运费和单位写入 note.json，字段为 shipping_fee（数值）和 weight_unit。"
        "修改 src/lookup.py 之前，当前会话必须先调用 delegate，对 src/lookup.py 做一次只读调查。"
        "调查返回后，当前会话还要亲自读取该文件的现行原文，再修改并运行 check.py。"
        "read、edit、write、bash 保持可用。调查里的完成声明不能代替文件修改。"
    )


def capabilities(result: JsonObject) -> dict[str, bool]:
    items = [
        item
        for turn in result.get("transcript", [])
        for item in turn.get("results", [])
        if item.get("executed") and not item.get("is_error")
    ]
    names = {item.get("name") for item in items}
    records = list((result.get("context") or {}).get("memories", []))
    records += [m for item in items if item.get("name") == "memory_read" for m in item.get("memories", [])]
    return {
        "retrieval": any(
            item.get("name") == "search_docs"
            and any(
                hit.get("source") == "docs/fee.md" and FEE in str(hit.get("text", ""))
                for hit in item.get("hits", [])
            )
            for item in items
        ),
        "memory": any(
            m.get("object") == "unit" and m.get("status") == "active" and m.get("statement") == UNIT
            for m in records
        ),
        "mcp": "mcp__marker__mark" in names,
        "browser": "mcp__shop__act" in names and "mcp__shop__query" in names,
        "delegate": verified_delegation(result),
    }


def _read_original(turn: JsonObject, item: JsonObject) -> bool:
    if item.get("name") != "read" or not item.get("executed") or item.get("is_error"):
        return False
    calls = turn.get("provider", {}).get("message", {}).get("tool_calls", [])
    return any(
        call.get("id") == item.get("tool_call_id")
        and json.loads(call["function"]["arguments"]).get("path", "").removeprefix("./") == "src/lookup.py"
        and item.get("content") == LOOKUP
        for call in calls
    )


def verified_delegation(result: JsonObject) -> bool:
    delegated = False
    digest = hashlib.sha256(LOOKUP.encode()).hexdigest()
    for turn in result.get("transcript", []):
        for item in turn.get("results", []):
            if not item.get("executed") or item.get("is_error"):
                continue
            if item.get("name") == "delegate" and item.get("joined"):
                proof = any(
                    p.get("path") == "src/lookup.py"
                    and p.get("sha256") == digest
                    and p.get("before_sha256") == digest
                    for p in item.get("proofs", [])
                )
                for survey in item.get("surveys", []):
                    identity = survey.get("run_id")
                    if proof and isinstance(identity, str):
                        child = json.loads((RUNS / identity / "result.json").read_text())
                        delegated |= any(
                            _read_original(t, r)
                            for t in child.get("transcript", [])
                            for r in t.get("results", [])
                        )
            if delegated and _read_original(turn, item):
                return True
            if item.get("name") in {"edit", "write"} and any(
                call.get("id") == item.get("tool_call_id")
                and json.loads(call["function"]["arguments"]).get("path", "").removeprefix("./")
                == "src/lookup.py"
                for call in turn.get("provider", {}).get("message", {}).get("tool_calls", [])
            ):
                return False
    return False


def marker_closed(state: Path) -> bool:
    pid = int((state / "pid").read_text())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def marker_count(state: Path) -> int:
    return sum(line == "mark" for line in (state / "calls.jsonl").read_text().splitlines())
