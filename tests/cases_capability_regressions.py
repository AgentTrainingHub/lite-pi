"""One real-model regression per affected capability; no full-suite effect claims."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path

import cases_delegation as delegate_cases
import cases_knowledge as memory_cases
from cases_extensions import ONE_TASK, dead, first_request, tool_names, write_mcp
from mcp_support import observed_pid
from support import FIX, AcceptanceBlocked, Suite, check, cli, require_available, save, snapshot

from lite_pi.cli import CLI_OUTPUT_TOKENS
from lite_pi.compaction import MODEL_CONTEXT_TOKENS
from lite_pi.journal import Journal
from lite_pi.session import RUNS, AgentSession, RunLimits
from lite_pi.spend.ledger import Ledger
from lite_pi.spend.pricing import INPUT_MISS_CNY, MAX_OUTPUT_TOKENS, MODEL, OUTPUT_CNY
from lite_pi.tools.contracts import JsonObject


@dataclass(frozen=True)
class Limits:
    calls: int
    turns: int
    output: int = 640
    context: int = 16384

    def session(self) -> RunLimits:
        return {
            "max_turns": self.turns,
            "max_model_calls": self.calls,
            "max_task_calls": self.calls,
            "max_tokens": self.output,
            "max_task_output_tokens": self.calls * self.output,
            "context_tokens": self.context,
            "max_task_seconds": 180,
        }


CODE = Limits(6, 6)
MCP = Limits(6, 6, 512)
PROPOSE = Limits(3, 3)
ADOPT = Limits(4, 4)
REVOKED = Limits(1, 1)
DELEGATE = Limits(14, 8, 800)
CASES = {"code": (CODE,), "mcp": (MCP,), "memory": (PROPOSE, ADOPT, REVOKED), "delegate": (DELEGATE,)}


def _code(work: Path, task: str, name: str, limits: Limits, *, knowledge: bool = False) -> JsonObject:
    if CLI_OUTPUT_TOKENS != limits.output:
        raise ValueError("CLI output limit differs from the frozen regression")
    args = [
        "code",
        "--root",
        str(work),
        "--session",
        str(work.parent / f"{name}.jsonl"),
        "--max-turns",
        str(limits.turns),
        "--max-task-calls",
        str(limits.calls),
        "--max-task-output",
        str(limits.calls * limits.output),
        "--context-tokens",
        str(limits.context),
        "--max-task-seconds",
        "180",
    ]
    if knowledge:
        args.append("--knowledge")
    receipt = cli(*args, task)
    return json.loads((RUNS / receipt["run_id"] / "result.json").read_text())


def code(suite: Suite) -> None:
    work = suite.project("code", fixture=True)
    original = (work / "check.py").read_bytes()
    before = snapshot(work)
    initial = check(work)
    save(work.parent / "initial-check.json", initial)
    if initial["returncode"] != 1:
        raise AssertionError("public unit-conversion fixture did not fail as expected")
    result = _code(work, FIX, "code", CODE)
    checked = check(work)
    save(work.parent / "final-check.json", checked)
    after = snapshot(work)
    changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
    safe = (work / "check.py").read_bytes() == original and changed <= {
        "weightnote/units.py",
        "weightnote/lines.py",
        "weightnote/invoice.py",
    }
    suite.record(
        "regression-code",
        checked["returncode"] == 0 and safe and result.get("model") == MODEL and result["status"] == "stop",
        safe=safe,
        run_id=result["run_id"],
        run_status=result["status"],
        changed=sorted(changed),
        initial_check=initial,
        final_check=checked,
    )


def mcp(suite: Suite) -> None:
    work = suite.project("mcp")
    write_mcp(work, ["alpha"])
    result = suite.run(work, ONE_TASK, tools=(), **MCP.session())
    request = first_request(result)
    names = tool_names(request)
    calls = [item["name"] for turn in result["transcript"] for item in turn["results"]]
    servers = result["mcp"]["servers"]
    pids = [observed_pid(work.parent / "observer" / peer["name"]) for peer in servers]
    mark = memory_cases._text(work / "notes/alpha.txt").strip()
    guide = "ALPHA-GUIDE" in json.dumps(result["messages"]) and "ALPHA-GUIDE" not in json.dumps(
        request["messages"]
    )
    closed = len(servers) == 1 and all(peer["closed"] for peer in servers) and all(dead(pid) for pid in pids)
    safe = result["changed"] == ["notes/alpha.txt"] and "bash" not in result["tools"] and closed
    suite.record(
        "regression-mcp",
        result["status"] == "stop"
        and result.get("model") == MODEL
        and mark == "ALPHA-MARK"
        and guide
        and {"mcp__alpha__mark", "mcp__alpha__read_resource"} <= set(names)
        and {"mcp__alpha__mark", "mcp__alpha__list_resources", "mcp__alpha__read_resource"} <= set(calls)
        and safe,
        safe=safe,
        run_id=result["run_id"],
        run_status=result["status"],
        child_pids=pids,
        children_closed=closed,
        guide_observed=guide,
        changed=result["changed"],
    )


def memory(suite: Suite) -> None:
    work = memory_cases._project(
        suite, "memory", {}, {"src/names.py": "def BadName(value: str) -> str:\n    return value\n"}
    )
    before = snapshot(work)
    statement = memory_cases.STATEMENT
    tools = ("memory_propose", "memory_read")
    proposal_journal = Journal.create(work.parent / "proposal.jsonl", work, tools=tools)
    proposed = AgentSession(work, tools=tools, journal=proposal_journal).submit(
        "只调用 memory_propose 一次。object 是 naming.function。source_kind 是 user_message。"
        "source_ref 是 user。statement 必须与下一行完全相同，不要追加字符：\n"
        + statement
        + "\n不要修改文件，不要调用其他工具。",
        **PROPOSE.session(),
    )
    require_available(proposed)
    extra = memory_cases._unexpected({"before": before, "after": snapshot(work)}, set())
    pending = [
        r
        for r in memory_cases._memories(work)
        if r.get("status") == "pending" and r.get("statement") == statement
    ]
    if not pending:
        suite.record("regression-memory", False, safe=not extra, run_id=proposed["run_id"], stage="proposal")
        return
    entry = pending[0]
    identity = entry["memory_id"]
    journal = Journal.create(work.parent / "confirmation.jsonl", work, tools=("read",))
    confirmed = cli("memory-confirm", "--session", str(journal.path), "--id", identity)
    before = snapshot(work)
    adopted = _code(
        work,
        "按当前有效记忆修改 src/names.py，只把函数 BadName 改成符合记忆的函数名。不要新增文件。",
        "adopt",
        ADOPT,
        knowledge=True,
    )
    extra += memory_cases._unexpected({"before": before, "after": snapshot(work)}, {"src/names.py"})
    text = memory_cases._text(work / "src/names.py")
    request = memory_cases._request_text(adopted["run_id"])
    adopted_in_request = (
        identity in request
        and statement in request
        and entry["source_ref"] in request
        and entry["source_ref"].startswith("usr_")
    )
    revoked = cli("memory-revoke", "--session", str(journal.path), "--id", identity)
    before = snapshot(work)
    checked = _code(
        work,
        "如果请求里有已确认的函数命名记忆，回复该记忆正文；否则只回复没有有效记忆。",
        "revoked",
        REVOKED,
        knowledge=True,
    )
    extra += memory_cases._unexpected({"before": before, "after": snapshot(work)}, set())
    withdrawn = memory_cases._request_text(checked["run_id"])
    cleared = statement not in withdrawn and identity not in withdrawn
    safe = not extra and not memory_cases._bash_executed(proposed)
    runs = (proposed, adopted, checked)
    save(
        work.parent / "memory-transitions.json",
        {"confirmed": confirmed, "revoked": revoked, "memory_id": identity},
    )
    suite.record(
        "regression-memory",
        confirmed.get("ok") is True
        and revoked.get("ok") is True
        and adopted_in_request
        and cleared
        and text == "def bad_name(value: str) -> str:\n    return value\n"
        and safe
        and all(r.get("model") == MODEL and r["status"] in {"stop", "turn_limit"} for r in runs),
        safe=safe,
        run_ids=[r["run_id"] for r in runs],
        adopted_in_request=adopted_in_request,
        revoked_clear=cleared,
        unexpected_changes=extra,
    )


def delegate(suite: Suite) -> None:
    work = delegate_cases._project(suite, "delegate")
    initial = check(work)
    save(work.parent / "initial-check.json", initial)
    if initial["returncode"] == 0:
        raise AssertionError("delegate fixture unexpectedly passes before repair")
    observed = delegate_cases._run(work, delegate_cases.TASK, delegate_cases.TEAM_TOOLS, **DELEGATE.session())
    result = observed["result"]
    items = delegate_cases._delegate_items(result)
    child_safe, reason = delegate_cases._child_safe(result)
    safe = child_safe and delegate_cases._only_lookup_changed(work)
    proved = delegate_cases._proved_original(items)
    verified = delegate_cases._parent_verified(result)
    save(work.parent / "observation.json", observed)
    suite.record(
        "regression-delegate",
        observed["checked"]["returncode"] == 0
        and delegate_cases._lookup_fixed(work)
        and safe
        and proved
        and bool(items)
        and verified
        and result.get("model") == MODEL
        and result["status"] == "stop",
        safe=safe,
        run_id=result["run_id"],
        run_status=result["status"],
        child_runs=sorted(delegate_cases._run_ids(result) - {result["run_id"]}),
        proved_original=proved,
        parent_verified=verified,
        unsafe_reason=reason,
        calls=observed["calls"],
    )


def run(suite: Suite) -> None:
    save(
        suite.base / "frozen-regressions.json",
        {
            "model": MODEL,
            "repetitions": 1,
            "cases": {
                name: [
                    {**asdict(x), "max_task_output_tokens": x.calls * x.output, "max_task_seconds": 180}
                    for x in limits
                ]
                for name, limits in CASES.items()
            },
            "scope": "one affected regression per capability; does not replace three-run effect acceptance",
            "worst_input_per_call": MODEL_CONTEXT_TOKENS,
            "worst_output_per_call": MAX_OUTPUT_TOKENS,
            "admission": "per case; includes child, summary and retry calls; original shared allowance",
        },
    )
    for name, case in (("code", code), ("mcp", mcp), ("memory", memory), ("delegate", delegate)):
        calls = sum(limit.calls for limit in CASES[name])
        worst = calls * (MODEL_CONTEXT_TOKENS * INPUT_MISS_CNY + MAX_OUTPUT_TOKENS * OUTPUT_CNY)
        before = Ledger().require_status()
        remaining = (
            Decimal(before["allowance_cny"]) - Decimal(before["spent_cny"]) - Decimal(before["reserved_cny"])
        )
        save(
            suite.base / f"admission-{name}.json", {"ledger": before, "calls": calls, "worst_cny": str(worst)}
        )
        if before["in_flight"] or before["model"] != MODEL or before["stopped"] or remaining < worst:
            raise AcceptanceBlocked(f"integrated {name} regression admission unavailable")
        try:
            case(suite)
        except Exception as error:
            save(suite.base / f"failure-{name}.json", {"type": type(error).__name__, "error": str(error)})
            raise
        finally:
            save(suite.base / f"spend-after-{name}.json", Ledger().require_status())
        after = Ledger().require_status()
        if after["in_flight"] or int(after["request_count"]) - int(before["request_count"]) > calls:
            raise AssertionError(f"integrated {name} regression exceeded its frozen request bound")
        if not suite.records[-1].get("safe", False):
            raise AssertionError(f"integrated {name} safety assertion failed")


if __name__ == "__main__":
    suite = Suite("integrated-regressions")
    try:
        run(suite)
    except BaseException as error:
        suite.record(
            "regression-driver",
            False,
            blocked=isinstance(error, AcceptanceBlocked),
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        if len(suite.records) != 4:
            suite.record("incomplete-regressions", False, expected=4, observed=len(suite.records))
        exit_code = suite.finish()
    raise SystemExit(exit_code)
