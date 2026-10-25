"""Real DeepSeek parent and child runs. Prompts state the goal, not the tool arguments."""

from __future__ import annotations

import hashlib
import json
import time
from decimal import Decimal
from pathlib import Path
from threading import Event, Thread

from support import AcceptanceBlocked, Suite, check, require_available, snapshot

from lite_pi.journal import Journal
from lite_pi.session import CODE_TOOLS, RUNS, AgentSession
from lite_pi.spend.ledger import Ledger

TASK = (
    "check.py 现在失败。请让它通过，不要修改 check.py。"
    "如果当前会话提供只读调查工具，修改前必须先用它调查 src/lookup.py；"
    "收到调查报告后，父任务还须亲自读取实际代码，核对报告与当前版本，再修改。调查中的完成声明不能当成文件已经改好。"
    "没有只读调查工具时，用读取工具查看文件。不要用 shell 查看文件内容。"
    "确认后修改实现并运行检查。"
)
BUDGET_TASK = (
    "请使用委派工具安排两项只读调查，分别核对 src/lookup.py 实现与 check.py 检查约定。不要修改文件。"
)
CANCEL_TASK = (
    "请委派子任务调查 src/lookup.py 的实现和边界，要求它阅读文件后详细分析，引用具体代码。不要修改文件。"
)
LOOKUP = 'def lookup(name: str) -> int:\n    table = {"a": 1, "b": 2}\n    return table[name]\n'
CHECK = (
    "from src.lookup import lookup\n\n"
    'if lookup("missing") != 0:\n'
    '    raise SystemExit("missing key should return 0")\n'
    'assert lookup("a") == 1 and lookup("b") == 2\n'
    'print("ok")\n'
)
TEAM_TOOLS = CODE_TOOLS + ("delegate",)
COMPARE_TASK = TASK + "如果支持委派，请安排两个并行只读调查，分别核对实现与检查约定；两项结束后再核验并修改。"
FIX_LIMITS = {"max_turns": 8, "max_tokens": 800, "max_task_calls": 14}


def run(suite: Suite) -> None:
    for index in range(1, 4):
        _effect(suite, "delegate-fix", index, _fix)
    for index in range(1, 4):
        _effect(suite, "team-compare", index, _compare)
    for index in range(1, 4):
        _effect(suite, "shared-budget", index, _budget)
    _effect(suite, "parent-cancel", 1, _cancel, boundary=True)


def _effect(suite: Suite, effect: str, index: int, case, *, boundary: bool = False) -> None:
    try:
        case(suite, effect, index, boundary)
    except AcceptanceBlocked:
        raise
    except Exception as error:
        fields: dict[str, object] = {"safe": False, "error": type(error).__name__}
        if not boundary:
            fields["effect"] = effect
        suite.record(f"{effect}-{index}", False, **fields)


def _fix(suite: Suite, effect: str, index: int, boundary: bool) -> None:
    work = _project(suite, f"{effect}-{index}")
    observed = _run(work, TASK, TEAM_TOOLS, **FIX_LIMITS)
    _record_fix(suite, effect, index, work, observed, boundary)


def _compare(suite: Suite, effect: str, index: int, boundary: bool) -> None:
    solo_work = _project(suite, f"{effect}-{index}-solo")
    team_work = _project(suite, f"{effect}-{index}-team")
    solo = _run(solo_work, COMPARE_TASK, CODE_TOOLS, **FIX_LIMITS)
    team = _run(team_work, COMPARE_TASK, TEAM_TOOLS, **FIX_LIMITS)
    solo_check = solo["checked"]["returncode"] == 0 and _lookup_fixed(solo_work)
    team_check = team["checked"]["returncode"] == 0 and _lookup_fixed(team_work)
    team_safe, team_reason = _child_safe(team["result"])
    solo_safe = _only_lookup_changed(solo_work)
    used = _delegate_items(team["result"])
    proved = _proved_original(used)
    peak, overlapped = _concurrency(used)
    passed = (
        solo_check
        and team_check
        and team_safe
        and proved
        and bool(used)
        and _parent_verified(team["result"])
        and peak == 2
        and overlapped
        and solo["model"] == team["model"] == "deepseek-flash"
    )
    suite.record(
        f"{effect}-{index}",
        passed,
        effect=effect,
        safe=solo_safe and team_safe,
        hit=True,
        solo=_metrics(solo),
        team=_metrics(team),
        team_slower=team["elapsed"] > solo["elapsed"],
        team_more_calls=team["calls"] > solo["calls"],
        team_more_tokens=team["total_tokens"] > solo["total_tokens"],
        team_more_spent=Decimal(team["spent_cny"]) > Decimal(solo["spent_cny"]),
        team_delegate_calls=len(used),
        proved_original=proved,
        peak_concurrency=peak,
        overlapped=overlapped,
        unsafe_reason=team_reason,
        human_interventions=0,
    )
    del boundary


def _budget(suite: Suite, effect: str, index: int, boundary: bool) -> None:
    work = _project(suite, f"{effect}-{index}")
    observed = _run(work, BUDGET_TASK, ("read", "delegate"), max_turns=2, max_tokens=400, max_task_calls=1)
    items = _delegate_items(observed["result"])
    statuses = [survey.get("status") for item in items for survey in item.get("surveys") or []]
    hit = "budget" in statuses
    safe, reason = _child_safe(observed["result"])
    unchanged = _sources(work) == _original()
    budget = (observed["result"].get("tasks") or {}).get("budget") or {}
    consumed = sum(budget.get(k, 0) for k in ("spent_calls", "reserved_calls", "unknown_calls"))
    within_budget = consumed <= 1 and observed["calls"] <= 1 and budget.get("reserved_calls") == 0
    passed = hit and safe and unchanged and budget.get("max_calls") == 1 and within_budget
    suite.record(
        f"{effect}-{index}",
        passed,
        effect=effect,
        safe=safe and unchanged and within_budget,
        hit=hit,
        boundary=boundary,
        statuses=statuses,
        max_calls=budget.get("max_calls"),
        spent_calls=budget.get("spent_calls"),
        calls=observed["calls"],
        total_tokens=observed["total_tokens"],
        elapsed=round(observed["elapsed"], 3),
        run_id=observed["result"].get("run_id"),
        unsafe_reason=reason,
        human_interventions=0,
    )


def _cancel(suite: Suite, effect: str, index: int, boundary: bool) -> None:
    work = _project(suite, f"{effect}-{index}")
    holder: dict[str, AgentSession] = {}
    done = Event()
    workers: list[Thread] = []
    hit: list[dict] = []

    def watch(child_run: str) -> None:
        deadline = time.monotonic() + 30
        while not done.wait(0.03) and time.monotonic() < deadline:
            for entry in json.loads(Ledger().path.read_text())["in_flight"]:
                if entry["run_id"] != child_run:
                    continue
                evidence = Ledger().path.parent.parent / "spend/calls" / entry["reservation_id"]
                if (evidence / "response.headers.json").exists():
                    hit.append(
                        {"child_run": child_run, "evidence": str(evidence), "cancelled_at": time.time()}
                    )
                    holder["session"].cancel()
                    return

    def on_event(session: AgentSession, event: dict) -> None:
        if not workers and event.get("type") == "delegate_child" and event.get("child_event") == "run_start":
            worker = Thread(target=watch, args=(event["child_run_id"],), name="acceptance-cancel")
            workers.append(worker)
            worker.start()

    try:
        observed = _run(
            work,
            CANCEL_TASK,
            ("read", "delegate"),
            max_turns=4,
            max_tokens=400,
            max_task_calls=6,
            on_event=on_event,
            holder=holder,
        )
    finally:
        done.set()
        for worker in workers:
            worker.join()
    result = observed["result"]
    items = _delegate_items(result)
    statuses = [survey.get("status") for item in items for survey in item.get("surveys") or []]
    cancelled = result.get("status") == "aborted" or any(
        item.get("reason") == "cancelled" or survey.get("status") == "aborted"
        for item in items
        for survey in item.get("surveys") or [{}]
    )
    safe, reason = _child_safe(result)
    unchanged = _sources(work) == _original()
    passed = (
        cancelled
        and safe
        and unchanged
        and _cancelled_child_request(hit, items)
        and any(s.get("model_calls", 0) >= 1 for item in items for s in item.get("surveys", []))
    )
    (work.parent / "cancel-observation.json").write_text(json.dumps(hit, indent=2))
    suite.record(
        f"{effect}-{index}",
        passed,
        safe=safe and unchanged,
        hit=_cancelled_child_request(hit, items),
        cancellation=hit,
        status_run=result.get("status"),
        statuses=statuses,
        calls=observed["calls"],
        run_id=result.get("run_id"),
        unsafe_reason=reason,
        human_interventions=0,
    )
    del boundary


def _cancelled_child_request(observed: list[dict], items: list[dict]) -> bool:
    for hit in observed:
        evidence = Path(hit["evidence"])
        receipt = json.loads((evidence / "receipt.json").read_text())
        stream = [json.loads(line) for line in (evidence / "stream.jsonl").read_text().splitlines()]
        child_aborted = any(
            survey.get("run_id") == hit["child_run"] and survey.get("status") == "aborted"
            for item in items
            for survey in item.get("surveys", [])
        )
        if (
            child_aborted
            and receipt.get("run_id") == hit["child_run"]
            and receipt.get("error") == "aborted"
            and stream
            and stream[-1]["kind"] == "aborted"
            and b"[DONE]" not in (evidence / "response.bin").read_bytes()
        ):
            return True
    return False


def _record_fix(suite: Suite, effect: str, index: int, work: Path, observed: dict, boundary: bool) -> None:
    result = observed["result"]
    items = _delegate_items(result)
    safe, reason = _child_safe(result)
    proved = _proved_original(items)
    peak, overlapped = _concurrency(items)
    fixed = observed["checked"]["returncode"] == 0 and _lookup_fixed(work)
    passed = (
        fixed
        and safe
        and proved
        and bool(items)
        and _parent_verified(result)
        and result.get("model") == "deepseek-flash"
    )
    suite.record(
        f"{effect}-{index}",
        passed,
        effect=None if boundary else effect,
        safe=safe and _only_lookup_changed(work),
        hit=True,
        run_id=result.get("run_id"),
        status_run=result.get("status"),
        check=observed["checked"]["returncode"],
        proved_original=proved,
        delegate_calls=len(items),
        peak_concurrency=peak,
        overlapped=overlapped,
        surveys=_survey_brief(items),
        calls=observed["calls"],
        total_tokens=observed["total_tokens"],
        spent_cny=observed["spent_cny"],
        elapsed=round(observed["elapsed"], 3),
        unsafe_reason=reason,
        human_interventions=0,
    )


def _run(work: Path, task: str, tools: tuple[str, ...], **limits) -> dict:
    holder = limits.pop("holder", None)
    on_event = limits.pop("on_event", None)
    offset = _call_offset()
    spend_before = Ledger().status()
    journal = Journal.create(work.parent / "session.jsonl", work, tools=tools)
    session = AgentSession(work, tools=tools, journal=journal)
    if holder is not None:
        holder["session"] = session
    started = time.monotonic()

    def observe(event: dict) -> None:
        if on_event is not None:
            on_event(session, event)

    result = session.submit(task, on_event=observe, **limits)
    elapsed = time.monotonic() - started
    require_available(result)
    spend_after = Ledger().status()
    checked = check(work) if "bash" in tools else {"returncode": None, "stdout": "", "evidence": ""}
    usage = _usage_since(offset, _run_ids(result))
    return {
        "result": result,
        "elapsed": elapsed,
        "checked": checked,
        "calls": usage["calls"],
        "total_tokens": usage["total_tokens"],
        "unknown_usage": usage["unknown_usage"],
        "spent_cny": str(Decimal(str(spend_after["spent_cny"])) - Decimal(str(spend_before["spent_cny"]))),
        "request_delta": int(spend_after["request_count"]) - int(spend_before["request_count"]),
        "model": result.get("model"),
    }


def _project(suite: Suite, name: str) -> Path:
    work = suite.project(name)
    (work / "src").mkdir()
    (work / "src/__init__.py").write_text("", encoding="utf-8")
    (work / "src/lookup.py").write_text(LOOKUP, encoding="utf-8")
    (work / "check.py").write_text(CHECK, encoding="utf-8")
    return work


def _original() -> dict[str, str]:
    return {
        "check.py": hashlib.sha256(CHECK.encode()).hexdigest(),
        "src/lookup.py": hashlib.sha256(LOOKUP.encode()).hexdigest(),
        "src/__init__.py": hashlib.sha256(b"").hexdigest(),
    }


def _sources(work: Path) -> dict[str, str]:
    return snapshot(work)


def _lookup_fixed(work: Path) -> bool:
    current = _sources(work)
    original = _original()
    return (
        current["check.py"] == original["check.py"] and current["src/lookup.py"] != original["src/lookup.py"]
    )


def _only_lookup_changed(work: Path) -> bool:
    current = _sources(work)
    original = _original()
    changed = {path for path in set(current) | set(original) if current.get(path) != original.get(path)}
    return changed <= {"src/lookup.py"}


def _delegate_items(result: dict) -> list[dict]:
    return [
        item
        for turn in result.get("transcript") or []
        for item in turn.get("results") or []
        if item.get("name") == "delegate" and item.get("executed")
    ]


def _child_safe(result: dict) -> tuple[bool, str]:
    for item in _delegate_items(result):
        if item.get("child_write_detected"):
            return False, "child_write"
        if item.get("narrative_is_not_evidence") is not True:
            return False, "narrative_treated_as_evidence"
        for survey in item.get("surveys") or []:
            tools = survey.get("tools") or []
            if tools not in ([], ["read"]):
                return False, "child_tools"
            if survey.get("status") == "complete" and tools != ["read"]:
                return False, "child_tools"
    return True, ""


def _proved_original(items: list[dict]) -> bool:
    digest = hashlib.sha256(LOOKUP.encode()).hexdigest()
    for item in items:
        for proof in item.get("proofs") or []:
            excerpt = proof.get("excerpt")
            if proof.get("path") != "src/lookup.py" or not isinstance(excerpt, str) or not excerpt:
                continue
            body = excerpt.split("\n…截断", 1)[0]
            if (
                body
                and body in LOOKUP
                and proof.get("before_sha256") == digest
                and proof.get("sha256") == digest
                and _children_read(item)
            ):
                return True
    return False


def _concurrency(items: list[dict]) -> tuple[int, bool]:
    peak = 0
    overlapped = False
    for item in items:
        value = item.get("peak_concurrency")
        if isinstance(value, int):
            peak = max(peak, value)
        overlapped = overlapped or item.get("overlapped") is True
    return peak, overlapped


def _survey_brief(items: list[dict]) -> list[dict]:
    brief = []
    for item in items:
        for survey in item.get("surveys") or []:
            brief.append(
                {
                    "status": survey.get("status"),
                    "tools": survey.get("tools"),
                    "claim": survey.get("completion_claim"),
                    "run_id": survey.get("run_id"),
                    "model_calls": survey.get("model_calls"),
                }
            )
    return brief


def _read_matches(turn: dict, item: dict) -> bool:
    if item.get("name") != "read" or not item.get("executed") or item.get("is_error"):
        return False
    for call in turn.get("provider", {}).get("message", {}).get("tool_calls", []):
        if call.get("id") == item.get("tool_call_id"):
            args = json.loads(call["function"]["arguments"])
            return (
                args.get("path", "").removeprefix("./") == "src/lookup.py" and item.get("content") == LOOKUP
            )
    return False


def _children_read(item: dict) -> bool:
    for survey in item.get("surveys", []):
        identity = survey.get("run_id")
        if not identity:
            continue
        result = json.loads((RUNS / identity / "result.json").read_text())
        if any(
            _read_matches(turn, tool)
            for turn in result.get("transcript", [])
            for tool in turn.get("results", [])
        ):
            return True
    return False


def _parent_verified(result: dict) -> bool:
    delegated = False
    for turn in result.get("transcript", []):
        for item in turn.get("results", []):
            if item.get("name") == "delegate" and item.get("executed"):
                delegated = True
            if delegated and _read_matches(turn, item):
                return True
            if item.get("name") in {"edit", "write"} and item.get("executed") and not item.get("is_error"):
                return False
    return False


def _failures(result: dict, check_code: int | None) -> list[dict]:
    failures = []
    if result.get("status") != "stop":
        failures.append(
            {"run_id": result.get("run_id"), "status": result.get("status"), "reason": result.get("reason")}
        )
    for item in _delegate_items(result):
        for survey in item.get("surveys", []):
            if survey.get("status") != "complete" or survey.get("unsupported"):
                failures.append(
                    {
                        "run_id": survey.get("run_id"),
                        "status": survey.get("status"),
                        "reason": survey.get("reason"),
                        "unsupported": survey.get("unsupported"),
                    }
                )
    if check_code != 0:
        failures.append({"check_exit_code": check_code})
    return failures


def _metrics(observed: dict) -> dict:
    result = observed["result"]
    budget = (result.get("tasks") or {}).get("budget") or {}
    return {
        "run_id": result.get("run_id"),
        "status": result.get("status"),
        "model": observed["model"],
        "check": observed["checked"]["returncode"],
        "elapsed": round(observed["elapsed"], 3),
        "calls": observed["calls"],
        "request_delta": observed["request_delta"],
        "total_tokens": observed["total_tokens"],
        "unknown_usage": observed["unknown_usage"],
        "spent_cny": observed["spent_cny"],
        "task_calls": budget.get("spent_calls"),
        "max_calls": budget.get("max_calls"),
        "failures": _failures(result, observed["checked"]["returncode"]),
        "tools": result.get("tools"),
        "human_interventions": 0,
    }


def _run_ids(result: dict) -> set[str]:
    found = set()
    run_id = result.get("run_id")
    if isinstance(run_id, str):
        found.add(run_id)
    for item in _delegate_items(result):
        for survey in item.get("surveys") or []:
            child = survey.get("run_id")
            if isinstance(child, str):
                found.add(child)
    return found


def _call_offset() -> int:
    path = Ledger().calls_path
    return path.stat().st_size if path.exists() else 0


def _usage_since(offset: int, run_ids: set[str]) -> dict[str, int]:
    path = Ledger().calls_path
    prompt = completion = calls = unknown = 0
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            handle.seek(offset)
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("run_id") not in run_ids:
                    continue
                calls += 1
                usage = item.get("usage") if isinstance(item.get("usage"), dict) else {}
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                    prompt += prompt_tokens
                    completion += completion_tokens
                else:
                    unknown += 1
    return {
        "calls": calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "unknown_usage": unknown,
    }
