"""Framework comparison with real DeepSeek runs. Heldout is copied only after the judgment rules exist."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from browser_support import browser_closed, load_shop, start_shop
from compare.arms import run_graph, run_handwritten
from compare.combo import (
    CHECK,
    TOOLS,
    capabilities,
    marker_closed,
    marker_count,
    prepare,
    task_text,
    write_servers,
)
from compare.judge import (
    HELDOUT_TASK,
    call_offset,
    check_unchanged,
    implementation_changed,
    product_run_ids,
    sha_tree,
    usage_since,
)
from compare.platform import docker_server, host_facts, wsl_available
from compare.upstream import pi_available, run_lite, run_pi
from support import ROOT, AcceptanceBlocked, Suite, check, require_available

from lite_pi.journal import Journal
from lite_pi.session import AgentSession
from lite_pi.spend.ledger import Ledger
from lite_pi.spend.proxy import serve

FIXTURE = ROOT / "baselines" / "coding" / "fixture"
HELDOUT = ROOT / "baselines" / "coding" / "heldout"


def run(suite: Suite) -> None:
    _note_heldout(suite)
    for index in range(1, 4):
        if not _room():
            _skip(suite, f"langgraph-compare-{index}", "langgraph-compare")
            continue
        _effect(suite, "langgraph-compare", index, _langgraph)
    for index in range(1, 4):
        if not _room():
            _skip(suite, f"heldout-{index}", "heldout")
            continue
        _effect(suite, "heldout", index, _heldout)
    for index in range(1, 4):
        if not _room():
            _skip(suite, f"combination-{index}", "combination")
            continue
        _effect(suite, "combination", index, _combination)
    _upstream(suite)
    _platforms(suite)
    if _room():
        _effect(suite, "public-entry", 1, _public_entry, boundary=True)
    else:
        _skip(suite, "public-entry-1", None)


def _langgraph(suite: Suite, effect: str, index: int, boundary: bool) -> None:
    del boundary
    hand_parent = suite.base / f"{effect}-{index}-hand"
    graph_parent = suite.base / f"{effect}-{index}-graph"
    hand_parent.mkdir()
    graph_parent.mkdir()
    offset = call_offset()
    before = Ledger().status()
    hand = run_handwritten(hand_parent, FIXTURE)
    graph = run_graph(graph_parent, FIXTURE, run_id=f"comparison-graph-{index}-{uuid4().hex[:8]}")
    after = Ledger().status()
    hand_ids = product_run_ids(hand["results"])
    graph_ids = {graph["run_id"]}
    hand_usage = usage_since(offset, hand_ids)
    graph_usage = usage_since(offset, graph_ids)
    hand_ok = _fix_ok(hand)
    graph_ok = _fix_ok(graph)
    models = _models(offset, hand_ids | graph_ids)
    passed = hand_ok and graph_ok and models == {"deepseek-flash"}
    suite.record(
        f"{effect}-{index}",
        passed,
        effect=effect,
        safe=check_unchanged(hand["before"], hand["after"])
        and check_unchanged(graph["before"], graph["after"]),
        hit=True,
        hand=_arm_metrics(hand, hand_usage),
        graph=_arm_metrics(graph, graph_usage),
        graph_slower=graph["elapsed"] > hand["elapsed"],
        graph_more_calls=graph_usage["calls"] > hand_usage["calls"],
        spent_cny=str(Decimal(str(after["spent_cny"])) - Decimal(str(before["spent_cny"]))),
        ledger=str(Ledger().path),
        human_interventions=0,
    )


def _heldout(suite: Suite, effect: str, index: int, boundary: bool) -> None:
    del boundary
    parent = suite.base / f"{effect}-{index}"
    parent.mkdir()
    observed = run_handwritten(parent, HELDOUT, task=HELDOUT_TASK)
    for result in observed["results"]:
        require_available(result)
    passed = _fix_ok(observed) and all(r.get("model") == "deepseek-flash" for r in observed["results"])
    suite.record(
        f"{effect}-{index}",
        passed,
        effect=effect,
        safe=check_unchanged(observed["before"], observed["after"]),
        hit=observed["resumed"],
        run_ids=sorted(product_run_ids(observed["results"])),
        resumed=observed["resumed"],
        check=observed["checked"]["returncode"],
        human_interventions=0,
    )


def _combination(suite: Suite, effect: str, index: int, boundary: bool) -> None:
    del boundary
    work = suite.project(f"{effect}-{index}")
    prepare(work)
    before = sha_tree(work)
    orders = work.parent / "orders.json"
    browser = work.parent / "browser"
    server = start_shop(orders)
    try:
        write_servers(work, server.origin, browser)
        journal = Journal.create(work.parent / "session.jsonl", work, tools=TOOLS)
        result = AgentSession(work, tools=TOOLS, journal=journal).submit(
            task_text(server.origin),
            max_turns=14,
            max_tokens=700,
            max_task_calls=20,
            max_retrievals=4,
        )
        require_available(result)
        checked = check(work)
        used = capabilities(result)
        unchanged = (work / "check.py").read_text(encoding="utf-8")
        file_ok = checked["returncode"] == 0 and unchanged == CHECK
        covered = all(used.values())
        shop = load_shop(orders)
        marker = work.parent / "observer/marker"
        effects = (
            shop["posts"] == 1
            and len(shop["orders"]) == 1
            and shop["orders"][0]["sku"] == "notebook"
            and marker_count(marker) == 1
        )
        closed = browser_closed(browser) and marker_closed(marker)
        after = sha_tree(work)
        allowed = {
            "src/lookup.py",
            "note.json",
            "notes/marker.txt",
            ".lite-pi/mcp.json",
            ".lite-pi/knowledge.sqlite",
            ".lite-pi/knowledge.sqlite-wal",
            ".lite-pi/knowledge.sqlite-shm",
        }
        changes_safe = all(
            path in allowed or before.get(path) == after.get(path) for path in before.keys() | after.keys()
        )
        suite.record(
            f"{effect}-{index}",
            file_ok and covered and effects and closed and result.get("status") == "stop",
            effect=effect,
            safe=unchanged == CHECK and changes_safe and closed and shop["posts"] <= 1,
            external_effects=effects,
            resources_closed=closed,
            hit=True,
            run_id=result.get("run_id"),
            model=result.get("model"),
            status_run=result.get("status"),
            check=checked["returncode"],
            capabilities=used,
            human_interventions=0,
        )
    finally:
        server.close()


def _upstream(suite: Suite) -> None:
    if not pi_available():
        for index in range(1, 4):
            _skip(
                suite, f"upstream-compare-{index}", "upstream-compare", reason="fixed upstream Pi unavailable"
            )
        return
    server = serve("127.0.0.1", 0)
    port = server.server_port
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            response.read()
        for index in range(1, 4):
            if not _room():
                _skip(suite, f"upstream-compare-{index}", "upstream-compare")
                continue
            _effect(suite, "upstream-compare", index, lambda s, e, i, b: _pair(s, e, i, b, port))
    finally:
        server.shutdown()
        server.server_close()


def _pair(suite: Suite, effect: str, index: int, boundary: bool, port: int) -> None:
    del boundary
    lite_parent = suite.base / f"{effect}-{index}-lite"
    pi_parent = suite.base / f"{effect}-{index}-pi"
    lite_parent.mkdir()
    pi_parent.mkdir()
    offset = call_offset()
    lite = run_lite(lite_parent)
    upstream = run_pi(pi_parent, index, port)
    for arm in (lite, upstream):
        run_ids = {str(arm["run_id"])}
        arm["usage"] = usage_since(offset, run_ids)
        arm["actual_models"] = sorted(_models(offset, run_ids))
    passed = bool(
        lite["passed"]
        and upstream["passed"]
        and all(arm["actual_models"] == ["deepseek-flash"] for arm in (lite, upstream))
    )
    suite.record(
        f"{effect}-{index}",
        passed,
        effect=effect,
        safe=bool(lite["safe"] and upstream["safe"]),
        hit=True,
        lite=lite,
        upstream=upstream,
        human_interventions=0,
    )


def _platforms(suite: Suite) -> None:
    facts = host_facts()
    suite.record(
        "platforms",
        False,
        blocked=True,
        hit=False,
        reason="Independent clean macOS, Linux and WSL2 host receipts required; see macos_run for local fresh-environment checks",
        macos=facts,
        linux="blocked",
        docker=docker_server(),
        wsl="blocked" if not wsl_available() else "present-unverified",
        human_interventions=0,
    )


def _public_entry(suite: Suite, effect: str, index: int, boundary: bool) -> None:
    del effect, index, boundary
    parent = suite.base / "public-entry"
    parent.mkdir()
    work = parent / "project"
    shutil_copy(HELDOUT, work)
    before = sha_tree(work)
    session = parent / "session.jsonl"
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
    if "HOME" in os.environ:
        env["HOME"] = os.environ["HOME"]
    started = time.monotonic()
    completed = subprocess.run(
        [
            "uv",
            "run",
            "lite-pi",
            "code",
            "--root",
            str(work),
            "--session",
            str(session),
            "--max-turns",
            "8",
            "--max-task-calls",
            "12",
            HELDOUT_TASK,
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    (parent / "stdout.json").write_text(completed.stdout)
    (parent / "stderr.txt").write_text(completed.stderr)
    receipt = json.loads(completed.stdout)
    run_path = receipt.get("evidence")
    run_result = json.loads((Path(run_path) / "result.json").read_text()) if isinstance(run_path, str) else {}
    checked = check(work)
    after = sha_tree(work)
    passed = (
        receipt.get("model") == "deepseek-flash"
        and run_result.get("request_count", 0) > 0
        and completed.returncode == 0
        and checked["returncode"] == 0
        and check_unchanged(before, after)
        and "DEEPSEEK_API_KEY" not in env
    )
    suite.record(
        "public-entry-1",
        passed,
        safe=check_unchanged(before, after) and "DEEPSEEK_API_KEY" not in env,
        hit=True,
        run_id=receipt.get("run_id"),
        request_count=run_result.get("request_count"),
        returncode=completed.returncode,
        check=checked["returncode"],
        elapsed=round(time.monotonic() - started, 3),
        ledger=str(Ledger().path),
        human_interventions=0,
    )


def _note_heldout(suite: Suite) -> None:
    path = suite.base / "heldout-first-exposure.json"
    if path.exists():
        return
    path.write_text(
        json.dumps(
            {
                "at": datetime.now(UTC).isoformat(),
                "path": str(HELDOUT),
                "rules": "design.md and compare/judge.py were written before this copy",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _fix_ok(observed: dict) -> bool:
    checked = observed.get("checked") or {}
    return (
        checked.get("returncode") == 0
        and check_unchanged(observed["before"], observed["after"])
        and implementation_changed(observed["before"], observed["after"])
        and observed.get("state_passed") is True
    )


def _arm_metrics(observed: dict, usage: dict[str, int]) -> dict:
    return {
        "elapsed": observed.get("elapsed"),
        "resumed": observed.get("resumed"),
        "check": (observed.get("checked") or {}).get("returncode"),
        "state_passed": observed.get("state_passed"),
        "calls": usage["calls"],
        "total_tokens": usage["total_tokens"],
        "unknown_usage": usage["unknown_usage"],
        "run_ids": sorted(product_run_ids(observed.get("results") or []) or {observed.get("run_id")}),
    }


def _models(offset: int, run_ids: set[str]) -> set[str]:
    path = Ledger().calls_path
    found: set[str] = set()
    if not path.exists():
        return found
    with path.open(encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("run_id") in run_ids and isinstance(item.get("actual_model"), str):
                found.add(item["actual_model"])
    return found


def _room() -> bool:
    status = Ledger().status()
    return status.get("in_flight") == 0 and status.get("stopped") is not True


def _skip(
    suite: Suite, name: str, effect: str | None, *, reason: str = "shared ledger is busy or stopped"
) -> None:
    fields: dict[str, object] = {
        "blocked": True,
        "reason": reason,
        "ledger": str(Ledger().path),
    }
    if effect is not None:
        fields["effect"] = effect
    suite.record(name, False, **fields)


def _effect(
    suite: Suite,
    effect: str,
    index: int,
    case: Callable[[Suite, str, int, bool], None],
    *,
    boundary: bool = False,
) -> None:
    try:
        case(suite, effect, index, boundary)
    except AcceptanceBlocked as error:
        fields: dict[str, object] = {"blocked": True, "reason": str(error)}
        if not boundary:
            fields["effect"] = effect
        suite.record(f"{effect}-{index}", False, **fields)
    except Exception as error:
        fields = {"safe": False, "error": f"{type(error).__name__}: {error}"}
        if not boundary:
            fields["effect"] = effect
        suite.record(f"{effect}-{index}", False, **fields)


def shutil_copy(source: Path, work: Path) -> None:
    import shutil

    shutil.copytree(source, work)
