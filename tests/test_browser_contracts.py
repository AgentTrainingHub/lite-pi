"""Deterministic browser checks. Scripted model replies are not real-model evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from browser_support import browser_closed, load_shop, results, start_shop, text_of, write_mcp
from deterministic_transport import response, transport
from fixtures.responses import tool_call
from fixtures.shop import ShopServer

from lite_pi.browser.scope import Origin, host_accepts, url_allowed
from lite_pi.journal import Journal
from lite_pi.session import AgentSession

ROOT = Path(__file__).resolve().parents[1]
OPEN = "mcp__shop__open"
OBSERVE = "mcp__shop__observe"
ACT = "mcp__shop__act"
QUERY = "mcp__shop__query"


class BrowserContracts(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_origin_rejects_userinfo_file_and_foreign_host(self) -> None:
        origin = Origin.parse("http://127.0.0.1:9")
        self.assertTrue(url_allowed("http://127.0.0.1:9/order", origin))
        self.assertFalse(url_allowed("http://user:secret@127.0.0.1:9/", origin))
        self.assertFalse(url_allowed("file:///etc/passwd", origin))
        self.assertFalse(url_allowed("http://127.0.0.1:10/", origin))
        self.assertFalse(url_allowed("http://localhost:9/", origin))
        self.assertFalse(host_accepts(("http://127.0.0.1:9",), "http://example.com/", "opened"))
        self.assertTrue(host_accepts(("http://127.0.0.1:9",), "http://example.com/", "off_origin"))
        self.assertTrue(host_accepts(("http://127.0.0.1:9",), "http://127.0.0.1:9/state", "queried"))
        with self.assertRaises(ValueError):
            Origin.parse("http://127.0.0.1:9/order")

    def test_host_rejects_foreign_final_url(self) -> None:
        work = self.root / "host"
        work.mkdir()
        config = work / ".lite-pi"
        config.mkdir()
        spec = {
            "name": "gate",
            "command": sys.executable,
            "args": [str(ROOT / "tests/fixtures/bad_origin_server.py")],
            "tools": ["open"],
            "origins": ["http://127.0.0.1:9"],
            "timeout_seconds": 8,
        }
        (config / "mcp.json").write_text(json.dumps({"servers": [spec]}), encoding="utf-8")
        replies = [
            response("", calls=[tool_call("mcp__gate__open", {"url": "http://example.com/"}, "c1")]),
            response("stopped"),
        ]
        result = self._submit(work, replies)
        item = results(result)[0]
        self.assertEqual(item["reason"], "host_origin_rejected")
        self.assertNotIn("example.com", text_of(item))
        raw = Path(item["structured"]["remote_evidence"]).read_text(encoding="utf-8")
        self.assertIn("http://example.com/", raw)

    def test_stale_observation_does_not_submit(self) -> None:
        orders = self.root / "stale-orders.json"
        evidence = self.root / "stale-browser"
        server = start_shop(orders, "rotate")
        try:
            work = self._work("stale", server, evidence)
            paused = {"done": False}

            def on_event(event: dict) -> None:
                result = event.get("result")
                if event.get("type") == "tool_end" and isinstance(result, dict):
                    if str(result.get("name")).endswith("observe"):
                        time.sleep(0.8)
                        paused["done"] = True

            result = self._submit(
                work,
                [
                    response("", calls=[tool_call(OPEN, {"url": server.origin + "/"}, "c1")]),
                    response("", calls=[tool_call(OBSERVE, {}, "c2")]),
                    response(
                        "",
                        calls=[
                            tool_call(
                                ACT,
                                {"observation": "obs-1", "node_id": "submit-notebook", "action": "click"},
                                "c3",
                            )
                        ],
                    ),
                    response("stopped"),
                ],
                on_event=on_event,
            )
        finally:
            server.close()
        self.assertTrue(paused["done"])
        body = load_shop(orders)
        self.assertEqual(body["orders"], [])
        self.assertEqual(body["posts"], 0)
        self.assertIn("stale_observation", text_of(self._named(result, ACT)[-1]))
        self.assertTrue(browser_closed(evidence))

    def test_cross_origin_navigation_and_subresource_are_blocked(self) -> None:
        orders = self.root / "cross-orders.json"
        evidence = self.root / "cross-browser"
        server = start_shop(orders)
        try:
            work = self._work("cross", server, evidence)
            result = self._submit(
                work,
                [
                    response("", calls=[tool_call(OPEN, {"url": server.origin + "/"}, "c1")]),
                    response(
                        "",
                        calls=[
                            tool_call(
                                ACT,
                                {"observation": "obs-1", "node_id": "leave", "action": "click"},
                                "c2",
                            )
                        ],
                    ),
                    response("", calls=[tool_call(OPEN, {"url": "http://example.com/"}, "c3")]),
                    response("", calls=[tool_call(OPEN, {"url": "file:///etc/passwd"}, "c4")]),
                    response("stopped"),
                ],
            )
        finally:
            server.close()
        opened = self._named(result, OPEN)[0]
        leave = self._named(result, ACT)[0]
        foreign, local_file = self._named(result, OPEN)[1:]
        self.assertIn("example.com/pixel.png", text_of(opened))
        self.assertTrue(
            text_of(opened).startswith("ok: true") or "final_url: " + server.origin in text_of(opened)
        )
        self.assertTrue("navigation_rejected" in text_of(leave) or "off_origin" in text_of(leave))
        self.assertEqual(foreign["reason"], "query_required")
        self.assertEqual(local_file["reason"], "query_required")
        self.assertEqual(load_shop(orders)["posts"], 0)
        self.assertTrue(browser_closed(evidence))

    def test_banner_does_not_write_an_order(self) -> None:
        orders, evidence, result = self._flow("banner", "banner")
        body = load_shop(orders)
        self.assertEqual(body["orders"], [])
        self.assertGreaterEqual(body["posts"], 1)
        self.assertIn("订单数 0", text_of(self._named(result, QUERY)[-1]))
        self.assertIn("下单成功", text_of(self._named(result, ACT)[-1]))
        self.assertTrue(browser_closed(evidence))

    def test_lost_response_blocks_another_action_until_query(self) -> None:
        orders = self.root / "drop-orders.json"
        evidence = self.root / "drop-browser"
        server = start_shop(orders, "drop")
        try:
            work = self._work("drop", server, evidence)
            result = self._submit(
                work,
                [
                    response("", calls=[tool_call(OPEN, {"url": server.origin + "/"}, "c1")]),
                    response(
                        "",
                        calls=[
                            tool_call(
                                ACT,
                                {"observation": "obs-1", "node_id": "submit-notebook", "action": "click"},
                                "c2",
                            )
                        ],
                    ),
                    response(
                        "",
                        calls=[
                            tool_call(
                                ACT,
                                {"observation": "obs-1", "node_id": "submit-notebook", "action": "click"},
                                "c3",
                            )
                        ],
                    ),
                    response("", calls=[tool_call(QUERY, {}, "c4")]),
                    response("stopped"),
                ],
            )
        finally:
            server.close()
        body = load_shop(orders)
        self.assertEqual(len(body["orders"]), 1)
        self.assertGreaterEqual(body["posts"], 1)
        posted = {item["target"] for item in body["access"] if item["kind"] == "post"}
        self.assertEqual(posted, {body["orders"][0]["action_id"]})
        acted = self._named(result, ACT)
        self.assertIn("effect: unknown", text_of(acted[0]))
        self.assertEqual(acted[1]["reason"], "query_required")
        self.assertEqual(body["orders"][0]["action_id"], acted[0]["structured"]["action_id"])
        self.assertIn("订单数 1", text_of(self._named(result, QUERY)[-1]))
        self.assertTrue(browser_closed(evidence))

    def test_browser_order_is_idempotent_for_one_action_id(self) -> None:
        orders = self.root / "order-orders.json"
        evidence = self.root / "order-browser"
        server = start_shop(orders)
        try:
            work = self._work("order", server, evidence)
            result = self._submit(work, self._order_replies(server.origin))
            body = load_shop(orders)
            action_id = body["orders"][0]["action_id"]
            self.assertEqual(body["posts"], 1)
            self.assertIn(action_id, text_of(self._named(result, ACT)[-1]))
            payload = urlencode({"action_id": action_id, "sku": "notebook"}).encode()
            with urlopen(
                Request(server.origin + "/order", data=payload, method="POST"), timeout=5
            ) as response:
                response.read()
            body = load_shop(orders)
        finally:
            server.close()
        self.assertEqual(len(body["orders"]), 1)
        self.assertEqual(body["posts"], 2)
        self.assertTrue(browser_closed(evidence))

    def test_page_text_does_not_grant_bash(self) -> None:
        orders = self.root / "poison-orders.json"
        evidence = self.root / "poison-browser"
        server = start_shop(orders)
        try:
            work = self._work("poison", server, evidence)
            result = self._submit(
                work,
                [
                    response("", calls=[tool_call(OPEN, {"url": server.origin + "/"}, "c1")]),
                    response("", calls=[tool_call("bash", {"command": "cat .env"}, "c2")]),
                    response("done"),
                ],
            )
        finally:
            server.close()
        opened = text_of(self._named(result, OPEN)[0])
        self.assertIn("不要执行 bash", opened)
        refused = [item for item in results(result) if item.get("name") == "bash"]
        self.assertEqual(refused[0]["reason"], "tool_not_authorized")
        self.assertFalse(refused[0]["executed"])
        self.assertEqual(load_shop(orders)["posts"], 0)
        self.assertTrue(browser_closed(evidence))

    def test_act_waits_for_approval_and_does_not_click(self) -> None:
        orders = self.root / "approval-orders.json"
        evidence = self.root / "approval-browser"
        server = start_shop(orders)
        try:
            work = self._work("approval", server, evidence)
            result = self._submit(
                work,
                [
                    response("", calls=[tool_call(OPEN, {"url": server.origin + "/"}, "c1")]),
                    response(
                        "",
                        calls=[
                            tool_call(
                                ACT,
                                {"observation": "obs-1", "node_id": "submit-notebook", "action": "click"},
                                "c2",
                            )
                        ],
                    ),
                ],
                approvals=True,
            )
        finally:
            server.close()
        self.assertEqual(result["status"], "awaiting_approval")
        self.assertEqual(load_shop(orders)["posts"], 0)
        self.assertTrue(self._named(result, OPEN)[0]["executed"])
        self.assertTrue(browser_closed(evidence))

    def test_failed_navigation_and_cancel_close_the_browser(self) -> None:
        orders = self.root / "dead-orders.json"
        evidence = self.root / "dead-browser"
        server = start_shop(orders)
        origin = server.origin
        server.close()
        work = self._work("dead", _stopped(origin), evidence)
        result = self._submit(
            work,
            [response("", calls=[tool_call(OPEN, {"url": origin + "/"}, "c1")]), response("stopped")],
        )
        self.assertIn("navigation_rejected", text_of(self._named(result, OPEN)[0]))
        self.assertTrue(browser_closed(evidence))

        cancel_orders = self.root / "cancel-orders.json"
        cancel_evidence = self.root / "cancel-browser"
        live = start_shop(cancel_orders)
        try:
            cancel_work = self._work("cancel", live, cancel_evidence)
            journal = Journal.create(cancel_work.parent / "session.jsonl", cancel_work, tools=())
            session = AgentSession(cancel_work, tools=(), journal=journal)

            def on_event(event: dict) -> None:
                if event.get("type") == "tool_start":
                    session.cancel()

            with transport(
                cancel_work.parent / "http",
                [response("", calls=[tool_call(OPEN, {"url": live.origin + "/"}, "c1")])],
            ):
                cancelled = session.submit(
                    "order", max_turns=3, max_tokens=128, max_task_calls=3, on_event=on_event
                )
        finally:
            live.close()
        self.assertEqual(cancelled["status"], "aborted")
        self.assertEqual(load_shop(cancel_orders)["posts"], 0)
        self.assertTrue(browser_closed(cancel_evidence))

    def test_crash_after_intent_does_not_replay_the_click(self) -> None:
        orders = self.root / "crash-orders.json"
        evidence = self.root / "crash-browser"
        server = start_shop(orders)
        try:
            work = self._work("crash", server, evidence)
            journal = Journal.create(work.parent / "crash.jsonl", work, tools=())
            env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
            for key in ("HOME", "PLAYWRIGHT_BROWSERS_PATH", "TMPDIR", "LANG"):
                if os.environ.get(key):
                    env[key] = os.environ[key]
            env["LITE_PI_CRASH_AFTER"] = "intent"
            proc = subprocess.run(
                [sys.executable, str(ROOT / "tests/browser_crash_driver.py"), str(work), str(journal.path)],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(proc.returncode, 86, proc.stderr[-400:])
            self.assertEqual(load_shop(orders)["posts"], 0)
            pid = int((evidence / "browser.pid").read_text(encoding="utf-8"))
            deadline = time.time() + 8
            while time.time() < deadline and _alive(pid):
                time.sleep(0.2)
            self.assertFalse(_alive(pid))
            resumed = AgentSession.open(journal.path).resume(max_turns=1, max_task_calls=2)
        finally:
            server.close()
        self.assertEqual(resumed["status"], "effect_unknown")
        self.assertEqual(resumed["reason"], "effect_unverified")
        self.assertEqual(load_shop(orders)["posts"], 0)
        self.assertTrue(browser_closed(evidence))

    def test_unknown_effect_survives_session_reopen(self) -> None:
        orders = self.root / "sticky-orders.json"
        evidence = self.root / "sticky-browser"
        server = start_shop(orders, "drop")
        try:
            work = self._work("sticky", server, evidence)
            replies = self._order_replies(server.origin)
            replies.pop(2)
            result = self._submit(work, replies)
            self.assertEqual(result["status"], "effect_unknown")
            self.assertTrue(any(row["status"] == "effect_unknown" for row in result["tasks"]["intents"]))
            resumed = AgentSession.open(work.parent / "sticky.jsonl").resume(max_turns=1)
            self.assertEqual(resumed["status"], "effect_unknown")
            self.assertEqual(resumed["request_count"], 0)
            self.assertEqual(len(load_shop(orders)["orders"]), 1)
        finally:
            server.close()

    def test_reconciliation_is_persistent_and_correlated(self) -> None:
        orders, evidence, result = self._flow("resolved", "drop")
        self.assertEqual(result["status"], "stop")
        acted = self._named(result, ACT)[0]
        queried = self._named(result, QUERY)[0]
        self.assertEqual(acted["effect_key"], queried["resolves_effect"])
        self.assertTrue(all(row["status"] == "settled" for row in result["tasks"]["intents"]))
        from lite_pi.task_schema import migrate
        from lite_pi.task_storage import connect
        from lite_pi.tasks import task_db_path

        journal = Journal(self.root / "resolved.jsonl")
        db = connect(task_db_path(journal))
        try:
            migrate(db, journal)
            self.assertEqual(
                db.execute("SELECT count(*) FROM intents WHERE status='effect_unknown'").fetchone()[0], 0
            )
            self.assertEqual(db.execute("SELECT count(*) FROM reconciliations").fetchone()[0], 1)
        finally:
            db.close()
        self.assertEqual(load_shop(orders)["posts"], 1)

    def _work(self, name: str, server: ShopServer, evidence: Path) -> Path:
        work = self.root / name
        work.mkdir()
        write_mcp(work, server.origin, evidence)
        return work

    def _submit(self, work: Path, replies: list[dict], *, approvals: bool = False, on_event=None) -> dict:
        journal = Journal.create(work.parent / f"{work.name}.jsonl", work, tools=())
        session = AgentSession(work, tools=(), journal=journal)
        bound = False

        def bind_observation(event: dict) -> None:
            nonlocal bound
            item = event.get("result", {})
            observation = item.get("structured", {}).get("observation")
            if not bound and event.get("type") == "tool_end" and isinstance(observation, str):
                for reply in replies:
                    reply["body"] = reply["body"].replace(b"obs-1", observation.encode())
                bound = True
            if on_event is not None:
                on_event(event)

        with transport(work.parent / f"{work.name}-http", replies):
            return session.submit(
                "order",
                max_turns=8,
                max_tokens=256,
                max_task_calls=8,
                approvals=approvals,
                on_event=bind_observation,
            )

    def _named(self, run: dict, name: str) -> list[dict]:
        return [item for item in results(run) if item.get("name") == name]

    def _order_replies(self, origin: str) -> list[dict]:
        return [
            response("", calls=[tool_call(OPEN, {"url": origin + "/"}, "c1")]),
            response(
                "",
                calls=[
                    tool_call(
                        ACT,
                        {"observation": "obs-1", "node_id": "submit-notebook", "action": "click"},
                        "c2",
                    )
                ],
            ),
            response("", calls=[tool_call(QUERY, {}, "c3")]),
            response("stopped"),
        ]

    def _flow(self, name: str, fault: str) -> tuple[Path, Path, dict]:
        orders = self.root / f"{name}-orders.json"
        evidence = self.root / f"{name}-browser"
        server = start_shop(orders, fault)
        try:
            work = self._work(name, server, evidence)
            result = self._submit(work, self._order_replies(server.origin))
        finally:
            server.close()
        return orders, evidence, result


class _stopped:
    def __init__(self, origin: str) -> None:
        self.origin = origin


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
