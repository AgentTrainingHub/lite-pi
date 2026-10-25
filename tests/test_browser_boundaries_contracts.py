"""Boundary regressions use real browser effects; no paid model calls."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from browser_support import load_shop, start_shop

from lite_pi.browser.driver import browser_alive, launch
from lite_pi.mcp_host import _checked


class BrowserBoundaryContracts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.orders = self.root / "orders.json"
        self.server = start_shop(self.orders, "drop")
        self.playwright, self.driver = await launch(self.server.origin, self.root / "browser", "chrome")

    async def asyncTearDown(self):
        try:
            await self.driver.close()
        finally:
            await self.playwright.stop()
            self.server.close()
            self.tmp.cleanup()

    async def test_failed_fill_can_observe_and_retry_without_query_lock(self):
        # Filling a non-editable real element raises Playwright Error; it cannot
        # create a POST receipt and must not lock every later action.
        opened = await self.driver.open(self.server.origin + "/")
        failed = await self.driver.act(opened.observation, "submit-notebook", "fill", "text")
        self.assertEqual(failed.reason, "fill_failed")
        self.assertIsNone(self.driver.unresolved)
        refreshed = await self.driver.observe()
        retried = await self.driver.act(refreshed.observation, "submit-notebook", "click")
        self.assertNotEqual(retried.reason, "query_required")
        self.assertEqual(load_shop(self.orders)["posts"], 1)

    async def test_replaced_identical_node_is_not_the_observed_element(self):
        opened = await self.driver.open(self.server.origin + "/")
        await self.driver.page.evaluate("""() => {
            const node = document.querySelector('[data-node="submit-notebook"]');
            node.replaceWith(node.cloneNode(true));
        }""")
        outcome = await self.driver.act(opened.observation, "submit-notebook", "click")
        self.assertEqual(outcome.reason, "stale_observation")
        self.assertEqual(load_shop(self.orders)["posts"], 0)

    async def test_changed_form_target_rejects_the_observed_action(self):
        opened = await self.driver.open(self.server.origin + "/")
        await self.driver.page.locator("[name=sku]").evaluate('(node) => node.value="other-product"')
        outcome = await self.driver.act(opened.observation, "submit-notebook", "click")
        self.assertEqual(outcome.reason, "stale_observation")
        self.assertEqual(load_shop(self.orders)["posts"], 0)

    async def test_unknown_query_response_does_not_allow_another_action(self):
        opened = await self.driver.open(self.server.origin + "/")
        acted = await self.driver.act(opened.observation, "submit-notebook", "click")
        self.assertEqual(acted.effect, "unknown")
        with patch.object(
            self.server.shop, "_state_html", return_value="<html><body>订单暂不可查</body></html>"
        ):
            queried = await self.driver.query()
        self.assertEqual(queried.effect, "unknown")
        self.assertEqual(self.driver.unresolved, acted.action_id)
        refused = await self.driver.act(opened.observation, "submit-notebook", "click")
        self.assertEqual(refused.reason, "query_required")
        queried = await self.driver.query()
        self.assertEqual(queried.reconciled_action, acted.action_id)
        self.assertIsNone(self.driver.unresolved)
        self.assertEqual(load_shop(self.orders)["posts"], 1)

    async def test_all_cleanup_callbacks_run_when_one_fails(self):
        async def fail():
            raise RuntimeError("cleanup probe")

        self.driver._resources.push_async_callback(fail)
        with self.assertRaisesRegex(RuntimeError, "cleanup probe"):
            await self.driver.close()
        pid = int((self.root / "browser/browser.pid").read_text())
        self.assertFalse(browser_alive(pid))
        self.assertFalse((self.root / "browser/closed.json").exists())

    async def test_origin_does_not_implicitly_authorize_read_only(self):
        spec = _checked(
            {
                "name": "sample",
                "command": "python",
                "args": [],
                "tools": ["open"],
                "origins": ["http://127.0.0.1:9"],
            }
        )
        self.assertEqual(spec.read_only_tools, ())

    async def test_websocket_and_popup_requests_are_blocked(self):
        await self.driver.open(self.server.origin + "/")
        await self.driver.page.evaluate(
            """(url) => {
            window.open(url + '/state');
            window.socket = new WebSocket(url.replace('http:', 'ws:') + '/ws');
        }""",
            self.server.origin,
        )
        await self.driver.page.wait_for_timeout(250)
        body = load_shop(self.orders)
        self.assertFalse(any(row["target"] in {"/state", "/ws"} for row in body["access"]))

    async def test_initialization_failure_closes_launched_browser(self):
        browser = await self.playwright.chromium.launch(headless=True, channel="chrome")
        with (
            patch.object(self.playwright.chromium, "launch", return_value=browser),
            patch.object(browser, "new_context", side_effect=RuntimeError("context probe")),
        ):
            with self.assertRaisesRegex(RuntimeError, "context probe"):
                await type(self.driver).start(
                    self.playwright, self.server.origin, self.root / "broken", "chrome"
                )
        self.assertFalse(browser.is_connected())

    async def test_click_handler_cannot_change_authorized_submission(self):
        opened = await self.driver.open(self.server.origin + "/")
        await self.driver.page.evaluate("""() => {
            const node = document.querySelector('[data-node="submit-notebook"]');
            node.addEventListener('pointerdown', () => {
                document.querySelector('[name="sku"]').value = 'other-product';
            });
        }""")
        acted = await self.driver.act(opened.observation, "submit-notebook", "click")
        self.assertEqual(acted.effect, "unknown")
        self.assertEqual(load_shop(self.orders)["posts"], 0)
        self.assertIsNotNone(self.driver.unresolved)

    async def test_capture_does_not_publish_mixed_page_states(self):
        from lite_pi.browser.observation import Observation

        await self.driver.open(self.server.origin + "/")
        title = self.driver.page.title

        async def change():
            await self.driver.page.locator("[name=sku]").evaluate('(node) => node.value="changed"')
            return await title()

        with patch.object(self.driver.page, "title", side_effect=change):
            with self.assertRaisesRegex(ValueError, "observation_changed_during_capture"):
                await Observation.capture(self.driver.page)

    async def test_host_rejects_success_without_a_final_url(self):
        from lite_pi.mcp_host import McpServer

        spec = _checked(
            {
                "name": "bad",
                "command": "python",
                "args": [],
                "tools": ["act"],
                "origins": [self.server.origin],
            }
        )
        server = McpServer(spec, cwd=self.root, state=self.root / "mcp")
        try:
            result = server._check_origin(
                {"text": "applied", "structured": {"text": "applied"}},
                {"ok": True, "effect": "applied", "reason": "driver_error"},
                "act",
            )
            self.assertEqual(result["reason"], "host_origin_rejected")
            self.assertEqual(result["effect"], "unknown")
        finally:
            server.close()

    async def test_post_then_rejected_redirect_preserves_unknown(self):
        self.server.shop.fault = "drop-redirect"
        opened = await self.driver.open(self.server.origin + "/")
        acted = await self.driver.act(opened.observation, "submit-notebook", "click")
        self.assertEqual(acted.effect, "unknown")
        self.assertEqual(load_shop(self.orders)["posts"], 1)
        queried = await self.driver.query()
        self.assertEqual(queried.reconciled_action, acted.action_id)
