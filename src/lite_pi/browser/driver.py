"""One browser owner, bounded navigation, and action-specific reconciliation."""

from __future__ import annotations

import json
import os
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl
from uuid import uuid4

from playwright.async_api import Browser, BrowserContext, Error, Page, Playwright, Route, async_playwright

from lite_pi.browser.observation import Observation, PageView
from lite_pi.browser.scope import Origin, url_allowed
from lite_pi.tools.contracts import JsonObject

Effect = Literal["none", "applied", "unknown"]


@dataclass(frozen=True, kw_only=True)
class DriverReceipt:
    ok: bool
    reason: str
    effect: Effect = "none"
    action_id: str | None = None
    final_url: str = ""
    observation: str | None = None
    view: PageView | None = None
    blocked_urls: tuple[str, ...] = ()
    diagnostic: str | None = None
    reconciled_action: str | None = None

    def public(self) -> JsonObject:
        return asdict(self)


@dataclass(frozen=True)
class Submission:
    url: str
    fields: tuple[tuple[str, str], ...]

    def accepts(self, url: str, data: str | None) -> bool:
        return url == self.url and tuple(sorted(parse_qsl(data or "", keep_blank_values=True))) == self.fields


class PageDriver:
    def __init__(
        self,
        browser: Browser,
        context: BrowserContext,
        page: Page,
        origin: Origin,
        evidence: Path,
        resources: AsyncExitStack,
        pid: int | None,
    ):
        self.browser = browser
        self.context = context
        self.page = page
        self.origin = origin
        self.evidence = evidence
        self._resources = resources
        self._pid = pid
        self.blocked: list[str] = []
        self.unresolved: str | None = None
        self._observation: Observation | None = None
        self._closed = False
        self._submission: Submission | None = None
        self._posted = False

    @classmethod
    async def start(
        cls, playwright: Playwright, origin: str, evidence: Path, channel: str = "chromium"
    ) -> PageDriver:
        parsed = Origin.parse(origin)
        evidence.mkdir(parents=True, exist_ok=True)
        async with AsyncExitStack() as resources:
            browser = await playwright.chromium.launch(
                headless=True, channel=None if channel == "chromium" else channel
            )
            resources.push_async_callback(browser.close)
            pid = await _browser_pid(browser)
            context = await browser.new_context(accept_downloads=False, service_workers="block")
            resources.push_async_callback(context.close)
            context.set_default_timeout(8000)
            page = await context.new_page()
            driver = cls(browser, context, page, parsed, evidence, resources, pid)
            await context.route("**/*", driver._route)
            await context.route_web_socket("**/*", lambda socket: socket.close())
            (evidence / "browser.json").write_text(json.dumps({"browser": channel}) + "\n")
            if pid is not None:
                (evidence / "browser.pid").write_text(str(pid))
            driver._resources = resources.pop_all()
            return driver

    async def open(self, url: str) -> DriverReceipt:
        if not url_allowed(url, self.origin):
            return self._reject("off_origin", url)
        try:
            response = await self.page.goto(url, wait_until="domcontentloaded", timeout=8000)
            if response is None or not response.ok or not url_allowed(self.page.url, self.origin):
                return self._reject("navigation_rejected", self.page.url)
            return await self._observed("opened")
        except Error as error:
            return self._reject("navigation_rejected", self.page.url, diagnostic=str(error))

    async def observe(self) -> DriverReceipt:
        if self.page.is_closed() or not url_allowed(self.page.url, self.origin):
            return self._reject("page_closed", self.page.url)
        return await self._observed("observed")

    async def query(self) -> DriverReceipt:
        opened = await self.open(self.origin.origin() + "/state")
        if not opened.ok:
            return opened
        identity = self.unresolved
        if identity is not None:
            # Only a matching backend receipt settles the action. Absence is not proof of non-execution.
            found = await self.page.locator("[data-order-id]").evaluate_all(
                '(nodes) => nodes.map(node => node.getAttribute("data-order-id"))'
            )
            if identity not in found:
                return self._unknown("effect_unverified", identity, opened.view)
            self.unresolved = None
        return DriverReceipt(
            ok=True,
            reason="queried",
            final_url=opened.final_url,
            observation=opened.observation,
            view=opened.view,
            reconciled_action=identity,
            blocked_urls=tuple(self.blocked),
        )

    async def act(
        self, observation: str, node_id: str, action: str, value: str | None = None
    ) -> DriverReceipt:
        if self.unresolved is not None:
            return self._reject("query_required", self.page.url)
        if action not in {"click", "fill", "submit"} or (action == "fill" and value is None):
            return self._reject("invalid_action", self.page.url)
        saved = self._observation
        if self.page.is_closed():
            return self._reject("page_closed", "")
        if (
            saved is None
            or saved.view.observation != observation
            or node_id not in saved.elements
            or not url_allowed(self.page.url, self.origin)
            or not await saved.matches(self.page)
        ):
            return self._reject("stale_observation", self.page.url)
        handle = saved.elements[node_id]
        if not await handle.is_enabled() or not await handle.is_visible():
            return self._reject("invalid_action", self.page.url)
        action_id = "act-" + uuid4().hex
        raw = await handle.evaluate(
            """(node, id) => {
            const form = node.form;
            if (!form || form.method.toLowerCase() !== 'post') return null;
            const fields = [...new FormData(form).entries()].map(([key, value]) => [key, String(value)]);
            return {url: form.action, fields: fields.map(([key, value]) => [key, key === 'action_id' ? id : value])};
        }""",
            action_id,
        )
        self._submission = (
            None
            if raw is None
            else Submission(
                str(raw["url"]), tuple(sorted((str(key), str(value)) for key, value in raw["fields"]))
            )
        )
        if not await saved.matches(self.page):
            return self._reject("stale_observation", self.page.url)
        self._posted = False
        self.unresolved = action_id if action != "fill" else None
        try:
            if action == "fill":
                await handle.fill(value or "", timeout=5000)
            else:
                await handle.evaluate(
                    """(node, id) => {
                    const form = node.form;
                    const field = form && form.querySelector('input[name="action_id"]');
                    if (field) field.value = id;
                }""",
                    action_id,
                )
                if self._submission is not None:
                    async with self.page.expect_navigation(wait_until="domcontentloaded", timeout=8000):
                        await handle.click(timeout=8000)
                else:
                    await handle.click(timeout=8000)
                    await self.page.wait_for_load_state("domcontentloaded", timeout=8000)
            if not url_allowed(self.page.url, self.origin):
                if action == "fill":
                    return self._reject("navigation_rejected", self.page.url)
                return self._unknown("navigation_rejected", action_id, saved.view)
            observed = await self._observed("applied", effect="applied", action_id=action_id)
            self.unresolved = None
            return observed
        except Error as error:
            if action == "fill":
                return self._reject("fill_failed", self.page.url, diagnostic=str(error))
            return self._unknown("response_lost", action_id, saved.view, str(error))
        except Exception as error:
            if action == "fill":
                return self._reject("fill_failed", self.page.url, diagnostic=str(error))
            return self._unknown("driver_error", action_id, saved.view, str(error))

    async def close(self) -> None:
        if self._closed:
            return
        try:
            await self._resources.aclose()
        finally:
            self._closed = True
        (self.evidence / "closed.json").write_text(json.dumps({"pid": self._pid, "closed": True}) + "\n")

    async def _route(self, route: Route) -> None:
        request = route.request
        # Popups and child frames cannot issue requests before they are closed.
        try:
            allowed_page = request.frame == self.page.main_frame
        except Error:
            allowed_page = False
        if allowed_page and url_allowed(request.url, self.origin):
            if request.method in {"GET", "HEAD"}:
                await route.continue_()
                return
            if (
                request.method == "POST"
                and self.unresolved is not None
                and not self._posted
                and self._submission is not None
                and self._submission.accepts(request.url, request.post_data)
            ):
                self._posted = True
                await route.continue_()
                return
        self.blocked.append(request.url)
        await route.abort()

    async def _observed(
        self, reason: str, *, effect: Effect = "none", action_id: str | None = None
    ) -> DriverReceipt:
        previous, self._observation = self._observation, None
        if previous is not None:
            await previous.close()
        self._observation = await Observation.capture(self.page)
        view = self._observation.view
        return DriverReceipt(
            ok=True,
            reason=reason,
            effect=effect,
            action_id=action_id,
            final_url=view.url,
            observation=view.observation,
            view=view,
            blocked_urls=tuple(self.blocked),
        )

    def _unknown(
        self, reason: str, identity: str, view: PageView | None, diagnostic: str | None = None
    ) -> DriverReceipt:
        return DriverReceipt(
            ok=False,
            reason=reason,
            effect="unknown",
            action_id=identity,
            final_url=self.page.url,
            observation=None if view is None else view.observation,
            view=view,
            blocked_urls=tuple(self.blocked),
            diagnostic=diagnostic,
        )

    def _reject(self, reason: str, url: str, *, diagnostic: str | None = None) -> DriverReceipt:
        return DriverReceipt(
            ok=False, reason=reason, final_url=url, blocked_urls=tuple(self.blocked), diagnostic=diagnostic
        )


async def _browser_pid(browser: Browser) -> int | None:
    client = await browser.new_browser_cdp_session()
    try:
        raw = await client.send("SystemInfo.getProcessInfo")
    finally:
        await client.detach()
    for item in raw.get("processInfo", []):
        if item.get("type") == "browser" and isinstance(item.get("id"), int):
            return int(item["id"])
    return None


async def launch(origin: str, evidence: Path, channel: str = "chromium") -> tuple[Playwright, PageDriver]:
    playwright = await async_playwright().start()
    try:
        driver = await PageDriver.start(playwright, origin, evidence, channel)
    except BaseException:
        await playwright.stop()
        raise
    return playwright, driver


def browser_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
