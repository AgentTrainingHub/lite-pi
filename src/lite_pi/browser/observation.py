"""Host-issued observations bind DOM content and actual element identities."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from playwright.async_api import ElementHandle, Page


@dataclass(frozen=True)
class PageNode:
    node_id: str
    name: str


@dataclass(frozen=True)
class PageView:
    observation: str
    url: str
    title: str
    text: str
    nodes: tuple[PageNode, ...]


class Observation:
    def __init__(self, view: PageView, html: str, elements: dict[str, ElementHandle]) -> None:
        self.view = view
        self.html = html
        self.elements = elements

    @classmethod
    async def capture(cls, page: Page) -> Observation:
        elements: dict[str, ElementHandle] = {}
        nodes: list[PageNode] = []
        before = await _fingerprint(page)
        handles = await page.query_selector_all("[data-node]")
        try:
            for handle in handles:
                identity = await handle.get_attribute("data-node")
                if not identity or identity in elements:
                    raise ValueError("missing_or_duplicate_node_identity")
                elements[identity] = handle
                nodes.append(PageNode(identity, (await handle.inner_text()).strip()))
            view = PageView(
                "obs-" + uuid4().hex,
                page.url,
                await page.title(),
                await page.locator("body").inner_text(),
                tuple(nodes),
            )
            captured = cls(view, before, elements)
            if not await captured.matches(page):
                raise ValueError("observation_changed_during_capture")
            return captured
        except BaseException:
            for handle in handles:
                await handle.dispose()
            raise

    async def matches(self, page: Page) -> bool:
        if page.url != self.view.url or await _fingerprint(page) != self.html:
            return False
        for handle in self.elements.values():
            if not await handle.evaluate("(node) => node.isConnected"):
                return False
        return True

    async def close(self) -> None:
        for handle in self.elements.values():
            await handle.dispose()


async def _fingerprint(page: Page) -> str:
    return str(
        await page.evaluate("""() => JSON.stringify({
        html: document.documentElement.outerHTML,
        fields: [...document.querySelectorAll('input,select,textarea')].map(node => ({
            value: node.value, checked: node.checked, disabled: node.disabled
        }))
    })""")
    )
