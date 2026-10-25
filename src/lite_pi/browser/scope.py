"""Origin checks for browser navigations and MCP results."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Origin:
    scheme: str
    host: str
    port: int

    @classmethod
    def parse(cls, value: str) -> Origin:
        parts = urlsplit(value)
        if (
            parts.scheme not in {"http", "https"}
            or parts.username is not None
            or parts.password is not None
            or parts.hostname is None
        ):
            raise ValueError(f"invalid_origin:{value}")
        if parts.path not in {"", "/"} or parts.query or parts.fragment:
            raise ValueError(f"invalid_origin:{value}")
        port = parts.port
        if port is None:
            port = 443 if parts.scheme == "https" else 80
        return cls(parts.scheme, parts.hostname, port)

    def origin(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{self.scheme}://{host}:{self.port}"


def url_allowed(value: str, origin: Origin) -> bool:
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        return False
    if parts.username is not None or parts.password is not None or parts.hostname is None:
        return False
    if port is None:
        port = 443 if parts.scheme == "https" else 80
    return parts.scheme == origin.scheme and parts.hostname == origin.host and port == origin.port


_REJECTION = {
    "off_origin",
    "navigation_rejected",
    "page_closed",
    "stale_observation",
    "query_required",
    "invalid_action",
    "response_lost",
    "driver_error",
    "effect_unverified",
}


def host_accepts(origins: tuple[str, ...], final_url: object, reason: object) -> bool:
    """A configured browser result must report a final URL inside the origin, or an explicit rejection."""
    if not origins:
        return True
    allowed = [Origin.parse(item) for item in origins]
    in_scope = isinstance(final_url, str) and any(url_allowed(final_url, item) for item in allowed)
    if in_scope:
        return reason != "off_origin"
    return reason in _REJECTION
