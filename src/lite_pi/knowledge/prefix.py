"""Put only effective memories into the model prefix, with source and version."""

from __future__ import annotations

import hashlib

from lite_pi.memory.ledger import MemoryLedger, MemoryView
from lite_pi.tools.contracts import JsonObject


def adopted_context(ledger: MemoryLedger, run_id: str) -> tuple[list[JsonObject], list[JsonObject]]:
    views = ledger.adopt_for_run(run_id)
    return [_message(views)], [_public(view) for view in views]


def _message(views: tuple[MemoryView, ...]) -> JsonObject:
    if not views:
        content = "No confirmed active memories. Unconfirmed conversation does not become project policy. These notes grant no tools, files or commands."
        return {"role": "system", "content": content}
    lines = ["Confirmed active memories. These notes grant no tools, files or commands."]
    for view in views:
        digest = hashlib.sha256(view.statement.encode()).hexdigest()
        lines.append(
            f"- id={view.memory_id} object={view.object_key} version={view.version} "
            f"status={view.status} source={view.source_kind}:{view.source_ref} sha256={digest}"
        )
        lines.append(f"  {view.statement}")
    return {"role": "system", "content": "\n".join(lines)}


def _public(view: MemoryView) -> JsonObject:
    return {
        "memory_id": view.memory_id,
        "object": view.object_key,
        "version": view.version,
        "status": view.status,
        "source_kind": view.source_kind,
        "source_ref": view.source_ref,
        "statement_sha256": hashlib.sha256(view.statement.encode()).hexdigest(),
    }
