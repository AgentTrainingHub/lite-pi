"""Layered acceptance; required cases must pass in their declared test layer."""

import argparse
import importlib

from support import AcceptanceBlocked, Suite

GROUPS = {
    "regressions": ["regressions", "container"],
    "container": ["container"],
    "session": ["session"],
    "tools": ["tool_contracts", "tools", "tool_recovery"],
    "unknown-tool-probe": ["unknown_probe"],
    "tool-recovery": ["tool_recovery"],
    "workspace": ["coding", "files", "shell"],
    "coding": ["coding"],
    "journal": ["journal"],
    "interaction": ["interaction"],
    "context": ["context", "context_boundaries"],
    "context-boundaries": ["context_boundaries"],
    "extensions": ["extensions"],
    "recovery": ["recovery", "recovery_boundaries", "concurrency"],
    "concurrency": ["concurrency"],
    "recovery-boundaries": ["recovery_boundaries"],
    "knowledge": ["knowledge"],
    "browser": ["browser"],
    "delegation": ["delegation"],
    "comparison": ["comparison"],
    "terminal": ["terminal"],
    "terminal-reading": ["terminal_reading"],
    "identity": ["identity"],
    "planning": ["planning"],
    "hybrid-retrieval": ["hybrid_retrieval"],
    "edit": ["files"],
    "shell": ["shell"],
    "all": [
        "tool_contracts",
        "session",
        "tools",
        "tool_recovery",
        "coding",
        "files",
        "shell",
        "journal",
        "interaction",
        "context",
        "context_boundaries",
        "extensions",
        "recovery",
        "recovery_boundaries",
        "concurrency",
        "knowledge",
        "browser",
        "delegation",
        "comparison",
        "terminal",
        "terminal_reading",
        "identity",
        "planning",
        "hybrid_retrieval",
        "regressions",
        "container",
    ],
}


def main(group: str | None = None) -> int:
    if group is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("group", choices=GROUPS)
        group = parser.parse_args().group
    suite = Suite(group)
    try:
        for name in GROUPS[group]:
            importlib.import_module("cases_" + name).run(suite)
    except AcceptanceBlocked as error:
        suite.record("runtime", False, blocked=True, reason=str(error))
    return suite.finish()


if __name__ == "__main__":
    raise SystemExit(main())
