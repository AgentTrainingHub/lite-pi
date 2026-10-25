"""Required deterministic tool layer; results stay distinct from live runs."""

import contracts
from support import Suite


def run(suite: Suite) -> None:
    result = contracts.run()
    suite.record(
        "tool-contracts",
        result["status"] == "pass",
        layer="deterministic",
        tests_run=result["tests_run"],
        evidence=result["evidence"],
    )
