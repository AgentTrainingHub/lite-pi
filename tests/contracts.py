"""Run deterministic contracts with their own evidence and no paid model calls."""

import argparse
import io
import json
import unittest
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


class ContractLoader(unittest.TestLoader):
    def loadTestsFromModule(self, module, *, pattern=None):
        # Imported base fixtures must not silently execute a second time.
        return self.suiteClass(
            self.loadTestsFromTestCase(item)
            for item in vars(module).values()
            if isinstance(item, type)
            and issubclass(item, unittest.TestCase)
            and item.__module__ == module.__name__
        )


class ContractResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocked: list[tuple[str, str]] = []

    def addError(self, test, err):
        from support import AcceptanceBlocked

        if isinstance(err[1], AcceptanceBlocked):
            self.blocked.append((test.id(), str(err[1])))
            self.addSkip(test, "environment blocked: " + str(err[1]))
        else:
            super().addError(test, err)


def run() -> dict:
    evidence = ROOT / ".artifacts/contracts" / uuid4().hex
    evidence.mkdir(parents=True)
    loader = ContractLoader()
    suite = loader.discover(str(ROOT / "tests"), pattern="test_*contracts.py")

    def identities(item):
        if isinstance(item, unittest.TestSuite):
            names = []
            for child in item:
                names.extend(identities(child))
            return names
        return [item.id()]

    test_ids = identities(suite)
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2, resultclass=ContractResult).run(suite)
    (evidence / "unittest.log").write_text(output.getvalue())
    summary = {
        "layer": "deterministic",
        "status": (
            "fail"
            if not result.wasSuccessful() or result.testsRun == 0
            else "blocked"
            if result.blocked
            else "pass"
        ),
        "tests_run": result.testsRun,
        "tests": test_ids,
        "failures": [test.id() for test, _ in result.failures],
        "errors": [test.id() for test, _ in result.errors],
        "blocked": result.blocked,
        "skipped": [(test.id(), reason) for test, reason in result.skipped],
        "evidence": str(evidence),
    }
    (evidence / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(output.getvalue(), end="")
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-blocked", action="store_true", help="允许 CI 无隔离环境；报告仍为 blocked")
    args = parser.parse_args()
    status = run()["status"]
    raise SystemExit(0 if status == "pass" or (args.allow_blocked and status == "blocked") else 1)
