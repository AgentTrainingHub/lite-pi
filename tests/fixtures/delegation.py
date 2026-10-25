"""Controlled child outcomes and calls for delegation tests."""

from threading import Lock

from fixtures.responses import tool_call

from lite_pi.delegation_report import ChildReport


def scripted(items: list[dict]):
    pending = list(items)
    lock = Lock()

    def read(*_args, **_kwargs):
        with lock:
            if not pending:
                raise AssertionError("unexpected model call")
            return pending.pop(0)

    return read


def delegate_results(result: dict) -> list[dict]:
    return [
        item for turn in result["transcript"] for item in turn["results"] if item.get("name") == "delegate"
    ]


def delegate_call(task: str, paths: list[str], identity: str) -> dict:
    return delegate_many([(task, paths)], identity)


def delegate_many(surveys: list[tuple[str, list[str]]], identity: str) -> dict:
    return tool_call(
        "delegate",
        {"surveys": [{"task": task, "paths": paths} for task, paths in surveys]},
        identity,
    )


def child_report(generation: int) -> ChildReport:
    return ChildReport(
        survey_id=f"{generation}:0",
        generation=generation,
        status="complete",
        reason=None,
        narrative="done",
        run_id=None,
        parent_run_id="parent",
        paths=("src/lookup.py",),
        tools=("read",),
        refusals=(),
        unsupported=(),
        completion_claim="none",
        model_calls=0,
        started=0.0,
        ended=1.0,
        started_at="start",
        ended_at="end",
        kept=False,
    )
