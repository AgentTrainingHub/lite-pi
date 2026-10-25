"""Process crash injection at persisted transitions, confined to test workers."""

import os
import runpy
from functools import wraps


def install() -> None:
    point = os.environ.get("LITE_PI_CRASH_AFTER")
    if point is None:
        return
    if point not in {"intent", "effect", "result"}:
        raise ValueError("unknown crash point")
    from lite_pi.tasks import TaskBook
    from lite_pi.tools.registry import ToolRegistry

    def after(owner, name, matches=lambda *args, **kwargs: True):
        original = getattr(owner, name)

        @wraps(original)
        def invoke(*args, **kwargs):
            result = original(*args, **kwargs)
            if matches(*args, **kwargs):
                os._exit(86)
            return result

        setattr(owner, name, invoke)

    if point == "intent":
        after(TaskBook, "_insert_intent", lambda *args, **kw: kw.get("status") == "intent")
        after(TaskBook, "_mark", lambda self, identity, status, *args: status == "intent")
    elif point == "effect":
        after(ToolRegistry, "complete")
    else:
        after(TaskBook, "_save_result")


if __name__ == "__main__":
    install()
    runpy.run_module("lite_pi", run_name="__main__")
