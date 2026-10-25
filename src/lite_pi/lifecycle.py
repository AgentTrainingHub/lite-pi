"""Close every owned resource without replacing the failure that caused unwinding."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class CleanupFailure:
    resource: str
    error: BaseException

    def diagnostic(self) -> str:
        return f"{self.resource}_cleanup_failed:{type(self.error).__name__}: {self.error}"


def close_resources(resources: list[tuple[str, Callable[[], object]]]) -> list[CleanupFailure]:
    """Run callbacks in list order; every callback runs even after an earlier failure."""
    failures: list[CleanupFailure] = []
    for name, callback in resources:
        try:
            callback()
        except BaseException as error:
            failures.append(CleanupFailure(name, error))
    return failures
