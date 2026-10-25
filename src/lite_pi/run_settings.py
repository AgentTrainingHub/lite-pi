"""Run settings and admission validation, shared by all public session entries."""

import math
from collections.abc import Callable
from dataclasses import dataclass

from lite_pi.compaction import MODEL_CONTEXT_TOKENS
from lite_pi.tools.contracts import JsonObject


@dataclass(frozen=True)
class RunSettings:
    max_turns: int = 6
    max_tokens: int = 512
    max_bytes: int = 50 * 1024
    shell_timeout: float = 10
    cancel_bash_after: float | None = None
    max_model_calls: int | None = None
    interrupt_after_chunks: int | None = None
    on_event: Callable[[JsonObject], None] | None = None
    context_tokens: int | None = None
    approvals: bool = False
    max_task_calls: int | None = None
    max_task_output_tokens: int | None = None
    max_task_seconds: float | None = None
    approval_ttl_seconds: float | None = None
    max_retrievals: int | None = None

    def validate(self) -> None:
        for name in ("shell_timeout", "cancel_bash_after", "max_task_seconds", "approval_ttl_seconds"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if min(self.max_turns, self.max_tokens, self.max_bytes, self.shell_timeout) <= 0:
            raise ValueError("run limits must be positive")
        if self.max_model_calls is not None and self.max_model_calls < 1:
            raise ValueError("max_model_calls must be positive")
        if self.max_task_calls is not None and self.max_task_calls < 1:
            raise ValueError("max_task_calls must be positive")
        if self.max_task_output_tokens is not None and self.max_task_output_tokens < 1:
            raise ValueError("max_task_output_tokens must be positive")
        if self.max_task_seconds is not None and self.max_task_seconds <= 0:
            raise ValueError("max_task_seconds must be positive")
        if self.approval_ttl_seconds is not None and self.approval_ttl_seconds < 0:
            raise ValueError("approval_ttl_seconds must not be negative")
        if self.max_retrievals is not None and self.max_retrievals < 1:
            raise ValueError("max_retrievals must be positive")
        if self.cancel_bash_after is not None and self.cancel_bash_after <= 0:
            raise ValueError("cancel_bash_after must be positive")
        if self.context_tokens is not None and self.context_tokens <= self.max_tokens:
            raise ValueError("context_tokens must exceed the output reserve")
        if self.context_tokens is not None and self.context_tokens > MODEL_CONTEXT_TOKENS:
            raise ValueError("context_tokens exceeds the model window")
