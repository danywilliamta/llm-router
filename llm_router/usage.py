"""
Optional usage-reporting hook.

llm_router is a lightweight LLM proxy with no storage dependency of its own
(see pyproject.toml — anthropic/openai/httpx only). Callers that want
token/cost tracking register a hook here; this module has no opinion on
where usage data ends up (Postgres, logs, metrics...) — that's the hook's
job, not this package's.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class UsageEvent:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    # Caller-supplied free-form tag (e.g. {"agent_id": ..., "tenant_id": ...,
    # "user_id": ...}) — llm_router doesn't interpret this, just forwards it.
    context: dict[str, Any] = field(default_factory=dict)


UsageHook = Callable[[UsageEvent], Awaitable[None]]

_hook: Optional[UsageHook] = None


def set_usage_hook(hook: Optional[UsageHook]) -> None:
    """Register (or clear, with `None`) the callback invoked after every
    completed LLM call with its token usage. A hook exception is logged and
    swallowed — a broken usage callback must never break the LLM call it's
    reporting on.
    """
    global _hook
    _hook = hook


async def emit(event: UsageEvent) -> None:
    if _hook is None:
        return
    try:
        await _hook(event)
    except Exception:
        logger.exception("usage hook raised for %s/%s", event.provider, event.model)
