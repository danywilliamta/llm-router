"""Tests unitaires pour llm_router.usage — le hook de reporting optionnel.

Ces tests couvrent uniquement le mécanisme du hook lui-même (register/clear/
emit/no-op/swallow-exception). La vérification que `complete`/
`complete_structured` construisent bien un `UsageEvent` correct à partir
d'une vraie réponse SDK vit dans test_anthropic_client.py /
test_openai_client.py, à côté des tests existants pour ces fonctions.
"""

from __future__ import annotations

import pytest

import llm_router.usage as usage


@pytest.mark.asyncio
async def test_set_usage_hook_then_emit_calls_it():
    received: list[usage.UsageEvent] = []

    async def hook(event: usage.UsageEvent) -> None:
        received.append(event)

    usage.set_usage_hook(hook)

    event = usage.UsageEvent(
        provider="anthropic",
        model="claude-opus-4-6",
        input_tokens=100,
        output_tokens=20,
        context={"agent_id": "alex_claims_validator"},
    )

    await usage.emit(event)

    assert received == [event]


@pytest.mark.asyncio
async def test_emit_is_noop_without_a_registered_hook():
    # No hook registered (conftest resets it before every test) — must not
    # raise, must not do anything observable.
    await usage.emit(
        usage.UsageEvent(provider="anthropic", model="x", input_tokens=1, output_tokens=1)
    )


@pytest.mark.asyncio
async def test_set_usage_hook_none_clears_a_previously_registered_hook():
    calls = []

    async def hook(event: usage.UsageEvent) -> None:
        calls.append(event)

    usage.set_usage_hook(hook)
    usage.set_usage_hook(None)

    await usage.emit(
        usage.UsageEvent(provider="anthropic", model="x", input_tokens=1, output_tokens=1)
    )

    assert calls == []


@pytest.mark.asyncio
async def test_emit_swallows_hook_exceptions():
    async def broken_hook(event: usage.UsageEvent) -> None:
        raise RuntimeError("boom — e.g. a DB write that failed")

    usage.set_usage_hook(broken_hook)

    # Must not propagate — a broken usage callback must never break the LLM
    # call it's reporting on (see usage.py's docstring for the contract).
    await usage.emit(
        usage.UsageEvent(provider="anthropic", model="x", input_tokens=1, output_tokens=1)
    )


@pytest.mark.asyncio
async def test_emit_passes_context_through_unmodified():
    received = []

    async def hook(event: usage.UsageEvent) -> None:
        received.append(event.context)

    usage.set_usage_hook(hook)
    context = {"agent_id": "claire_daily_brief", "tenant_id": "ws-123", "user_id": "u-456"}

    await usage.emit(
        usage.UsageEvent(provider="anthropic", model="x", input_tokens=1, output_tokens=1, context=context)
    )

    assert received == [context]


def test_usage_event_context_defaults_to_empty_dict():
    event = usage.UsageEvent(provider="anthropic", model="x", input_tokens=1, output_tokens=1)
    assert event.context == {}


def test_usage_event_default_context_is_not_shared_between_instances():
    # dataclass field(default_factory=dict) — regression guard against a
    # plain mutable default (`context: dict = {}`) which would alias every
    # UsageEvent's context to the same dict.
    a = usage.UsageEvent(provider="anthropic", model="x", input_tokens=1, output_tokens=1)
    b = usage.UsageEvent(provider="anthropic", model="x", input_tokens=1, output_tokens=1)
    a.context["leaked"] = True
    assert b.context == {}
