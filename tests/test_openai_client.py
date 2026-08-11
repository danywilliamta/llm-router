"""Tests unitaires pour llm_router.openai_llm_client.

`openai.AsyncOpenAI` n'est jamais réellement instancié pour `complete` : on
injecte un faux client dont `.chat.completions.create` est un AsyncMock.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import llm_router.openai_llm_client as openai_client
import llm_router.usage as usage


def _fake_response(content: str | None, usage_obj=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=usage_obj,
    )


def _fake_client(content: str | None = "ok", usage_obj=None):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_fake_response(content, usage_obj))
    return client


# ---------------------------------------------------------------------------
# get_client
# ---------------------------------------------------------------------------

def test_get_client_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        openai_client.get_client()


def test_get_client_returns_singleton(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    first = openai_client.get_client()
    second = openai_client.get_client()
    assert first is second


# ---------------------------------------------------------------------------
# complete
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_returns_message_content():
    openai_client._client = _fake_client("bonjour")

    result = await openai_client.complete(user_prompt="salut")

    assert result == "bonjour"


@pytest.mark.asyncio
async def test_complete_returns_empty_string_when_content_is_none():
    openai_client._client = _fake_client(None)

    result = await openai_client.complete(user_prompt="salut")

    assert result == ""


@pytest.mark.asyncio
async def test_complete_without_system_prompt_or_history():
    client = _fake_client()
    openai_client._client = client

    await openai_client.complete(user_prompt="salut")

    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["messages"] == [{"role": "user", "content": "salut"}]


@pytest.mark.asyncio
async def test_complete_prepends_system_prompt_and_history():
    client = _fake_client()
    openai_client._client = client
    history = [
        {"role": "user", "content": "premier"},
        {"role": "assistant", "content": "reponse"},
    ]

    await openai_client.complete(
        user_prompt="deuxieme",
        system_prompt="tu es un assistant",
        history=history,
        model="gpt-x",
        max_tokens=77,
    )

    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["model"] == "gpt-x"
    assert kwargs["max_tokens"] == 77
    assert kwargs["messages"] == [
        {"role": "system", "content": "tu es un assistant"},
        {"role": "user", "content": "premier"},
        {"role": "assistant", "content": "reponse"},
        {"role": "user", "content": "deuxieme"},
    ]


@pytest.mark.asyncio
async def test_complete_uses_default_model_and_max_tokens():
    client = _fake_client()
    openai_client._client = client

    await openai_client.complete(user_prompt="salut")

    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["max_tokens"] == 4096


# ---------------------------------------------------------------------------
# usage hook reporting
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_reports_usage_to_registered_hook():
    usage_obj = SimpleNamespace(prompt_tokens=80, completion_tokens=15)
    openai_client._client = _fake_client("bonjour", usage_obj=usage_obj)

    received: list[usage.UsageEvent] = []

    async def hook(event: usage.UsageEvent) -> None:
        received.append(event)

    usage.set_usage_hook(hook)

    await openai_client.complete(
        user_prompt="salut",
        model="gpt-4o-mini",
        usage_context={"agent_id": "brand_brain_importer", "tenant_id": "ws-1"},
    )

    assert len(received) == 1
    event = received[0]
    assert event.provider == "openai"
    assert event.model == "gpt-4o-mini"
    assert event.input_tokens == 80
    assert event.output_tokens == 15
    assert event.context == {"agent_id": "brand_brain_importer", "tenant_id": "ws-1"}


@pytest.mark.asyncio
async def test_complete_does_not_call_hook_when_usage_is_none():
    # OpenAI response with usage=None (e.g. streaming without usage tracking
    # enabled) — must not raise, must simply skip reporting.
    openai_client._client = _fake_client("bonjour", usage_obj=None)

    received: list[usage.UsageEvent] = []

    async def hook(event: usage.UsageEvent) -> None:
        received.append(event)

    usage.set_usage_hook(hook)

    result = await openai_client.complete(user_prompt="salut")

    assert result == "bonjour"
    assert received == []


@pytest.mark.asyncio
async def test_complete_does_not_call_hook_when_none_registered():
    usage_obj = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    openai_client._client = _fake_client("bonjour", usage_obj=usage_obj)

    result = await openai_client.complete(user_prompt="salut")

    assert result == "bonjour"


@pytest.mark.asyncio
async def test_complete_survives_a_hook_that_raises():
    usage_obj = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    openai_client._client = _fake_client("bonjour", usage_obj=usage_obj)

    async def broken_hook(event: usage.UsageEvent) -> None:
        raise RuntimeError("simulated failure")

    usage.set_usage_hook(broken_hook)

    result = await openai_client.complete(user_prompt="salut")

    assert result == "bonjour"
