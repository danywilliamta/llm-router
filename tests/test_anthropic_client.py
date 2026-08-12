"""Tests unitaires pour llm_router.anthropic_client.

Le client `anthropic.AsyncAnthropic` n'est jamais réellement instancié pour les
tests de `complete`/`complete_structured` : on injecte un faux client dont
`.messages.stream(...)` retourne un context manager async minimal simulant le
SDK (`get_final_message`). Les tests de `get_client` vérifient uniquement la
gestion de la clé d'API et le comportement singleton, sans appel réseau.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import llm_router.anthropic_client as anthropic_client
import llm_router.usage as usage


class FakeStream:
    """Simule le context manager retourné par `client.messages.stream(...)`."""

    def __init__(self, message):
        self._message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc_info):
        return False

    async def get_final_message(self):
        return self._message


def _fake_client_with_message(message, capture: dict | None = None):
    client = MagicMock()

    def _stream(**kwargs):
        if capture is not None:
            capture.update(kwargs)
        return FakeStream(message)

    client.messages.stream = _stream
    return client


# ---------------------------------------------------------------------------
# get_client
# ---------------------------------------------------------------------------

def test_get_client_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        anthropic_client.get_client()


def test_get_client_returns_singleton(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    first = anthropic_client.get_client()
    second = anthropic_client.get_client()
    assert first is second


# ---------------------------------------------------------------------------
# _build_kwargs
# ---------------------------------------------------------------------------

def test_build_kwargs_without_system_prompt():
    kwargs = anthropic_client._build_kwargs(
        user_prompt="hi",
        system_prompt="",
        model="claude-x",
        max_tokens=100,
        use_thinking=False,
        cache_system=True,
    )
    assert "system" not in kwargs
    assert "thinking" not in kwargs
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert kwargs["model"] == "claude-x"
    assert kwargs["max_tokens"] == 100


def test_build_kwargs_system_prompt_with_caching():
    kwargs = anthropic_client._build_kwargs(
        user_prompt="hi",
        system_prompt="tu es un assistant",
        model="claude-x",
        max_tokens=100,
        use_thinking=False,
        cache_system=True,
    )
    assert kwargs["system"] == [
        {
            "type": "text",
            "text": "tu es un assistant",
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_build_kwargs_system_prompt_without_caching():
    kwargs = anthropic_client._build_kwargs(
        user_prompt="hi",
        system_prompt="tu es un assistant",
        model="claude-x",
        max_tokens=100,
        use_thinking=False,
        cache_system=False,
    )
    assert kwargs["system"] == [{"type": "text", "text": "tu es un assistant"}]


def test_build_kwargs_thinking_enabled():
    kwargs = anthropic_client._build_kwargs(
        user_prompt="hi",
        system_prompt="",
        model="claude-x",
        max_tokens=100,
        use_thinking=True,
        cache_system=True,
    )
    assert kwargs["thinking"] == {"type": "adaptive"}


def test_build_kwargs_prepends_history():
    history = [
        {"role": "user", "content": "premier message"},
        {"role": "assistant", "content": "premiere reponse"},
    ]
    kwargs = anthropic_client._build_kwargs(
        user_prompt="deuxieme message",
        system_prompt="",
        model="claude-x",
        max_tokens=100,
        use_thinking=False,
        cache_system=True,
        history=history,
    )
    assert kwargs["messages"] == history + [
        {"role": "user", "content": "deuxieme message"}
    ]


def test_build_kwargs_does_not_mutate_history_argument():
    history = [{"role": "user", "content": "premier"}]
    anthropic_client._build_kwargs(
        user_prompt="deuxieme",
        system_prompt="",
        model="claude-x",
        max_tokens=100,
        use_thinking=False,
        cache_system=True,
        history=history,
    )
    assert history == [{"role": "user", "content": "premier"}]


def test_build_kwargs_without_images_keeps_plain_string_content():
    kwargs = anthropic_client._build_kwargs(
        user_prompt="hi",
        system_prompt="",
        model="claude-x",
        max_tokens=100,
        use_thinking=False,
        cache_system=True,
        images=None,
    )
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_build_kwargs_with_images_wraps_content_in_blocks():
    kwargs = anthropic_client._build_kwargs(
        user_prompt="decris cette image",
        system_prompt="",
        model="claude-x",
        max_tokens=100,
        use_thinking=False,
        cache_system=True,
        images=[(b"fake-bytes", "image/png")],
    )
    content = kwargs["messages"][0]["content"]
    assert content == [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "ZmFrZS1ieXRlcw==",  # base64("fake-bytes")
            },
        },
        {"type": "text", "text": "decris cette image"},
    ]


def test_build_kwargs_with_multiple_images_preserves_order():
    kwargs = anthropic_client._build_kwargs(
        user_prompt="compare",
        system_prompt="",
        model="claude-x",
        max_tokens=100,
        use_thinking=False,
        cache_system=True,
        images=[(b"one", "image/png"), (b"two", "image/jpeg")],
    )
    content = kwargs["messages"][0]["content"]
    assert [b["source"]["media_type"] for b in content[:2]] == ["image/png", "image/jpeg"]
    assert content[2] == {"type": "text", "text": "compare"}


def test_build_kwargs_images_go_in_same_message_as_history():
    # Les rôles doivent alterner strictement côté API Anthropic — les images
    # doivent rejoindre le nouveau message utilisateur, jamais former un
    # message "user" séparé qui suivrait un message "user" de l'historique.
    history = [
        {"role": "user", "content": "premier message"},
        {"role": "assistant", "content": "premiere reponse"},
    ]
    kwargs = anthropic_client._build_kwargs(
        user_prompt="deuxieme message",
        system_prompt="",
        model="claude-x",
        max_tokens=100,
        use_thinking=False,
        cache_system=True,
        history=history,
        images=[(b"img", "image/png")],
    )
    assert len(kwargs["messages"]) == 3
    assert kwargs["messages"][:2] == history
    assert kwargs["messages"][2]["role"] == "user"
    assert isinstance(kwargs["messages"][2]["content"], list)


# ---------------------------------------------------------------------------
# complete
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_joins_text_blocks_and_filters_others():
    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking"),  # pas de .text -> filtre par hasattr
            SimpleNamespace(type="text", text="Bonjour"),
            SimpleNamespace(type="tool_use", text="ignore-moi", name="x", input={}),
            SimpleNamespace(type="text", text="le monde"),
        ]
    )
    anthropic_client._client = _fake_client_with_message(message)

    result = await anthropic_client.complete(user_prompt="salut")

    assert result == "Bonjour\nle monde"


@pytest.mark.asyncio
async def test_complete_returns_empty_string_when_no_text_blocks():
    message = SimpleNamespace(content=[SimpleNamespace(type="thinking")])
    anthropic_client._client = _fake_client_with_message(message)

    result = await anthropic_client.complete(user_prompt="salut")

    assert result == ""


@pytest.mark.asyncio
async def test_complete_forwards_build_kwargs_output_to_stream():
    message = SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])
    captured: dict = {}
    anthropic_client._client = _fake_client_with_message(message, captured)

    await anthropic_client.complete(
        user_prompt="salut",
        system_prompt="ctx",
        model="claude-y",
        max_tokens=42,
        use_thinking=True,
        cache_system=True,
    )

    assert captured["model"] == "claude-y"
    assert captured["max_tokens"] == 42
    assert captured["thinking"] == {"type": "adaptive"}
    assert captured["system"][0]["text"] == "ctx"
    assert captured["messages"] == [{"role": "user", "content": "salut"}]


@pytest.mark.asyncio
async def test_complete_forwards_images_to_stream():
    message = SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])
    captured: dict = {}
    anthropic_client._client = _fake_client_with_message(message, captured)

    await anthropic_client.complete(
        user_prompt="decris cette image",
        images=[(b"fake-bytes", "image/png")],
    )

    content = captured["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[-1] == {"type": "text", "text": "decris cette image"}


# ---------------------------------------------------------------------------
# complete_structured
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_structured_parses_json_text_block():
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text='{"nom": "Alice"}')]
    )
    anthropic_client._client = _fake_client_with_message(message)

    result = await anthropic_client.complete_structured(
        user_prompt="extrait le nom",
        input_schema={"type": "object", "additionalProperties": False},
    )

    assert result == {"nom": "Alice"}


@pytest.mark.asyncio
async def test_complete_structured_forces_model_and_output_config():
    message = SimpleNamespace(content=[SimpleNamespace(type="text", text="{}")])
    captured: dict = {}
    anthropic_client._client = _fake_client_with_message(message, captured)

    schema = {"type": "object", "additionalProperties": False}
    await anthropic_client.complete_structured(user_prompt="x", input_schema=schema)

    # Le modèle est fixé en dur (Haiku 4.5) — jamais le défaut claude-opus-4-6
    # de complete(), et pas un paramètre exposé à l'appelant.
    assert captured["model"] == anthropic_client.STRUCTURED_MODEL
    assert captured["output_config"] == {"format": {"type": "json_schema", "schema": schema}}
    assert "tools" not in captured
    assert "tool_choice" not in captured
    assert "thinking" not in captured


@pytest.mark.asyncio
async def test_complete_structured_raises_when_no_text_block_returned():
    message = SimpleNamespace(content=[SimpleNamespace(type="thinking")])
    anthropic_client._client = _fake_client_with_message(message)

    with pytest.raises(RuntimeError, match="output_config.format"):
        await anthropic_client.complete_structured(user_prompt="x", input_schema={})


# ---------------------------------------------------------------------------
# usage hook reporting
# ---------------------------------------------------------------------------

def _fake_message_with_usage(text: str, input_tokens: int, output_tokens: int):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


@pytest.mark.asyncio
async def test_complete_reports_usage_to_registered_hook():
    message = _fake_message_with_usage("ok", input_tokens=123, output_tokens=45)
    anthropic_client._client = _fake_client_with_message(message)

    received: list[usage.UsageEvent] = []

    async def hook(event: usage.UsageEvent) -> None:
        received.append(event)

    usage.set_usage_hook(hook)

    await anthropic_client.complete(
        user_prompt="salut",
        model="claude-opus-4-6",
        usage_context={"agent_id": "alex_claims_validator", "tenant_id": "ws-1"},
    )

    assert len(received) == 1
    event = received[0]
    assert event.provider == "anthropic"
    assert event.model == "claude-opus-4-6"
    assert event.input_tokens == 123
    assert event.output_tokens == 45
    assert event.context == {"agent_id": "alex_claims_validator", "tenant_id": "ws-1"}


@pytest.mark.asyncio
async def test_complete_reports_usage_with_empty_context_by_default():
    message = _fake_message_with_usage("ok", input_tokens=1, output_tokens=1)
    anthropic_client._client = _fake_client_with_message(message)

    received: list[usage.UsageEvent] = []

    async def hook(event: usage.UsageEvent) -> None:
        received.append(event)

    usage.set_usage_hook(hook)

    await anthropic_client.complete(user_prompt="salut")

    assert received[0].context == {}


@pytest.mark.asyncio
async def test_complete_structured_reports_usage_with_structured_model():
    # complete_structured has no `model` param — the reported model must be
    # STRUCTURED_MODEL (Haiku 4.5), never the caller's default/override.
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="{}")],
        usage=SimpleNamespace(input_tokens=10, output_tokens=2),
    )
    anthropic_client._client = _fake_client_with_message(message)

    received: list[usage.UsageEvent] = []

    async def hook(event: usage.UsageEvent) -> None:
        received.append(event)

    usage.set_usage_hook(hook)

    await anthropic_client.complete_structured(
        user_prompt="x",
        input_schema={"type": "object", "additionalProperties": False},
        usage_context={"agent_id": "alex_brand_voice"},
    )

    assert len(received) == 1
    assert received[0].model == anthropic_client.STRUCTURED_MODEL
    assert received[0].input_tokens == 10
    assert received[0].output_tokens == 2
    assert received[0].context == {"agent_id": "alex_brand_voice"}


@pytest.mark.asyncio
async def test_complete_does_not_call_hook_when_none_registered():
    # No hook registered (conftest resets between tests) — complete() must
    # still return normally, usage reporting is silently skipped.
    message = _fake_message_with_usage("ok", input_tokens=1, output_tokens=1)
    anthropic_client._client = _fake_client_with_message(message)

    result = await anthropic_client.complete(user_prompt="salut")

    assert result == "ok"


@pytest.mark.asyncio
async def test_complete_survives_message_without_usage_attribute():
    # Regression guard: older/mocked responses lacking `.usage` entirely (as
    # every fake message elsewhere in this file does) must not break
    # `complete()` even with a hook registered — extraction failure is
    # swallowed, not propagated. See anthropic_client._report_usage.
    message = SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])
    anthropic_client._client = _fake_client_with_message(message)

    hook_calls = []

    async def hook(event: usage.UsageEvent) -> None:
        hook_calls.append(event)

    usage.set_usage_hook(hook)

    result = await anthropic_client.complete(user_prompt="salut")

    assert result == "ok"
    assert hook_calls == []


@pytest.mark.asyncio
async def test_complete_survives_a_hook_that_raises():
    message = _fake_message_with_usage("ok", input_tokens=1, output_tokens=1)
    anthropic_client._client = _fake_client_with_message(message)

    async def broken_hook(event: usage.UsageEvent) -> None:
        raise RuntimeError("simulated DB failure")

    usage.set_usage_hook(broken_hook)

    result = await anthropic_client.complete(user_prompt="salut")

    assert result == "ok"
