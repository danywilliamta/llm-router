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
