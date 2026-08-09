"""Tests unitaires pour llm_router.router — résolution du provider/modèle et
dispatch de llm_complete / llm_complete_structured vers le bon client.

Les clients réels (anthropic_client / openai_llm_client) sont mockés : ces tests
vérifient le comportement de routage, pas les appels réseau.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import llm_router.anthropic_client as anthropic_client
import llm_router.openai_llm_client as openai_client
import llm_router.router as router_mod
from llm_router import LLMResponse


# ---------------------------------------------------------------------------
# _get_provider
# ---------------------------------------------------------------------------

def test_get_provider_defaults_to_anthropic():
    assert router_mod._get_provider() == "anthropic"


def test_get_provider_reads_env_var(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert router_mod._get_provider() == "openai"


def test_get_provider_is_case_and_whitespace_insensitive(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "  OpenAI  ")
    assert router_mod._get_provider() == "openai"


def test_get_provider_override_takes_precedence_over_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert router_mod._get_provider("anthropic") == "anthropic"


def test_get_provider_invalid_value_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mistral")
    with pytest.raises(ValueError, match="invalide"):
        router_mod._get_provider()


def test_get_provider_invalid_override_raises():
    with pytest.raises(ValueError):
        router_mod._get_provider("bedrock")


# ---------------------------------------------------------------------------
# _default_model
# ---------------------------------------------------------------------------

def test_default_model_anthropic_fallback():
    assert router_mod._default_model("anthropic") == "claude-opus-4-6"


def test_default_model_anthropic_env_override(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    assert router_mod._default_model("anthropic") == "claude-sonnet-5"


def test_default_model_openai_fallback():
    assert router_mod._default_model("openai") == "gpt-4o"


def test_default_model_openai_env_override(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    assert router_mod._default_model("openai") == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# llm_complete — dispatch vers Anthropic
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_complete_dispatches_to_anthropic(monkeypatch):
    mock_complete = AsyncMock(return_value="reponse claude")
    monkeypatch.setattr(anthropic_client, "complete", mock_complete)

    response = await router_mod.llm_complete(
        user_prompt="bonjour",
        system_prompt="sois concis",
        provider_override="anthropic",
    )

    assert isinstance(response, LLMResponse)
    assert response.text == "reponse claude"
    assert response.provider == "anthropic"
    assert response.model == "claude-opus-4-6"
    assert response.cached is True  # cache_system=True par défaut + system_prompt non vide

    mock_complete.assert_awaited_once()
    _, kwargs = mock_complete.call_args
    assert kwargs["user_prompt"] == "bonjour"
    assert kwargs["system_prompt"] == "sois concis"
    assert kwargs["model"] == "claude-opus-4-6"


@pytest.mark.asyncio
async def test_llm_complete_cached_false_when_no_system_prompt(monkeypatch):
    monkeypatch.setattr(anthropic_client, "complete", AsyncMock(return_value="x"))

    response = await router_mod.llm_complete(
        user_prompt="hi", provider_override="anthropic"
    )

    assert response.cached is False


@pytest.mark.asyncio
async def test_llm_complete_cached_false_when_cache_system_disabled(monkeypatch):
    monkeypatch.setattr(anthropic_client, "complete", AsyncMock(return_value="x"))

    response = await router_mod.llm_complete(
        user_prompt="hi",
        system_prompt="contexte",
        cache_system=False,
        provider_override="anthropic",
    )

    assert response.cached is False


@pytest.mark.asyncio
async def test_llm_complete_model_override_bypasses_default(monkeypatch):
    mock_complete = AsyncMock(return_value="x")
    monkeypatch.setattr(anthropic_client, "complete", mock_complete)
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-should-not-be-used")

    response = await router_mod.llm_complete(
        user_prompt="hi", model="claude-custom", provider_override="anthropic"
    )

    assert response.model == "claude-custom"
    _, kwargs = mock_complete.call_args
    assert kwargs["model"] == "claude-custom"


@pytest.mark.asyncio
async def test_llm_complete_forwards_thinking_and_history(monkeypatch):
    mock_complete = AsyncMock(return_value="x")
    monkeypatch.setattr(anthropic_client, "complete", mock_complete)
    history = [{"role": "user", "content": "prev"}]

    await router_mod.llm_complete(
        user_prompt="hi",
        use_thinking=False,
        history=history,
        provider_override="anthropic",
    )

    _, kwargs = mock_complete.call_args
    assert kwargs["use_thinking"] is False
    assert kwargs["history"] == history


# ---------------------------------------------------------------------------
# llm_complete — dispatch vers OpenAI
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_complete_dispatches_to_openai(monkeypatch):
    mock_complete = AsyncMock(return_value="reponse gpt")
    monkeypatch.setattr(openai_client, "complete", mock_complete)

    response = await router_mod.llm_complete(
        user_prompt="hi", provider_override="openai"
    )

    assert response.provider == "openai"
    assert response.text == "reponse gpt"
    assert response.model == "gpt-4o"
    # LLMResponse.cached est laissé à sa valeur par défaut (False) pour OpenAI
    assert response.cached is False

    mock_complete.assert_awaited_once()
    _, kwargs = mock_complete.call_args
    assert kwargs["user_prompt"] == "hi"
    assert kwargs["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_llm_complete_respects_env_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    mock_complete = AsyncMock(return_value="x")
    monkeypatch.setattr(openai_client, "complete", mock_complete)

    response = await router_mod.llm_complete(user_prompt="hi")

    assert response.provider == "openai"


# ---------------------------------------------------------------------------
# llm_complete_structured
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_complete_structured_dispatches_to_anthropic(monkeypatch):
    mock_structured = AsyncMock(return_value={"nom": "Alice"})
    monkeypatch.setattr(anthropic_client, "complete_structured", mock_structured)

    schema = {"type": "object", "properties": {"nom": {"type": "string"}}}
    result = await router_mod.llm_complete_structured(
        user_prompt="extrait le nom",
        input_schema=schema,
        provider_override="anthropic",
    )

    assert result == {"nom": "Alice"}
    mock_structured.assert_awaited_once()
    _, kwargs = mock_structured.call_args
    assert kwargs["input_schema"] == schema
    # Pas de "model" transmis — anthropic_client.complete_structured fixe le
    # modèle en dur (Haiku 4.5), llm_complete_structured n'a pas ce paramètre.
    assert "model" not in kwargs


@pytest.mark.asyncio
async def test_llm_complete_structured_openai_not_implemented():
    with pytest.raises(NotImplementedError):
        await router_mod.llm_complete_structured(
            user_prompt="x",
            input_schema={},
            provider_override="openai",
        )


@pytest.mark.asyncio
async def test_llm_complete_structured_env_provider_openai_not_implemented(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(NotImplementedError):
        await router_mod.llm_complete_structured(user_prompt="x", input_schema={})
