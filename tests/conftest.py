"""Fixtures partagées : reset des singletons clients + des env vars pertinentes
avant chaque test, pour que les tests restent indépendants de l'ordre d'exécution
et de l'environnement de la machine qui les lance."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_client_singletons(monkeypatch):
    import llm_router.anthropic_client as anthropic_client
    import llm_router.openai_llm_client as openai_client

    monkeypatch.setattr(anthropic_client, "_client", None)
    monkeypatch.setattr(openai_client, "_client", None)
    yield


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "LLM_PROVIDER",
        "ANTHROPIC_MODEL",
        "OPENAI_MODEL",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    yield
