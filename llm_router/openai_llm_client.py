"""
Client OpenAI async pour les appels LLM (GPT-4o etc.).
Supporte les conversations multi-tours via le paramètre `history`.
"""

from __future__ import annotations

import json
import logging
import os

import openai

from llm_router.usage import UsageEvent, emit

logger = logging.getLogger(__name__)

# Modèle par défaut pour complete_structured (response_format json_schema,
# strict=True) — contrairement à Anthropic (output_config.format restreint à
# une liste fixe de modèles, voir anthropic_client.STRUCTURED_MODEL), le mode
# strict JSON schema d'OpenAI est supporté par toute la famille gpt-4o à
# partir des snapshots datés 2024-07-18/2024-08-06, donc pas besoin de le
# figer en dur : env var pour changer le défaut sans toucher au code, +
# paramètre `model` sur complete_structured() pour un override par appel.
STRUCTURED_MODEL = os.getenv("OPENAI_STRUCTURED_MODEL", "gpt-4o-mini")

_client: openai.AsyncOpenAI | None = None


def get_client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY manquant dans l'environnement")
        _client = openai.AsyncOpenAI(api_key=api_key)
    return _client


async def complete(
    user_prompt: str,
    system_prompt: str = "",
    model: str = "gpt-4o",
    max_tokens: int = 4096,
    history: list[dict] | None = None,
    usage_context: dict | None = None,
    **_kwargs,
) -> str:
    client = get_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_prompt})

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )

    await _report_usage(model=model, response=response, usage_context=usage_context)

    return response.choices[0].message.content or ""


async def complete_structured(
    user_prompt: str,
    input_schema: dict,
    system_prompt: str = "",
    model: str | None = None,
    max_tokens: int = 4096,
    history: list[dict] | None = None,
    usage_context: dict | None = None,
) -> dict:
    """Force une réponse JSON validée par schéma via `response_format`
    (`{"type": "json_schema", "json_schema": {"name", "schema", "strict": True}}`)
    — le mode structured outputs strict de l'API Chat Completions OpenAI.
    Miroir de `anthropic_client.complete_structured` (même contrat : un dict
    déjà conforme à `input_schema`, jamais de nettoyage markdown-fence côté
    appelant) mais, contrairement à lui, expose `model` — le mode strict
    d'OpenAI n'est pas limité à un seul modèle (voir `STRUCTURED_MODEL`
    ci-dessus pour le défaut et comment le changer sans toucher au code).

    `input_schema` doit respecter les contraintes du mode strict OpenAI :
    `additionalProperties: false` à chaque niveau objet, et chaque propriété
    listée dans `required` (pas de notion de champ optionnel en mode strict —
    un champ non pertinent doit être un type nullable, ex. `["string", "null"]`,
    plutôt qu'omis de `required`).
    """
    client = get_client()
    resolved_model = model or STRUCTURED_MODEL

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_prompt})

    response = await client.chat.completions.create(
        model=resolved_model,
        messages=messages,
        max_tokens=max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": input_schema, "strict": True},
        },
    )

    await _report_usage(model=resolved_model, response=response, usage_context=usage_context)

    message = response.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"complete_structured: le modèle a refusé la requête : {message.refusal}")
    if not message.content:
        raise RuntimeError("complete_structured: réponse vide (response_format json_schema)")
    return json.loads(message.content)


async def _report_usage(*, model: str, response, usage_context: dict | None) -> None:
    """See `anthropic_client._report_usage` — same fail-soft contract:
    extracting usage must never break the completion it's reporting on."""
    try:
        usage = response.usage
        if usage is None:
            return
        await emit(UsageEvent(
            provider="openai",
            model=model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            context=usage_context or {},
        ))
    except Exception:
        logger.exception("failed to extract/report usage for model=%s", model)
