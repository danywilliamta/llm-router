"""
Couche LLM abstraite — point d'entrée unique pour tous les appels.

Le provider actif se choisit via la variable d'env LLM_PROVIDER :
    LLM_PROVIDER=anthropic   → Claude  (défaut)
    LLM_PROVIDER=openai      → GPT

Le modèle par défaut de chaque provider se configure via :
    ANTHROPIC_MODEL=claude-opus-4-6
    OPENAI_MODEL=gpt-4o

Usage :
    from llm_router import llm_complete

    response = await llm_complete(user_prompt="...", system_prompt="...")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Résolution du provider au démarrage (pas à chaque appel)
# ---------------------------------------------------------------------------

def _get_provider(override: str | None = None) -> str:
    provider = (override or os.getenv("LLM_PROVIDER", "anthropic")).lower().strip()
    if provider not in ("anthropic", "openai"):
        raise ValueError(
            f"LLM_PROVIDER='{provider}' invalide. Valeurs acceptées : anthropic, openai"
        )
    return provider


def _default_model(provider: str) -> str:
    """Retourne le modèle par défaut du provider donné."""
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o")
    return os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")


# ---------------------------------------------------------------------------
# Dataclass de réponse unifiée
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    cached: bool = False
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------

async def llm_complete(
    user_prompt: str,
    system_prompt: str = "",
    model: str | None = None,
    max_tokens: int = 4096,
    use_thinking: bool = True,
    cache_system: bool = True,
    history: list[dict] | None = None,
    provider_override: str | None = None,
    usage_context: dict | None = None,
    images: list[tuple[bytes, str]] | None = None,
    **extra_kwargs,
) -> LLMResponse:
    """
    Appel LLM async — dispatche vers le provider défini par LLM_PROVIDER.

    Args:
        user_prompt:      Requête / instruction.
        system_prompt:    Contexte système. Mis en cache si Anthropic.
        model:            Surcharge le modèle par défaut du provider.
        max_tokens:       Tokens max en sortie.
        use_thinking:     Adaptive thinking (Anthropic uniquement, ignoré ailleurs).
        cache_system:     Prompt caching sur le system_prompt (Anthropic uniquement).
        provider_override: Force un provider spécifique pour cet appel.
        usage_context:    Transmis tel quel au hook enregistré via `set_usage_hook`
                           (voir `llm_router.usage`) — ignoré si aucun hook n'est
                           configuré. Sert à identifier l'appelant (agent_id,
                           tenant_id, user_id...) pour l'accounting côté appelant.
        images:           Liste de (bytes, media_type) à joindre au message utilisateur
                           (vision). Anthropic uniquement pour l'instant — lève
                           `NotImplementedError` sous provider='openai' plutôt que de
                           silencieusement ignorer les images (voir openai_llm_client).
        **extra_kwargs:   Paramètres additionnels transmis au client.
    """
    provider = _get_provider(provider_override)
    resolved_model = model or _default_model(provider)

    if provider == "anthropic":
        from llm_router import anthropic_client
        text = await anthropic_client.complete(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model=resolved_model,
            max_tokens=max_tokens,
            use_thinking=use_thinking,
            cache_system=cache_system,
            history=history,
            usage_context=usage_context,
            images=images,
        )
        return LLMResponse(
            text=text,
            model=resolved_model,
            provider="anthropic",
            cached=cache_system and bool(system_prompt),
        )

    # openai — pas de support vision dans openai_llm_client aujourd'hui ; échoue
    # explicitement plutôt que de silencieusement laisser tomber les images
    # (même principe que llm_complete_structured sous provider='openai').
    if images:
        raise NotImplementedError(
            f"llm_complete(images=...) non implémenté pour provider='{provider}' "
            "(vision Anthropic uniquement pour l'instant)"
        )

    from llm_router import openai_llm_client
    text = await openai_llm_client.complete(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        model=resolved_model,
        max_tokens=max_tokens,
        history=history,
        usage_context=usage_context,
    )
    return LLMResponse(text=text, model=resolved_model, provider="openai")


async def llm_complete_structured(
    user_prompt: str,
    input_schema: dict,
    system_prompt: str = "",
    max_tokens: int = 4096,
    cache_system: bool = True,
    history: list[dict] | None = None,
    provider_override: str | None = None,
    usage_context: dict | None = None,
) -> dict:
    """Équivalent structuré de `llm_complete` : réponse JSON validée par schéma
    via les structured outputs natifs de l'API Anthropic (`output_config.format`)
    — voir `anthropic_client.complete_structured`. Anthropic uniquement pour
    l'instant (pas câblé côté OpenAI dans ce package) — échoue explicitement
    plutôt que de silencieusement retomber sur un comportement différent.

    Pas de paramètre `model` : `anthropic_client.complete_structured` fixe le
    modèle en dur (Haiku 4.5) — `output_config.format` n'est supporté que par
    une liste restreinte de modèles Anthropic, et Haiku 4.5 couvre tous les
    usages actuels de cette fonction.

    `usage_context` : voir `llm_complete` — transmis tel quel au hook de
    tracking optionnel (`llm_router.usage.set_usage_hook`).
    """
    provider = _get_provider(provider_override)

    if provider != "anthropic":
        raise NotImplementedError(
            f"llm_complete_structured non implémenté pour provider='{provider}' (Anthropic uniquement)"
        )

    from llm_router import anthropic_client
    return await anthropic_client.complete_structured(
        user_prompt=user_prompt,
        input_schema=input_schema,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        cache_system=cache_system,
        history=history,
        usage_context=usage_context,
    )
