"""
Client Anthropic async avec prompt caching sur le system prompt.
Utilise claude-opus-4-6 par défaut avec adaptive thinking et streaming.
Supporte les conversations multi-tours via le paramètre `history`.
"""

from __future__ import annotations

import base64
import json
import logging
import os

import anthropic
import httpx

from llm_router.usage import UsageEvent, emit

logger = logging.getLogger(__name__)

# output_config.format (structured outputs natifs) n'est supporté que par une
# liste restreinte de modèles Anthropic (Fable 5, Opus 5, Opus 4.8, Sonnet 5,
# Haiku 4.5, Opus 4.5/4.1 legacy) — voir la doc Anthropic sur les structured
# outputs. Fixé en dur plutôt qu'exposé en paramètre : tous les appelants de
# `complete_structured` dans ce package font du scoring/classification léger,
# jamais de génération créative, donc Haiku 4.5 convient systématiquement —
# pas besoin de rendre ce choix configurable.
STRUCTURED_MODEL = "claude-haiku-4-5"

_client: anthropic.AsyncAnthropic | None = None


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY manquant dans l'environnement")
        # timeout explicite : sans ça, une connexion coincée dans le pool httpx
        # sous-jacent (ex : suite à l'annulation d'une requête précédente via
        # asyncio.wait_for/GeneratorExit) peut bloquer indéfiniment un nouvel
        # appel qui attend d'obtenir une connexion — aucune limite ne le rattrape.
        # Ce timeout donne une deadline dure au niveau du SDK/transport lui-même,
        # au lieu de dépendre uniquement de l'annulation asyncio côté appelant.
        #
        # max_keepalive_connections=0 : désactive la réutilisation de connexions
        # TCP/TLS entre deux appels. Sur certains réseaux (VPN/NAT), une connexion
        # gardée "keep-alive" par httpx peut être fermée silencieusement côté
        # intermédiaire réseau pendant le délai d'exécution d'un outil sans
        # FIN/RST — le prochain appel la réutilise alors en pensant qu'elle est
        # vivante, envoie sa requête dans le vide, et plus rien ne revient jamais
        # (ni erreur ni timeout, à aucune couche). Forcer une connexion fraîche à
        # chaque appel coûte ~100-200ms de handshake TLS mais élimine cette classe
        # de blocage silencieux.
        _client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=100.0,
            max_retries=1,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=0, max_connections=100),
            ),
        )
    return _client


async def _report_usage(*, model: str, message, usage_context: dict | None) -> None:
    """Extract token usage from a completed `Message` and forward it to the
    usage hook, if one is registered. Deliberately fails soft — a malformed
    or missing `.usage` (e.g. an older/mocked SDK response) must never break
    the completion this usage was for. See `llm_router.usage` for the hook
    contract itself, which has the same swallow-and-log guarantee for the
    hook's own body; this covers the extraction step ahead of it."""
    try:
        usage = message.usage
        await emit(UsageEvent(
            provider="anthropic",
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            context=usage_context or {},
        ))
    except Exception:
        logger.exception("failed to extract/report usage for model=%s", model)


def _build_kwargs(
    user_prompt: str,
    system_prompt: str,
    model: str,
    max_tokens: int,
    use_thinking: bool,
    cache_system: bool,
    history: list[dict] | None = None,
    images: list[tuple[bytes, str]] | None = None,
) -> dict:
    system_content: list[dict] = []
    if system_prompt:
        block: dict = {"type": "text", "text": system_prompt}
        if cache_system:
            block["cache_control"] = {"type": "ephemeral"}
        system_content.append(block)

    # `images` : liste de (bytes, media_type), injectées en blocs "image" avant
    # le texte dans le même message utilisateur — jamais un message séparé (les
    # rôles doivent alterner strictement côté API Anthropic). Content reste une
    # simple str quand `images` est vide/None, comportement identique à avant
    # l'ajout de ce paramètre (aucun appelant existant n'est affecté).
    if images:
        user_content: str | list[dict] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.b64encode(data).decode("utf-8"),
                },
            }
            for data, media_type in images
        ] + [{"type": "text", "text": user_prompt}]
    else:
        user_content = user_prompt

    # Historique + nouveau message utilisateur
    messages = list(history or []) + [{"role": "user", "content": user_content}]

    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system_content:
        kwargs["system"] = system_content
    if use_thinking:
        kwargs["thinking"] = {"type": "adaptive"}
    return kwargs


async def complete(
    user_prompt: str,
    system_prompt: str = "",
    model: str = "claude-opus-4-6",
    max_tokens: int = 4096,
    use_thinking: bool = True,
    cache_system: bool = True,
    history: list[dict] | None = None,
    usage_context: dict | None = None,
    images: list[tuple[bytes, str]] | None = None,
) -> str:
    client = get_client()
    kwargs = _build_kwargs(user_prompt, system_prompt, model, max_tokens, use_thinking, cache_system, history, images)

    async with client.messages.stream(**kwargs) as stream:
        message = await stream.get_final_message()

    await _report_usage(model=model, message=message, usage_context=usage_context)

    parts = [
        block.text
        for block in message.content
        if hasattr(block, "text") and block.type == "text"
    ]
    return "\n".join(parts)


async def complete_structured(
    user_prompt: str,
    input_schema: dict,
    system_prompt: str = "",
    max_tokens: int = 4096,
    cache_system: bool = True,
    history: list[dict] | None = None,
    usage_context: dict | None = None,
) -> dict:
    """Force une réponse JSON validée par schéma via `output_config.format` —
    les structured outputs natifs de l'API Anthropic. Ni tool "bidon" ni
    `tool_choice` forcé : le premier bloc texte de la réponse est déjà du JSON
    garanti conforme à `input_schema`, jamais de nettoyage markdown-fence côté
    appelant (contrairement à `complete`, qui retourne du texte libre).

    `input_schema` doit avoir `additionalProperties: false` (même exigence que
    l'ancien strict tool use — voir la doc Anthropic sur les structured
    outputs). Modèle fixé à `STRUCTURED_MODEL` (Haiku 4.5) — voir sa docstring.
    """
    client = get_client()
    kwargs = _build_kwargs(user_prompt, system_prompt, STRUCTURED_MODEL, max_tokens, False, cache_system, history)
    kwargs["output_config"] = {"format": {"type": "json_schema", "schema": input_schema}}

    async with client.messages.stream(**kwargs) as stream:
        message = await stream.get_final_message()

    await _report_usage(model=STRUCTURED_MODEL, message=message, usage_context=usage_context)

    text_block = next((block for block in message.content if block.type == "text"), None)
    if text_block is None:
        raise RuntimeError("complete_structured: aucun bloc texte dans la réponse (output_config.format)")
    return json.loads(text_block.text)
