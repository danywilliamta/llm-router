"""
Client Anthropic async avec prompt caching sur le system prompt.
Utilise claude-opus-4-6 par défaut avec adaptive thinking et streaming.
Supporte les conversations multi-tours via le paramètre `history`.
"""

from __future__ import annotations

import logging
import os

import anthropic
import httpx

logger = logging.getLogger(__name__)

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


def _build_kwargs(
    user_prompt: str,
    system_prompt: str,
    model: str,
    max_tokens: int,
    use_thinking: bool,
    cache_system: bool,
    history: list[dict] | None = None,
) -> dict:
    system_content: list[dict] = []
    if system_prompt:
        block: dict = {"type": "text", "text": system_prompt}
        if cache_system:
            block["cache_control"] = {"type": "ephemeral"}
        system_content.append(block)

    # Historique + nouveau message utilisateur
    messages = list(history or []) + [{"role": "user", "content": user_prompt}]

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
) -> str:
    client = get_client()
    kwargs = _build_kwargs(user_prompt, system_prompt, model, max_tokens, use_thinking, cache_system, history)

    async with client.messages.stream(**kwargs) as stream:
        message = await stream.get_final_message()

    parts = [
        block.text
        for block in message.content
        if hasattr(block, "text") and block.type == "text"
    ]
    return "\n".join(parts)


async def complete_structured(
    user_prompt: str,
    tool_name: str,
    input_schema: dict,
    system_prompt: str = "",
    model: str = "claude-opus-4-6",
    max_tokens: int = 4096,
    use_thinking: bool = False,
    cache_system: bool = True,
    history: list[dict] | None = None,
) -> dict:
    """Force Claude à répondre via UN appel à `tool_name` (tool_choice forcé +
    strict: true) — `input_schema` est validé côté serveur, `tool_call.input`
    est déjà un dict Python parsé par le SDK. Jamais de JSON à parser/nettoyer
    côté appelant (contrairement à `complete`, qui retourne du texte libre).

    `input_schema` doit avoir `additionalProperties: false` (exigence du
    strict tool use — voir la doc Anthropic sur les structured outputs).
    """
    if use_thinking:
        raise ValueError(
            "complete_structured: use_thinking incompatible avec tool_choice forcé "
            "(l'API Anthropic refuse la combinaison, 400) — laisse use_thinking=False."
        )
    client = get_client()
    kwargs = _build_kwargs(user_prompt, system_prompt, model, max_tokens, use_thinking, cache_system, history)
    kwargs["tools"] = [{
        "name": tool_name,
        "description": f"Soumets le résultat structuré pour {tool_name}.",
        "input_schema": input_schema,
        "strict": True,
    }]
    kwargs["tool_choice"] = {"type": "tool", "name": tool_name}

    async with client.messages.stream(**kwargs) as stream:
        message = await stream.get_final_message()

    tool_call = next(
        (block for block in message.content if block.type == "tool_use" and block.name == tool_name),
        None,
    )
    if tool_call is None:
        raise RuntimeError(f"Claude n'a pas appelé le tool '{tool_name}' malgré tool_choice forcé")
    return tool_call.input
