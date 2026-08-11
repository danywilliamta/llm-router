"""
Client OpenAI async pour les appels LLM (GPT-4o etc.).
Supporte les conversations multi-tours via le paramètre `history`.
"""

from __future__ import annotations

import logging
import os

import openai

from llm_router.usage import UsageEvent, emit

logger = logging.getLogger(__name__)

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

    if response.usage is not None:
        await emit(UsageEvent(
            provider="openai",
            model=model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            context=usage_context or {},
        ))

    return response.choices[0].message.content or ""
