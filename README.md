# llm-router

Lightweight async LLM proxy for Python. One entry point, dispatches between
Anthropic and OpenAI behind a unified response type — no per-call branching
on which provider you're using.

Extracted from a production multi-tenant SaaS ([Shiftaura](https://github.com/danywilliamta/marketing-agency-ia)),
where it sits alongside [agent-harness](https://github.com/danywilliamta/Agentic-OS)
(agent-harness owns multi-turn agent conversations via LangGraph; llm-router
handles one-off completions outside that — structured extraction, daily
briefs, headless analysis).

## Why not LiteLLM

LiteLLM covers 100+ providers generically. This package covers two, on
purpose, and encodes production lessons instead of hiding them:

- **Fails loud, never silently degrades.** `llm_complete_structured` (forced
  tool-call output) raises `NotImplementedError` on providers that don't
  support it, rather than falling back to unstructured text you'd have to
  notice was different.
- **Anthropic prompt caching wired in.** `cache_system=True` marks the system
  prompt `ephemeral` automatically.
- **A specific httpx transport fix baked into the Anthropic client**
  (`max_keepalive_connections=0` + an explicit SDK-level timeout) — without
  it, a connection silently closed by a network intermediary during a long
  tool call gets reused, the next request goes into a void, and nothing ever
  times out at any layer. Cost ~100-200ms of TLS handshake per call to
  eliminate a whole class of silent hangs.

If you need more providers, use LiteLLM. If you need these two providers to
behave predictably and explicitly, this is smaller and easier to reason
about.

## Usage

```python
from llm_router import llm_complete, llm_complete_structured

response = await llm_complete(
    user_prompt="...",
    system_prompt="...",
)
print(response.text, response.provider, response.model)

# Forced structured output (Anthropic only, raises on other providers)
result = await llm_complete_structured(
    user_prompt="...",
    tool_name="extract_fields",
    input_schema={...},
)
```

## Configuration (env vars)

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` \| `openai` |
| `ANTHROPIC_API_KEY` | — | required if provider is `anthropic` |
| `ANTHROPIC_MODEL` | `claude-opus-4-6` | |
| `OPENAI_API_KEY` | — | required if provider is `openai` |
| `OPENAI_MODEL` | `gpt-4o` | |

Any call can override the provider (`provider_override=`) or model
(`model=`) explicitly, independent of the env defaults.

## Install

```
pip install "llm-router @ git+https://github.com/danywilliamta/llm-router.git"
```

Pin to a commit in production, same pattern as `agent-harness`:

```
llm-router @ git+https://github.com/danywilliamta/llm-router.git@<commit>
```

## License

MIT
