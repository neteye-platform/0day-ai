"""Shared ChatOpenAI instances. `import settings` loads .env before any key read."""

from langchain_openai import ChatOpenAI

import settings

_KEYS = dict(
    base_url=settings.llm_base_url,
    model=settings.llm_model,
    api_key=settings.llm_api_key,
    stream_usage=True,
    max_completion_tokens=settings.llm_max_completion_tokens,
)

fast_llm = ChatOpenAI(temperature=0.2, reasoning_effort="none", **_KEYS)
smart_llm = ChatOpenAI(temperature=0.8, reasoning_effort="medium", **_KEYS)
reviewer_llm = ChatOpenAI(temperature=0.8, reasoning_effort="low", **_KEYS)
