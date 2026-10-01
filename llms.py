"""Shared ChatOpenAI instances. `import settings` loads .env before any key read."""

import logging

from langchain_openai import ChatOpenAI
from openai import LengthFinishReasonError

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


def invoke_structured_capped(llm, messages, description: str):
    """Structured-output invoke that survives the completion-token cap.

    Hub-sized prompts can push the model into a degeneration that burns the
    whole `llm_max_completion_tokens` budget and dies mid-JSON
    (LengthFinishReasonError). Re-roll once; on a second cap hit log an ERROR
    and return None so the caller skips its unit of work instead of exhausting
    LangGraph's task retries and crashing the whole run.
    """
    for attempt in (1, 2):
        try:
            return llm.invoke(messages)
        except LengthFinishReasonError:
            if attempt == 2:
                logging.error(
                    f"{description}: LLM hit the {settings.llm_max_completion_tokens}-token "
                    "output cap twice; skipping this call to keep the run alive."
                )
                return None
            logging.warning(
                f"{description}: output cap reached, retrying the request once."
            )
