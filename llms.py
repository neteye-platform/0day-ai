"""Shared ChatOpenAI instances. `import settings` loads .env before any key read."""

import logging

from langchain_openai import ChatOpenAI
from openai import BadRequestError, LengthFinishReasonError

import settings

# Marker inside the gateway's hard input-limit rejection ("This model's maximum
# context length is ... tokens"). Unlike the output cap, re-rolling cannot help:
# every retry resends the identical oversized prompt.
_CONTEXT_LENGTH_MARKER = "maximum context length"

_KEYS = dict(
    base_url=settings.llm_base_url,
    model=settings.llm_model,
    api_key=settings.llm_api_key,
    stream_usage=True,
    max_completion_tokens=settings.llm_max_completion_tokens,
)

fast_llm = ChatOpenAI(temperature=0.2, reasoning_effort="none", **_KEYS)
smart_llm = ChatOpenAI(temperature=0.8, reasoning_effort="medium", **_KEYS)
reviewer_llm = ChatOpenAI(temperature=0.8, reasoning_effort="medium", **_KEYS)
# The validator's per-turn output is tool calls, not reasoning; the sandbox is
# the oracle, so extended reasoning is pure latency on every loop turn.
validator_llm = ChatOpenAI(temperature=0.8, reasoning_effort="low", **_KEYS)


def invoke_structured_capped(llm, messages, description: str):
    """Structured-output invoke that survives the completion-token cap.

    Hub-sized prompts can push the model into a degeneration that burns the
    whole `llm_max_completion_tokens` budget and dies mid-JSON
    (LengthFinishReasonError). Re-roll once; on a second cap hit log an ERROR
    and return None so the caller skips its unit of work instead of exhausting
    LangGraph's task retries and crashing the whole run. A context-window 400
    (input over the model limit) is deterministic: no retry is attempted —
    log an ERROR and return None for the same skip-the-unit behavior. Any other
    BadRequestError propagates.
    """
    for attempt in (1, 2):
        try:
            return llm.invoke(messages)
        except BadRequestError as exc:
            if _CONTEXT_LENGTH_MARKER not in str(exc):
                raise
            logging.error(
                f"{description}: prompt exceeds the model context window; "
                "skipping this call to keep the run alive."
            )
            return None
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
