"""Per-agent ChatOpenAI factory (get_llm). `import settings` loads .env before any key read."""

import logging

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from openai import BadRequestError, LengthFinishReasonError

import settings
from run_stats import new_usage, record_usage

# Marker inside the gateway's hard input-limit rejection ("This model's maximum
# context length is ... tokens"). Unlike the output cap, re-rolling cannot help:
# every retry resends the identical oversized prompt.
_CONTEXT_LENGTH_MARKER = "maximum context length"

_INSTANCES: dict[str, ChatOpenAI] = {}

# Registry keys consumed by the pipeline itself, not ChatOpenAI kwargs.
_CONFIG_ONLY_KEYS = ("context_window",)


def get_config(agent: str) -> dict:
    """Effective per-agent LLM config: settings.llm_defaults merged with
    settings.llm_overrides[agent] (model/base_url/api_key/temperature/
    reasoning_effort/max_completion_tokens/context_window all overridable per
    agent). Re-read per call, so edits are seen live by config consumers
    (compaction caps, cache fingerprints) — but get_llm builds its client once
    per key, so registry edits must happen before that agent's first get_llm
    (a long-lived `langgraph dev` server needs a restart)."""
    cfg = dict(settings.llm_defaults)
    cfg.update(settings.llm_overrides.get(agent, {}))
    return cfg


def get_llm(agent: str) -> ChatOpenAI:
    """ChatOpenAI for one pipeline agent, built once per key and reused from
    the get_config merge (config-only keys like context_window excluded)."""
    llm = _INSTANCES.get(agent)
    if llm is None:
        cfg = {
            k: v
            for k, v in get_config(agent).items()
            if k not in _CONFIG_ONLY_KEYS
        }
        llm = ChatOpenAI(stream_usage=True, **cfg)
        _INSTANCES[agent] = llm
    return llm


class UsageCapture(BaseCallbackHandler):
    """Per-invoke token-usage sink. The chat model fires on_llm_end with the
    raw AIMessage before any output parser runs, so this captures usage even
    for `with_structured_output` chains whose parsed result carries none.
    Failed attempts fire on_llm_error instead and contribute nothing."""

    def __init__(self):
        super().__init__()
        self.usage = new_usage()

    def on_llm_end(self, response, **kwargs):
        for generation_list in getattr(response, "generations", []) or []:
            for generation in generation_list:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if usage:
                    self.usage["calls"] += 1
                    self.usage["input_tokens"] += int(usage.get("input_tokens") or 0)
                    self.usage["output_tokens"] += int(usage.get("output_tokens") or 0)


def invoke_tracked(llm, messages, agent: str):
    """Invoke `llm` (plain or structured chain), capturing token usage through
    a per-invoke callback. Records the usage under `agent` in the run ledger
    and returns ``(result, usage)`` — `result` is exactly what the chain's
    own invoke would return (None is never synthesized here; the caller's
    exceptions propagate untouched)."""
    capture = UsageCapture()
    result = llm.invoke(messages, config={"callbacks": [capture]})
    record_usage(agent, capture.usage)
    return result, capture.usage


def invoke_structured_capped(llm, messages, description: str, agent: str = ""):
    """Structured-output invoke that survives the completion-token cap.

    Hub-sized prompts can push the model into a degeneration that burns the
    whole `max_completion_tokens` budget and dies mid-JSON
    (LengthFinishReasonError). Re-roll once; on a second cap hit log an ERROR
    and return (None, usage) so the caller skips its unit of work instead of
    exhausting LangGraph's task retries and crashing the whole run. A
    context-window 400 (input over the model limit) is deterministic: no retry
    is attempted — log an ERROR and return (None, usage) for the same
    skip-the-unit behavior. Any other BadRequestError propagates.

    Returns ``(result, usage)``: per-invoke token usage ({calls,
    input_tokens, output_tokens}) accumulated across all successfully-answered
    attempts (failed attempts fire on_llm_error and count nothing). When
    `agent` is set the usage is added to the run token ledger here; callers
    that also CACHE the result pass the returned usage to their cache write so
    the entry records what it cost to produce.
    """
    capture = UsageCapture()
    try:
        for attempt in (1, 2):
            try:
                return llm.invoke(messages, config={"callbacks": [capture]}), capture.usage
            except BadRequestError as exc:
                if _CONTEXT_LENGTH_MARKER not in str(exc):
                    raise
                logging.error(
                    f"{description}: prompt exceeds the model context window; "
                    "skipping this call to keep the run alive."
                )
                return None, capture.usage
            except LengthFinishReasonError:
                if attempt == 2:
                    logging.error(
                        f"{description}: LLM hit its completion-token output cap twice; "
                        "skipping this call to keep the run alive."
                    )
                    return None, capture.usage
                logging.warning(
                    f"{description}: output cap reached, retrying the request once."
                )
    finally:
        if agent:
            record_usage(agent, capture.usage)
