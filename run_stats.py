"""Run-local progress/statistics ledgers and small shared pipeline helpers.

Counters feed the report's "Pipeline Statistics" section; reset at bootstrap so
one process can run the pipeline repeatedly. All counters are deterministic and
never derived from LLM output. Thread-safe (fan-out nodes run concurrently).
"""

import logging
import re
import threading
import uuid

import settings
from dedup import Embeddings

_agent_progress: dict[str, dict[str, int]] = {}
_agent_progress_lock = threading.Lock()

_pipeline_stats: dict[str, int] = {}
_pipeline_stats_lock = threading.Lock()


def _record_stat(key: str, amount: int = 1) -> None:
    with _pipeline_stats_lock:
        _pipeline_stats[key] = _pipeline_stats.get(key, 0) + (amount or 0)


def _reset_pipeline_stats() -> None:
    with _pipeline_stats_lock:
        _pipeline_stats.clear()


def _snapshot_pipeline_stats() -> dict[str, int]:
    with _pipeline_stats_lock:
        return dict(_pipeline_stats)


def _start_agent_progress(total: int) -> str:
    """Register a fan-out progress ledger; returns the id ("" when total == 0)."""
    progress_id = uuid.uuid4().hex
    if total:
        with _agent_progress_lock:
            _agent_progress[progress_id] = {"total": total, "completed": 0}
    return progress_id


def _log_agent_completion(progress_id: str, agent_name: str, detail: str) -> None:
    """Advance one progress ledger and log the completion line; id may be empty."""
    if not progress_id:
        return

    with _agent_progress_lock:
        progress = _agent_progress.get(progress_id)
        if progress is None:
            return
        progress["completed"] += 1
        completed = progress["completed"]
        total = progress["total"]
        remaining = total - completed
        if remaining == 0:
            _agent_progress.pop(progress_id, None)

    logging.info(
        "%s progress: %d/%d complete, %d remaining (%s).",
        agent_name,
        completed,
        total,
        remaining,
        detail,
    )


def as_dict(record) -> dict:
    """Normalize a graph-state record (dict or pydantic model) to a dict."""
    return record if isinstance(record, dict) else record.model_dump()


def as_dicts(records) -> list[dict]:
    return [as_dict(r) for r in records or []]


def get_embedder(gate: bool, warn_prefix: str, exact_mode_note: str):
    """Embeddings client when `gate` is on and reachable, else None (warns)."""
    if not gate:
        return None
    embedder = Embeddings(
        settings.embeddings_base_url,
        settings.embeddings_model,
        settings.embeddings_timeout,
    )
    if embedder.available():
        return embedder
    logging.warning(
        "%s: embeddings unavailable (Ollama idle or model %r not pulled?); using %s.",
        warn_prefix,
        settings.embeddings_model,
        exact_mode_note,
    )
    return None


_STEP_NUM_RE = re.compile(r'^\s*\d+[\.\)]\s+')


def strip_step_numbering(step) -> str:
    """Drop any leading 'N.' / 'N)' the reviewer embedded, so steps never double-number."""
    return _STEP_NUM_RE.sub('', str(step))
