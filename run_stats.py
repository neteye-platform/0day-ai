"""Run-local progress/statistics ledgers and small shared pipeline helpers.

Counters feed the report's "Pipeline Statistics" section; reset at bootstrap so
one process can run the pipeline repeatedly. All counters are deterministic and
never derived from LLM output. Thread-safe (fan-out nodes run concurrently).

The progress ledgers are mirrored on disk (states/progress/) so a `langgraph
dev` hot reload mid-fan-out — which restarts the worker into fresh module state
while the run resumes from checkpoints carrying the dispatch's progress id —
keeps logging completion lines instead of silently dropping every remaining
one.
"""

import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path

import settings
from dedup import Embeddings

_agent_progress: dict[str, dict[str, int]] = {}
_agent_progress_lock = threading.Lock()
_PROGRESS_DIR = Path("states") / "progress"

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


def _progress_file(progress_id: str) -> Path:
    return _PROGRESS_DIR / f"{progress_id}.json"


def _start_agent_progress(total: int) -> str:
    """Register a fan-out progress ledger; returns the id ("" when total == 0)."""
    progress_id = uuid.uuid4().hex
    if total:
        with _agent_progress_lock:
            _agent_progress[progress_id] = {"total": total, "completed": 0}
        try:
            _PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
            for stale in _PROGRESS_DIR.glob("*.json"):
                try:
                    if time.time() - stale.stat().st_mtime > 86400:
                        stale.unlink()
                except OSError:
                    pass
            _progress_file(progress_id).write_text(
                json.dumps({"total": total, "completed": 0})
            )
        except OSError as exc:
            logging.debug("Progress ledger for %s is memory-only: %s", progress_id, exc)
    return progress_id


def _log_agent_completion(progress_id: str, agent_name: str, detail: str) -> None:
    """Advance one progress ledger and log the completion line; id may be empty."""
    if not progress_id:
        return

    with _agent_progress_lock:
        progress = _agent_progress.get(progress_id)
        if progress is None:
            # This module may have been hot-reloaded mid-fan-out by `langgraph
            # dev`; the ledger survives on disk under the id carried in state.
            try:
                progress = json.loads(_progress_file(progress_id).read_text())
            except (OSError, ValueError):
                return
        completed = progress["completed"] = progress.get("completed", 0) + 1
        total = progress["total"]
        remaining = total - completed
        if remaining == 0:
            _agent_progress.pop(progress_id, None)
        else:
            _agent_progress[progress_id] = progress
        try:
            if remaining == 0:
                _progress_file(progress_id).unlink(missing_ok=True)
            else:
                _progress_file(progress_id).write_text(json.dumps(progress))
        except OSError:
            pass

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
        budget_sec=settings.embeddings_fallback_budget_sec,
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


def affected_nodes_label(report: dict, fallback: str) -> str:
    """Comma-joined non-empty affected_nodes of a record, else `fallback`."""
    affected = [n for n in (report.get("affected_nodes") or []) if n]
    return ", ".join(affected) if affected else fallback


_STEP_NUM_RE = re.compile(r'^\s*\d+[\.\)]\s+')


def strip_step_numbering(step) -> str:
    """Drop any leading 'N.' / 'N)' the reviewer embedded, so steps never double-number."""
    return _STEP_NUM_RE.sub('', str(step))


def steps_block(steps, strip_numbering: bool = False) -> str:
    """Numbered reproduction-steps block for agent prompts, shared by validator/auditor."""
    return (
        "\n".join(
            f"  {i}. {strip_step_numbering(s) if strip_numbering else s}"
            for i, s in enumerate(steps, 1)
        )
        if steps else "  None provided by reviewer"
    )
