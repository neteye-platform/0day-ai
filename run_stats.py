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

# ---------------------------------------------------------------------------
# Cooperative run stop (Ctrl+C). The first SIGINT only sets this event: every
# queued fan-out task raises RunStopped at its entry, langgraph then cancels
# the rest of the queue, waits for the in-flight agents to finish (each one
# commits its durable writes as it completes), and re-raises out of the run.
# In-flight agents never read the flag, so a running reviewer/validator still
# finishes its whole loop and submits normally. Everything checkpointed stays
# in sqlite, so the next `python graph.py` resumes exactly where this stopped.

_stop_requested = threading.Event()


class RunStopped(Exception):
    """Pipeline stop requested; raised at task entry to unwind the run."""


def request_stop() -> None:
    _stop_requested.set()


def stop_requested() -> bool:
    return _stop_requested.is_set()


def raise_if_stopping() -> None:
    if _stop_requested.is_set():
        raise RunStopped("Pipeline stop requested (Ctrl+C).")


_sigint_armed = threading.Event()


def install_signal_handlers(run_live: threading.Event) -> None:
    """Two-phase Ctrl+C handling for the pipeline entrypoint.

    The first SIGINT arms the cooperative stop (``request_stop()``): queued
    fan-out tasks raise RunStopped at their entry, langgraph cancels the rest
    of the queue and waits for the in-flight agents to finish and commit their
    results, then the exception unwinds in the caller and the process exits
    with a resumable checkpoint. The second SIGINT exits straight from the
    handler, because background threads cannot be killed and the run's
    executor would otherwise block waiting for them. While ``run_live`` is
    unset (startup/shutdown windows) there is no run to unwind, so even the
    first SIGINT exits immediately. SIGTERM/SIGHUP always exit immediately.
    """
    import os
    import signal

    def _hard_exit(message: str, code: int):
        print(f"\n{message}", flush=True)
        logging.info(message)
        for handler in logging.getLogger().handlers:
            handler.flush()
        os._exit(code)

    def _on_sigint(signum, frame):
        if _sigint_armed.is_set() or not run_live.is_set():
            _hard_exit(
                "Terminating immediately: agents still executing are abandoned "
                "(their in-flight work is lost; everything checkpointed earlier "
                "is kept and will resume on the next start).",
                130,
            )
        _sigint_armed.set()
        request_stop()
        print(
            "\nStop requested: the agents currently executing will finish and "
            "their results are checkpointed; queued agents are skipped. "
            "Press Ctrl+C again to terminate immediately.",
            flush=True,
        )

    def _on_sigterm(signum, frame):
        _hard_exit(
            f"Signal {signum} received: terminating immediately "
            "(checkpoints are kept; the next start resumes this scan).",
            143,
        )

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigterm)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _on_sigterm)


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


def gen_run_id() -> str:
    return uuid.uuid4().hex[:12]


def payload_subject(state) -> str:
    """Best identifying label of a subagent dispatch payload (reviewer node /
    validator or auditor vuln id), used as LangSmith metadata/tag on the
    detached trace so runs of one finding are greppable inside the project."""
    if not isinstance(state, dict):
        return ""
    report = state.get("report_to_test") or state.get("expert_report") or {}
    if isinstance(report, dict) and report.get("vuln_id"):
        return str(report["vuln_id"])[:120]
    return str(state.get("node_id") or "")[:120]


def langsmith_split_active() -> bool:
    return bool(settings.langsmith_split_subagents and settings.langsmith_tracing)


def langsmith_detached_node(subgraph, agent: str):
    """Wrap a compiled subgraph so every dispatch runs it as its OWN root
    trace in its OWN LangSmith project (agent keys: reviewer / validator /
    integration_auditor). Returns the subgraph untouched when off.

    LangSmith routes a whole callback tree through the single
    ``LangChainTracer`` instantiated at the root invoke — its project
    (LANGSMITH_PROJECT) is stamped on every nested run, so a nested
    ``tracing_context(project_name=...)`` or a config ``project_name`` on the
    subtree is IGNORED (verified empirically against langsmith 0.10.5 +
    langchain-core 1.5.4). The one working lever: the parent task hides its
    config — carrying that tracer — in the ``var_child_runnable_config``
    contextvar, which an inner ``ensure_config`` seeds callbacks from; briefly
    clearing it makes the nested invoke build a fresh callback manager whose
    tracer picks up ``tracing_context(project_name=..., parent=False)`` —
    a detached root trace in the agent project.

    Correlation rides on the invoke config: ``pipeline_run_id`` + subject land
    in the metadata of every run of the detached tree (config metadata
    propagates to child runs), and the ``agent:<name>`` tag comes from the
    tracing-context tags read by the fresh tracer. The node config's internal
    ``configurable`` keys (``__pregel_checkpointer``, ``checkpoint_ns``) pass
    through, so the subgraph still inherits the parent checkpointer; an
    omitted ``recursion_limit`` falls back to the same langgraph default a
    subgraph-as-node gets. The detached subtree loses the parent's OTHER
    callbacks (e.g. a caller-supplied custom tracing handler).
    """
    if not langsmith_split_active():
        def node(state, config):
            raise_if_stopping()
            return subgraph.invoke(state, config)

        node.__name__ = f"{agent}_node"
        return node

    from langsmith.run_helpers import tracing_context

    project = settings.langsmith_split_projects[agent]

    def node(state, config):
        from langchain_core.runnables.config import var_child_runnable_config

        raise_if_stopping()
        run_id = state.get("pipeline_run_id") if isinstance(state, dict) else None
        subject = payload_subject(state)
        metadata = {"agent": agent}
        if run_id:
            metadata["pipeline_run_id"] = run_id
        if subject:
            metadata["subject"] = subject
        bare = {k: v for k, v in (config or {}).items() if k != "callbacks"}
        bare["metadata"] = {**(config or {}).get("metadata", {}), **metadata}
        token = var_child_runnable_config.set(None)
        try:
            with tracing_context(
                project_name=project,
                parent=False,
                tags=[f"agent:{agent}"],
            ):
                return subgraph.invoke(state, bare)
        finally:
            var_child_runnable_config.reset(token)

    node.__name__ = f"{agent}_traced"
    return node


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
