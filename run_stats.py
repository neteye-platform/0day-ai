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
import os
import re
import threading
import time
import uuid
from pathlib import Path

import settings
from dedup import Embeddings

logger = logging.getLogger(__name__)

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
        logger.info(message)
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
    _flush_ledger()


def _reset_pipeline_stats() -> None:
    # The token ledger rides the same bootstrap reset (one ledger per run).
    with _pipeline_stats_lock:
        _pipeline_stats.clear()
    with _token_lock:
        _token_totals.clear()


def _snapshot_pipeline_stats() -> dict[str, int]:
    with _pipeline_stats_lock:
        return dict(_pipeline_stats)


# ---------------------------------------------------------------------------
# Token usage ledger — per-agent LLM input/output token totals for the run.

USAGE_FIELDS = ("calls", "input_tokens", "output_tokens")

_token_totals: dict[str, dict[str, int]] = {}
_token_lock = threading.Lock()


def new_usage() -> dict:
    """Fresh zeroed usage accumulator ({calls, input_tokens, output_tokens})."""
    return {field: 0 for field in USAGE_FIELDS}


def normalize_usage(usage) -> dict | None:
    """Normalize any token-usage carrier (AIMessage usage_metadata dict or a
    stored usage accumulator) to {calls, input_tokens, output_tokens}; None
    when it carries no tokens at all (so zeroed payloads — e.g. an endpoint
    that omits usage_metadata — never bookkeep or render as all-zero rows)."""
    if not isinstance(usage, dict):
        return None
    tokens = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
    if not tokens:
        return None
    calls = usage.get("calls")
    if calls is None:
        calls = 1
    return {
        "calls": int(calls),
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
    }


def add_usage(a: dict | None, b: dict | None) -> dict | None:
    """Field-wise sum of two usage dicts (either may be None)."""
    na, nb = normalize_usage(a), normalize_usage(b)
    if na is None:
        return nb
    if nb is None:
        return na
    return {field: na[field] + nb[field] for field in USAGE_FIELDS}


def extract_llm_usage(message) -> dict | None:
    """Token usage of one LLM response message (usage_metadata), normalized."""
    return normalize_usage(getattr(message, "usage_metadata", None))


def record_usage(agent: str, usage) -> dict | None:
    """Add one call's normalized usage to the agent's run totals; returns the
    normalized dict (state/ledger bookkeeping) or None when nothing to add."""
    usage = normalize_usage(usage)
    if usage is None:
        return None
    with _token_lock:
        totals = _token_totals.setdefault(agent, new_usage())
        for field in USAGE_FIELDS:
            totals[field] += usage[field]
    _flush_ledger()
    return usage


def record_llm_usage(agent: str, message) -> dict | None:
    """Record one chat-model response's token usage under `agent`; the
    response's AIMessage carries usage_metadata (stream_usage is on)."""
    return record_usage(agent, extract_llm_usage(message))


def record_cached_token_usage(agent: str, usage) -> None:
    """Merge token totals restored from a cache entry (work spent by an
    earlier run) into the ledger, so report totals stay comparable across
    cached and fresh runs. Entries predating token tracking carry nothing."""
    record_usage(agent, usage)


def take_cached_usage(agent: str, entry) -> None:
    """Raw-`cache()` payload hook: pop the entry's 'token_usage' key (so it
    never leaks into downstream consumers) and restore it into the ledger."""
    if not isinstance(entry, dict):
        return
    record_cached_token_usage(agent, entry.pop("token_usage", None))


def snapshot_token_totals() -> dict[str, dict[str, int]]:
    with _token_lock:
        return {agent: dict(totals) for agent, totals in _token_totals.items()}


# ---------------------------------------------------------------------------
# Durable scan-scoped ledgers (token totals + pipeline stats).
#
# A `python graph.py` scan spans MANY process starts (restarts/resumes): with
# the ledgers kept purely in memory, every stopped process took its live LLM
# spend (and the usage banked by the cache hits it read) down with it — the
# report of a scan resumed N times could only ever show the LAST process's
# numbers. init_usage_ledger() pins both ledgers to
# states/token_usage_<thread>.json; every record flushes the file, so even a
# hard-exited process leaves its spend durable, and a resume restores the
# totals BEFORE bootstrap's reset would wipe them (the reset is skipped once
# the ledger is initialized). A fresh scan starts from zero and drops the
# previous scans' files (their reports were already written). Un-initialized
# (langgraph dev, in-process runs) => process-local ledgers, exactly the old
# per-invocation behavior.

_LEDGER_DIR = Path("states")
_ledger_file: Path | None = None
_ledger_lock = threading.Lock()


def ledger_initialized() -> bool:
    return _ledger_file is not None


def _ledger_write_locked() -> None:
    """Atomically persist both ledgers; must be called holding _ledger_lock."""
    if _ledger_file is None:
        return
    payload = {
        "pipeline_stats": _snapshot_pipeline_stats(),
        "token_totals": snapshot_token_totals(),
    }
    try:
        tmp = _ledger_file.with_name(_ledger_file.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, _ledger_file)
    except OSError as exc:
        logger.warning(f"Failed to persist ledger {_ledger_file}: {exc}")


def _flush_ledger() -> None:
    if _ledger_file is None:
        return
    with _ledger_lock:
        _ledger_write_locked()


def init_usage_ledger(thread_id: str, fresh: bool) -> None:
    """Pin the token + pipeline-stats ledgers to states/token_usage_<thread>.json
    (called once by the pipeline entrypoint after the thread decision).
    fresh=True starts from zero and discards previous scans' ledger files;
    fresh=False (resume) restores the persisted totals into the in-memory
    ledgers so the eventual report carries the whole scan's work."""
    global _ledger_file
    ledger = _LEDGER_DIR / f"token_usage_{thread_id}.json"
    with _ledger_lock:
        if fresh:
            _LEDGER_DIR.mkdir(parents=True, exist_ok=True)
            with _pipeline_stats_lock:
                _pipeline_stats.clear()
            with _token_lock:
                _token_totals.clear()
            for stale in _LEDGER_DIR.glob("token_usage_*.json"):
                try:
                    stale.unlink()
                except OSError:
                    pass
        _ledger_file = ledger
        if fresh or not ledger.exists():
            _ledger_write_locked()
            return
        try:
            data = json.loads(ledger.read_text())
        except (OSError, ValueError) as exc:
            logger.warning(f"Corrupt ledger {ledger} ({exc}); starting from zero.")
            _ledger_write_locked()
            return
        with _pipeline_stats_lock:
            _pipeline_stats.update(
                {str(k): int(v) for k, v in (data.get("pipeline_stats") or {}).items()}
            )
        with _token_lock:
            for agent, totals in (data.get("token_totals") or {}).items():
                live = _token_totals.setdefault(agent, new_usage())
                for field in USAGE_FIELDS:
                    live[field] += int(totals.get(field) or 0)
    logger.info(
        f"Restored scan ledger {ledger.name} (agents: "
        + ", ".join(
            f"{a}: {t['input_tokens']}in/{t['output_tokens']}out"
            for a, t in sorted(snapshot_token_totals().items())
        )
        + ")."
    )


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
            logger.debug("Progress ledger for %s is memory-only: %s", progress_id, exc)
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

    logger.info(
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
        batch_size=settings.embeddings_batch_size,
        parallel_chunks=settings.embeddings_parallel_chunks,
        prewarm_timeout=settings.embeddings_prewarm_timeout,
        keep_alive=settings.embeddings_keep_alive,
        stall_budget_sec=settings.embeddings_stall_budget_sec,
    )
    if embedder.available():
        return embedder
    logger.warning(
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


_STEP_NUM_RE = re.compile(r"^\s*\d+[\.\)]\s+")


def strip_step_numbering(step) -> str:
    """Drop any leading 'N.' / 'N)' the reviewer embedded, so steps never double-number."""
    return _STEP_NUM_RE.sub("", str(step))


def steps_block(steps, strip_numbering: bool = False) -> str:
    """Numbered reproduction-steps block for agent prompts, shared by validator/auditor."""
    return (
        "\n".join(
            f"  {i}. {strip_step_numbering(s) if strip_numbering else s}"
            for i, s in enumerate(steps, 1)
        )
        if steps
        else "  None provided by reviewer"
    )
