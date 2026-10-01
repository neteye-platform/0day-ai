import json
import logging
import argparse
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

# Keep successful HTTP transport requests out of the application logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy
from openai import BadRequestError, LengthFinishReasonError

import settings
import tools
import browser_tools
import attacker_tools
import patch_tools
from stage_aggregate import aggregate_demands_node
from stage_cve import cve_analyzer_node, dispatch_cve_analyzers
from stage_dedup import dedup_agent_node
from stage_edge_traversal import edge_traversal_node
from stage_explorer import dispatch_explorers, expert_explorer_node
from stage_integration_auditor import (
    ask_integration_auditor_for_tool,
    dispatch_integration_audits,
    integration_auditor_fallback_node,
    integration_auditor_node,
    integration_auditor_router,
    route_integration_audit,
)
from stage_manager import manager_agent_node
from stage_patcher import (
    ask_patcher_for_tool,
    dispatch_patch_reviews,
    dispatch_patchers,
    patcher_agent_node,
    patcher_fallback_node,
    patcher_router,
    route_patch_reviews,
    sandbox_resync_node,
)
from stage_preprocess import bootstrap_node, preprocessor_node
from stage_reporter import dispatch_reporters, report_assembler_node, reporter_node
from stage_reviewer import (
    ask_reviewer_for_tool,
    dispatch_reviewers,
    reviewer_agent_node,
    reviewer_fallback_node,
    reviewer_router,
)
from stage_threat_intel import dispatch_threat_intel, threat_intel_gate_node, threat_intel_node
from stage_validator import (
    ask_validator_for_tool,
    dispatch_validators,
    route_validator_feedback,
    validator_agent_node,
    validator_fallback_node,
    validator_router,
)
from stage_verifier import contract_verifier_node, dispatch_verifiers
from run_stats import RunStopped, gen_run_id, install_signal_handlers, langsmith_detached_node
from tool_loop import SequentialToolNode, concise_tool_error
from credential_finder import credential_finder_node
from state import (
    IntegrationAuditorState,
    MasterState,
    PatcherState,
    ReviewerState,
    ValidatorState,
)
from schemas import PatcherOutput, ReviewerOutput, ValidatorOutput


def compile_reviewer():
    reviewer_workflow = StateGraph(ReviewerState, output_schema=ReviewerOutput)
    reviewer_workflow.add_node("reviewer_agent", reviewer_agent_node, retry_policy=RETRY)
    reviewer_workflow.add_node("ask_reviewer_for_tool", ask_reviewer_for_tool)
    reviewer_workflow.add_node("reviewer_fallback", reviewer_fallback_node)
    reviewer_workflow.add_node("reviewer_tools", ToolNode([
        tools.read_source_code,
        tools.read_file,
        tools.get_node_connections,
        tools.search_codebase,
        tools.get_definition,
        tools.list_container_artifacts,
        tools.find_in_container,
        tools.read_container_artifact,
        tools.submit_evaluation
    ], handle_tool_errors=concise_tool_error))
    reviewer_workflow.add_edge(START, "reviewer_agent")
    reviewer_workflow.add_conditional_edges(
        "reviewer_agent",
        reviewer_router,
        {
            "reviewer_tools": "reviewer_tools",
            "ask_reviewer_for_tool": "ask_reviewer_for_tool",
            "reviewer_fallback": "reviewer_fallback",
            "__end__": END
        }
    )
    reviewer_workflow.add_conditional_edges(
        "reviewer_tools",
        reviewer_router,
        {
            "reviewer_agent": "reviewer_agent",
            "reviewer_fallback": "reviewer_fallback",
            "__end__": END
        }
    )
    reviewer_workflow.add_edge("ask_reviewer_for_tool", "reviewer_agent")
    reviewer_workflow.add_edge("reviewer_fallback", END)
    compiled_reviewer_agent = reviewer_workflow.compile()

    return compiled_reviewer_agent


def compile_validator():
    validator_workflow = StateGraph(ValidatorState, output_schema=ValidatorOutput)
    validator_workflow.add_node("validator_agent", validator_agent_node, retry_policy=RETRY)
    validator_workflow.add_node("ask_validator_for_tool", ask_validator_for_tool)
    validator_workflow.add_node("validator_fallback", validator_fallback_node)
    validator_workflow.add_node("validator_tools", SequentialToolNode([
        tools.send_http_request,
        browser_tools.browser_navigate,
        browser_tools.browser_click,
        browser_tools.browser_fill,
        browser_tools.browser_evaluate,
        browser_tools.browser_console,
        attacker_tools.run_command,
        attacker_tools.write_attacker_file,
        attacker_tools.read_attacker_file,
        tools.ask_for_context,
        tools.mark_validation_complete
    ], handle_tool_errors=concise_tool_error))
    validator_workflow.add_edge(START, "validator_agent")
    validator_workflow.add_conditional_edges(
        "validator_agent",
        validator_router,
        {
            "validator_tools": "validator_tools",
            "ask_validator_for_tool": "ask_validator_for_tool",
            "validator_fallback": "validator_fallback",
            "__end__": END
        }
    )
    validator_workflow.add_conditional_edges(
        "validator_tools",
        validator_router,
        {
            "validator_agent": "validator_agent",
            "validator_fallback": "validator_fallback",
            "__end__": END
        }
    )
    validator_workflow.add_edge("ask_validator_for_tool", "validator_agent")
    validator_workflow.add_edge("validator_fallback", END)
    compiled_validator_agent = validator_workflow.compile()

    return compiled_validator_agent 


def compile_patcher():
    patcher_workflow = StateGraph(PatcherState, output_schema=PatcherOutput)
    patcher_workflow.add_node("patcher_agent", patcher_agent_node, retry_policy=RETRY)
    patcher_workflow.add_node("ask_patcher_for_tool", ask_patcher_for_tool)
    patcher_workflow.add_node("patcher_fallback", patcher_fallback_node)
    # Sequential: same-response batches may chain multi-hunk edits of one file,
    # whose line ranges must apply strictly in the listed order.
    patcher_workflow.add_node("patcher_tools", SequentialToolNode([
        tools.read_source_code,
        tools.read_file,
        tools.search_codebase,
        tools.get_definition,
        patch_tools.patch_source_file,
        patch_tools.submit_patch,
    ], handle_tool_errors=concise_tool_error))
    patcher_workflow.add_edge(START, "patcher_agent")
    patcher_workflow.add_conditional_edges(
        "patcher_agent",
        patcher_router,
        {
            "patcher_tools": "patcher_tools",
            "ask_patcher_for_tool": "ask_patcher_for_tool",
            "patcher_fallback": "patcher_fallback",
            "__end__": END
        }
    )
    patcher_workflow.add_conditional_edges(
        "patcher_tools",
        patcher_router,
        {
            "patcher_agent": "patcher_agent",
            "patcher_fallback": "patcher_fallback",
            "__end__": END
        }
    )
    patcher_workflow.add_edge("ask_patcher_for_tool", "patcher_agent")
    patcher_workflow.add_edge("patcher_fallback", END)
    compiled_patcher_agent = patcher_workflow.compile()

    return compiled_patcher_agent


def compile_integration_auditor():
    integration_auditor_workflow = StateGraph(IntegrationAuditorState)
    integration_auditor_workflow.add_node("integration_auditor_agent", integration_auditor_node, retry_policy=RETRY)
    integration_auditor_workflow.add_node("ask_integration_auditor_for_tool", ask_integration_auditor_for_tool)
    integration_auditor_workflow.add_node("integration_auditor_fallback", integration_auditor_fallback_node)
    integration_auditor_workflow.add_node("integration_auditor_tools", ToolNode([
        tools.get_vulnerability_details,
        tools.get_node_connections,
        tools.get_path,
        tools.submit_integration_audit
    ], handle_tool_errors=concise_tool_error))
    integration_auditor_workflow.add_edge(START, "integration_auditor_agent")
    integration_auditor_workflow.add_conditional_edges(
        "integration_auditor_agent",
        integration_auditor_router,
        {
            "integration_auditor_tools": "integration_auditor_tools",
            "ask_integration_auditor_for_tool": "ask_integration_auditor_for_tool",
            "integration_auditor_fallback": "integration_auditor_fallback",
            "__end__": END
        }
    )
    integration_auditor_workflow.add_conditional_edges(
        "integration_auditor_tools",
        integration_auditor_router,
        {
            "integration_auditor_agent": "integration_auditor_agent",
            "integration_auditor_fallback": "integration_auditor_fallback",
            "__end__": END
        }
    )
    integration_auditor_workflow.add_edge("ask_integration_auditor_for_tool", "integration_auditor_agent")
    integration_auditor_workflow.add_edge("integration_auditor_fallback", END)
    compiled_integration_auditor = integration_auditor_workflow.compile()

    return compiled_integration_auditor


# Retry only transient errors: LengthFinishReasonError and other deterministic failures fail fast.
# Context-length 400s belong in the same bucket: the prompt size is recomputed
# identically on every attempt, so retrying just burns the attempt budget.
try:  # lives in a private langgraph module; fall back to "retry others" if moved
    from langgraph._internal._retry import default_retry_on as default_retry_on
except ImportError:  # pragma: no cover
    def default_retry_on(exc):
        return True


def _retry_on(exc):
    # RunStopped is the cooperative Ctrl+C unwind: retrying it would defeat the stop.
    if isinstance(exc, RunStopped):
        return False
    # A context-window 400 is deterministic: every retry resends the identical
    # oversized prompt. Any other BadRequestError keeps the default behavior.
    if isinstance(exc, BadRequestError) and "maximum context length" in str(exc):
        return False
    return default_retry_on(exc) and not isinstance(exc, LengthFinishReasonError)


RETRY = RetryPolicy(
    initial_interval=1.0,
    backoff_factor=2.0,
    max_interval=60.0,
    max_attempts=5,
    jitter=True,
    retry_on=_retry_on,
)


def build_graph(checkpointer=None, interrupt_before=None):
    workflow = StateGraph(MasterState).set_node_defaults(retry_policy=RETRY)
    workflow.add_node("bootstrap", bootstrap_node)
    workflow.add_node("preprocessor", preprocessor_node)
    workflow.add_node("credential_finder", credential_finder_node)
    workflow.add_node("manager", manager_agent_node)
    workflow.add_node("explorer_agent", expert_explorer_node)
    workflow.add_node("cve_analyzer", cve_analyzer_node)
    workflow.add_node("threat_intel_gate", threat_intel_gate_node)
    workflow.add_node("threat_intel", threat_intel_node)
    workflow.add_node("aggregate_demands", aggregate_demands_node)
    workflow.add_node("contract_verifier", contract_verifier_node)
    workflow.add_node("dedup_agent", dedup_agent_node)
    workflow.add_node("edge_traversal", edge_traversal_node)
    # Subgraphs own their per-message retries; disable wholesale replay retry from set_node_defaults.
    # The wrapper always starts with the cooperative-stop check (raise_if_stopping);
    # with trace splitting off it is a plain subgraph.invoke pass-through, when on each
    # dispatch runs the subgraph as its own root trace in the agent's dedicated LangSmith project.
    workflow.add_node("reviewer_agent", langsmith_detached_node(compiled_reviewer_agent, "reviewer"), retry_policy=RetryPolicy(max_attempts=1))
    workflow.add_node("validator_agent", langsmith_detached_node(compiled_validator_agent, "validator"), retry_policy=RetryPolicy(max_attempts=1))
    workflow.add_node("patcher_agent", langsmith_detached_node(compiled_patcher_agent, "patcher"), retry_policy=RetryPolicy(max_attempts=1))
    workflow.add_node("integration_auditor", langsmith_detached_node(compiled_integration_auditor, "integration_auditor"), retry_policy=RetryPolicy(max_attempts=1))
    # Barrier for the contract-verifier fan-out: edge_traversal runs once after every verifier task has written.
    workflow.add_node("synchronization", lambda state: {})
    # Barrier so dispatch_validators sees the fully-merged record set, not a mid-superstep snapshot.
    workflow.add_node("validator_dispatch_gate", lambda state: {})
    # Barrier after the validator superstep: fans deferred `requires_integration` records to the auditor once their peers carry proven poc_payloads.
    workflow.add_node("integration_audit_dispatch", lambda state: {})
    # Barrier after a validator superstep whose exploitable records are due a fix:
    # one fan-out point so the patch loop cannot collide with the audit phase's
    # superstep bookkeeping. Unreachable while settings.patcher_enabled is False.
    workflow.add_node("patch_dispatch", lambda state: {})
    # Deterministic pause where the sandbox adopts freshly patched source
    # (image rebuild or docker-cp + restart) before the patched records return
    # to the Reviewer.
    workflow.add_node("sandbox_resync", sandbox_resync_node)
    # Terminal barrier: one single-shot reporter per reportable vuln, then report_assembler writes the report dir.
    workflow.add_node("reporter_dispatch", lambda state: {})
    workflow.add_node("reporter", reporter_node)
    workflow.add_node("report_assembler", report_assembler_node)

    workflow.add_edge(START, "bootstrap")
    workflow.add_edge("bootstrap", "preprocessor")
    workflow.add_edge("bootstrap", "manager")

    workflow.add_conditional_edges("manager", dispatch_explorers, ["explorer_agent"])
    workflow.add_edge("preprocessor", "credential_finder")
    workflow.add_conditional_edges("credential_finder", dispatch_cve_analyzers, ["cve_analyzer"])

    workflow.add_edge("cve_analyzer", "threat_intel_gate")
    # AND-join barrier: fires once only after BOTH branches write (separate edges would trigger it on the first writer).
    workflow.add_conditional_edges(
        "threat_intel_gate",
        dispatch_threat_intel,
        ["threat_intel"],
    )
    workflow.add_edge(["explorer_agent", "threat_intel"], "aggregate_demands")

    workflow.add_conditional_edges("aggregate_demands", dispatch_verifiers, ["contract_verifier", "synchronization", END])
    workflow.add_edge("contract_verifier", "synchronization")
    workflow.add_edge("synchronization", "edge_traversal")
    workflow.add_edge("edge_traversal", "dedup_agent")
    workflow.add_conditional_edges("dedup_agent", dispatch_reviewers, ["reviewer_agent", "reporter_dispatch"])
    workflow.add_edge("reviewer_agent", "validator_dispatch_gate")
    # Stage 1: prove direct_to_validator records; requires_integration records are DEFERRED so the auditor only sees proven poc_payloads.
    workflow.add_conditional_edges(
        "validator_dispatch_gate",
        dispatch_validators,
        {
            "validator_agent": "validator_agent",
            "integration_audit_dispatch": "integration_audit_dispatch",
            # Never returned today, but reporter dispatch must remain the sole sink.
            "__end__": "reporter_dispatch",
        },
    )
    # Stage 2: after the validator superstep, bounce insufficient_context records to the Reviewer;
    # otherwise route due-for-fix exploitable records to the Patcher (flag off => never) or advance.
    workflow.add_conditional_edges(
        "validator_agent",
        route_validator_feedback,
        {
            "reviewer_agent": "reviewer_agent",
            "patch_dispatch": "patch_dispatch",
            "integration_audit_dispatch": "integration_audit_dispatch",
            "__end__": "reporter_dispatch",
        },
    )
    # Stage 2a: patcher fan-out; banked patches first resync the sandbox, then
    # return to the Reviewer as cache-exempt PATCH APPLIED re-adjudications.
    workflow.add_conditional_edges(
        "patch_dispatch",
        dispatch_patchers,
        ["patcher_agent", "integration_audit_dispatch"],
    )
    workflow.add_conditional_edges(
        "patcher_agent",
        route_patch_reviews,
        ["sandbox_resync", "integration_audit_dispatch"],
    )
    workflow.add_conditional_edges(
        "sandbox_resync",
        dispatch_patch_reviews,
        ["reviewer_agent", "integration_audit_dispatch"],
    )
    # Stage 2b: fan deferred records to the auditor, or advance to reporter dispatch when none remain.
    workflow.add_conditional_edges(
        "integration_audit_dispatch",
        dispatch_integration_audits,
        ["integration_auditor", "reporter_dispatch"],
    )
    # Stage 3: chained records re-validate with peers' proven poc_payloads (see route_integration_audit); the rest advance to reporter dispatch.
    workflow.add_conditional_edges("integration_auditor", route_integration_audit, ["validator_agent", "reporter_dispatch"])
    workflow.add_conditional_edges("reporter_dispatch", dispatch_reporters, ["reporter", "report_assembler"])
    workflow.add_edge("reporter", "report_assembler")
    workflow.add_edge("report_assembler", END)

    app = workflow.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
    app = app.with_config({"max_concurrency": settings.agents_concurrency})

    return app


compiled_reviewer_agent = compile_reviewer()
compiled_validator_agent = compile_validator()
compiled_patcher_agent = compile_patcher()
compiled_integration_auditor = compile_integration_auditor()
graph = build_graph()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Vulnerability Analyzer")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging.")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = "%(asctime)s [%(levelname)s] %(module)s - %(message)s"
    log_datefmt = "%H:%M:%S"
    log_file = Path(os.getenv("LOG_FILE", "logs.txt"))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Keep console output while also retaining a complete run log for failures
    # that occur after the terminal output has scrolled away.
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=log_datefmt,
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logging.info(
        "\n\n===== Pipeline run started at %s =====",
        datetime.now().astimezone().isoformat(timespec="seconds"),
    )

    run_id = gen_run_id()
    initial_state = MasterState(
        pipeline_run_id=run_id,
        known_vulns=[],
        expert_tasks=[],
        sandbox_url=None,
        notes=[],
        cve_demands=[],
        grouped_demands={},
        vulnerabilities=[],
        reporter_findings=[],
    )

    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import Command

    Path("states").mkdir(parents=True, exist_ok=True)
    done_flag = Path("states/scan-complete.flag")
    thread_file = Path("states/current_thread.txt")

    # Every scan runs on its own checkpoint thread (pointer kept in
    # states/current_thread.txt). A completed scan (flag present) means this
    # start is a NEW scan: mint a fresh thread so the old run's append-channel
    # values (notes, vulnerabilities, ...) can never accumulate into it. If the
    # flag is absent, the pointer thread is the interrupted scan and gets
    # resumed from its sqlite checkpoints.
    pointer_thread = thread_file.read_text().strip() if thread_file.exists() else None
    if done_flag.exists():
        done_flag.unlink()
        pointer_thread = None
    thread_id = pointer_thread or f"scan-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    thread_file.write_text(thread_id)

    # Stamp the entry config so this run's ROOT trace correlates with the
    # project-split subagent traces (run_stats.langsmith_detached_node).
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"pipeline_run_id": run_id, "target": settings.app_path.name},
        "tags": ["pipeline"],
    }

    # Two-phase Ctrl+C (see run_stats.install_signal_handlers): 1st press
    # cooperatively stops after the running agents, 2nd press exits now;
    # SIGTERM/SIGHUP exit immediately. Exits leave resumable checkpoints.
    run_live = threading.Event()
    install_signal_handlers(run_live)

    # A force-killed previous process leaked per-agent attacker containers
    # (close_agent_sessions never ran); sweep them before the scan starts.
    attacker_tools.manager.sweep_stale_containers()

    with SqliteSaver.from_conn_string("states/pipeline_checkpoints.sqlite") as checkpointer:
        app = build_graph(checkpointer=checkpointer)
        try:
            snapshot = app.get_state(config)
            if snapshot.next:
                # Resume needs a None/Command input: a plain dict is treated as
                # a NEW run and restarts from bootstrap. The Command UPDATE
                # patches pipeline_run_id for pre-change checkpoints while the
                # pending tasks keep running (task writes already committed are
                # durable and never re-executed).
                run_id = snapshot.values.get("pipeline_run_id") or run_id
                config["metadata"]["pipeline_run_id"] = run_id
                run_input = Command(update={"pipeline_run_id": run_id})
                logging.info(
                    "Resuming interrupted scan on thread %s (pending step: %s).",
                    thread_id, ", ".join(dict.fromkeys(snapshot.next)),
                )
            else:
                if snapshot.values and thread_id == pointer_thread:
                    # The pointed thread already ran to END (e.g. killed just
                    # before the done flag was touched): reusing it would
                    # accumulate the old run's channel values into the new one.
                    thread_id = f"scan-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    thread_file.write_text(thread_id)
                    config["configurable"]["thread_id"] = thread_id
                    logging.info(
                        "Thread %s already completed; starting fresh scan on thread %s.",
                        pointer_thread, thread_id,
                    )
                run_input = initial_state
                logging.info("Starting fresh scan on thread %s.", thread_id)

            run_live.set()
            try:
                final_state = app.invoke(run_input, config)
            finally:
                run_live.clear()

            out_file = "results.json"
            with open(out_file, "w") as f:
                json.dump(final_state, f)
                logging.info(f"Final state saved to {out_file}.")
            done_flag.touch()

        except FileNotFoundError:
            print("Waiting for actual graph.json to execute.")
        except RunStopped:
            logging.info(
                "Stopped after the in-flight agents finished. Progress is "
                "checkpointed: rerun `python graph.py` to resume this scan."
            )
            sys.exit(130)
        except KeyboardInterrupt:
            # Only reachable when SIGINT races the run_live transition; while
            # the loop is live the signal handler itself acts.
            logging.info(
                "Interrupted. Progress is checkpointed: "
                "rerun `python graph.py` to resume this scan."
            )
            sys.exit(130)
