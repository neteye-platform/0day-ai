import json
import logging
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Keep successful HTTP transport requests out of the application logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy

import settings
import tools
import browser_tools
import attacker_tools
from nodes import bootstrap_node, preprocessor_node, manager_agent_node, expert_explorer_node, cve_analyzer_node, threat_intel_gate_node, threat_intel_node, reviewer_agent_node, ask_reviewer_for_tool, reviewer_fallback_node, dispatch_explorers, dispatch_cve_analyzers, dispatch_threat_intel, dispatch_reviewers, dispatch_validators, dispatch_integration_audits, integration_auditor_node, integration_auditor_router, integration_auditor_fallback_node, ask_integration_auditor_for_tool, route_integration_audit, route_validator_feedback, dispatch_verifiers, reviewer_router, validator_agent_node, ask_validator_for_tool, validator_fallback_node, validator_router, aggregate_demands_node, contract_verifier_node, synchronization_node, edge_traversal_node, reporter_node
from credential_finder import credential_finder_node
from state import MasterState, ReviewerState, ValidatorState, IntegrationAuditorState
from schemas import ReviewerOutput, ValidatorOutput


def compile_reviewer():
    reviewer_workflow = StateGraph(ReviewerState, output_schema=ReviewerOutput)
    # The agent node is a single LLM invocation that is pure w.r.t. state until
    # it returns, so its retry policy re-runs just the failing message -- never
    # the whole tool loop.
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
    ]))
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
    # Loop length is bounded by the explicit iteration counter in
    # reviewer_router/fallback_node, well under the default recursion limit.
    compiled_reviewer_agent = reviewer_workflow.compile()

    return compiled_reviewer_agent


def compile_validator():
    validator_workflow = StateGraph(ValidatorState, output_schema=ValidatorOutput)
    # Same scoped-retry rationale as the reviewer agent node.
    validator_workflow.add_node("validator_agent", validator_agent_node, retry_policy=RETRY)
    validator_workflow.add_node("ask_validator_for_tool", ask_validator_for_tool)
    validator_workflow.add_node("validator_fallback", validator_fallback_node)
    validator_workflow.add_node("validator_tools", ToolNode([
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
    ]))
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


def compile_integration_auditor():
    integration_auditor_workflow = StateGraph(IntegrationAuditorState)
    # Same scoped-retry rationale as the reviewer/validator agent nodes.
    integration_auditor_workflow.add_node("integration_auditor_agent", integration_auditor_node, retry_policy=RETRY)
    integration_auditor_workflow.add_node("ask_integration_auditor_for_tool", ask_integration_auditor_for_tool)
    integration_auditor_workflow.add_node("integration_auditor_fallback", integration_auditor_fallback_node)
    integration_auditor_workflow.add_node("integration_auditor_tools", ToolNode([
        tools.get_vulnerability_details,
        tools.get_node_connections,
        tools.get_path,
        tools.submit_integration_audit
    ]))
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


RETRY = RetryPolicy(
    initial_interval=1.0,
    backoff_factor=2.0,
    max_interval=60.0,
    max_attempts=5,
    jitter=True,
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
    # Single synchronous Edge Traversal node: synthesizes trust-boundary edges
    # from the explorer notes and emits composite hypotheses for the reviewer's
    # cross_boundary track. Runs on the linear chain after the contract-verifier
    # barrier (synchronization) and before dispatch_reviewers, so its findings
    # land in the vulnerabilities channel before review dispatch.
    workflow.add_node("edge_traversal", edge_traversal_node)
    # The reviewer/validator outer nodes are compiled subgraphs with internal
    # per-message retries; explicitly disable wholesale (subgraph replay)
    # retries here (set_node_defaults would otherwise apply RETRY to them).
    workflow.add_node("reviewer_agent", compiled_reviewer_agent, retry_policy=RetryPolicy(max_attempts=1))
    workflow.add_node("validator_agent", compiled_validator_agent, retry_policy=RetryPolicy(max_attempts=1))
    workflow.add_node("integration_auditor", compiled_integration_auditor, retry_policy=RetryPolicy(max_attempts=1))
    workflow.add_node("synchronization", synchronization_node)
    # Barrier after the reviewer superstep, so dispatch_validators sees the fully
    # merged record set (not a partial mid-superstep snapshot, which previously
    # sent the auditor an empty `confirmed_vulns` peer list).
    workflow.add_node("validator_dispatch_gate", lambda state: {})
    # Runs strictly AFTER the validator superstep: a barrier node whose
    # conditional edges fan the deferred `requires_integration` records to the
    # auditor once their `direct_to_validator` peers carry proven poc_payloads
    # (dispatch_integration_audits is the conditional path function).
    workflow.add_node("integration_audit_dispatch", lambda state: {})
    # Single-shot terminal Reporter: after the validator/audit phases drain, it
    # writes the final markdown report (report.md under the target app dir)
    # covering the proven-exploitable + accepted-static findings.
    workflow.add_node("reporter", reporter_node)
    # workflow.add_node("reviewer_sync", synchronization_node)

    workflow.add_edge(START, "bootstrap")
    workflow.add_edge("bootstrap", "preprocessor")
    workflow.add_edge("bootstrap", "manager")

    workflow.add_conditional_edges("manager", dispatch_explorers, ["explorer_agent"])
    # The credential finder runs right after the preprocessor finishes the
    # container build/sandbox/artifact-setup phase, on the same linear branch
    # that fans out into the CVE analyzers.
    workflow.add_edge("preprocessor", "credential_finder")
    workflow.add_conditional_edges("credential_finder", dispatch_cve_analyzers, ["cve_analyzer"])

    workflow.add_edge("cve_analyzer", "threat_intel_gate")
    # AND-join barrier: aggregate_demands only fires once both the explorer
    # branch and the CVE/threat-intel branch have written. A naive set of
    # separate edges would trigger it on the FIRST writer (both write to the
    # same EphemeralValue trigger channel), running it twice with partial input.
    workflow.add_conditional_edges(
        "threat_intel_gate",
        dispatch_threat_intel,
        ["threat_intel"],
    )
    workflow.add_edge(["explorer_agent", "threat_intel"], "aggregate_demands")

    workflow.add_conditional_edges("aggregate_demands", dispatch_verifiers, ["contract_verifier", "synchronization", END])
    workflow.add_edge("contract_verifier", "synchronization")
    # Contract-verifier barrier -> Edge Traversal (composite hypotheses are wired
    # synchronously on this chain, no extra fan-out/join) -> reviewer dispatch.
    workflow.add_edge("synchronization", "edge_traversal")
    # Reporter is the single sink: every early/terminal exit routes there so the
    # final markdown report is written exactly once, even with zero hypotheses.
    workflow.add_conditional_edges("edge_traversal", dispatch_reviewers, ["reviewer_agent", "reporter"])
    # workflow.add_edge("reviewer_agent", "reviewer_sync")
    # Evaluate dispatch from the barrier (never mid-superstep) so it reads the
    # fully-merged confirmed set before emitting validator/auditor Sends.
    workflow.add_edge("reviewer_agent", "validator_dispatch_gate")
    # Stage 1: prove every direct_to_validator record. requires_integration records
    # are DEFERRED by dispatch_validators — the auditor must run only AFTER these
    # are proven so its chain candidates carry real validator poc_payloads.
    workflow.add_conditional_edges(
        "validator_dispatch_gate",
        dispatch_validators,
        {
            "validator_agent": "validator_agent",
            "integration_audit_dispatch": "integration_audit_dispatch",
            # Never returned today, but the reporter must remain the sole sink.
            "__end__": "reporter",
        },
    )
    # Stage 2: from the validator (once its whole superstep finished) either bounce
    # insufficient_context records back to the Reviewer OR, when no feedback is
    # pending, advance into the integration-audit phase.
    workflow.add_conditional_edges(
        "validator_agent",
        route_validator_feedback,
        {
            "reviewer_agent": "reviewer_agent",
            "integration_audit_dispatch": "integration_audit_dispatch",
            "__end__": "reporter",
        },
    )
    # Stage 2b: fan the audited records to the auditor (or advance to the
    # Reporter when none remain — covers the no-direct-tasks / post-chain drain
    # cases, so report.md is always written).
    workflow.add_conditional_edges(
        "integration_audit_dispatch",
        dispatch_integration_audits,
        ["integration_auditor", "reporter"],
    )
    # Stage 3: chained records go to the Validator with the peers' proven
    # poc_payloads injected (see route_integration_audit); unchainable are
    # terminal. When nothing chains the pipeline advances to the Reporter.
    workflow.add_conditional_edges("integration_auditor", route_integration_audit, ["validator_agent", "reporter"])
    # Single terminal sink: the Reporter runs exactly once and the graph ends.
    workflow.add_edge("reporter", END)

    app = workflow.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
    app = app.with_config({"max_concurrency": settings.agents_concurrency})

    return app


compiled_reviewer_agent = compile_reviewer()
compiled_validator_agent = compile_validator()
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

    initial_state = MasterState(
        known_vulns=[],
        expert_tasks=[],
        sandbox_url=None,
        notes=[],
        cve_demands=[],
        grouped_demands={},
        vulnerabilities=[]
    )

    from langgraph.checkpoint.sqlite import SqliteSaver

    Path("states").mkdir(parents=True, exist_ok=True)

    config = {"configurable": {"thread_id": "scan-1"}}
    done_flag = Path("states/scan-complete.flag")

    with SqliteSaver.from_conn_string("states/pipeline_checkpoints.sqlite") as checkpointer:
        app = build_graph(checkpointer=checkpointer)
        try:
            if app.get_state(config).values and not done_flag.exists():
                logging.info("Resuming previously interrupted run from checkpoint.")
                final_state = app.invoke(None, config)
            else:
                final_state = app.invoke(initial_state, config)

            out_file = "results.json"
            with open(out_file, "w") as f:
                json.dump(final_state, f)
                logging.info(f"Final state saved to {out_file}.")
            done_flag.touch()

        except FileNotFoundError:
            print("Waiting for actual graph.json to execute.")
        except KeyboardInterrupt:
            done_flag.unlink(missing_ok=True)
            exit(1)
