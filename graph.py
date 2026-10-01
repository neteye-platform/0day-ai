import json
import logging
import argparse
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy

import settings
import tools
from nodes import bootstrap_node, preprocessor_node, manager_agent_node, expert_explorer_node, cve_analyzer_node, reviewer_agent_node, ask_reviewer_for_tool, dispatch_explorers, dispatch_cve_analyzers, dispatch_reviewers, dispatch_validators, dispatch_verifiers, reviewer_router, validator_agent_node, ask_validator_for_tool, validator_router, aggregate_demands_node, contract_verifier_node, synchronization_node
from reachability import reachability_filter_node
from state import MasterState, ReviewerState, ValidatorState
from schemas import ReviewerOutput, ValidatorOutput


def compile_reviewer():
    reviewer_workflow = StateGraph(ReviewerState, output_schema=ReviewerOutput)
    reviewer_workflow.add_node("reviewer_agent", reviewer_agent_node)
    reviewer_workflow.add_node("ask_reviewer_for_tool", ask_reviewer_for_tool)
    reviewer_workflow.add_node("reviewer_tools", ToolNode([
        tools.read_source_code,
        tools.read_file,
        tools.get_node_connections,
        tools.search_codebase,
        tools.get_definition,
        tools.submit_evaluation
    ]))
    reviewer_workflow.add_edge(START, "reviewer_agent")
    reviewer_workflow.add_conditional_edges(
        "reviewer_agent",
        reviewer_router,
        {
            "reviewer_tools": "reviewer_tools",
            "ask_reviewer_for_tool": "ask_reviewer_for_tool",
            "__end__": END
        }
    )
    reviewer_workflow.add_conditional_edges(
        "reviewer_tools",
        reviewer_router,
        {
            "reviewer_agent": "reviewer_agent",
            "__end__": END
        }
    )
    reviewer_workflow.add_edge("ask_reviewer_for_tool", "reviewer_agent")
    compiled_reviewer_agent = reviewer_workflow.compile()

    return compiled_reviewer_agent


def compile_validator():
    validator_workflow = StateGraph(ValidatorState)
    validator_workflow.add_node("validator_agent", validator_agent_node)
    validator_workflow.add_node("ask_validator_for_tool", ask_validator_for_tool)
    validator_workflow.add_node("validator_tools", ToolNode([
        tools.send_http_request,
        tools.list_files,
        tools.read_sandbox_file,
        tools.mark_validation_complete
    ]))
    validator_workflow.add_edge(START, "validator_agent")
    validator_workflow.add_conditional_edges(
        "validator_agent",
        validator_router,
        {
            "validator_tools": "validator_tools",
            "ask_validator_for_tool": "ask_validator_for_tool"
        }
    )
    validator_workflow.add_conditional_edges(
        "validator_tools",
        validator_router,
        {
            "validator_agent": "validator_agent",
            "__end__": END
        }
    )
    validator_workflow.add_edge("ask_validator_for_tool", "validator_agent")
    compiled_validator_agent = validator_workflow.compile()

    return compiled_validator_agent 


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
    workflow.add_node("manager", manager_agent_node)
    workflow.add_node("explorer_agent", expert_explorer_node)
    workflow.add_node("cve_analyzer", cve_analyzer_node)
    workflow.add_node("aggregate_demands", aggregate_demands_node)
    workflow.add_node("contract_verifier", contract_verifier_node)
    workflow.add_node("reviewer_agent", compiled_reviewer_agent)
    workflow.add_node("validator_agent", compiled_validator_agent)
    workflow.add_node("synchronization", synchronization_node)
    workflow.add_node("reachability_filter", reachability_filter_node)
    # workflow.add_node("reviewer_sync", synchronization_node)

    workflow.add_edge(START, "bootstrap")
    workflow.add_edge("bootstrap", "preprocessor")
    workflow.add_edge("bootstrap", "manager")

    workflow.add_conditional_edges("manager", dispatch_explorers, ["explorer_agent"])
    workflow.add_conditional_edges("preprocessor", dispatch_cve_analyzers, ["cve_analyzer"])

    workflow.add_edge("explorer_agent", "aggregate_demands")
    workflow.add_edge("cve_analyzer", "aggregate_demands")

    workflow.add_conditional_edges("aggregate_demands", dispatch_verifiers, ["contract_verifier", "synchronization", END])
    workflow.add_edge("contract_verifier", "synchronization")
    workflow.add_edge("synchronization", "reachability_filter")
    workflow.add_conditional_edges("reachability_filter", dispatch_reviewers, ["reviewer_agent", END])
    # workflow.add_edge("reviewer_agent", "reviewer_sync")
    workflow.add_conditional_edges("reviewer_agent", dispatch_validators, ["validator_agent", END])
    workflow.add_edge("validator_agent", END)

    app = workflow.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
    app = app.with_config({"max_concurrency": settings.simple_agents_concurrency})

    return app


compiled_reviewer_agent = compile_reviewer()
compiled_validator_agent = compile_validator()
graph = build_graph()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Vulnerability Analyzer")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging.")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(module)s - %(message)s",
        datefmt="%H:%M:%S"
    )

    initial_state = MasterState(
        graph={},
        app_summary="",
        known_vulns=[],
        expert_tasks=[],
        sandbox_url=None,
        container_name=None,
        notes=[],
        cve_demands=[],
        grouped_demands={},
        vulnerability_hypothesis=[],
        confirmed_vulnerabilities=[]
    )

    try:
        final_state = graph.invoke(initial_state)

        out_file = "results.json"
        with open(out_file, "w") as f:
            json.dump(final_state, f)
            logging.info(f"Final state saved to {out_file}.")

    except FileNotFoundError:
        print("Waiting for actual graph.json to execute.")
    except KeyboardInterrupt:
        exit(1)
