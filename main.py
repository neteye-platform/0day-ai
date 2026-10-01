import os
import settings
import logging
import argparse
from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

import tools
from utils import build_networkx_graph, run_stream
from state import MasterState, ExpertState
from schemas import ExpertTask, VulnerabilityEvaluation, VulnerabilityReport, ManagerOutput, ReviewerOutput, EXPERT_AGENTS, REVIEWER_AGENT, TOOLS

logger = logging.getLogger(__name__)

# ==========================================
# Nodes
# ==========================================

def preprocessor_node(state: MasterState) -> Dict[str, Any]:
    """Reads graph.json, builds a NetworkX graph, and summarizes it for the manager node."""
    logger.debug(f"Entering preprocessor_node. Reading graph from: {state['graph_path']}")

    G = build_networkx_graph(state["graph_path"])

    # Group nodes by community
    communities_map = {}
    for node_id, data in G.nodes(data=True):
        comm_id = str(data.get('community', 'unknown'))
        if comm_id not in communities_map:
            communities_map[comm_id] = []
        communities_map[comm_id].append((node_id, data))

    # Generate a summary for the Manager
    summary = (
        f"Application Topology Summary:\n"
        f"- Total Nodes: {G.number_of_nodes()} | Total Edges: {G.number_of_edges()}\n"
        f"- Total Communities: {len(communities_map)}\n\n"
        f"Community Breakdown:\n"
    )

    for comm_id, nodes_data in communities_map.items():
        node_ids = [n[0] for n in nodes_data]

        # Node Types
        node_types = {"functions": 0, "classes/files": 0, "docs/other": 0}
        for _, data in nodes_data:
            f_type = data.get('file_type', 'unknown')
            label = data.get('label', '')
            if f_type == 'code':
                if '()' in label:
                    node_types["functions"] += 1
                else:
                    node_types["classes/files"] += 1
            else:
                node_types["docs/other"] += 1
        type_str = ", ".join([f"{v} {k}" for k, v in node_types.items() if v > 0])

        # Find the "God Node" (the most connected file/function in this community)
        subgraph = G.subgraph(node_ids)
        if len(subgraph.nodes) > 0:
            degrees = dict(subgraph.degree())
            god_node = max(degrees, key=degrees.__getitem__)
        else:
            god_node = "None"

        summary += (
            f"- Community {comm_id} ({len(node_ids)} nodes): [{type_str}]\n"
            f"  -> Central Hub Node: {god_node}\n"
        )
        communities_map[comm_id] = node_ids

    logger.debug(f"Preprocessor complete. Found {len(communities_map)} communities.")
    return {
        "app_summary": summary,
        "communities_map": communities_map
    }


def manager_agent_node(state: MasterState) -> Dict[str, Any]:
    """The Manager LLM reads the programmatic summary and dispatches tasks."""
    logger.debug("Entering manager_agent_node. Invoking Lead Security Architect LLM.")

    # llm = ChatOllama(model="qwen3.6:35b", temperature=0)
    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        model="glm-5-2-3-bit",
        temperature=0
    )
    parser = PydanticOutputParser(pydantic_object=ManagerOutput)

    roles_docs = "\n".join([
        f"- {role}: {config['manager_description']}"
        for role, config in EXPERT_AGENTS.items()
    ])
    sys_msg = SystemMessage(content=(
        "You are the Lead Security Architect. Analyze the topology summary. "
        "Assign communities to one or more appropriate PREDEFINED Expert Agents. "
        "You may ONLY assign roles from the following list based on their capabilities:\n\n"
        f"{roles_docs}\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- You must populate the 'target_communities' array for every task with the exact Community IDs (as strings, e.g., '0', '1') provided in the topology summary. Never leave the 'target_communities' array empty.\n"
        "- Do not assign more than 3 communities to a single task. If a complex logic flow spans, for example, 7 communities, break it down into overlapping tasks (e.g., Task 1: Comm 6,7,8. Task 2: Comm 8,9,10). This prevents context overload."
        f"{parser.get_format_instructions()}"
    ))
    human_msg = HumanMessage(content=f"Here is the app topology:\n{state.get('app_summary')}")

    response_msg = llm.invoke([sys_msg, human_msg])
    response = parser.invoke(response_msg)

    logger.debug(f"Manager created {len(response.tasks)} expert tasks.")
    return {"expert_tasks": response.tasks, "messages": response_msg}


def expert_agent_node(state: ExpertState) -> dict:
    role_name = state["task"].agent_role
    logger.debug(f"Entering expert_agent_node for role: {role_name}")

    if not state.get("subgraph_nodes"):
        logger.debug(f"No subgraph nodes assigned to {role_name}. Bypassing execution.")
        return {"vulnerability_reports": []}

    # llm = ChatOllama(model="qwen3.6:35b", temperature=0)
    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        model="glm-5-2-3-bit",
        temperature=0
    )

    agent_tools = [tools.submit_report]
    tool_names = EXPERT_AGENTS[role_name].get("tools", [])
    for name in tool_names:
        if hasattr(tools, name):
            agent_tools.append(getattr(tools, name))

    llm_with_tools = llm.bind_tools(agent_tools, tool_choice="any")

    if not state.get("messages"):
        logger.debug(f"Initializing new conversation for {role_name}.")
        tool_rules = ""
        for tool_name in EXPERT_AGENTS[role_name].get("tools", []):
            rule = TOOLS[tool_name].get("rule", "")
            tool_rules += f"- {tool_name}: {rule}\n"

        sys_msg_content = (
            f"{EXPERT_AGENTS[role_name]['prompt']}\n\n"
            f"### Operational Rules\n\n{tool_rules}\n"
        )
        sys_msg = SystemMessage(content=sys_msg_content)

        human_msg = HumanMessage(content=(
            f"Your Task: {state['task'].task_description}\n\n"
            f"Your Assigned Nodes: {state['subgraph_nodes']}\n\n"
        ))

        messages = [sys_msg, human_msg]
        response = llm_with_tools.invoke(messages)
        messages = [sys_msg, human_msg, response]
    else:
        logger.debug(f"Continuing existing conversation for {role_name}. Message count: {len(state['messages'])}")
        messages = state["messages"]

        response = llm_with_tools.invoke(messages)
        messages = [response]

    logger.debug(f"Expert LLM finished generating response for {role_name}.")
    return {"messages": messages}


def expert_agent_router(state: ExpertState) -> str:
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        # Check if it called the termination tool
        for tc in last_message.tool_calls:
            if tc["name"] == "mark_task_complete":
                return "__end__"
            elif tc["name"] == "submit_report":
                return "save_report"

        # Go to tools node
        return "tools"

    # The LLM output plain text but DID NOT call a tool!
    # Instead of ending, we route to a "nagging" node.
    return "nag_agent"


def save_report_node(state: ExpertState) -> dict:
    """Intercepts the submit_report tool call to save findings to the graph state."""
    role_name = state["task"].agent_role
    logger.debug(f"Entering save_report_node for role: {role_name}")

    last_message = state["messages"][-1]
    reports = []
    tool_responses = []

    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "submit_report":
            args = tool_call["args"]
            finding_data = args.get("finding", args)

            reports.append({
                "role": role_name,
                "vulnerability": finding_data.get("cwe_class", "Unknown"),
                "details": finding_data.get("details", ""),
                "sink_node": finding_data.get("sink_node", ""),
                "trace_nodes": finding_data.get("trace_nodes", [])
            })
            logger.debug(f"Saved finding by {role_name}")

            tool_responses.append(
                ToolMessage(
                    content=f"Successfully saved finding. Please continue your audit.",
                    tool_call_id=tool_call["id"]
                )
            )

    return {
        "vulnerability_reports": reports, 
        "messages": tool_responses
    }


def nag_agent_node(state: ExpertState):
    """If the agent tries to chat instead of working, hit it with a system prompt."""
    nag_message = HumanMessage(
        content=f"You did not invoke any tools. You must either use `{'`, `'.join(TOOLS.keys())}` to proceed."
    )
    return {"messages": [nag_message]}


def dispatch_experts(state: MasterState):
    """Reads the Manager's instructions and creates a list of 'Send' objects."""
    logger.debug("Entering dispatch_experts. Preparing isolated threads for assigned tasks.")

    commands: List[Send] = []
    for task in state["expert_tasks"]:
        nodes_for_task = []
        for comm_id in task.target_communities:
            clean_id = comm_id.lower().replace("community ", "").strip()
            nodes_for_task.extend(state["communities_map"].get(clean_id, []))

        logger.debug(f"Dispatching task to '{task.agent_role}' spanning communities {task.target_communities} ({len(nodes_for_task)} nodes).")

        payload = ExpertState(
            task=task,
            subgraph_nodes=nodes_for_task
        )
        commands.append(Send("expert_agent", payload))

    return commands


def reviewer_node(state: MasterState):
    """Review the vulnerability reports and keep only what is actually relevant"""
    logger.debug("Entering reviewer_node.")

    reports = state.get("vulnerability_reports", [])
    if not reports:
        logger.debug("No reports to review. Skipping.")
        return {"filtered_reports": []}

    # llm = ChatOllama(model="qwen3.6:35b", temperature=0)
    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        model="glm-5-2-3-bit",
        temperature=0
    )
    parser = PydanticOutputParser(pydantic_object=VulnerabilityEvaluation)

    sys_msg = SystemMessage(content=(
        f"{REVIEWER_AGENT.get('prompt')}\n\n"
        f"{parser.get_format_instructions()}"
    ))

    valid_reports = []
    for report in reports:
        human_msg = HumanMessage(content=f"Vulnerability report to evaluate:\n{report}")

        response_msg = llm.invoke([sys_msg, human_msg])
        response = parser.invoke(response_msg)

        if response.is_exploitable:
            valid_reports.append(response)

    logger.debug(f"Reviewer kept {len(valid_reports)} out of {len(reports)} reports.")
    return {"filtered_reports": valid_reports}

# ==========================================
# Build and Compile the Graph
# ==========================================

def build_graph(checkpointer=None, interrupt_before=None):

    # Expert Sub-Graph
    expert_workflow = StateGraph(ExpertState)
    expert_workflow.add_node("expert", expert_agent_node)
    expert_workflow.add_node("tools", ToolNode([tools.submit_report, tools.read_source_code, tools.check_package_vulnerability]))
    expert_workflow.add_node("save_report", save_report_node)
    expert_workflow.add_node("nag_agent", nag_agent_node)
    expert_workflow.add_edge(START, "expert")
    expert_workflow.add_conditional_edges("expert", expert_agent_router)
    expert_workflow.add_edge("tools", "expert")
    expert_workflow.add_edge("save_report", "expert")
    expert_workflow.add_edge("nag_agent", "expert")
    compiled_expert_agent = expert_workflow.compile()

    # Master Graph
    workflow = StateGraph(MasterState)
    workflow.add_node("preprocessor", preprocessor_node)
    workflow.add_node("manager", manager_agent_node)
    workflow.add_node("expert_agent", compiled_expert_agent)
    # workflow.add_node("expert_agent", expert_agent_wrapper)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_edge(START, "preprocessor")
    workflow.add_edge("preprocessor", "manager")
    workflow.add_conditional_edges("manager", dispatch_experts, ["expert_agent"])
    workflow.add_edge("expert_agent", "reviewer")
    workflow.add_edge("reviewer", END)
    app = workflow.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)

    return app

# ==========================================
# Execution
# ==========================================
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

    app = build_graph()
    initial_state = MasterState(
        graph_path=os.path.join(settings.app_path, "graphify-out/graph.json"),
        app_summary="",
        communities_map={},
        expert_tasks=[],
        vulnerability_reports=[],
        filtered_reports=[],
        messages=[]
    )

    try:
        final_state = run_stream(app, initial_state)

        # Print the aggregated findings
        print("\n\n" + "="*60)
        print("🛡️  FINAL VULNERABILITY AUDIT REPORT")
        print("="*60)

        reports = final_state.get("filtered_reports", [])

        if not reports:
            print("No vulnerabilities reported by the expert agents.")
        else:
            for idx, report in enumerate(reports, 1):
                print(f"\n[{idx}] Found by: {report.get('role')}")
                print(f"    Type:     {report.get('vulnerability')}")
                print(f"    Nodes:    {', '.join(report.get('nodes', []))}")
                print(f"    Details:  {report.get('details')}")
                print("-" * 60)

    except FileNotFoundError:
        print("Waiting for actual graph.json to execute.")
    except KeyboardInterrupt:
        exit(1)
