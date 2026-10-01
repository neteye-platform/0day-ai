import json
import os
import settings
import operator
import logging
import argparse
from typing import TypedDict, List, Dict, Any, Annotated, Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import tools
from utils import *

import warnings
warnings.filterwarnings("ignore", message=".*allowed_objects.*")
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))

logger = logging.getLogger(__name__)

# ==========================================
# State
# ==========================================

class MasterState(TypedDict):
    graph_path: str
    app_summary: str
    communities_map: Dict[str, List[str]] # Maps community ID to list of node IDs
    expert_tasks: List[ExpertTask]
    vulnerability_reports: Annotated[List[Dict[str, Any]], operator.add] # Aggregated findings
    filtered_reports: list[dict]

class ExpertState(TypedDict):
    task: ExpertTask
    subgraph_nodes: List[str]
    messages: Annotated[list, add_messages] # Tracks the conversation and tool calls
    vulnerability_reports: List[Dict[str, Any]]

# ==========================================
# Output Schemas
# ==========================================

with open("agents.json", "r") as f:
    data = json.load(f)
    EXPERT_AGENTS = data.get("agents")
    TOOLS = data.get("tools")

class ExpertTask(BaseModel):
    agent_role: str = Field(
        description="The PREDEFINED role best suited for this specific architectural area.",
        json_schema_extra={"enum": list(EXPERT_AGENTS.keys())}
    )
    target_communities: List[str] = Field(
        description="List of EXACT community IDs this agent must focus on (e.g., ['1', '3', '4']). Extract these exact IDs from the summary. It MUST NOT be empty."
    )
    task_description: str = Field(
        description="Detailed instructions on what specific vulnerability classes, architectural risks, or cross-component interactions to investigate within this subgraph."
    )

class ManagerOutput(BaseModel):
    strategic_overview: str = Field(description="The manager's brief (max 200 words) reasoning on the app's attack surface.")
    tasks: List[ExpertTask] = Field(description="List of tasks matching predefined roles.")

class VulnerabilityReport(BaseModel):
    vulnerability_type: str = Field(description="Type of logic bug or misconfiguration found.")
    description: str = Field(description="Detailed explanation of the flaw.")
    affected_nodes: List[str] = Field(description="List of node IDs involved in the vulnerability.")

class SubmitReport(BaseModel):
    findings: List[VulnerabilityReport]
    audit_summary: str = Field(description="Brief summary of what was checked.")

class VulnerabilityEvaluation(BaseModel):
    report_id: str = Field(description="The unique identifier or title of the vulnerability report.")
    is_exploitable: bool = Field(
        description="True if the vulnerability has a realistic path to exploitation. False if it is a false positive, purely theoretical, or blocked by standard mitigations."
    )
    confidence_score: int = Field(description="Confidence in this assessment from 1 to 10.")
    reasoning: str = Field(description="Brief technical explanation for the decision.")

class ReviewerOutput(BaseModel):
    evaluations: List[VulnerabilityEvaluation]

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

    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        model="mistral-3.5-128b",
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
    return {"expert_tasks": response.tasks}


def expert_agent_node(state: ExpertState) -> dict:
    role_name = state["task"].agent_role
    logger.debug(f"Entering expert_agent_node for role: {role_name}")

    if not state.get("subgraph_nodes"):
        logger.debug(f"No subgraph nodes assigned to {role_name}. Bypassing execution.")
        return {"vulnerability_reports": []}

    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        model="mistral-3.5-128b",
        temperature=0
    )

    agent_tools = [SubmitReport]
    tool_names = EXPERT_AGENTS[role_name].get("tools", [])
    for name in tool_names:
        if hasattr(tools, name):
            agent_tools.append(getattr(tools, name))

    llm_with_tools = llm.bind_tools(agent_tools)

    if not state.get("messages"):
        logger.debug(f"Initializing new conversation for {role_name}.")
        tool_rules = ""
        for tool_name in EXPERT_AGENTS[role_name].get("tools", []):
            rule = TOOLS[tool_name].get("rule", "")
            tool_rules += f"- {tool_name}: {rule}\n"

        sys_msg_content = (
            f"{EXPERT_AGENTS[role_name]['prompt']}\n\n"
            f"Operational Rules:\n{tool_rules}\n"
            "When you have found vulnerabilities OR finished your audit, you MUST call the 'SubmitReport' tool to output your findings."
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

        # Strip dynamically generated IDs for cache
        for msg in messages:
            msg.id = None

        response = llm_with_tools.invoke(messages)
        messages = [response]

    logger.debug(f"Expert LLM finished generating response for {role_name}.")
    return {"messages": messages}


def expert_router(state: ExpertState) -> Literal["execute_tools", "save_report", "__end__"]:
    """Routes the sub-graph based on which tool the LLM decided to call."""
    last_message = state["messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        logger.debug(f"Router ending execution: No tool calls found in the last message.")
        return "__end__"

    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "SubmitReport":
            logger.debug("Router directing to 'save_report_node'.")
            return "save_report"

    logger.debug(f"Router directing to 'execute_tools' for tools: {[tc['name'] for tc in last_message.tool_calls]}")
    return "execute_tools"


def save_report_node(state: ExpertState) -> dict:
    """Intercepts the SubmitReport tool call, formats it, and prepares it for the Master Graph."""
    role_name = state["task"].agent_role
    logger.debug(f"Entering save_report_node for role: {role_name}")

    last_message = state["messages"][-1]
    reports = []

    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "SubmitReport":
            args = tool_call["args"]

            for finding in args.get("findings", []):
                reports.append({
                    "role": role_name,
                    "vulnerability": finding.get("vulnerability_type"),
                    "details": finding.get("description"),
                    "nodes": finding.get("affected_nodes")
                })
                logger.debug(f"Saved finding: {finding.get('vulnerability_type')} by {role_name}")

    return {"vulnerability_reports": reports}


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
    reports = reports[:1] # Analyze only the first one for testing the node

    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        model="mistral-3.5-128b",
        temperature=0
    )
    parser = PydanticOutputParser(pydantic_object=ReviewerOutput)

    # Format reports with a generated ID so the LLM can reference them reliably
    formatted_reports = ""
    for i, report in enumerate(reports):
        rep_id = f"report_{i}"
        formatted_reports += (
            f"--- Report ID: {rep_id} ---\n"
            f"Found By: {report.get('role', 'Unknown')}\n"
            f"Vulnerability: {report.get('vulnerability', 'Unknown')}\n"
            f"Affected Nodes: {', '.join(report.get('nodes', []))}\n"
            f"Details: {report.get('details', '')}\n\n"
        )

    sys_msg = SystemMessage(content=(
        "You are a Senior Application Security Reviewer. "
        "Analyze the vulnerability reports generated by automated Expert Agents. "
        "Your job is to filter out the noise and identify truly exploitable vulnerabilities.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- Reject reports (is_exploitable=False) if they are false positives, require unrealistic prerequisites, "
        "are blocked by standard framework mitigations, or exist in unreachable code.\n"
        "- You must provide an evaluation for EVERY report provided in the input.\n"
        "- Use the exact 'Report ID' provided in the human message for your evaluation (e.g., 'report_0').\n\n"
        f"{parser.get_format_instructions()}"
    ))

    human_msg = HumanMessage(content=f"Here are the vulnerability reports to evaluate:\n{formatted_reports}")

    response_msg = llm.invoke([sys_msg, human_msg])
    response = parser.invoke(response_msg)

    # Map the LLM's evaluations back to the original report dictionaries
    evaluations_by_id = {eval.report_id: eval for eval in response.evaluations}
    valid_reports = []

    for i, report in enumerate(reports):
        rep_id = f"report_{i}"
        evaluation = evaluations_by_id.get(rep_id)

        # Keep only if marked exploitable with a high confidence score
        if evaluation and evaluation.is_exploitable and evaluation.confidence_score >= 7:
            # We preserve your exact original dictionary and append the reviewer's context
            validated_report = report.copy()
            validated_report.update({
                "reviewer_reasoning": evaluation.reasoning,
                "confidence_score": evaluation.confidence_score
            })
            valid_reports.append(validated_report)

    logger.debug(f"Reviewer kept {len(valid_reports)} out of {len(reports)} reports.")
    return {"filtered_reports": valid_reports}

# ==========================================
# Build and Compile the Graph
# ==========================================

# Force sequential execution
import threading
expert_lock = threading.Lock()
def expert_agent_wrapper(state, config: RunnableConfig):
    with expert_lock:
        # Create a fresh copy of the config so we don't mutate the master graph's config
        child_config = config.copy()
        # Remove the concurrency limit so the sub-graph has room to execute
        child_config.pop("max_concurrency", None)
        # Invoke the compiled sub-graph manually
        return compiled_expert_agent.invoke(state, child_config)

# Expert Sub-Graph
expert_workflow = StateGraph(ExpertState)
expert_workflow.add_node("expert", expert_agent_node)
expert_workflow.add_node("execute_tools", ToolNode([tools.read_source_code]))
expert_workflow.add_node("save_report", save_report_node)
expert_workflow.add_edge(START, "expert")
expert_workflow.add_conditional_edges("expert", expert_router)
expert_workflow.add_edge("execute_tools", "expert")
expert_workflow.add_edge("save_report", END)
compiled_expert_agent = expert_workflow.compile()

# Master Graph
workflow = StateGraph(MasterState)
workflow.add_node("preprocessor", preprocessor_node)
workflow.add_node("manager", manager_agent_node)
# workflow.add_node("expert_agent", compiled_expert_agent)
workflow.add_node("expert_agent", expert_agent_wrapper)
workflow.add_node("reviewer", reviewer_node)
workflow.add_edge(START, "preprocessor")
workflow.add_edge("preprocessor", "manager")
workflow.add_conditional_edges("manager", dispatch_experts, ["expert_agent"])
workflow.add_edge("expert_agent", "reviewer")
workflow.add_edge("reviewer", END)
app = workflow.compile()

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

    initial_state = MasterState(
        graph_path=os.path.join(settings.app_path, "graphify-out/graph.json"),
        app_summary="",
        communities_map={},
        expert_tasks=[]
    )

    try:
        final_state = run_stream(app, initial_state)

        # Print the aggregated findings
        print("\n\n" + "="*60)
        print("🛡️  FINAL VULNERABILITY AUDIT REPORT")
        print("="*60)

        reports = final_state.get("vulnerability_reports", [])

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
