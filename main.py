import json
import sys
import os
import settings
import operator
from typing import TypedDict, List, Dict, Any, Annotated, Literal
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Annotated
from langgraph.graph.message import add_messages
import tools
from utils import *

import warnings
warnings.filterwarnings("ignore", message=".*allowed_objects.*")

from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))

# ==========================================
# State
# ==========================================

class MasterState(TypedDict):
    graph_path: str
    app_summary: str
    communities_map: Dict[str, List[str]] # Maps community ID to list of node IDs
    expert_tasks: List[ExpertTask]
    vulnerability_reports: Annotated[List[Dict[str, Any]], operator.add] # Aggregated findings

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
    # thought_process: str = Field(description="Step-by-step reasoning on why certain communities are vulnerable and how to distribute tasks.")
    strategic_overview: str = Field(description="The manager's brief (max 200 words) reasoning on the app's attack surface.")
    tasks: List[ExpertTask] = Field(description="List of tasks matching predefined roles.")

class VulnerabilityReport(BaseModel):
    vulnerability_type: str = Field(description="Type of logic bug or misconfiguration found.")
    description: str = Field(description="Detailed explanation of the flaw.")
    affected_nodes: List[str] = Field(description="List of node IDs involved in the vulnerability.")

class SubmitReport(BaseModel):
    findings: List[VulnerabilityReport]
    audit_summary: str = Field(description="Brief summary of what was checked.")

# ==========================================
# Nodes
# ==========================================

def preprocessor_node(state: MasterState) -> Dict[str, Any]:
    """
    Reads graph.json, builds a NetworkX graph, and summarizes it
    for the manager node.
    """
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

        # Build a highly dense, low-token summary block
        summary += (
            f"- Community {comm_id} ({len(node_ids)} nodes): [{type_str}]\n"
            f"  -> Central Hub Node: {god_node}\n"
        )

        # Store the list of IDs in the state map for routing later
        communities_map[comm_id] = node_ids

    return {
        "app_summary": summary,
        "communities_map": communities_map
    }


def manager_agent_node(state: MasterState) -> Dict[str, Any]:
    """
    The Manager LLM reads the programmatic summary and dispatches tasks.
    """

    # llm = ChatOllama(model="gemma4:26b", temperature=0)
    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        model="mistral-3.5-128b",
        temperature=0
    )
    structured_llm = llm.with_structured_output(ManagerOutput)

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
    ))
    human_msg = HumanMessage(content=f"Here is the app topology:\n{state.get('app_summary')}")
    response = structured_llm.invoke([sys_msg, human_msg])
    assert isinstance(response, ManagerOutput), "LLM failed to return structured output!"

    return {"expert_tasks": response.tasks}


def expert_agent_node(state: ExpertState) -> dict:
    role_name = state["task"].agent_role

    # Safety catch for empty tasks
    if not state.get("subgraph_nodes"):
        return {"vulnerability_reports": []}

    # llm = ChatOllama(model="gemma4:26b", temperature=0)
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
        tool_rules = ""
        for tool_name in EXPERT_AGENTS[role_name].get("tools", []):
            rule = TOOLS[tool_name].get("rule", "")
            tool_rules += f"- {tool_name}: {rule}\n"

        sys_msg_content = (
            f"{EXPERT_AGENTS[role_name]["prompt"]}\n\n"
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
        messages = state["messages"]

        # Strip dynamically generated IDs for cache
        for msg in messages:
            msg.id = None

        response = llm_with_tools.invoke(messages)
        messages = [response]

    return {"messages": messages}


def expert_router(state: ExpertState) -> Literal["execute_tools", "save_report", "__end__"]:
    """Routes the sub-graph based on which tool the LLM decided to call."""
    last_message = state["messages"][-1]

    # If LLM didn't call tools, end the loop (Safety catch)
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "__end__"

    # Check what tool was called
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "SubmitReport":
            return "save_report"

    # If it's not SubmitReport, it must be read_source_code
    return "execute_tools"


def save_report_node(state: ExpertState) -> dict:
    """Intercepts the SubmitReport tool call, formats it, and prepares it for the Master Graph."""
    role_name = state["task"].agent_role

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

    return {"vulnerability_reports": reports}


def dispatch_experts(state: MasterState):
    """
    This edge reads the Manager's instructions and creates a list of 'Send' objects.
    LangGraph will execute all returned Send objects concurrently in parallel threads.
    """
    commands: List[Send] = []
    for task in state["expert_tasks"]:
        # Extract the node IDs belonging to the targeted communities
        nodes_for_task = []
        for comm_id in task.target_communities:
            clean_id = comm_id.lower().replace("community ", "").strip()
            nodes_for_task.extend(state["communities_map"].get(clean_id, []))

        # Create an isolated sub-state payload for this specific worker
        payload = ExpertState(
            task=task,
            subgraph_nodes=nodes_for_task
        )
        # Instruct LangGraph to send this payload to the 'expert_agent' node
        commands.append(Send("expert_agent", payload))

    return commands


# ==========================================
# Build and Compile the Graph
# ==========================================

# Expert Sub-Graph
expert_workflow = StateGraph(ExpertState)

expert_workflow.add_node("expert", expert_agent_node)
expert_workflow.add_node("execute_tools", ToolNode([tools.read_source_code]))
expert_workflow.add_node("save_report", save_report_node)

expert_workflow.add_edge(START, "expert")
expert_workflow.add_conditional_edges("expert", expert_router)
expert_workflow.add_edge("execute_tools", "expert")    # Loop back to LLM after reading code
expert_workflow.add_edge("save_report", END)           # Exit Sub-Graph after submission

# Compile the Sub-Graph
compiled_expert_agent = expert_workflow.compile()

# ==========================================

# Initialize the state graph
workflow = StateGraph(MasterState)

# Add nodes
workflow.add_node("preprocessor", preprocessor_node)
workflow.add_node("manager", manager_agent_node)
workflow.add_node("expert_agent", compiled_expert_agent)

# Define edges
workflow.add_edge(START, "preprocessor")
workflow.add_edge("preprocessor", "manager")
workflow.add_conditional_edges(
    "manager",           # From the manager
    dispatch_experts,    # Run this function to determine where to go
    ["expert_agent"]     # The potential destinations
)
workflow.add_edge("expert_agent", END)

# Compile the graph
app = workflow.compile()

# # Print the graph with Mermaid syntax
# print(app.get_graph(xray=1).draw_ascii())
# exit()

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    os.makedirs("agent_logs", exist_ok=True)
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
        
        # Now we safely extract reports from the captured final state
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
