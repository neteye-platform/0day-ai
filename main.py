import json
import sys
import os
import settings
from enum import Enum
import asyncio
import operator
from typing import TypedDict, List, Dict, Any, Annotated, Literal
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import httpx
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from typing import Annotated
from langgraph.graph.message import add_messages
from tools import *
from utils import *

# ==========================================
# State
# ==========================================

class MasterState(TypedDict):
    graph_path: str
    app_summary: str
    communities_map: Dict[str, List[str]] # Maps community ID to list of node IDs
    expert_tasks: List[ExpertTask]
    vulnerability_reports: Annotated[List[Dict[str, Any]], operator.add] # Aggregated findings

class ExpertState(TypedDict, total=False):
    task: ExpertTask
    subgraph_nodes: List[str]
    messages: Annotated[list, add_messages] # Tracks the conversation and tool calls
    vulnerability_reports: List[Dict[str, Any]]

# ==========================================
# Output Schemas
# ==========================================

with open("agents.json", "r") as f:
    EXPERT_AGENTS = json.load(f)

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
    to avoid overloading the LLM's context window.
    """
    G = build_networkx_graph(state["graph_path"])

    communities_map = {}
    for node_id, data in G.nodes(data=True):
        comm_id = str(data.get('community', 'unknown'))

        if comm_id not in communities_map:
            communities_map[comm_id] = []
        communities_map[comm_id].append(node_id)

    # Generate a lightweight summary for the Manager
    summary = (
        f"Application Topology Summary:\n"
        f"- Total Nodes (Files/Functions/Classes): {G.number_of_nodes()}\n"
        f"- Total Edges (Calls/Imports): {G.number_of_edges()}\n"
        f"- Number of distinct Communities (Modules): {len(communities_map)}\n\n"
        f"Community Breakdown:\n"
    )

    for comm_id, node_names in communities_map.items():
        # Just send a sample of nodes per community to save context
        sample_nodes = ", ".join(node_names[:5])
        summary += f"- Community ID {comm_id}: {len(node_names)} nodes. Samples: {sample_nodes}\n"

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
        api_key=os.getenv("API_KEY"),
        model="mistral-3.5-128b",
        temperature=0,
        http_client=httpx.Client(verify=False)
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

    return {"expert_tasks": response.tasks}


def expert_agent_node(state: ExpertState) -> dict:
    role_name = state["task"].agent_role

    # Safety catch for empty tasks
    if not state.get("subgraph_nodes"):
        return {"vulnerability_reports": []}

    # llm = ChatOllama(model="gemma4:26b", temperature=0)
    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key=os.getenv("API_KEY"),
        model="mistral-3.5-128b",
        temperature=0,
        http_client=httpx.Client(verify=False)
    )
    llm_with_tools = llm.bind_tools([read_source_code, SubmitReport])

    if not state.get("messages"):
        sys_msg = SystemMessage(content=EXPERT_AGENTS[role_name]["prompt"])
        human_msg = HumanMessage(content=(
            f"Your Task: {state['task'].task_description}\n\n"
            f"Your Assigned Nodes: {state['subgraph_nodes']}\n\n"
            "Instructions:\n"
            "1. Analyze the context of your assigned nodes.\n"
            "2. Use the 'read_source_code' tool to investigate specific logic implementations. CRITICAL: Do not read more than 3 files at the same time.\n"
            "3. When you have found vulnerabilities OR finished your audit, you MUST call the 'SubmitReport' tool to output your findings."
        ))

        messages = [sys_msg, human_msg]
        response = llm_with_tools.invoke(messages)
        messages = [sys_msg, human_msg, response]
    else:
        messages = state["messages"]
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
expert_workflow.add_node("execute_tools", ToolNode([read_source_code]))
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
        # Run the async stream and capture the returned final state
        # final_state = asyncio.run(run_stream(app, initial_state))
        if "-v" in sys.argv:
            final_state = run_stream(app, initial_state)
        elif "-vv" in sys.argv:
            final_state = run_stream(app, initial_state, vv=True)
        else:
            final_state = app.invoke(initial_state)

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
