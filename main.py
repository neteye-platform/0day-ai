import json
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
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
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
    print("--- [Node] Preprocessing Graphify Data ---")

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
    logger = get_master_logger()
    logger.info("Manager Agent Routing")

    # llm = ChatOllama(model="gemma4:26b", temperature=0, reasoning=True)
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
    logger.info(f"Strategy: {response.strategic_overview}")
    logger.info(f"Dispatching {len(response.tasks)} Expert Agents...")

    return {"expert_tasks": response.tasks}


def expert_agent_node(state: ExpertState) -> dict:
    role_name = state["task"].agent_role

    logger = get_agent_logger(role_name, state["task"].target_communities)

    # Safety catch for empty tasks
    if not state.get("subgraph_nodes"):
        logger.info(f"Aborted: No nodes to analyze.")
        return {"vulnerability_reports": []}

    # llm = ChatOllama(model="gemma4:26b", temperature=0, reasoning=True)
    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key=os.getenv("API_KEY"),
        model="mistral-3.5-128b",
        temperature=0,
        http_client=httpx.Client(verify=False)
    )
    llm_with_tools = llm.bind_tools([read_source_code, SubmitReport])

    # Check if we just came back from executing a tool
    if state.get("messages"):
        # LLMs execute tools in parallel. Iterate backwards to catch ALL new ToolMessages.
        recent_tool_msgs = []
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage):
                recent_tool_msgs.append(msg)
            else:
                break # Stop looking when we hit the AIMessage that requested the tools

        # Log them in chronological order
        for t_msg in reversed(recent_tool_msgs):
            logger.debug(f"--- Tool Execution Result [{t_msg.name} | {t_msg.tool_call_id}] ---")

            # Truncate massive source code dumps in the log to keep it readable
            content = t_msg.content
            if len(content) > 1000:
                logger.debug(f"{content[:1000]}\n... [TRUNCATED FOR LOGS]")
            else:
                logger.debug(f"{content}")

    if not state.get("messages"):
        logger.info("=== Booting Agent ===")
        sys_msg = SystemMessage(content=EXPERT_AGENTS[role_name]["prompt"])
        human_msg = HumanMessage(content=(
            f"Your Task: {state['task'].task_description}\n\n"
            f"Your Assigned Nodes: {state['subgraph_nodes']}\n\n"
            "Instructions:\n"
            "1. Analyze the context of your assigned nodes.\n"
            "2. Use the 'read_source_code' tool to investigate specific logic implementations. CRITICAL: Do not read more than 3 files at the same time.\n"
            "3. When you have found vulnerabilities OR finished your audit, you MUST call the 'SubmitReport' tool to output your findings."
        ))
        logger.info(f"User prompt: {human_msg.content}")

        messages = [sys_msg, human_msg]
        response = llm_with_tools.invoke(messages)
        messages = [sys_msg, human_msg, response]
    else:
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        messages = [response]

    # Log any tool calls the LLM is about to make
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            logger.info(f"LLM requested tool: {tc['name']} with args: {tc['args']}")

    if hasattr(response, "reasoning") and response.reasoning:
         logger.debug(f"LLM Thought Process: {response.reasoning}")

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
    logger = get_agent_logger(role_name, state["task"].target_communities)

    last_message = state["messages"][-1]
    reports = []

    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "SubmitReport":
            args = tool_call["args"]

            logger.info("=== Audit Completed ===")
            logger.info(f"Summary: {args.get('audit_summary', 'None')}")

            for finding in args.get("findings", []):
                logger.warning(f"VULNERABILITY FOUND: {finding.get('vulnerability_type')}")
                logger.warning(f"Nodes: {finding.get('affected_nodes')}")

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

# ==========================================
# Execution
# ==========================================
async def run_stream(app, inputs):
    final_state = None
    
    # Request BOTH messages (for token streaming) and values (for state updates)
    async for event_type, data in app.astream(
        inputs,
        stream_mode=["messages", "values"]
    ):
        if event_type == "messages":
            # Unpack the chunk and metadata
            chunk, metadata = data
            
            reasoning = chunk.additional_kwargs.get("reasoning_content", "")
            if reasoning:
                # Print reasoning tokens in gray (using ANSI escape codes)
                print(f"\033[90m{reasoning}\033[0m", end="", flush=True)
            elif chunk.content:
                # Print actual answer in default terminal color
                print(chunk.content, end="", flush=True)
            elif hasattr(chunk, "tool_calls") and chunk.tool_calls:
                for tool_call in chunk.tool_calls:
                    tool_name = tool_call.get("name", "UnknownTool")
                    tool_args = tool_call.get("args", {})

                    # Print the tool name in cyan
                    print(f"\n\033[96m[Tool Call Intercepted] -> {tool_name}\033[0m")
                    # Pretty-print the structured arguments
                    print(f"\033[96m{json.dumps(tool_args, indent=2)}\033[0m\n", flush=True)
                    
        elif event_type == "values":
            # The "values" stream yields the entire state dictionary every time a node finishes.
            # We continuously overwrite final_state so that when the loop finishes, 
            # it holds the absolute final state of the graph.
            final_state = data

    return final_state

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
        final_state = app.invoke(initial_state)
        # final_state = asyncio.run(run_stream(app, initial_state))

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
