import json
import os
import settings
import logging
import argparse
import hashlib
from typing import Any
from collections import defaultdict
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

import tools
from utils import build_networkx_graph, run_stream, compact_tool_history, format_notes, get_node_source_code
from state import MasterState, ExpertState, ReviewerState, ValidatorState
from schemas import ExpertTask, ManagerOutput, MANAGER_AGENT, EXPERT_AGENTS, REVIEWER_AGENT, VALIDATOR_AGENT, TOOLS, TakeNoteInput

# ==========================================
# Preprocessor
# ==========================================

def preprocessor_node(state: MasterState) -> dict[str, Any]:
    """Reads graph.json, builds a NetworkX graph, and summarizes it for the manager node."""

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

    counter = 0
    max_comm = 50

    sorted_communities = sorted(
        communities_map.items(),
        key=lambda x: int(x[0]) if x[0].isdigit() else x[0]
    )

    for comm_id, nodes_data in sorted_communities:
        counter += 1
        if counter > max_comm:
            logging.warning("Truncated the number of communities of this application")
            break

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
            god_node = max(degrees, key=lambda x: (degrees[x], x))
        else:
            god_node = "None"

        summary += (
            f"- Community {comm_id} ({len(node_ids)} nodes): [{type_str}]\n"
            f"  -> Central Hub Node: {god_node}\n"
        )
        communities_map[comm_id] = node_ids

    return {
        "app_summary": summary,
        "communities_map": communities_map
    }

# ==========================================
# Manager
# ==========================================

def manager_agent_node(state: MasterState) -> dict[str, Any]:
    """The Manager LLM reads the programmatic summary and dispatches tasks."""
    # ----------------------
    # Check cache
    # ----------------------
    # Create a hash of the app_summary so the cache auto-invalidates if the graph changes
    summary_hash = hashlib.md5(state.get("app_summary", "").encode()).hexdigest()
    cache_file = settings.manager_cache_dir / f"manager_tasks_{summary_hash}.json"

    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)

            # Reconstruct the Pydantic objects and the AIMessage
            expert_tasks = [ExpertTask(**task) for task in cached_data.get("expert_tasks", [])]
            manager_message = AIMessage(content=cached_data.get("manager_message", ""))

            logging.debug("Loaded Manager tasks from cache.")
            return {"expert_tasks": expert_tasks, "manager_message": manager_message}
        except Exception as e:
            logging.warning(f"Manager cache corrupted or schema changed. Re-generating... ({e})")

    # ----------------------
    # LLM invocation
    # ----------------------
    llm = ChatOllama(model="qwen36", temperature=0, reasoning=False, num_ctx=32768)
    # llm = ChatOpenAI(base_url="http://localhost:11434/v1", model="glm-5-2", temperature=0)

    parser = PydanticOutputParser(pydantic_object=ManagerOutput)

    sys_msg = SystemMessage(content=(
        f"{MANAGER_AGENT.get('prompt')}\n\n"
        f"{parser.get_format_instructions()}"
    ))
    human_msg = HumanMessage(content=f"Here is the app topology:\n{state.get('app_summary')}")

    response_msg = llm.invoke([sys_msg, human_msg])
    response = parser.invoke(response_msg)

    # ----------------------
    # Save to cache
    # ----------------------
    try:
        cache_data = {
            # Convert Pydantic objects to dicts
            "expert_tasks": [task.model_dump() for task in response.tasks],
            # Extract the raw text from the AIMessage
            "manager_message": response_msg.content
        }
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
        logging.debug("Saved Manager tasks to cache.")
    except Exception as e:
        logging.warning(f"Failed to write Manager cache: {e}")

    return {"expert_tasks": response.tasks, "manager_message": response_msg}

# ==========================================
# Expert agents
# ==========================================

def expert_explorer_node(state: ExpertState) -> dict:
    if isinstance(state["task"], dict):
        task = ExpertTask(**state["task"])
    else:
        task = state["task"]

    unprocessed = state.get("unprocessed_nodes")
    if unprocessed is None:
        unprocessed = list(state.get("subgraph_nodes", []))

    if not unprocessed:
        return {}

    current_node = unprocessed.pop(0)
    role_name = task.agent_role

    # ----------------------
    # Check cache
    # ----------------------
    cache_file = settings.expert_explorer_cache_dir / f"{current_node}-{role_name}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cached_note = json.load(f)
            logging.debug(f"Loaded note for {current_node} from cache.")
            return {
                "unprocessed_nodes": unprocessed,
                "notes": [cached_note]
            }
        except json.JSONDecodeError:
            logging.warning(f"Cache file {cache_file} corrupted. Re-generating...")

    # ----------------------
    # LLM invocation
    # ----------------------
    source_code = get_node_source_code(settings.graph, current_node)

    llm = ChatOllama(model="qwen36", temperature=0, reasoning=False, num_ctx=32768)
    # llm = ChatOpenAI(base_url="http://localhost:11434/v1", model="glm-5-2", temperature=0)

    parser = PydanticOutputParser(pydantic_object=TakeNoteInput)
    sys_msg = SystemMessage(content=(
        f"{EXPERT_AGENTS[role_name]['prompt']}\n\n"
        "Your ONLY job is to read the code and take a structured note. "
        "You must extract sources, sinks, and assumptions."
    ))
    human_msg = HumanMessage(content=(
        f"Analyze this node: {current_node}\n\n```python\n{source_code}\n```\n\n"
        f"{parser.get_format_instructions()}"
    ))

    # structured_llm = llm.with_structured_output(TakeNoteInput)
    response = llm.invoke([sys_msg, human_msg])
    note = parser.invoke(response)
    dict_note = note if isinstance(note, dict) else note.model_dump()

    # ----------------------
    # Node ID resolution
    # ----------------------
    with open(settings.graph, "r") as f:
        graph_data = json.load(f)
        nodes_list = graph_data.get("nodes", [])

    items_to_resolve = dict_note.get("assumptions_to_verify", []) + dict_note.get("potential_issues", [])
    for item in items_to_resolve:
        depends_on = item.get("depends_on")
        if not depends_on:
            continue

        module = depends_on.get("module")
        symbol = depends_on.get("symbol")
        if not module or not symbol:
            continue

        node = next(
            (n for n in nodes_list 
             if n.get("source_file", "").endswith(module.replace(".", "/") + ".py")
             and n.get("label") in [symbol, f"{symbol}()"]),
            None
        )

        if not node:
            logging.warning(f"Failed to find graph node for local symbol '{module}.{symbol}'.")
            continue

        depends_on["resolved_node_id"] = node.get("id")

    # ----------------------
    # Save to cache
    # ----------------------
    try:
        with open(cache_file, "w") as f:
            json.dump(dict_note, f, indent=2)
        logging.debug(f"Saved note for {current_node} to cache.")
    except Exception as e:
        logging.warning(f"Failed to write cache file for {current_node}: {e}")

    return {
        "unprocessed_nodes": unprocessed,
        "notes": [dict_note]
    }


def expert_researcher_node(state: ExpertState) -> dict:
    if isinstance(state["task"], dict):
        task = ExpertTask(**state["task"])
    else:
        task = state["task"]

    role_name = task.agent_role
    community_id = task.target_community

    # ----------------------
    # Check cache
    # ----------------------
    cache_file = settings.expert_researcher_cache_dir / f"{community_id}-{role_name}.json"

    # Only check the cache if this is the very first time we enter this node 
    if not state.get("messages") and cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cached_reports = json.load(f)
            logging.debug(f"Loaded cached reports for Community {community_id} ({role_name}).")

            # Create a mock AIMessage that forces the router to end the subgraph
            mock_completion = AIMessage(
                content="Loaded from cache.",
                tool_calls=[{"name": "mark_task_complete", "args": {}, "id": "cache_hit_id"}]
            )
            return {
                "vulnerability_reports": cached_reports,
                "messages": [mock_completion]
            }
        except json.JSONDecodeError:
            logging.warning(f"Cache file {cache_file} corrupted. Re-generating...")

    # ----------------------
    # LLM invocation
    # ----------------------
    llm = ChatOllama(model="qwen36", temperature=0, reasoning=False, num_ctx=32768)
    # llm = ChatOpenAI(base_url="http://localhost:11434/v1", model="glm-5-2", temperature=0)

    agent_tools = [
        tools.submit_report,
        tools.mark_task_complete
    ]
    tool_names = EXPERT_AGENTS[role_name].get("tools", [])
    for name in tool_names:
        if hasattr(tools, name):
            agent_tools.append(getattr(tools, name))

    llm_with_tools = llm.bind_tools(agent_tools)

    if not state.get("messages"):
        notes_str = format_notes(state.get("notes", []))
        sys_msg = SystemMessage(content=(
            f"{EXPERT_AGENTS[role_name]['prompt']}\n\n"
            f"{EXPERT_AGENTS['prompt']}"
        ))
        human_msg = HumanMessage(content=(
            f"Your Task: {task.task_description}\n\n"
            f"## Audit Notes for Assigned Community\n\n{notes_str}"
        ))

        messages = [sys_msg, human_msg]
        response = llm_with_tools.invoke(messages)
        messages = [sys_msg, human_msg, response]
    else:
        compacted_messages = compact_tool_history(state["messages"], safe_window=8)
        response = llm_with_tools.invoke(compacted_messages)
        messages = [response]

    # ----------------------
    # Save to cache
    # ----------------------
    # If the LLM decides to finish the task, save the accumulated reports to disk
    is_complete = any(tc["name"] == "mark_task_complete" for tc in response.tool_calls)
    if is_complete:
        try:
            with open(cache_file, "w") as f:
                json.dump(state.get("vulnerability_reports", []), f, indent=2)
            logging.debug(f"Saved reports to cache for Community {community_id} ({role_name}).")
        except Exception as e:
            logging.warning(f"Failed to write cache for Community {community_id}: {e}")

    return {"messages": messages}


def expert_agent_router(state: ExpertState):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]

        # If it has other tools to run (like submit_report), force it to the ToolNode first
        if "mark_task_complete" in tool_names and len(tool_names) == 1:
            return "__end__"

        # Go to tools node
        return "tools"

    # The LLM failed to call a tool
    return "ask_expert_for_tool"


def route_expert_phases(state: ExpertState) -> str:
    unprocessed = state.get("unprocessed_nodes")

    # Initialize check on first run
    if unprocessed is None:
        unprocessed = state.get("subgraph_nodes", [])

    if len(unprocessed) == 0:
        return "expert_researcher_node"
    return "expert_explorer_node"


def ask_expert_for_tool(state: ExpertState):
    """Fallback node to force the LLM to use a tool."""
    message = HumanMessage(
        content=f"You did not invoke any tools. You must either use `{'`, `'.join(TOOLS.keys())}` to proceed."
    )
    return {"messages": [message]}


def dispatch_experts(state: MasterState):
    """Reads the Manager's instructions and creates a list of 'Send' objects."""

    # G = build_networkx_graph(state["graph_path"])
    commands: list[Send] = []

    for task in state["expert_tasks"]:
        clean_id = task.target_community.lower().replace("community ", "").strip()
        nodes_for_task = state["communities_map"].get(clean_id, [])

        # # Filter out skeleton nodes, but keep functions, classes, and non-code files
        # for node in community_nodes:
        #     for node_id, data in G.nodes(data=True):
        #         if node_id == node:
        #             if not data.get("source_file", "").endswith(data.get("label")):
        #                 nodes_for_task.append(node)
        #                 break

        payload = ExpertState(
            task=task,
            subgraph_nodes=nodes_for_task,
            unprocessed_nodes=nodes_for_task.copy(),
            messages=[],
            vulnerability_reports=[],
            notes=[]
        )
        commands.append(Send("expert_agent", payload))

    return commands

# ==========================================
# Reviewer agent
# ==========================================

def dispatch_reviewers(state: MasterState):
    """Groups reports and dispatches parallel reviewer threads using the Send API."""

    reports = state.get("vulnerability_reports", [])
    if not reports:
        return END

    # Group the reports by vulnerability and sink_node
    grouped_reports = defaultdict(list)
    for report in reports:
        vuln = report.get("vulnerability", "Unknown")
        source = report.get("source_node", "Unknown")
        sink = report.get("sink_node", "Unknown")
        grouped_reports[(vuln, source, sink)].append(report)

    commands = []
    sys_msg = SystemMessage(content=REVIEWER_AGENT.get('prompt'))
    for (vuln, source, sink), group in grouped_reports.items():
        # Format the group into a single, clean string for the LLM
        formatted_group_text = f"Vulnerability: {vuln}\nSource Node: {source}\nSink Node: {sink}\n\nInstances found:\n"
        for idx, item in enumerate(group, 1):
            traces = item.get("trace_nodes", [])
            trace_str = ", ".join(traces) if traces else "None"
            formatted_group_text += (
                f"  --- Instance {idx} ---\n"
                f"  Role: {item.get('role', 'Unknown')}\n"
                f"  Details: {item.get('details', '')}\n"
                f"  Trace Nodes: {trace_str}\n"
            )

        report_id = f"{vuln} @ {sink}"
        human_msg = HumanMessage(content=f"Vulnerability report to evaluate:\n{formatted_group_text}")

        payload = ReviewerState(
            report_id=report_id,
            expert_report=group,
            messages=[sys_msg, human_msg]
        )
        commands.append(Send("reviewer_agent", payload))

    return commands


def reviewer_agent_node(state: ReviewerState) -> dict:
    """Review the vulnerability reports and keep only what is actually relevant"""
    llm = ChatOllama(model="qwen36", temperature=0, reasoning=False, num_ctx=32768)
    # llm = ChatOpenAI(base_url="http://localhost:11434/v1", model="glm-5-2", temperature=0)

    llm_with_tools = llm.bind_tools([
        tools.read_source_code,
        tools.search_codebase,
        tools.get_node_connections,
        tools.submit_evaluation
    ])

    # Use a slightly larger safe_window for the reviewer
    # so it can compare a search result with a file read
    compacted_messages = compact_tool_history(state["messages"], safe_window=8)

    sys_msg = compacted_messages[0]
    human_msg = compacted_messages[1]
    dynamic_msgs = []
    if state.get("notes"):
        notes_str = "\n".join([f"- {n}" for n in state["notes"]])
        saved_notes = f"\n\n### Persistent Scratchpad\n{notes_str}\n"
        dynamic_msgs.append(HumanMessage(content=saved_notes))

    messages_to_pass = [sys_msg, human_msg] + dynamic_msgs + compacted_messages[2:]

    response = llm_with_tools.invoke(messages_to_pass)
    return {"messages": [response]}


def reviewer_router(state: ReviewerState):
    """Routes based on the tool called by the reviewer LLM."""
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "reviewer_tools"

    # The LLM failed to call a tool
    return "ask_reviewer_for_tool"


def ask_reviewer_for_tool(state: ReviewerState):
    """Fallback node to force the LLM to use a tool."""
    message = HumanMessage(content=f"You did not invoke any tools. You must use a tool to proceed.")
    return {"messages": [message]}

# ==========================================
# Validator agent
# ==========================================

def dispatch_validators(state: MasterState):
    """Creates a parallel validation thread for each vulnerability that survived the reviewer."""

    commands = []
    # Loop over the Pydantic models generated by the reviewer
    for evaluation in state.get("filtered_reports", []):
        if evaluation.is_exploitable:

            payload = ValidatorState(
                report_to_test=evaluation,
                sandbox_url=settings.sandbox_url,
                messages=[]
            )
            commands.append(Send("validator_agent", payload))

    if not commands:
        # If nothing to validate, skip straight to the end
        return END

    return commands


def validator_agent_node(state: ValidatorState) -> dict:
    llm = ChatOllama(model="qwen36", temperature=0, reasoning=False, num_ctx=32768)
    # llm = ChatOpenAI(base_url="http://localhost:11434/v1", model="glm-5-2", temperature=0)

    llm_with_tools = llm.bind_tools([
        tools.send_http_request,
        tools.mark_validation_complete,
        tools.take_notes
    ])
    current_cookies = state.get("cookies", {})

    if not state.get("messages"):
        sys_msg = SystemMessage(content=VALIDATOR_AGENT.get('prompt'))
        human_msg = HumanMessage(content=(
            f"Target Sandbox: {state['sandbox_url']}\n\n"
            f"Vulnerability to Prove:\n{state['report_to_test']}\n"
        ))
        messages = [sys_msg, human_msg]
        response = llm_with_tools.invoke(messages)
        return {"messages": [sys_msg, human_msg, response]}
    else:
        # Update cookies from the recent history
        for msg in reversed(state["messages"]):
            if getattr(msg, "type", "") == "ai":
                break
            if getattr(msg, "type", "") == "tool" and getattr(msg, "name", "") == "send_http_request":
                if hasattr(msg, "artifact") and msg.artifact:
                    # Merge the new cookies into the current state
                    current_cookies.update(msg.artifact)

        compacted_messages = compact_tool_history(state["messages"], safe_window=8)
        sys_msg = compacted_messages[0]
        human_msg = compacted_messages[1]

        dynamic_msgs = []
        if state.get("notes"):
            notes_str = "\n".join([f"- {n}" for n in state["notes"]])
            saved_notes = f"\n\n### Persistent Scratchpad\n{notes_str}\n"
            dynamic_msgs.append(HumanMessage(content=saved_notes))

        messages_to_pass = [sys_msg, human_msg] + dynamic_msgs + compacted_messages[2:]

        response = llm_with_tools.invoke(messages_to_pass)
        return {"messages": [response], "cookies": current_cookies}


def validator_router(state: ValidatorState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "validator_tools"

    # The LLM failed to call a tool
    return "ask_validator_for_tool"


def ask_validator_for_tool(state: ValidatorState):
    """Fallback node to force the LLM to use a tool."""
    message = HumanMessage(content=f"You did not invoke any tools. You must use a tool to proceed.")
    return {"messages": [message]}

# ==========================================
# Build and Compile the Graph
# ==========================================

def compile_expert():
    expert_workflow = StateGraph(ExpertState)
    expert_workflow.add_node("expert_explorer", expert_explorer_node)
    expert_workflow.add_node("expert_researcher", expert_researcher_node)
    expert_workflow.add_node("ask_expert_for_tool", ask_expert_for_tool)
    expert_workflow.add_node("tools", ToolNode([
        tools.submit_report,
        tools.mark_task_complete,
        tools.check_package_vulnerability
    ]))
    expert_workflow.add_edge(START, "expert_explorer")
    expert_workflow.add_conditional_edges(
        "expert_explorer",
        route_expert_phases,
        {
            "expert_explorer_node": "expert_explorer",
            "expert_researcher_node": "expert_researcher"
        }
    )
    expert_workflow.add_conditional_edges(
        "expert_researcher",
        expert_agent_router,
        {
            "tools": "tools",
            "ask_expert_for_tool": "ask_expert_for_tool",
            "__end__": END
        }
    )
    expert_workflow.add_edge("tools", "expert_researcher")
    expert_workflow.add_edge("ask_expert_for_tool", "expert_researcher")
    compiled_expert_agent = expert_workflow.compile()

    return compiled_expert_agent


def compile_reviewer():
    reviewer_workflow = StateGraph(ReviewerState)
    reviewer_workflow.add_node("reviewer_agent", reviewer_agent_node)
    reviewer_workflow.add_node("ask_reviewer_for_tool", ask_reviewer_for_tool)
    reviewer_workflow.add_node("reviewer_tools", ToolNode([
        tools.read_source_code,
        tools.get_node_connections,
        tools.search_codebase,
        tools.submit_evaluation,
        tools.take_notes
    ]))
    reviewer_workflow.add_edge(START, "reviewer_agent")
    reviewer_workflow.add_conditional_edges("reviewer_agent", reviewer_router)
    reviewer_workflow.add_edge("reviewer_tools", "reviewer_agent")
    reviewer_workflow.add_edge("ask_reviewer_for_tool", "reviewer_agent")
    compiled_reviewer_agent = reviewer_workflow.compile()

    return compiled_reviewer_agent


def compile_validator():
    validator_workflow = StateGraph(ValidatorState)
    validator_workflow.add_node("validator_agent", validator_agent_node)
    validator_workflow.add_node("ask_validator_for_tool", ask_validator_for_tool)
    validator_workflow.add_node("validator_tools", ToolNode([
        tools.send_http_request,
        tools.mark_validation_complete,
        tools.take_notes
    ]))
    validator_workflow.add_edge(START, "validator_agent")
    validator_workflow.add_conditional_edges("validator_agent", validator_router)
    validator_workflow.add_edge("validator_tools", "validator_agent")
    validator_workflow.add_edge("ask_validator_for_tool", "validator_agent")
    compiled_validator_agent = validator_workflow.compile()

    return compiled_validator_agent 

def build_graph(checkpointer=None, interrupt_before=None):
    compiled_expert_agent = compile_expert()
    compiled_reviewer_agent = compile_reviewer()
    compiled_validator_agent = compile_validator()

    # Master Graph
    workflow = StateGraph(MasterState)
    workflow.add_node("preprocessor", preprocessor_node)
    workflow.add_node("manager", manager_agent_node)
    workflow.add_node("expert_agent", compiled_expert_agent)
    workflow.add_node("reviewer_agent", compiled_reviewer_agent)
    workflow.add_node("validator_agent", compiled_validator_agent)
    workflow.add_edge(START, "preprocessor")
    workflow.add_edge("preprocessor", "manager")
    workflow.add_conditional_edges("manager", dispatch_experts, ["expert_agent"])
    workflow.add_conditional_edges("expert_agent", dispatch_reviewers, ["reviewer_agent", END])
    workflow.add_conditional_edges("reviewer_agent", dispatch_validators, ["validator_agent", END])
    workflow.add_edge("validator_agent", END)

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
    config = {
        "max_concurrency": 2
    }

    try:
        final_state = run_stream(app, initial_state, config=config)

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
