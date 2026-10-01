from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Send
from langgraph.graph import END
from typing import Any
from collections import defaultdict
import networkx as nx
import hashlib
import logging
from json_repair import repair_json

import settings
import tools
from state import MasterState, ExplorerState, CVEAnalyzerState, VerifierState, ReviewerState, ValidatorState
from schemas import ManagerOutput, ExpertTask, AnalysisNote, CVEDemand, VerifierOutput, MANAGER_AGENT, EXPERT_AGENTS, CVE_ANALYZER_AGENT, VERIFIER_AGENT, REVIEWER_AGENT, VALIDATOR_AGENT
from utils import build_networkx_graph, get_cached_graph_data, get_graph_summary, get_node_code, run_osv_scanner, deduplicate_cves, cache, resolve_node_id, uses_namespace_in_ast


# llm = ChatOllama(model="qwen36", temperature=0, reasoning=False, num_ctx=32768)
llm = ChatOpenAI(base_url="http://localhost:11434/v1", model="mistral-3.5-128b", stream_usage=True, temperature=0)

# ==========================================
# Preprocessor
# ==========================================

def preprocessor_node(state: MasterState) -> dict[str, Any]:
    """Reads graph.json, builds a NetworkX graph, and summarizes it for the manager node."""

    # communities_to_analyze = [0, 1, 2, 5, 7, 29, 81, 18, 127, 238, 298]
    communities_to_analyze = None
    G = build_networkx_graph(settings.graph, communities_to_analyze)
    summary = get_graph_summary(G)

    # Run the OSV scanner to parse manifests and query the database
    raw_vulns = run_osv_scanner(str(settings.app_path))
    logging.info(f"Found {len(raw_vulns)} raw vulns")
    clean_vulns = deduplicate_cves(raw_vulns)
    logging.info(f"{len(clean_vulns)} remaining CVEs after deduplication")

    return {
        "app_summary": summary,
        # "graph": nx.node_link_data(G),
        "known_vulns": clean_vulns
    }

# ==========================================
# Manager
# ==========================================

def manager_agent_node(state: MasterState) -> dict[str, Any]:
    """The Manager LLM reads the programmatic summary and dispatches tasks."""
    # Check cache
    # Create a hash of the app_summary so the cache auto-invalidates if the graph changes
    summary_hash = hashlib.md5(state.get("app_summary", "").encode()).hexdigest()
    cache_file = settings.cache_dir / "manager" / f"manager_tasks_{summary_hash}.json"
    cached_data = cache(cache_file, "read")
    if cached_data:
        expert_tasks = [ExpertTask(**task) for task in cached_data.get("expert_tasks", [])]
        return {"expert_tasks": expert_tasks}

    # LLM invocation
    parser = PydanticOutputParser(pydantic_object=ManagerOutput)

    sys_msg = SystemMessage(content=(
        f"{MANAGER_AGENT.get('prompt')}\n\n"
        f"{parser.get_format_instructions()}"
    ))
    human_msg = HumanMessage(content=f"Here is the app topology:\n{state.get('app_summary')}")

    response_msg = llm.invoke([sys_msg, human_msg])
    response = parser.invoke(response_msg)

    # Save to cache
    cache(cache_file, "write", {"expert_tasks": [task.model_dump() for task in response.tasks]})

    return {"expert_tasks": response.tasks}

# ==========================================
# Explorer agents
# ==========================================

def dispatch_explorers(state: MasterState):
    """Reads the Manager's instructions and creates a list of 'Send' objects."""

    # G = nx.node_link_graph(state["graph"])
    G = build_networkx_graph(settings.graph)
    commands: list[Send] = []

    for task in state["expert_tasks"]:
        task = task if isinstance(task, dict) else task.model_dump()
        clean_id = task.get("target_community", "").lower().replace("community ", "").strip()
        community_nodes = [n for n, attr in G.nodes(data=True) if str(attr.get("community")) == clean_id]

        # Filter out skeleton nodes, but keep functions, classes, and non-code files
        # TODO: If the file contains only the skeleton we should keep it
        for node_id in community_nodes:
            node_data = G.nodes[node_id]

            if node_data.get("source_file", "").endswith(node_data.get("label")):
                continue

            source_file = Path(node_data.get("source_file", ""))
            if source_file.name in ["requirements.txt", "packages.json"]:
                continue

            payload = ExplorerState(
                node_id=node_id,
                role=task.get("agent_role", ""),
                task_description=task.get("task_description", "")
            )

            commands.append(Send("explorer_agent", payload))

    logging.info(f"Dispatching {len(commands)} explorers.")
    return commands


def dispatch_all_tasks(state: MasterState):
    commands = []
    commands.extend(dispatch_explorers(state))
    commands.extend(dispatch_cve_analyzers(state))
    return commands


def expert_explorer_node(state: ExplorerState) -> dict:
    node_id = state.get("node_id")
    role_name = state.get("role")

    # Check cache
    cache_file = settings.cache_dir / "notes" / f"{node_id}-{role_name}.json"
    cached_note = cache(cache_file, "read")
    if cached_note:
        return cached_note

    # LLM invocation
    source_code = get_node_code(node_id)

    parser = PydanticOutputParser(pydantic_object=AnalysisNote)
    sys_msg = SystemMessage(content=(
        f"{EXPERT_AGENTS[role_name]['prompt']}\n\n"
        f"{EXPERT_AGENTS['explorer_prompt']}\n\n"
        f"{parser.get_format_instructions()}"
    ))

    graph_data = get_cached_graph_data(settings.graph)
    target_node = next((node for node in graph_data.get("nodes", []) if node.get("id") == node_id), {})

    if target_node.get("source_file", "").endswith(target_node.get("label", "")):
        user_prompt = (
            f"Analyze this entire file skeleton.\n\n"
            f"```python\n{source_code}\n```"
        )
    else:
        label = target_node.get("label", "node")
        user_prompt = (
            f"Analyze the specific logic inside '{label}'.\n"
            f"The rest of the file is provided solely as context. "
            f"Do NOT look for vulnerabilities outside of '{label}'.\n\n"
            f"```python\n{source_code}\n```"
        )
    human_msg = HumanMessage(content=user_prompt)

    response = llm.invoke([sys_msg, human_msg])

    try:
        fixed_json_string = repair_json(response.content)
        note = parser.invoke(fixed_json_string)
    except Exception as e:
        logging.error(f"Failed to parse LLM output. Agent: {role_name}. Node: {node_id}. Error: {e}\n\n{response.content}")
        return {
            "notes": []
        }

    dict_note = note if isinstance(note, dict) else note.model_dump()
    dict_note["node_id"] = node_id

    extracted_vulns = []

    # Extract and remove the list from dict_note
    raw_hypotheses = dict_note.pop("vulnerability_hypothesis", [])
    for hyp in raw_hypotheses:
        extracted_vulns.append({
            "node_id": node_id,
            "cwe_id": hyp.get("cwe_id", "OTHER_UNCATEGORIZED"), 
            "description": hyp.get("description", ""),
            "status": "hypothesis"
        })

    # Save to cache
    cache(cache_file, "write", {"notes": [dict_note], "vulnerabilities": extracted_vulns})

    return {
        "notes": [dict_note],
        "vulnerabilities": extracted_vulns
    }

# ==========================================
# CVE Analyzer
# ==========================================

def dispatch_cve_analyzers(state: MasterState):
    """Reads the deduplicated SCA results and dispatches tasks to the CVE Analyzer."""
    commands: list[Send] = []

    known_vulns = state.get("known_vulns", [])

    for cve_record in known_vulns:
        payload = CVEAnalyzerState(
            cve=cve_record
        )
        commands.append(Send("cve_analyzer", payload))

    logging.info(f"Dispatching {len(commands)} cve analyzers.")
    return commands


def cve_analyzer_node(state: CVEAnalyzerState) -> dict:
    """LLM node that extracts security assumptions from a single CVE description."""
    cve = state.get("cve", {})
    package_name = ", ".join(cve.get("packages", []))
    cve_id = cve.get("id", "UNKNOWN-CVE")
    details = cve.get("details")
    if not details or len(details) == 0:
        # Without details the LLM would just hallucinate
        logging.warning(f"{cve_id}: no details provided")
        return {"cve_demands": []}

    # Check cache
    cache_file = settings.cache_dir / "cve_analyzer" / f"{cve_id}.json"
    cached_demand = cache(cache_file, "read")
    if cached_demand:
        return {"cve_demands": [cached_demand]}

    # LLM invocation
    parser = PydanticOutputParser(pydantic_object=CVEDemand)

    sys_msg = SystemMessage(content=(
        f"{CVE_ANALYZER_AGENT['prompt']}\n\n"
        f"{parser.get_format_instructions()}"
    ))
    human_msg = HumanMessage(content=(
        f"Analyze this CVE affecting the package '{package_name}':\n\n"
        f"CVE ID: {cve_id}\n"
        f"Description: {details}\n\n"
    ))

    response = llm.invoke([sys_msg, human_msg])
    demand = parser.invoke(response)

    dict_demand = demand.model_dump()
    # Tag the resulting demand with the CVE ID for traceability during the Verification Phase
    dict_demand["source_cve"] = cve_id
    dict_demand["package"] = package_name

    # Save to cache
    cache(cache_file, "write", dict_demand)

    # Return the extracted demand to be appended to the global state
    return {
        "cve_demands": [dict_demand]
    }

# ==========================================
# Aggregate demands node
# ==========================================

def aggregate_demands_node(state: MasterState):
    grouped_demands = {}
    node_imports_map = {}

    # Process Explorer Notes & Build Import Map
    for note in state.get("notes", []):
        dict_note = note if isinstance(note, dict) else note.model_dump()
        node_id = dict_note.get("node_id")

        if node_id not in node_imports_map:
            node_imports_map[node_id] = set()

        # .update() safely adds items to the set, automatically ignoring duplicates
        node_imports_map[node_id].update(dict_note.get("imports", []))

        # Process explorer assumptions
        for assumption in dict_note.get("assumptions_to_verify", []):
            target_node_id = resolve_node_id(assumption.get("module"), assumption.get("symbol"))
            if target_node_id:
                if target_node_id not in grouped_demands:
                    grouped_demands[target_node_id] = []

                grouped_demands[target_node_id].append({
                    "source": dict_note.get("node_id"),
                    "type": "explorer_assumption",
                    "description": assumption.get("description")
                })

    # Process CVE Demands
    for demand in state.get("cve_demands", []):
        target_import = demand.get("import_namespace", "")

        for node_id, imports in node_imports_map.items():
            # Matches exact namespace (e.g., 'bs4' in ['os', 'bs4', 'sys'])
            if target_import in imports:

                # AST-Verification: Confirm the node's body actually uses the namespace.
                if uses_namespace_in_ast(node_id, target_import):
                    if node_id not in grouped_demands:
                        grouped_demands[node_id] = []

                    grouped_demands[node_id].append({
                        "source": demand.get("source_cve"),
                        "type": "cve_assumption",
                        "description": demand.get("security_assumption")
                    })

    logging.info(f"Grouped {len(grouped_demands)} demands.")
    return {"grouped_demands": grouped_demands}

# ==========================================
# Contract verifier node
# ==========================================

def dispatch_verifiers(state: MasterState):
    grouped_demands = state.get("grouped_demands", {})

    # If no demands were found across the whole codebase, skip straight to the end
    if not grouped_demands:
        return END

    commands: list[Send] = []

    for target_node_id, demands_list in grouped_demands.items():
        target_code = get_node_code(target_node_id)

        # If we don't have code (e.g., it's a 3rd party library), skip it
        if not target_code:
            continue

        payload = {
            "target_node_id": target_node_id,
            "target_code": target_code,
            "incoming_demands": demands_list
        }

        commands.append(Send("contract_verifier", payload))

    # If all targets were 3rd party libraries and we generated 0 commands
    if not commands:
        return "synchronization"

    logging.info(f"Dispatching {len(commands)} contract verifiers.")
    return commands


def contract_verifier_node(state: VerifierState) -> dict:
    target_node_id = state.get("target_node_id")
    target_code = state.get("target_code")
    demands = state.get("incoming_demands", [])

    if not demands or not target_code:
        return {"vulnerability_hypothesis": []}

    # Check cache
    cache_file = settings.cache_dir / "contract_verifier" / f"{target_node_id}.json"
    cached_data = cache(cache_file, "read")
    if cached_data:
        return {"vulnerability_hypothesis": cached_data.get("hypothesis", [])}

    # LLM invocation
    parser = PydanticOutputParser(pydantic_object=VerifierOutput)

    sys_msg = SystemMessage(content=(
        f"{VERIFIER_AGENT['prompt']}\n\n"
        f"{parser.get_format_instructions()}"
    ))

    formatted_demands = "\n".join([f"- {d.get('description')} (Requested by {d.get('source_node')})" for d in demands])
    human_msg = HumanMessage(content=(
        f"```\n{target_code}\n```\n\n"
        f"Security Demands to Verify:\n{formatted_demands}"
    ))

    response = llm.invoke([sys_msg, human_msg])
    fixed_json_string = repair_json(response.content)
    parsed_output: VerifierOutput = parser.invoke(fixed_json_string)

    # Process the Results
    new_vulnerabilities = []

    for eval in parsed_output.evaluations:
        if eval.status == "FAILED":
            # Map the failed contract to the new unified schema
            new_vulnerabilities.append({
                "node_id": target_node_id,
                "cwe_id": eval.cwe_id,
                "description": f"Fails to satisfy demand: '{eval.demand_description}'. Reasoning: {eval.reasoning}",
                "status": "hypothesis"
            })

    # Save to cache (update cache keys as needed)
    cache(cache_file, "write", {"hypothesis": new_vulnerabilities})

    return {
        "vulnerabilities": new_vulnerabilities
    }

# ==========================================
# Reviewer agent
# ==========================================

def synchronization_node(state: MasterState):
    """Dummy node to act as a Map-Reduce barrier."""
    return {}


def dispatch_reviewers(state: MasterState):
    """Groups reports and dispatches parallel reviewer threads using the Send API."""
    raw_vulns = state.get("vulnerabilities", [])
    all_vulns = [
        v if isinstance(v, dict) else v.model_dump() 
        for v in raw_vulns
    ]
    hypotheses = [v for v in all_vulns if v.get("status") == "hypothesis"]

    if not hypotheses:
        logging.warning(f"No vulnerabilities hypotheses to dispatch.")
        return END

    commands = []
    for hypothesis in hypotheses:
        payload = ReviewerState(
            node_id=hypothesis.get("node_id", "Unknown"),
            expert_report=hypothesis,
            messages=[]
        )
        commands.append(Send("reviewer_agent", payload))

    logging.info(f"Dispatching {len(commands)} reviewers.")
    return commands


def reviewer_agent_node(state: ReviewerState) -> dict:
    """Review the vulnerability reports and keep only what is actually relevant"""
    llm_with_tools = llm.bind_tools([
        tools.read_source_code,
        tools.search_codebase,
        tools.get_node_connections,
        tools.submit_evaluation
    ])

    if not state.get("messages"):
        sys_msg = SystemMessage(content=REVIEWER_AGENT.get('prompt'))

        report = state.get("expert_report", {})

        formatted_vuln = (
            f"Target {state['node_id']}\n\n"
            f"Potential Issue to Investigate:\n"
            f"CWE ID: {report.get('cwe_id', 'Unknown')}\n"
            f"Component: {report.get('vulnerable_component', 'Unknown')}\n"
            f"Description: {report.get('description', '')}\n"
        )

        human_msg = HumanMessage(content=(
            f"Review the following potential issues found in the target node.\n\n"
            f"{formatted_vuln}"
        ))

        response = llm_with_tools.invoke([sys_msg, human_msg])
        return {"messages": [sys_msg, human_msg, response]}

    else:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}


def reviewer_router(state: ReviewerState):
    """Routes based on the tool called by the reviewer LLM."""
    last_message = state["messages"][-1]

    if last_message.type == "ai":
        if last_message.tool_calls:
            return "reviewer_tools"
        # The LLM failed to call a tool
        return "ask_reviewer_for_tool"

    elif last_message.type == "tool":
        if getattr(last_message, "name", "") == "submit_evaluation":
            return "__end__"
        return "reviewer_agent"

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
    raw_vulns = state.get("vulnerabilities", [])
    all_vulns = [
        v if isinstance(v, dict) else v.model_dump() 
        for v in raw_vulns
    ]
    # Filter for vulnerabilities that were confirmed by the Reviewer
    confirmed_vulns = [v for v in all_vulns if v.get("status") == "confirmed"]

    commands = []
    # Loop over the Pydantic models generated by the reviewer
    for evaluation in confirmed_vulns:
        payload = ValidatorState(
            report_to_test=evaluation, 
            sandbox_url=settings.sandbox_url,
            messages=[],
            confirmed_vulnerabilities=[], 
            cookies={}
        )
        commands.append(Send("validator_agent", payload))

    if not commands:
        # If nothing to validate, skip straight to the end
        return END

    return commands


def validator_agent_node(state: ValidatorState) -> dict:
    llm_with_tools = llm.bind_tools([
        tools.send_http_request,
        tools.list_files,
        tools.read_file,
        tools.mark_validation_complete
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

        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response], "cookies": current_cookies}


def validator_router(state: ValidatorState):
    last_message = state["messages"][-1]

    if last_message.type == "ai":
        if last_message.tool_calls:
            return "validator_tools"
        # The LLM failed to call a tool
        return "ask_validator_for_tool"

    elif last_message.type == "tool":
        if getattr(last_message, "name", "") == "mark_validation_complete":
            return "__end__"
        return "validator_agent"

    # The LLM failed to call a tool
    return "ask_validator_for_tool"


def ask_validator_for_tool(state: ValidatorState):
    """Fallback node to force the LLM to use a tool."""
    message = HumanMessage(content=f"You did not invoke any tools. You must use a tool to proceed.")
    return {"messages": [message]}

