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
from utils import build_networkx_graph, get_graph_summary, run_osv_scanner, deduplicate_cves, cache, get_node_source_code, resolve_node_id


llm = ChatOllama(model="qwen36", temperature=0, reasoning=False, num_ctx=32768)
# llm = ChatOpenAI(base_url="http://localhost:11434/v1", model="kimi-k2-7-code", temperature=0)

# ==========================================
# Preprocessor
# ==========================================

# def preprocessor_node(state: MasterState) -> dict[str, Any]:
#     """Reads graph.json, builds a NetworkX graph, and summarizes it for the manager node."""
#
#     G = build_networkx_graph(settings.graph)
#     summary = get_graph_summary(G)
#
#     # Run the OSV scanner to parse manifests and query the database
#     raw_vulns = run_osv_scanner(str(settings.app_path))
#     clean_vulns = deduplicate_cves(raw_vulns)
#     logging.debug(f"Found {len(raw_vulns)} raw vulns")
#     logging.debug(f"{len(clean_vulns)} remaining CVEs after deduplication")
#
#     return {
#         "app_summary": summary,
#         "graph": nx.node_link_data(G),
#         "known_vulns": clean_vulns
#     }


def preprocessor_node(state: MasterState) -> dict[str, Any]:
    """Reads graph.json, builds a NetworkX graph, filters for top 10 communities, and summarizes."""

    G = build_networkx_graph(settings.graph)

    # 1. Group nodes by community
    communities = defaultdict(list)
    for node_id, data in G.nodes(data=True):
        com_id = data.get("community")
        if com_id is not None:
            communities[com_id].append((node_id, data))

    # 2. Score communities based on security relevance
    # GLPI is PHP, so we look for standard web/auth/db terminology
    security_keywords = [
        'auth', 'login', 'session', 'db', 'sql', 'query', 
        'upload', 'admin', 'api', 'exec', 'password', 'token', 
        'crypto', 'hash', 'csrf', 'plugin'
    ]

    community_scores = {}
    for com_id, nodes in communities.items():
        score = 0
        for node_id, data in nodes:
            # Search both the node label and its file path for sensitive keywords
            text_to_search = (str(data.get("label", "")) + " " + str(data.get("source_file", ""))).lower()

            for kw in security_keywords:
                if kw in text_to_search:
                    score += 10  # Heavy weight for security concepts

            # Add minor weight for community size (ignore tiny, 1-node orphan communities)
            score += 1 

        community_scores[com_id] = score

    # 3. Select the Top 10 highest-scoring communities
    top_10_coms = sorted(community_scores, key=community_scores.get, reverse=True)[:10]
    logging.info(f"Selected top 10 communities: {top_10_coms}")

    # 4. Filter the NetworkX Graph
    nodes_to_keep = [
        node_id for node_id, data in G.nodes(data=True) 
        if data.get("community") in top_10_coms
    ]
    G_filtered = G.subgraph(nodes_to_keep).copy()

    # 5. Generate summary ONLY for the filtered graph
    summary = get_graph_summary(G_filtered)

    # Run the OSV scanner to parse manifests and query the database
    raw_vulns = run_osv_scanner(str(settings.app_path))
    clean_vulns = deduplicate_cves(raw_vulns)
    logging.debug(f"Found {len(raw_vulns)} raw vulns")
    logging.debug(f"{len(clean_vulns)} remaining CVEs after deduplication")

    return {
        "app_summary": summary,
        "graph": nx.node_link_data(G_filtered), # Pass the filtered graph to the state!
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

    G = nx.node_link_graph(state["graph"])
    commands: list[Send] = []

    for task in state["expert_tasks"]:
        clean_id = task.target_community.lower().replace("community ", "").strip()
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
                role=task.agent_role,
                task_description=task.task_description
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
        return {"notes": [cached_note]}

    # LLM invocation
    source_code = get_node_source_code(node_id)

    parser = PydanticOutputParser(pydantic_object=AnalysisNote)
    sys_msg = SystemMessage(content=(
        f"{EXPERT_AGENTS[role_name]['prompt']}\n\n"
        f"{EXPERT_AGENTS['explorer_prompt']}\n\n"
        f"{parser.get_format_instructions()}"
    ))
    human_msg = HumanMessage(content=(
        f"Analyze this node: {node_id}\n\n```python\n{source_code}\n```"
    ))

    response = llm.invoke([sys_msg, human_msg])
    note = parser.invoke(response)
    dict_note = note if isinstance(note, dict) else note.model_dump()

    # Save to cache
    cache(cache_file, "write", dict_note)

    return {
        "notes": [dict_note]
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
        target_import = demand.get("import_namespace") 

        for node_id, imports in node_imports_map.items():
            # Matches exact namespace (e.g., 'bs4' in ['os', 'bs4', 'sys'])
            if target_import in imports:
                if node_id not in grouped_demands:
                    grouped_demands[node_id] = []

                grouped_demands[node_id].append({
                    "source": demand.get("source_cve"),
                    "type": "cve_assumption",
                    "description": demand.get("security_assumption")
                })

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
        target_code = get_node_source_code(target_node_id)

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
        f"Target Node: {target_node_id}\n\n"
        f"Source Code:\n```python\n{target_code}\n```\n\n"
        f"Security Demands to Verify:\n{formatted_demands}"
    ))

    response = llm.invoke([sys_msg, human_msg])
    fixed_json_string = repair_json(response.content)
    parsed_output: VerifierOutput = parser.invoke(fixed_json_string)

    # Process the Results
    new_hypothesis = []

    for eval in parsed_output.evaluations:
        if eval.status == "FAILED":
            # If the contract is broken, it's a vulnerability
            new_hypothesis.append({
                "vulnerability_type": "Broken Trust Assumption / Interface Desync",
                "description": f"Node {target_node_id} fails to satisfy demand: '{eval.demand_description}'. Reasoning: {eval.reasoning}",
                "source_node": target_node_id
            })

    # Save to cache
    cache(cache_file, "write", {"hypothesis": new_hypothesis})

    # The operator.add reducer in MasterState safely appends this list
    return {
        "vulnerability_hypothesis": new_hypothesis
    }

# ==========================================
# Reviewer agent
# ==========================================

def synchronization_node(state: MasterState):
    """Dummy node to act as a Map-Reduce barrier."""
    return {}


def dispatch_reviewers(state: MasterState):
    """Groups reports and dispatches parallel reviewer threads using the Send API."""
    logging.warning(f"Running dispatch_reviewers")

    hypotheses = state.get("vulnerability_hypothesis", [])
    if not hypotheses:
        logging.warning(f"No vulnerabilities hypotheses to dispatch.")
        return END

    # Group all findings by the node they occurred in
    grouped_reports = defaultdict(list)
    for hypothesis in hypotheses:
        # Extract the node ID where the issue was found
        node_id = hypothesis.get("source_node", "Unknown")
        grouped_reports[node_id].append(hypothesis)

    commands = []
    for node_id, group in grouped_reports.items():
        payload = ReviewerState(
            node_id=node_id,
            expert_report=group,
            messages=[],
            filtered_reports=[]
        )
        commands.append(Send("reviewer_agent", payload))

        logging.warning(f"Sending payload: {payload}")

    logging.info(f"Dispatching {len(commands)} reviewers for {len(grouped_reports)} unique nodes.")
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

        # Format the group into a single, clean string for the LLM
        formatted_group_text = f"Target {state['node_id']}\n\nPotential Issues to Investigate:\n"

        for idx, item in enumerate(state.get("expert_report", []), 1):
            # Check if this is an Explorer Hypothesis
            if "vulnerability_type" in item:
                formatted_group_text += (
                    f"  --- Issue {idx} ---\n"
                    f"  Type: {item.get('vulnerability_type', 'Unknown')}\n"
                    f"  Description: {item.get('description', '')}\n\n"
                )
            # Check if this is a Failed Contract Demand
            elif "demand_description" in item:
                formatted_group_text += (
                    f"  --- Issue {idx} (Failed Security Contract) ---\n"
                    f"  Unmet Demand: {item.get('demand_description', '')}\n"
                    f"  Verifier Reasoning: {item.get('reasoning', '')}\n\n"
                )
            else:
                # Fallback for any other structure
                formatted_group_text += f"  --- Issue {idx} ---\n  {str(item)}\n\n"

        human_msg_content = (
            f"Review the following potential issues found in the target node.\n\n"
            f"{formatted_group_text}\n"
            f"INSTRUCTION: You must actively trace the execution path using your tools to prove or disprove reachability. "
            f"You MUST call `submit_evaluation` for EACH distinct issue listed above to log whether it is exploitable or a false positive."
        )

        human_msg = HumanMessage(content=human_msg_content)

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
    return {}
    # llm_with_tools = llm.bind_tools([
    #     tools.send_http_request,
    #     tools.mark_validation_complete,
    #     tools.take_notes
    # ])
    # current_cookies = state.get("cookies", {})
    #
    # if not state.get("messages"):
    #     sys_msg = SystemMessage(content=VALIDATOR_AGENT.get('prompt'))
    #     human_msg = HumanMessage(content=(
    #         f"Target Sandbox: {state['sandbox_url']}\n\n"
    #         f"Vulnerability to Prove:\n{state['report_to_test']}\n"
    #     ))
    #     messages = [sys_msg, human_msg]
    #     response = llm_with_tools.invoke(messages)
    #     return {"messages": [sys_msg, human_msg, response]}
    # else:
    #     # Update cookies from the recent history
    #     for msg in reversed(state["messages"]):
    #         if getattr(msg, "type", "") == "ai":
    #             break
    #         if getattr(msg, "type", "") == "tool" and getattr(msg, "name", "") == "send_http_request":
    #             if hasattr(msg, "artifact") and msg.artifact:
    #                 # Merge the new cookies into the current state
    #                 current_cookies.update(msg.artifact)
    #
    #     compacted_messages = compact_tool_history(state["messages"], safe_window=8)
    #     sys_msg = compacted_messages[0]
    #     human_msg = compacted_messages[1]
    #
    #     dynamic_msgs = []
    #     if state.get("notes"):
    #         notes_str = "\n".join([f"- {n}" for n in state["notes"]])
    #         saved_notes = f"\n\n### Persistent Scratchpad\n{notes_str}\n"
    #         dynamic_msgs.append(HumanMessage(content=saved_notes))
    #
    #     messages_to_pass = [sys_msg, human_msg] + dynamic_msgs + compacted_messages[2:]
    #
    #     response = llm_with_tools.invoke(messages_to_pass)
    #     return {"messages": [response], "cookies": current_cookies}


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

