from pathlib import Path
import json
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command, Send
from langgraph.graph import END
from typing import Any
from collections import defaultdict
import networkx as nx
import hashlib
import logging
from json_repair import repair_json

from languages import SYMBOL_QUERIES
import settings
import tools
from state import MasterState, ExplorerState, CVEAnalyzerState, VerifierState, ReviewerState, ValidatorState
from schemas import ManagerOutput, ExpertTask, AnalysisNote, CVEDemand, VerifierOutput, MANAGER_AGENT, EXPERT_AGENTS, CVE_ANALYZER_AGENT, VERIFIER_AGENT, REVIEWER_AGENT, VALIDATOR_AGENT
from utils import build_networkx_graph, compact_tool_history, extract_imports, get_cached_graph_data, get_node_code, index_file, run_osv_scanner, deduplicate_cves, cache, resolve_node_id, uses_namespace_in_ast

# llm = ChatOllama(model="qwen36", temperature=0, reasoning=False, num_ctx=32768)
llm = ChatOpenAI(base_url="http://localhost:11434/v1", model="laguna", stream_usage=True, temperature=0.4, max_retries=5)

# ==========================================
# Preprocessor
# ==========================================

def preprocessor_node(state: MasterState) -> dict[str, Any]:
    # Run the OSV scanner to parse manifests and query the database
    raw_vulns = run_osv_scanner(str(settings.app_path))
    logging.info(f"Found {len(raw_vulns)} raw vulns")
    clean_vulns = deduplicate_cves(raw_vulns)
    logging.info(f"{len(clean_vulns)} remaining CVEs after deduplication")

    # Build the Global Symbol Index
    global_symbol_index = []

    # Recursively find all files in the application directory
    for filepath in settings.app_path.rglob("*"):
        if filepath.is_file():
            # Quick check to avoid passing irrelevant files (like images or binaries)
            if filepath.suffix.lower() in SYMBOL_QUERIES:
                try:
                    file_symbols = index_file(filepath)
                    if file_symbols:
                        global_symbol_index.extend(file_symbols)
                except Exception as e:
                    logging.warning(f"Failed to index {filepath}: {e}")

    # Write the index to disk instead of putting it in the state
    index_file_path = settings.app_path / ".ast_symbol_index.json"

    with open(index_file_path, "w", encoding="utf-8") as f:
        json.dump(global_symbol_index, f)

    logging.info(f"Saved {len(global_symbol_index)} symbols to {index_file_path}.")

    return {
        "known_vulns": clean_vulns
    }

# ==========================================
# Manager
# ==========================================

def manager_agent_node(state: MasterState) -> dict[str, Any]:
    """The Manager agent assigns tasks completely deterministically using heuristics."""
    G = build_networkx_graph(settings.graph, settings.communities_to_analyze)

    # Group nodes by community
    community_groups = defaultdict(list)
    for node_id, data in G.nodes(data=True):
        comm_id = data.get("community")
        if comm_id is not None:
            community_groups[comm_id].append(data)

    # Keyword mapping for the expert agents
    expert_keywords = MANAGER_AGENT["expert_keywords"]
    expert_descriptions = MANAGER_AGENT["expert_descriptions"]
    heuristic_tasks = []

    for comm_id, nodes in community_groups.items():
        scores = {agent: 0 for agent in expert_keywords}

        for node in nodes:
            # Direct node matches get higher weight (e.g., +2)
            direct_string = f"{node.get('label', '')} {node.get('source_file', '')} {node.get('id', '')}".lower()

            # Neighbor matches get lower weight (e.g., +1) to prevent inheritence skew
            neighbor_string = ""
            if G.has_node(node.get("id")):
                for neighbor in G.successors(node.get("id")):
                    neighbor_data = G.nodes.get(neighbor, {})
                    neighbor_string += f" {neighbor_data.get('label', '')}".lower()

            for agent, keywords in expert_keywords.items():
                for keyword in keywords:
                    # Direct match
                    if keyword in direct_string:
                        scores[agent] += 2
                    # Neighbor match
                    if keyword in neighbor_string:
                        scores[agent] += 1

        max_score = max(scores.values())
        ASSIGNMENT_THRESHOLD = max(2, int(max_score * 0.5))

        assigned = False
        # Assign ALL agents that pass the threshold, adhering to the architecture
        for agent, score in scores.items():
            if score >= ASSIGNMENT_THRESHOLD:
                heuristic_tasks.append(
                    ExpertTask(
                        target_community=f"Community {comm_id}",
                        agent_role=agent,
                        task_description=expert_descriptions[agent]
                    )
                )
                assigned = True

        # Fallback if no agents scored anything
        if not assigned:
            heuristic_tasks.append(
                ExpertTask(
                    target_community=f"Community {comm_id}",
                    agent_role="LogicFlowAuditor",
                    task_description=expert_descriptions["LogicFlowAuditor"]
                )
            )

    return {"expert_tasks": heuristic_tasks}

# ==========================================
# Explorer agents
# ==========================================

def dispatch_explorers(state: MasterState):
    """Reads the Manager's instructions and creates a list of 'Send' objects."""

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

            # if node_data.get("source_file", "").endswith(node_data.get("label")):
            #     continue

            source_file = Path(node_data.get("source_file", ""))
            if source_file.name in ["requirements.txt", "packages.json"]:
                continue

            payload = ExplorerState(
                node_id=node_id,
                role=task.get("agent_role", ""),
                task_description=task.get("task_description", "")
            )
            logging.info(f"Dispatching explorer {task.get('agent_role', '')} on node {node_id}.")

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

    # Drop assumptions about the node under analysis
    valid_assumptions = []
    for assumption in dict_note.get("assumptions_to_verify", []):
        if node_id == resolve_node_id(assumption.get("module"), assumption.get("symbol")):
            continue # Drop it

        valid_assumptions.append(assumption)
    dict_note["assumptions_to_verify"] = valid_assumptions

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
    fixed_json_string = repair_json(response.content)
    demand: CVEDemand = parser.invoke(fixed_json_string)

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
    updated_notes = []

    # Extract graph data and build a caller map for upstream routing
    graph_data = get_cached_graph_data(settings.graph)
    callers_map = {}  # Maps a node_id to a list of its callers (incoming edges)
    for edge in graph_data.get("edges", []):
        src = edge.get("source")
        dst = edge.get("target")
        if dst not in callers_map:
            callers_map[dst] = []
        callers_map[dst].append(src)

    # Extract imports for CVE matching
    for node in graph_data.get("nodes", []):
        node_id = node.get("id")
        source_file_path = node.get("source_file")

        if not node_id or not source_file_path:
            continue

        source_file = settings.app_path / Path(source_file_path)
        if source_file.exists() and node_id not in node_imports_map:
            try:
                with open(source_file, "r", encoding="utf-8") as f:
                    code_content = f.read()
                node_imports_map[node_id] = set(extract_imports(code_content, source_file_path))
            except Exception:
                node_imports_map[node_id] = set()

    # Process Explorer Notes
    for note in state.get("notes", []):
        # Ensure we are working with a mutable dictionary
        dict_note = note if isinstance(note, dict) else note.model_dump()
        current_node_id = dict_note.get("node_id")

        if "upstream_assumptions" not in dict_note:
            dict_note["upstream_assumptions"] = []

        # Check if the node is an internal sink (has 0 source interfaces)
        has_source = any(
            iface.get("interface_type") == "source" 
            for iface in dict_note.get("business_interfaces", [])
        )

        if not has_source and dict_note.get("vulnerability_hypotheses"):
            # Convert localized vulnerabilities into upstream demands
            for vuln in dict_note.get("vulnerability_hypotheses", []):
                dict_note["upstream_assumptions"].append({
                    "description": f"Must prevent {vuln.get('cwe_id')}: {vuln.get('description')}",
                    "parameter_name": "context (auto-converted internal sink)"
                })
            # Wipe the localized vulnerabilities so they don't reach the Reviewer
            dict_note["vulnerability_hypotheses"] = []

        # --- DOWNSTREAM ASSUMPTIONS ---
        for assumption in dict_note.get("downstream_assumptions", []):
            target_node_id = resolve_node_id(assumption.get("module"), assumption.get("symbol"))
            if target_node_id:
                if target_node_id not in grouped_demands:
                    grouped_demands[target_node_id] = []

                grouped_demands[target_node_id].append({
                    "source": current_node_id,
                    "type": "explorer_downstream_assumption",
                    "description": assumption.get("description")
                })

        # --- UPSTREAM ASSUMPTIONS ---
        for assumption in dict_note.get("upstream_assumptions", []):
            # Fetch all nodes that call this current_node_id
            callers = callers_map.get(current_node_id, [])
            for caller_id in callers:
                if caller_id not in grouped_demands:
                    grouped_demands[caller_id] = []

                grouped_demands[caller_id].append({
                    "source": current_node_id,
                    "type": "explorer_upstream_assumption",
                    "description": assumption.get("description"),
                    "parameter_name": assumption.get("parameter_name", "unknown")
                })

        updated_notes.append(dict_note)

    # Process CVE Demands (unchanged)
    for demand in state.get("cve_demands", []):
        target_import = demand.get("import_namespace", "")

        for node_id, imports in node_imports_map.items():
            if target_import in imports:
                if uses_namespace_in_ast(node_id, target_import):
                    if node_id not in grouped_demands:
                        grouped_demands[node_id] = []

                    grouped_demands[node_id].append({
                        "source": demand.get("source_cve"),
                        "type": "cve_assumption",
                        "description": demand.get("security_assumption")
                    })

    logging.info(f"Grouped {len(grouped_demands)} demands.")

    # Return both the grouped demands and the filtered notes
    return {
        "grouped_demands": grouped_demands,
        "notes": updated_notes 
    }

# def aggregate_demands_node(state: MasterState):
#     grouped_demands = {}
#     node_imports_map = {}
#
#     # Extract imports for all graph nodes
#     graph_data = get_cached_graph_data(settings.graph)
#     for node in graph_data.get("nodes", []):
#         node_id = node.get("id")
#         source_file_path = node.get("source_file")
#
#         if not node_id or not source_file_path:
#             continue
#
#         # Resolve full path on disk
#         source_file = settings.app_path / Path(source_file_path)
#         if source_file.exists() and node_id not in node_imports_map:
#             try:
#                 with open(source_file, "r", encoding="utf-8") as f:
#                     code_content = f.read()
#                 # Use your tree-sitter helper
#                 node_imports_map[node_id] = set(extract_imports(code_content, source_file_path))
#             except Exception:
#                 node_imports_map[node_id] = set()
#
#     # Process Explorer Notes (for assumptions_to_verify)
#     for note in state.get("notes", []):
#         dict_note = note if isinstance(note, dict) else note.model_dump()
#         dict_note.get("node_id")
#
#         for assumption in dict_note.get("assumptions_to_verify", []):
#             target_node_id = resolve_node_id(assumption.get("module"), assumption.get("symbol"))
#             if target_node_id:
#                 if target_node_id not in grouped_demands:
#                     grouped_demands[target_node_id] = []
#
#                 grouped_demands[target_node_id].append({
#                     "source": dict_note.get("node_id"),
#                     "type": "explorer_assumption",
#                     "description": assumption.get("description")
#                 })
#
#     # Process CVE Demands (now checked against ALL nodes in the codebase)
#     for demand in state.get("cve_demands", []):
#         target_import = demand.get("import_namespace", "")
#
#         for node_id, imports in node_imports_map.items():
#             if target_import in imports:
#                 if uses_namespace_in_ast(node_id, target_import):
#                     if node_id not in grouped_demands:
#                         grouped_demands[node_id] = []
#
#                     grouped_demands[node_id].append({
#                         "source": demand.get("source_cve"),
#                         "type": "cve_assumption",
#                         "description": demand.get("security_assumption")
#                     })
#
#     logging.info(f"Grouped {len(grouped_demands)} demands.")
#     return {"grouped_demands": grouped_demands}

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
        return {"vulnerabilities": []}

    # Deterministic Cache Invalidation
    # Hash the demands so if upstream/downstream contracts change, the cache busts.
    demands_hash = hashlib.md5(json.dumps(demands, sort_keys=True).encode()).hexdigest()
    cache_file = settings.cache_dir / "contract_verifier" / f"{target_node_id}_{demands_hash}.json"

    cached_data = cache(cache_file, "read")
    if cached_data:
        return {"vulnerabilities": cached_data.get("hypothesis", [])}

    # Build a lookup map of demand_id -> original description
    demand_lookup = {}
    formatted_demands = []

    for d in demands:
        # Determine the ID you assigned in the prompt formatting
        # (e.g., matching whatever you put inside the [ID: ...] tag)
        d_id = d.get("parameter_name") or d.get("source") or "unknown"
        demand_lookup[d_id] = d.get("description")

        dtype = d.get("type")
        source = d.get("source")
        desc = d.get("description")

        if dtype == "explorer_upstream_assumption":
            param = d.get("parameter_name", "unknown")
            formatted_demands.append(
                f"- [ID: {param}] [DEMAND FROM CALLEE] You call '{source}'. It demands: '{desc}'"
            )
        elif dtype == "cve_assumption":
            formatted_demands.append(
                f"- [ID: {source}] [LIBRARY CVE MITIGATION] Known constraint: '{desc}'"
            )
        else:
            formatted_demands.append(f"- [ID: {source}] {desc}")

    demands_string = "\n".join(formatted_demands)

    # LLM Invocation
    parser = PydanticOutputParser(pydantic_object=VerifierOutput)
    sys_msg = SystemMessage(content=f"{VERIFIER_AGENT['prompt']}\n\n{parser.get_format_instructions()}")
    human_msg = HumanMessage(content=f"```python\n{target_code}\n```\n\nSecurity Demands:\n{demands_string}")

    response = llm.invoke([sys_msg, human_msg])
    fixed_json_string = repair_json(response.content)
    parsed_output: VerifierOutput = parser.invoke(fixed_json_string)

    new_vulnerabilities = []
    for eval in parsed_output.evaluations:
        # DELEGATED, OUT_OF_SCOPE, and MET will be safely ignored
        if eval.status == "FAILED":
            # Retrieve the clean, original description from Python memory
            original_desc = demand_lookup.get(eval.demand_id, "No description found.")

            new_vulnerabilities.append({
                "node_id": target_node_id,
                "cwe_id": eval.cwe_id,
                "description": f"Fails to satisfy demand: '{original_desc}'. Reasoning: {eval.reasoning}",
                "status": "hypothesis",
                "demand_id": eval.demand_id,
                "vulnerable_component": eval.demand_id
            })

    # Save to cache
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
            vulnerabilities=[],
            messages=[]
        )
        commands.append(Send("reviewer_agent", payload))

    logging.info(f"Dispatching {len(commands)} reviewers.")
    return commands


def reviewer_agent_node(state: ReviewerState) -> dict | Command:
    """Review the vulnerability reports and keep only what is actually relevant"""
    node_id = state.get("node_id")
    report = state.get("expert_report", {})

    # Check Cache
    if not state.get("messages"):
        report_hash = hashlib.md5(json.dumps(report, sort_keys=True).encode()).hexdigest()
        cache_file = settings.cache_dir / "reviewer" / f"{node_id}_{report_hash}.json"
        # cache_file = settings.cache_dir / "reviewer" / "converter_convert_job_convert_html_d1f9ef5218429efd22c5c9187d2c7e53.json"
        cached_data = cache(cache_file, "read")
        if cached_data:
            logging.info("Reviewer cache hit.")
            return Command(
                update={
                    "vulnerabilities": [cached_data]
                }
            )

    llm_with_tools = llm.bind_tools([
            tools.read_source_code,
            tools.search_codebase,
            tools.get_node_connections,
            tools.get_definition,
            tools.submit_evaluation
        ],
        parallel_tool_calls=False
    )

    if not state.get("messages"):
        sys_msg = SystemMessage(content=REVIEWER_AGENT.get('prompt'))

        report = state.get("expert_report", {})
        node_id = state.get("node_id")

        formatted_vuln = (
            f"Target: {state['node_id']}\n\n"
            f"Potential Issue to Investigate:\n"
            f"- CWE ID: {report.get('cwe_id', 'Unknown')}\n"
            f"- Component: {report.get('vulnerable_component', 'Unknown')}\n"
            f"- Description: {report.get('description', '')}\n"
        )

        if node_id:
            target_node_source = get_node_code(node_id)
            formatted_vuln += (
                f"--- TARGET NODE SOURCE CODE ---\n"
                f"```\n"
                f"{target_node_source}\n"
                f"```\n"
            )

        human_msg = HumanMessage(content=(
            f"Review the following potential issues found in the target node.\n\n"
            f"{formatted_vuln}"
        ))

        response = llm_with_tools.invoke([sys_msg, human_msg])
        return {"messages": [sys_msg, human_msg, response]}

    else:
        compacted_messages = compact_tool_history(state["messages"])
        response = llm_with_tools.invoke(compacted_messages)
        return {"messages": [response]}


def reviewer_router(state: ReviewerState):
    """Routes based on the tool called by the reviewer LLM."""
    messages = state["messages"]
    if len(messages) == 0 and state.get("vulnerabilities"):
        return "__end__"

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
            vulnerabilities=[], 
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
        # Build a structured string for the LLM
        report = state['report_to_test']
        params = report.get('required_parameters', [])
        params_str = "\n".join([f"  - {p}" for p in params]) if params else "  None specified"
        formatted_report = (
            f"--- CORE VULNERABILITY ---\n"
            f"Vulnerability ID: {report.get('vuln_id', 'Unknown')}\n"
            f"CWE ID: {report.get('cwe_id', 'Unknown')}\n"
            f"Target Node: {report.get('node_id', 'Unknown')}\n\n"
            f"--- CONTEXT & REASONING ---\n"
            f"Description: {report.get('description', 'None')}\n\n"
            f"Reviewer Reasoning: {report.get('reviewer_reasoning', 'None')}\n\n"
            f"--- ATTACK VECTOR ---\n"
            f"Entry Point: {report.get('entry_point_url', 'Unknown')}\n"
            f"Method: {report.get('http_method', 'Unknown')}\n"
            f"Auth Required: {report.get('auth_required', False)}\n"
            f"Required Parameters:\n{params_str}"
        )
        human_msg = HumanMessage(content=(
            f"Target Sandbox: {state['sandbox_url']}\n\n"
            f"Vulnerability to Prove:\n{formatted_report}\n"
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

