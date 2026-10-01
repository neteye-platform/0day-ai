from pathlib import Path
import json
import re
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
from schemas import ExpertTask, AnalysisNote, BatchedAnalysisResult, CVEDemand, VerifierOutput, MANAGER_AGENT, EXPERT_AGENTS, CVE_ANALYZER_AGENT, VERIFIER_AGENT, REVIEWER_AGENT, VALIDATOR_AGENT
from utils import build_networkx_graph, compact_tool_history, extract_imports, get_cached_graph_data, get_node_code, index_file, run_osv_scanner, deduplicate_cves, cache, resolve_node_id, uses_namespace_in_ast, is_node_worth_scanning, format_node_context

# llm = ChatOllama(model="qwen36", temperature=0, reasoning=False, num_ctx=32768)
# Maximum combined code size (in chars) for a batched explorer dispatch.
EXPLORER_BATCH_CHAR_THRESHOLD = 7500

model = "deepseek-v4-flash"
fast_llm = ChatOpenAI(base_url="http://localhost:11434/v1", model=model, stream_usage=True, temperature=0.0, max_retries=5, max_tokens=4096, reasoning_effort="none") # Used for explorer and cve_analyzer
smart_llm = ChatOpenAI(base_url="http://localhost:11434/v1", model=model, stream_usage=True, temperature=0.0, max_retries=5, max_tokens=4096, reasoning_effort="none") # Used for explorer and cve_analyzer
# smart_llm = ChatOpenAI(base_url="http://localhost:11434/v1", model=model, stream_usage=True, temperature=0.3, max_retries=5, max_tokens=8192, reasoning_effort="none")

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

def _pack_node_batches(file_nodes: list[str], threshold: int) -> list[list[str]]:
    """Greedily pack node ids into batches whose combined code size stays below `threshold`.

    Nodes are packed in the order given. A batch is closed as soon as adding the next
    node would exceed the threshold. A single node whose code already exceeds the
    threshold becomes its own batch.
    """
    batches: list[list[str]] = []
    current: list[str] = []
    current_len = 0

    graph_data = get_cached_graph_data(settings.graph)

    for node_id in file_nodes:
        code = get_node_code(node_id) or ""
        context = format_node_context(graph_data, node_id)
        length = len(code) + len(context)
        if current and current_len + length > threshold:
            batches.append(current)
            current = []
            current_len = 0
        current.append(node_id)
        current_len += length

    if current:
        batches.append(current)

    return batches


def dispatch_explorers(state: MasterState):
    """Reads the Manager's instructions and creates a list of 'Send' objects.

    Nodes are batched per (community, source_file) so that multiple small nodes
    sharing a file are analyzed in a single explorer dispatch, as long as their
    combined code size stays under EXPLORER_BATCH_CHAR_THRESHOLD characters.
    """

    G = build_networkx_graph(settings.graph)
    commands: list[Send] = []
    skipped_nodes = 0
    batched_batches = 0
    single_batches = 0

    for task in state["expert_tasks"]:
        task = task if isinstance(task, dict) else task.model_dump()
        clean_id = task.get("target_community", "").lower().replace("community ", "").strip()
        community_nodes = [n for n, attr in G.nodes(data=True) if str(attr.get("community")) == clean_id]

        # Group eligible nodes by source_file (the batching key within this task).
        files: dict[str, list[str]] = defaultdict(list)

        # Filter out skeleton nodes, but keep functions, classes, and non-code files
        for node_id in community_nodes:
            node_data = G.nodes[node_id]

            if node_data.get("source_file", "").endswith(node_data.get("label")):
                # Skip skeletons only if they are not the only node in that file
                nodes_in_file = [n for n, attr in G.nodes(data=True) if attr.get("source_file") == node_data.get("source_file")]
                if len(nodes_in_file) <= 1:
                    continue

            source_file = Path(node_data.get("source_file", ""))
            if source_file.name in ["requirements.txt", "packages.json"]:
                continue

            # Drop inert nodes (pure types, empty skeletons, flat constants) to save LLM budget
            if not is_node_worth_scanning(node_id):
                skipped_nodes += 1
                logging.debug(f"Skipping inert node {node_id} (no executable signals).")
                continue

            files[node_data.get("source_file", "")].append(node_id)

        # Pack each file's nodes into batches whose combined code size stays under the threshold.
        for file_path, file_nodes in files.items():
            for batch in _pack_node_batches(file_nodes, EXPLORER_BATCH_CHAR_THRESHOLD):
                payload = ExplorerState(
                    node_ids=batch,
                    role=task.get("agent_role", ""),
                    task_description=task.get("task_description", "")
                )
                if len(batch) > 1:
                    batched_batches += 1
                else:
                    single_batches += 1
                logging.debug(
                    f"Dispatching explorer {task.get('agent_role', '')} on batch "
                    f"{batch} (from {file_path})."
                )
                commands.append(Send("explorer_agent", payload))

    logging.info(
        f"Dispatching {len(commands)} explorers ({batched_batches} multi-node batches, "
        f"{single_batches} single-node), skipped {skipped_nodes} inert nodes."
    )
    return commands


def dispatch_all_tasks(state: MasterState):
    commands = []
    commands.extend(dispatch_explorers(state))
    commands.extend(dispatch_cve_analyzers(state))
    return commands


def expert_explorer_node(state: ExplorerState) -> dict:
    node_ids = state.get("node_ids", [])
    role_name = state.get("role")

    if len(node_ids) == 1:
        return _explore_single(node_ids[0], role_name)
    return _explore_batch(node_ids, role_name)


def _explore_single(node_id: str, role_name: str) -> dict:
    # Check cache
    cache_file = settings.cache_dir / "notes" / f"{node_id}-{role_name}.json"
    cached_note = cache(cache_file, "read")
    if cached_note:
        return cached_note

    # LLM invocation
    source_code = get_node_code(node_id)

    sys_msg = SystemMessage(content=(
        f"{EXPERT_AGENTS[role_name]['prompt']}\n\n"
        f"{EXPERT_AGENTS['explorer_prompt']}\n\n"
    ))

    graph_data = get_cached_graph_data(settings.graph)
    target_node = next((node for node in graph_data.get("nodes", []) if node.get("id") == node_id), {})

    context = format_node_context(graph_data, node_id)
    context_block = f"{context}\n\n" if context else ""

    if target_node.get("source_file", "").endswith(target_node.get("label", "")):
        user_prompt = (
            f"{context_block}"
            f"Analyze this entire file skeleton.\n\n"
            f"```python\n{source_code}\n```"
        )
    else:
        label = target_node.get("label", "node")
        user_prompt = (
            f"{context_block}"
            f"Analyze the specific logic inside '{label}'.\n"
            f"The rest of the file is provided solely as context. "
            f"Do NOT look for vulnerabilities outside of '{label}'.\n\n"
            f"```python\n{source_code}\n```"
        )
    human_msg = HumanMessage(content=user_prompt)

    explorer_llm = fast_llm.with_structured_output(AnalysisNote, method="json_schema", strict=True)
    note = explorer_llm.invoke([sys_msg, human_msg])

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


def _explore_batch(node_ids: list[str], role_name: str) -> dict:
    # Deterministic cache key: sorted node ids joined by '__'
    batch_key = "__".join(sorted(node_ids))
    cache_file = settings.cache_dir / "notes" / f"batch-{batch_key}-{role_name}.json"
    cached_note = cache(cache_file, "read")
    if cached_note:
        return cached_note

    graph_data = get_cached_graph_data(settings.graph)

    sys_msg = SystemMessage(content=(
        f"{EXPERT_AGENTS[role_name]['prompt']}\n\n"
        f"{EXPERT_AGENTS['explorer_prompt']}\n\n"
        "You are analyzing MULTIPLE nodes in a single dispatch. "
        "Analyze each node independently and produce exactly one note per node."
    ))

    # Build a prompt that lists each node with its id, label, and code.
    sections = []
    for node_id in node_ids:
        target_node = next((node for node in graph_data.get("nodes", []) if node.get("id") == node_id), {})
        source_code = get_node_code(node_id)
        label = target_node.get("label", node_id)
        context_block = format_node_context(graph_data, node_id)
        if context_block:
            context_block += "\n\n"
        if target_node.get("source_file", "").endswith(target_node.get("label", "")):
            sections.append(
                f"### Node '{node_id}' ({label}) — entire file skeleton\n"
                f"{context_block}"
                f"```python\n{source_code}\n```"
            )
        else:
            sections.append(
                f"### Node '{node_id}' ({label})\n"
                f"{context_block}"
                f"Analyze the specific logic inside '{label}'. "
                f"The rest of the file is provided solely as context; "
                f"do NOT look for vulnerabilities outside of '{label}'.\n"
                f"```python\n{source_code}\n```"
            )

    user_prompt = (
        "Analyze each of the following nodes independently. "
        "For every node, produce a separate analysis note tagged with the matching node_id.\n\n"
        + "\n\n".join(sections)
    )
    human_msg = HumanMessage(content=user_prompt)

    explorer_llm = fast_llm.with_structured_output(BatchedAnalysisResult, method="json_schema", strict=True)
    result = explorer_llm.invoke([sys_msg, human_msg])

    result = result if isinstance(result, dict) else result.model_dump()
    raw_notes = result.get("notes", [])

    notes = []
    extracted_vulns = []

    for note in raw_notes:
        dict_note = note if isinstance(note, dict) else note.model_dump()
        node_id = dict_note.get("node_id")
        if not node_id:
            continue

        # Drop assumptions about the node under analysis
        valid_assumptions = []
        for assumption in dict_note.get("assumptions_to_verify", []):
            if node_id == resolve_node_id(assumption.get("module"), assumption.get("symbol")):
                continue # Drop it

            valid_assumptions.append(assumption)
        dict_note["assumptions_to_verify"] = valid_assumptions

        # Extract and remove the list from dict_note
        raw_hypotheses = dict_note.pop("vulnerability_hypothesis", [])
        for hyp in raw_hypotheses:
            extracted_vulns.append({
                "node_id": node_id,
                "cwe_id": hyp.get("cwe_id", "OTHER_UNCATEGORIZED"),
                "description": hyp.get("description", ""),
                "status": "hypothesis"
            })

        notes.append(dict_note)

    # Save to cache
    cache(cache_file, "write", {"notes": notes, "vulnerabilities": extracted_vulns})

    return {
        "notes": notes,
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
    sys_msg = SystemMessage(content=(
        f"{CVE_ANALYZER_AGENT['prompt']}\n\n"
        # f"{parser.get_format_instructions()}"
    ))
    human_msg = HumanMessage(content=(
        f"Analyze this CVE affecting the package '{package_name}':\n\n"
        f"CVE ID: {cve_id}\n"
        f"Description: {details}\n\n"
    ))

    cve_analyzer_llm = fast_llm.with_structured_output(CVEDemand, method="json_schema", strict=True)
    demand = cve_analyzer_llm.invoke([sys_msg, human_msg])

    dict_demand = demand if isinstance(demand, dict) else demand.model_dump()
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

def build_caller_map(graph_data: dict):
    callers_map = defaultdict(list)
    for edge in graph_data.get("links", []):
        callers_map[edge.get("target")].append(edge.get("source"))
    return callers_map


def build_import_map(graph_data: dict):
    node_imports_map = {}
    for node in graph_data.get("nodes", []):
        node_id, src_path = node.get("id"), node.get("source_file")

        if not node_id or not src_path:
            continue

        source_file = settings.app_path / Path(src_path)
        if source_file.exists() and node_id not in node_imports_map:
            try:
                with open(source_file, "r", encoding="utf-8") as f:
                    node_imports_map[node_id] = set(extract_imports(f.read(), src_path))
            except Exception:
                node_imports_map[node_id] = set()
    return node_imports_map


def aggregate_demands_node(state: MasterState):
    grouped_demands = defaultdict(list)
    updated_notes = []

    graph_data = get_cached_graph_data(settings.graph)
    callers_map = build_caller_map(graph_data)
    node_imports_map = build_import_map(graph_data)

    logging.info(f"Loaded graph data: {len(callers_map)} caller entries, {len(node_imports_map)} import entries.")
    notes = state.get("notes", [])
    cves = state.get("cve_demands", [])
    logging.info(f"Processing {len(notes)} notes and {len(cves)} CVE demands.")

    # Process Explorer Notes
    for note in notes:
        dict_note = note if isinstance(note, dict) else note.model_dump()
        current_node_id = dict_note.get("node_id")
        demands = dict_note.setdefault("demands", [])

        has_source = any(
            isinstance(iface, str) and (iface.upper().startswith("[SOURCE]") or iface.lower().startswith("source"))
            for iface in dict_note.get("business_interfaces", [])
        )

        # Convert localized vulnerabilities into upstream demands
        if not has_source and dict_note.get("vulnerability_hypotheses"):
            vulns = dict_note.pop("vulnerability_hypotheses")
            logging.info(f"[{current_node_id}] Treated as internal sink. Converting {len(vulns)} vulnerabilities into upstream demands.")
            
            for vuln in vulns:
                demands.append({
                    "direction": "upstream",
                    "target": "context (auto-converted internal sink)",
                    "description": f"Must prevent {vuln.get('cwe_id')} at {vuln.get('vulnerable_component')}"
                })
            dict_note["vulnerability_hypotheses"] = []

        # Process the unified SecurityDemand objects
        for demand in demands:
            direction = demand.get("direction")
            target_str = demand.get("target", "")
            desc = demand.get("description")

            if direction == "downstream":
                # STRIP LLM HALLUCINATIONS: Remove backticks, parentheses, and arguments
                clean_target = re.sub(r'\(.*?\)', '', target_str).replace('`', '').strip()

                # Handle correct `::` format, OR fallback to `module.symbol` dot notation
                if "::" in clean_target:
                    module, symbol = clean_target.split("::", 1)
                elif "." in clean_target:
                    module, symbol = clean_target.rsplit(".", 1)
                else:
                    module, symbol = clean_target, "unknown"

                if target_node_id := resolve_node_id(module, symbol):
                    grouped_demands[target_node_id].append({
                        "source": current_node_id,
                        "type": "explorer_downstream_assumption",
                        "description": desc
                    })
                else:
                    logging.warning(f"[{current_node_id}] DOWNSTREAM DROP: Could not resolve '{module}' / '{symbol}' (Original: {target_str})")

            elif direction == "upstream":
                # DEBUG: Check if incoming edges are missing
                callers = callers_map.get(current_node_id, [])
                if callers:
                    for caller_id in callers:
                        grouped_demands[caller_id].append({
                            "source": current_node_id,
                            "type": "explorer_upstream_assumption",
                            "description": desc,
                            "parameter_name": target_str
                        })
                else:
                    logging.warning(f"[{current_node_id}] UPSTREAM DROP: No callers found in graph for this node.")
                    
            else:
                logging.warning(f"[{current_node_id}] UNKNOWN DIRECTION: '{direction}'. Demand dropped.")

        updated_notes.append(dict_note)

    # Process CVE Demands
    for demand in cves:
        target_import = demand.get("import_namespace", "")
        source_cve = demand.get("source_cve", "unknown")

        combined_desc = (
            f"Security Context: {demand.get('security_assumption')} | "
            f"Trigger: {demand.get('trigger_condition')}"
        )

        matched_any = False
        for node_id, imports in node_imports_map.items():
            if target_import in imports:
                # DEBUG: Check if AST strict matching is rejecting the import
                if uses_namespace_in_ast(node_id, target_import):
                    grouped_demands[node_id].append({
                        "source": source_cve,
                        "type": "cve_assumption",
                        "description": combined_desc
                    })
                    matched_any = True
                else:
                    logging.info(f"[{node_id}] CVE SKIP: '{target_import}' found in imports but uses_namespace_in_ast() returned False.")

        if not matched_any:
            logging.warning(f"[CVE DROP] {source_cve} for '{target_import}' matched 0 nodes in the graph.")

    # Log the accurate total by summing the lengths of the lists
    total_demands = sum(len(d) for d in grouped_demands.values())
    logging.info(f"Summary: Grouped {total_demands} total demands across {len(grouped_demands)} target nodes.")

    return {
        "grouped_demands": dict(grouped_demands), 
        "notes": updated_notes 
    }

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
    sys_msg = SystemMessage(content=f"{VERIFIER_AGENT['prompt']}")
    human_msg = HumanMessage(content=f"```python\n{target_code}\n```\n\nSecurity Demands:\n{demands_string}")

    structured_llm = smart_llm.with_structured_output(VerifierOutput, method="json_schema", strict=True)
    response = structured_llm.invoke([sys_msg, human_msg])
    response = response if isinstance(response, dict) else response.model_dump()

    new_vulnerabilities = []
    for eval in response.get("evaluations", []):
        # DELEGATED, OUT_OF_SCOPE, and MET will be safely ignored
        if eval.get("status") == "FAILED":
            # Retrieve the clean, original description from Python memory
            original_desc = demand_lookup.get(eval.get("demand_id"), "No description found.")

            new_vulnerabilities.append({
                "node_id": target_node_id,
                "cwe_id": eval.get("cwe_id"),
                "description": f"Fails to satisfy demand: '{original_desc}'. Reasoning: {eval.get("reasoning")}",
                "status": "hypothesis",
                "demand_id": eval.get("demand_id"),
                "vulnerable_component": eval.get("demand_id")
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

    llm_with_tools = smart_llm.bind_tools([
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
    llm_with_tools = smart_llm.bind_tools([
        tools.send_http_request,
        # tools.list_files,
        # tools.read_file,
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

