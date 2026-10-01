from pathlib import Path
import json
import os
import re
import subprocess
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from tavily import TavilyClient
from langgraph.types import Command, Send
from langgraph.graph import END
from typing import Any
from collections import defaultdict
import hashlib
import logging
import threading
import uuid

from languages import SYMBOL_QUERIES
import settings
import tools
from state import MasterState, ExplorerState, CVEAnalyzerState, ThreatIntelState, VerifierState, ReviewerState, ValidatorState
from schemas import ExpertTask, AnalysisNote, BatchedAnalysisResult, CVEAnalysis, VerifierOutput, MANAGER_AGENT, EXPERT_AGENTS, CVE_ANALYZER_AGENT, THREAT_INTEL_AGENT, VERIFIER_AGENT, REVIEWER_AGENT, VALIDATOR_AGENT
from utils import build_networkx_graph, compact_tool_history, extract_imports, get_cached_graph_data, get_node_code, index_file, run_osv_scanner, run_osv_scanner_image, deduplicate_cves, cache, safe_cache_filename, resolve_node_id, uses_namespace_in_ast, is_node_worth_scanning, format_node_context, find_container_builds, build_images, start_sandbox, extract_container_artifacts, load_code_corpus, find_unsupported_code_files, read_file_text, clear_aggregate_caches, is_high_severity

# fast_llm = ChatOllama(model="gemma4:cloud", temperature=0.2, reasoning=False, num_ctx=32768)
# smart_llm = ChatOllama(model="gemma4:cloud", temperature=0.6, reasoning=False, num_ctx=32768)
# Maximum combined code size (in chars) for a batched explorer dispatch.
EXPLORER_BATCH_CHAR_THRESHOLD = 10000

_agent_progress: dict[str, dict[str, int]] = {}
_agent_progress_lock = threading.Lock()

class ErrorLoggingCallbackHandler(BaseCallbackHandler):
    def on_llm_error(self, error: BaseException, **kwargs: Any) -> Any:
        """Run when LLM errors out completely."""

        # The callback kwargs contain the payloads sent to the LLM
        payload = kwargs.get('prompts') or kwargs.get('messages')

        # Combine the message into a single formatted block
        error_message = (
            "\n" + "="*40 + "\n"
            "LLM EXHAUSTED ALL RETRIES\n"
            + "="*40 + "\n"
            f"REQUEST PAYLOAD:\n{payload}\n\n"
            f"ERROR DETAILS:\n{error}\n"
            + "="*40
        )

        logging.error(error_message)

base_llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    model="deepseek-v4-flash",
    stream_usage=True,
    temperature=0.4
)
# NOTE: Retry is handled at the graph level via RetryPolicy on every node
# (see graph.py build_graph -> set_node_defaults), so no per-call retry wrapper
# is needed here. This avoids double retry layers on top of the openai client.

fast_llm = base_llm.bind(temperature=0.2, max_tokens=4096, reasoning_effort="none")
smart_llm = base_llm.bind(temperature=0.8, max_tokens=16384, reasoning_effort="medium")


# ==========================================
# Bootstrap
# ==========================================

def bootstrap_node(state: MasterState) -> dict[str, Any]:
    """Ensure the knowledge graph exists before the parallel branches start."""
    if not settings.graph.exists():
        logging.info(f"Graph {settings.graph} not found. Running graphify extract...")
        subprocess.run(
            ["graphify", "extract", str(settings.app_path), "--code-only"],
            check=True,
        )
    return {}

# ==========================================
# Preprocessor
# ==========================================

def preprocessor_node(state: MasterState) -> dict[str, Any]:
    raw_vulns = []

    # Build any container image(s) found in the app repo, scan those, then start
    # the sandbox in the background and record its runtime data in the state.
    builds = find_container_builds(settings.app_path)
    sandbox_data = None
    built_images = []
    if builds:
        for kind, build_file in builds:
            tag = settings.docker_image_tag or f"vulnscan-{settings.app_path.name}:latest"
            images = build_images(kind, build_file, tag)
            if images:
                built_images.extend(images)
                for image in images:
                    logging.info(f"Scanning container image {image} with osv-scanner.")
                    raw_vulns.extend(run_osv_scanner_image(image))
                sandbox_data = start_sandbox(kind, build_file, images[0], settings.app_path.name)
            else:
                logging.warning(f"Failed to build image from {build_file}. Falling back to repo scan.")
                raw_vulns = run_osv_scanner(settings.app_path)
    else:
        logging.warning("No Dockerfile or compose file found. Falling back to repo scan.")
        raw_vulns = run_osv_scanner(settings.app_path)

    # Snapshot the built container image(s): extract curated config/build
    # artifacts + a full filesystem index so the reviewer can inspect the
    # effective runtime configuration without touching the live sandbox.
    # Deterministic (image-based) and independent of sandbox startup success.
    if built_images:
        artifact_summary = extract_container_artifacts(built_images)
        logging.info(f"Extracted container artifacts for {len(artifact_summary)} image(s).")
    else:
        logging.info("No container images built; skipping container artifact extraction.")

    logging.info(f"Found {len(raw_vulns)} raw vulns")
    clean_vulns = deduplicate_cves(raw_vulns)
    logging.info(f"{len(clean_vulns)} remaining CVEs after deduplication")

    # Log an error for any code files in a language we cannot analyze, so the
    # operator knows some of the codebase is invisible to AST-based analysis.
    unsupported = find_unsupported_code_files(get_cached_graph_data(settings.graph))
    for ext, files in unsupported.items():
        logging.error(
            f"Unsupported language '{ext}': {len(files)} code file(s) "
            f"cannot be analyzed ({', '.join(files)})."
        )

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
        "known_vulns": clean_vulns,
        "sandbox_url": sandbox_data["sandbox_url"] if sandbox_data else None,
        "container_name": sandbox_data["container_name"] if sandbox_data else None,
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

        assigned = 0
        # Assign only the top-K best-scoring roles (settings.max_experts_per_community)
        # that clear the threshold, instead of every role above it, so distinct
        # expert roles do not re-scan the same nodes.
        for agent, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            if assigned >= settings.max_experts_per_community or score < ASSIGNMENT_THRESHOLD:
                break
            heuristic_tasks.append(
                ExpertTask(
                    target_community=f"Community {comm_id}",
                    agent_role=agent,
                    task_description=expert_descriptions[agent]
                ).model_dump()
            )
            assigned += 1

        # Fallback if no agents scored anything
        if not assigned:
            heuristic_tasks.append(
                ExpertTask(
                    target_community=f"Community {comm_id}",
                    agent_role="LogicFlowAuditor",
                    task_description=expert_descriptions["LogicFlowAuditor"]
                ).model_dump()
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
    progress_id = uuid.uuid4().hex
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

            # Drop inert nodes (pure types, empty skeletons, flat constants) to save LLM budget
            if not is_node_worth_scanning(node_id):
                skipped_nodes += 1
                logging.debug(f"Skipping inert node {node_id} (no executable signals).")
                continue

            files[node_data.get("source_file", "")].append(node_id)

        # Pack each file's nodes into batches whose combined code size stays under the threshold,
        # unless batching is disabled, in which case every node is its own dispatch.
        for file_path, file_nodes in files.items():
            if settings.explorer_batching_enabled:
                batches = _pack_node_batches(file_nodes, EXPLORER_BATCH_CHAR_THRESHOLD)
            else:
                batches = [[node_id] for node_id in file_nodes]
            for batch in batches:
                payload = ExplorerState(
                    node_ids=batch,
                    role=task.get("agent_role", ""),
                    task_description=task.get("task_description", ""),
                    progress_id=progress_id,
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

    if commands:
        with _agent_progress_lock:
            _agent_progress[progress_id] = {"total": len(commands), "completed": 0}

    logging.info(
        "Starting explorer scan: 0/%d complete, %d remaining "
        "(%d multi-node batches, %d single-node), skipped %d inert nodes.",
        len(commands),
        len(commands),
        batched_batches,
        single_batches,
        skipped_nodes,
    )

    return commands


def _log_explorer_completion(state: ExplorerState) -> None:
    progress_id = state.get("progress_id")
    if not progress_id:
        return

    with _agent_progress_lock:
        progress = _agent_progress.get(progress_id)
        if progress is None:
            return
        progress["completed"] += 1
        completed = progress["completed"]
        total = progress["total"]
        remaining = total - completed
        if remaining == 0:
            _agent_progress.pop(progress_id, None)

    logging.info(
        "Explorer progress: %d/%d complete, %d remaining (role=%s, nodes=%s).",
        completed,
        total,
        remaining,
        state.get("role", "unknown"),
        ", ".join(state.get("node_ids", [])),
    )


def _start_agent_progress(total: int) -> str:
    progress_id = uuid.uuid4().hex
    if total:
        with _agent_progress_lock:
            _agent_progress[progress_id] = {"total": total, "completed": 0}
    return progress_id


def _log_agent_completion(progress_id: str, agent_name: str, detail: str) -> None:
    if not progress_id:
        return

    with _agent_progress_lock:
        progress = _agent_progress.get(progress_id)
        if progress is None:
            return
        progress["completed"] += 1
        completed = progress["completed"]
        total = progress["total"]
        remaining = total - completed
        if remaining == 0:
            _agent_progress.pop(progress_id, None)

    logging.info(
        "%s progress: %d/%d complete, %d remaining (%s).",
        agent_name,
        completed,
        total,
        remaining,
        detail,
    )


def expert_explorer_node(state: ExplorerState) -> dict:
    node_ids = state.get("node_ids", [])
    role_name = state.get("role")

    if len(node_ids) == 1:
        result = _explore_single(node_ids[0], role_name)
    else:
        result = _explore_batch(node_ids, role_name)

    _log_explorer_completion(state)
    return result


def _explore_single(node_id: str, role_name: str) -> dict:
    # Check cache
    cache_file = settings.cache_dir / "notes" / safe_cache_filename(f"{node_id}-{role_name}.json")
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

    # Extract hypotheses from the real 'vulns' key (kept in dict_note for later consumers)
    raw_hypotheses = dict_note.get("vulns", [])
    for hyp in raw_hypotheses:
        extracted_vulns.append({
            "node_id": node_id,
            "cwe_id": hyp.get("cwe", "OTHER_UNCATEGORIZED"),
            "description": hyp.get("component", ""),
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
    cache_file = settings.cache_dir / "notes" / safe_cache_filename(f"batch-{batch_key}-{role_name}.json")
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

        # Extract hypotheses from the real 'vulns' key (kept in dict_note for later consumers)
        raw_hypotheses = dict_note.get("vulns", [])
        for hyp in raw_hypotheses:
            extracted_vulns.append({
                "node_id": node_id,
                "cwe_id": hyp.get("cwe", "OTHER_UNCATEGORIZED"),
                "description": hyp.get("component", ""),
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
    progress_id = _start_agent_progress(len(state.get("known_vulns", [])))

    known_vulns = state.get("known_vulns", [])

    for cve_record in known_vulns:
        payload = CVEAnalyzerState(
            cve=cve_record,
            progress_id=progress_id,
        )
        commands.append(Send("cve_analyzer", payload))

    logging.info(
        "Starting CVE analyzer scan: 0/%d complete, %d remaining.",
        len(commands),
        len(commands),
    )
    return commands


def _finalize_cve_analysis(dict_analysis: dict, cve: dict, *, enriched_by: str | None = None) -> dict | None:
    """Apply deterministic CVE output guards and attach routing metadata."""
    cve_id = cve.get("id", "UNKNOWN-CVE")
    fix_category = dict_analysis.get("fix_category")
    if fix_category == "application_mitigation" and not dict_analysis.get("security_assumption"):
        logging.warning(f"{cve_id}: classified as application_mitigation but no security_assumption. Dropping.")
        return None
    if fix_category == "upgrade_only" and not dict_analysis.get("hypothesis"):
        logging.warning(f"{cve_id}: classified as upgrade_only but no hypothesis. Dropping.")
        return None
    if fix_category not in ("application_mitigation", "upgrade_only"):
        logging.warning(f"{cve_id}: invalid fix_category '{fix_category}'. Dropping.")
        return None

    dict_analysis["required_keywords"] = list(dict.fromkeys(
        kw.strip()
        for kw in (dict_analysis.get("required_keywords") or [])
        if kw and kw.strip()
    ))
    dict_analysis["source_cve"] = cve_id
    dict_analysis["package"] = cve.get("package") or "unknown"
    dict_analysis["fixed_version"] = cve.get("fixed_version")
    if enriched_by:
        dict_analysis["enriched_by"] = enriched_by
    return dict_analysis


def _cve_analyzer_node(state: CVEAnalyzerState) -> dict:
    """LLM node that classifies a single CVE description and either extracts a
    security demand (application_mitigation) or emits a vulnerability hypothesis
    (upgrade_only, e.g. RCE in the HTTP server where only a package upgrade fixes it)."""
    cve = state.get("cve", {})
    package_name = cve.get("package") or "unknown"
    cve_id = cve.get("id", "UNKNOWN-CVE")
    # Up to 3 distinct descriptions of the same CVE (deduplicated by the
    # preprocessor); fall back to the single `details` field for legacy records.
    descriptions = cve.get("descriptions")
    if not descriptions:
        details = cve.get("details")
        descriptions = [details] if details else []
    if not descriptions:
        # Without descriptions the LLM would just hallucinate
        logging.warning(f"{cve_id}: no descriptions provided")
        return {"cve_demands": []}

    # Check cache
    cache_file = settings.cache_dir / "cve_analyzer" / f"{cve_id}.json"
    cached_demand = cache(cache_file, "read")
    if cached_demand:
        return {"cve_demands": [cached_demand]}

    # LLM invocation
    sys_msg = SystemMessage(content=(
        f"{CVE_ANALYZER_AGENT['prompt']}\n\n"
    ))

    enrichment = ""
    fixed_version = cve.get("fixed_version")
    if fixed_version:
        enrichment += f"\nFixed version: {fixed_version}\n"
    cwe_ids = cve.get("cwe_ids") or []
    if cwe_ids:
        enrichment += f"OSV CWE classifications: {', '.join(cwe_ids)}\n"
    if cve.get("severity_label"):
        enrichment += f"OSV severity: {cve['severity_label']}\n"
    if cve.get("cvss_vector"):
        enrichment += f"OSV CVSS vector: {cve['cvss_vector']}\n"
    if enrichment:
        enrichment = f"\n--- OSV ENRICHMENT ---\n{enrichment}"

    desc_block = "\n".join(
        f"Description {i + 1}: {d}\n"
        for i, d in enumerate(descriptions)
    )

    human_msg = HumanMessage(content=(
        f"Analyze this CVE affecting the package '{package_name}':\n\n"
        f"CVE ID: {cve_id}\n"
        f"{desc_block}"
        f"{enrichment}"
    ))

    cve_analyzer_llm = fast_llm.with_structured_output(CVEAnalysis, method="json_schema", strict=True)
    analysis = cve_analyzer_llm.invoke([sys_msg, human_msg])

    dict_analysis = analysis if isinstance(analysis, dict) else analysis.model_dump()

    dict_analysis = _finalize_cve_analysis(dict_analysis, cve)
    if dict_analysis is None:
        return {"cve_demands": []}

    # Save to cache
    cache(cache_file, "write", dict_analysis)

    # Return the extracted analysis to be appended to the global state
    return {
        "cve_demands": [dict_analysis]
    }


def cve_analyzer_node(state: CVEAnalyzerState) -> dict:
    result = _cve_analyzer_node(state)
    cve = state.get("cve", {})
    _log_agent_completion(
        state.get("progress_id", ""),
        "CVE analyzer",
        f"cve={cve.get('id', 'UNKNOWN-CVE')}",
    )
    return result


# ==========================================
# Threat Intel agent
# ==========================================

def threat_intel_gate_node(state: MasterState) -> dict:
    """Barrier after all per-CVE analyzer tasks have completed."""
    return {}


def _analysis_needs_threat_intel(cve: dict, analysis: dict | None) -> bool:
    if is_high_severity(cve):
        return True
    if analysis is None:
        return True
    if "required_keywords" in analysis and not analysis.get("required_keywords"):
        return True
    trigger_keys = {"trigger_condition", "attacker_request_primitive"}
    if trigger_keys & analysis.keys() and not any(analysis.get(key) for key in trigger_keys):
        return True
    return False


def dispatch_threat_intel(state: MasterState):
    """Dispatch only CVEs whose mechanics need external threat intelligence."""
    if not os.environ.get("TAVILY_API_KEY"):
        logging.warning("Threat Intel disabled: TAVILY_API_KEY is not configured.")
        return "aggregate_demands"

    analyzed = {}
    for record in state.get("cve_demands", []):
        record = record if isinstance(record, dict) else record.model_dump()
        source_cve = record.get("source_cve")
        if source_cve:
            current = analyzed.get(source_cve)
            if current is None or record.get("enriched_by") == "threat_intel":
                analyzed[source_cve] = record

    candidates = []
    for cve in state.get("known_vulns", []):
        cve_id = cve.get("id", "UNKNOWN-CVE")
        prior = analyzed.get(cve_id)
        if _analysis_needs_threat_intel(cve, prior):
            candidates.append((cve, prior))

    if not candidates:
        logging.info("Threat Intel: no CVEs met the enrichment criteria.")
        return "aggregate_demands"

    progress_id = _start_agent_progress(len(candidates))
    logging.info("Starting Threat Intel scan: 0/%d complete, %d remaining.", len(candidates), len(candidates))
    return [
        Send("threat_intel", ThreatIntelState(
            cve=cve,
            prior_analysis=prior,
            progress_id=progress_id,
        ))
        for cve, prior in candidates
    ]


def _format_threat_intel_results(search_data: dict) -> str:
    sections = []
    answer = search_data.get("answer")
    if answer:
        sections.append(f"Tavily answer:\n{str(answer)[:2500]}")
    for index, result in enumerate(search_data.get("results", [])[:5], 1):
        if not isinstance(result, dict):
            continue
        sections.append(
            f"Result {index}: {result.get('title', '')}\n"
            f"URL: {result.get('url', '')}\n"
            f"Content: {str(result.get('content', ''))[:1500]}"
        )
    return "\n\n".join(sections)[:9000]


def _threat_intel_node(state: ThreatIntelState) -> dict:
    cve = state.get("cve", {})
    prior = state.get("prior_analysis")
    cve_id = cve.get("id", "UNKNOWN-CVE")
    package_name = cve.get("package") or "unknown"
    cache_file = settings.cache_dir / "threat_intel" / f"{cve_id}.json"
    cached = cache(cache_file, "read")
    if cached:
        return {"cve_demands": [cached]}

    query = f"{cve_id} {package_name} root cause writeup exploit analysis"
    try:
        search_data = TavilyClient().search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
        )
    except Exception as exc:
        logging.warning(f"{cve_id}: Tavily search failed; keeping analyzer output: {exc}")
        return {"cve_demands": [prior] if prior else []}

    descriptions = cve.get("descriptions") or ([cve.get("details")] if cve.get("details") else [])
    enrichment = []
    if cve.get("fixed_version"):
        enrichment.append(f"Fixed version: {cve['fixed_version']}")
    if cve.get("cwe_ids"):
        enrichment.append(f"OSV CWE classifications: {', '.join(cve['cwe_ids'])}")
    if cve.get("severity_label"):
        enrichment.append(f"OSV severity: {cve['severity_label']}")
    if cve.get("cvss_vector"):
        enrichment.append(f"OSV CVSS vector: {cve['cvss_vector']}")
    human_msg = HumanMessage(content=(
        f"Analyze and complete this CVE using the external threat intelligence.\n\n"
        f"CVE ID: {cve_id}\nPackage: {package_name}\n"
        f"OSV descriptions:\n{chr(10).join(descriptions)}\n"
        f"{' '.join(enrichment)}\n\n"
        f"Prior CVE analyzer output (may be null or incomplete):\n"
        f"{json.dumps(prior or {}, indent=2)}\n\n"
        f"--- WEB INTEL ---\n{_format_threat_intel_results(search_data)}"
    ))
    sys_msg = SystemMessage(content=THREAT_INTEL_AGENT.get("prompt", ""))
    structured_llm = fast_llm.with_structured_output(CVEAnalysis, method="json_schema", strict=True)
    response = structured_llm.invoke([sys_msg, human_msg])
    response = response if isinstance(response, dict) else response.model_dump()
    enriched = _finalize_cve_analysis(response, cve, enriched_by="threat_intel")
    if enriched is None:
        logging.warning(f"{cve_id}: Threat Intel returned an invalid analysis; keeping prior output.")
        return {"cve_demands": [prior] if prior else []}

    cache(cache_file, "write", enriched)
    return {"cve_demands": [enriched]}


def threat_intel_node(state: ThreatIntelState) -> dict:
    result = _threat_intel_node(state)
    cve = state.get("cve", {})
    _log_agent_completion(
        state.get("progress_id", ""),
        "Threat Intel",
        f"cve={cve.get('id', 'UNKNOWN-CVE')}",
    )
    return result

# ==========================================
# Aggregate demands node
# ==========================================

def build_caller_map(graph_data: dict):
    callers_map = defaultdict(list)
    for edge in graph_data.get("links", []):
        callers_map[edge.get("target")].append(edge.get("source"))
    return callers_map


def build_import_map(graph_data: dict):
    """Map every graph node to the import namespace set of its source file.

    Imports are a property of the file, not of an individual node, so each
    unique file is read and parsed (``extract_imports``) exactly once and the
    resulting set is shared by all nodes mapped to it — instead of re-reading
    and re-parsing the file once per node.
    """
    node_imports_map = {}
    nodes_by_file = defaultdict(list)
    for node in graph_data.get("nodes", []):
        node_id, src_path = node.get("id"), node.get("source_file")
        if node_id and src_path:
            nodes_by_file[src_path].append(node_id)

    for src_path in nodes_by_file:
        source_file = settings.app_path / Path(src_path)
        if not source_file.exists():
            continue
        content = read_file_text(src_path)
        try:
            imports = set(extract_imports(content, src_path)) if content is not None else set()
        except Exception:
            imports = set()
        for node_id in nodes_by_file[src_path]:
            node_imports_map[node_id] = imports

    return node_imports_map


def build_sub_nodes_index(graph_data: dict):
    """Precompute ``{source_file: [(start_line, node_id), ...]`` once so each
    ``get_node_code`` fold (in the CVE usage checks) does not rescan the whole
    graph for sibling nodes sharing the same file. Only nodes with a parseable
    ``source_location`` are indexed, matching the original scan."""
    index = defaultdict(list)
    for node in graph_data.get("nodes", []):
        src_path = node.get("source_file")
        source_location = node.get("source_location")
        if not src_path or not source_location:
            continue
        try:
            index[src_path].append((int(str(source_location).replace("L", "")), node.get("id")))
        except ValueError:
            continue
    return dict(index)


def _note_demands(dict_note: dict) -> list[dict]:
    """Normalize an AnalysisNote into a flat list of unified demand dicts.

    The explorer emits `upstream`/`downstream` (schema keys), each entry shaped
    ``{"target": str, "description": str}``. Returns
    ``{"direction": ..., "target": ..., "description": ...}`` dicts.
    """
    current_node_id = dict_note.get("node_id")
    demands: list[dict] = []

    for direction in ("upstream", "downstream"):
        for entry in dict_note.get(direction, []) or []:
            if not isinstance(entry, dict):
                logging.warning(f"[{current_node_id}] NOTE DEMAND SKIP: malformed {direction} entry (not a dict): {entry!r}")
                continue
            demands.append({
                "direction": direction,
                "target": entry.get("target", ""),
                "description": entry.get("description"),
            })

    return demands


def _route_downstream(demand: dict, current_node_id: str, graph_data: dict, grouped_demands: defaultdict) -> None:
    """Route a downstream demand to its resolved target node (callee)."""
    target_str = demand.get("target", "")
    desc = demand.get("description")

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


def _route_upstream(demand: dict, current_node_id: str, callers_map: dict, grouped_demands: defaultdict) -> None:
    """Route an upstream demand to every caller of the current node."""
    target_str = demand.get("target", "")
    desc = demand.get("description")

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


def _route_explorer_notes(notes: list, graph_data: dict, callers_map: dict, grouped_demands: defaultdict) -> list[dict]:
    """Process explorer notes into grouped demands. Returns the updated notes."""
    updated_notes = []

    for note in notes:
        dict_note = note if isinstance(note, dict) else note.model_dump()
        current_node_id = dict_note.get("node_id")

        for demand in _note_demands(dict_note):
            direction = demand.get("direction")
            if direction == "downstream":
                _route_downstream(demand, current_node_id, graph_data, grouped_demands)
            elif direction == "upstream":
                _route_upstream(demand, current_node_id, callers_map, grouped_demands)
            else:
                logging.warning(f"[{current_node_id}] UNKNOWN DIRECTION: '{direction}'. Demand dropped.")

        updated_notes.append(dict_note)

    return updated_notes


def _process_cve_demands(cves: list, node_imports_map: dict, grouped_demands: defaultdict,
                         node_map: dict | None = None,
                         sub_nodes_index: dict | None = None) -> list[dict]:
    """Route CVE analyzer outputs.

    Two output kinds are handled (mirroring the `fix_category` classifier):
    - `application_mitigation` records become `cve_assumption` demands routed to
      every node that imports AND uses the affected namespace.
    - `upgrade_only` records bypass the contract verifier and become direct
      vulnerability hypotheses (like explorer `vulns`), anchored to a synthetic
      `dependency:<package>` node. Exactly ONE hypothesis is emitted per CVE.
      Returns the list of hypothesis dicts for the `vulnerabilities` channel.

    A one-time inverted index (``imports_by_namespace``) turns the per-CVE
    all-node scan into a lookup of only the nodes importing the namespace, so
    the cost scales with the number of *matching* nodes, not the codebase size.
    """
    hypotheses: list[dict] = []

    imports_by_namespace: dict[str, list[str]] = defaultdict(list)
    for node_id, imports in node_imports_map.items():
        for imp in imports:
            imports_by_namespace[imp].append(node_id)

    for record in cves:
        target_import = record.get("import_namespace", "")
        source_cve = record.get("source_cve", "unknown")
        package = record.get("package", "unknown")
        fix_category = record.get("fix_category", "application_mitigation")

        if fix_category == "upgrade_only":
            hypothesis = _build_cve_hypothesis(record, imports_by_namespace, node_map, sub_nodes_index)
            if hypothesis:
                hypotheses.append(hypothesis)
            else:
                logging.warning(f"[CVE HYPOTHESIS DROP] {source_cve} for '{target_import}' produced no usable hypothesis.")
            continue

        # --- application_mitigation: route as a verifiable demand ---
        trigger = record.get("attacker_request_primitive") or record.get("trigger_condition")
        combined_desc = (
            f"Security Context: {record.get('security_assumption')} | "
            f"Trigger: {trigger}"
        )

        matched_any = False
        for node_id in imports_by_namespace.get(target_import, []):
            if not uses_namespace_in_ast(node_id, target_import, node_map, sub_nodes_index):
                logging.info(f"[{node_id}] CVE SKIP: '{target_import}' imported but not used in the node's AST.")
                continue
            grouped_demands[node_id].append({
                "source": source_cve,
                "type": "cve_assumption",
                "description": combined_desc,
            })
            matched_any = True

        if not matched_any:
            logging.warning(f"[CVE DROP] {source_cve} for '{target_import}' matched 0 nodes in the graph.")

    return hypotheses


def _build_cve_hypothesis(record: dict, imports_by_namespace: dict, node_map: dict | None = None,
                          sub_nodes_index: dict | None = None) -> dict | None:
    """Build a single vulnerability hypothesis for an upgrade-only CVE.

    Anchored to a synthetic `dependency:<package>` node (a library-internal flaw
    has no application node of its own). Usage-site hints are appended to the
    description so the Reviewer (which has no library source) knows where to look,
    including the transitive-dependency case where nothing imports the package.
    """
    source_cve = record.get("source_cve", "unknown")
    package = record.get("package", "unknown")
    target_import = record.get("import_namespace", "")
    hypothesis = record.get("hypothesis") or {}
    cwe_id = hypothesis.get("cwe", "OTHER_UNCATEGORIZED")
    description = hypothesis.get("description", "")
    affected_component = hypothesis.get("affected_component", "")

    if not description:
        return None

    full_desc = f"[{source_cve}] {description} Affected package: '{package}'."
    exposure = hypothesis.get("framework_exposure_mechanism")
    if exposure:
        full_desc += f" Framework exposure: {exposure}"

    # Usage hints: nodes that import AND use the namespace, capped to bound prompt size.
    if target_import:
        matching_nodes = sorted(
            node_id for node_id in imports_by_namespace.get(target_import, [])
            if uses_namespace_in_ast(node_id, target_import, node_map, sub_nodes_index)
        )
        if matching_nodes:
            shown = matching_nodes[:5]
            hint = f" Nodes importing/using '{target_import}': {', '.join(shown)}"
            if len(matching_nodes) > 5:
                hint += f" (+{len(matching_nodes) - 5} more)"
            full_desc += hint
        else:
            full_desc += (
                f" No application node imports '{target_import}' directly — the package is likely a "
                f"transitive dependency pulled in by a framework. Look for usage of the parent-framework "
                f"feature instead (see 'Component')."
            )

    return {
        "node_id": f"dependency:{package}",
        "cwe_id": cwe_id,
        "description": full_desc,
        "status": "hypothesis",
        "demand_id": source_cve,
        "source_cve": source_cve,
        "vulnerable_component": affected_component,
        "vulnerability_type": "Known Dependency Vulnerability",
    }


def filter_cve_demands_by_keywords(cves: list[dict]) -> list[dict]:
    """Deterministic pre-filter: drop CVE records whose ``required_keywords``
    are all absent from the application's code files (graph nodes with
    ``file_type == "code"``).

    A record is kept if ANY of its keywords appears as an exact, case-sensitive
    substring in ANY code file. Records with an empty or missing keyword list
    are kept unchanged (fail-open — protects stale caches generated before the
    ``required_keywords`` field existed). Applies to BOTH fix categories, so a
    dropped record never produces a contract-verifier demand nor an
    upgrade-only reviewer hypothesis.
    """
    if not cves:
        return cves

    corpus = load_code_corpus()
    if not corpus:
        logging.warning(
            "CVE keyword filter: no code files indexed — skipping filter "
            "(all CVE records forwarded)."
        )
        return cves

    kept: list[dict] = []
    dropped: list[dict] = []

    for record in cves:
        source_cve = record.get("source_cve", record.get("id", "unknown"))
        keywords = record.get("required_keywords") or []

        if not keywords:
            logging.debug(f"{source_cve}: no required_keywords — keeping (cannot filter).")
            kept.append(record)
            continue

        match = next(
            ((kw, source_file) for kw in keywords for source_file, content in corpus.items()
             if kw in content),
            None
        )
        if match:
            keyword, source_file = match
            logging.debug(f"{source_cve}: keyword '{keyword}' found in '{source_file}' — keeping.")
            kept.append(record)
        else:
            logging.info(
                f"[CVE KEYWORD DROP] {source_cve}: none of {keywords} found in code files — dropped."
            )
            dropped.append(record)

    logging.info(
        f"CVE keyword filter: kept {len(kept)}/{len(cves)} record(s), "
        f"dropped {len(dropped)} before downstream LLM stages."
    )
    return kept


def _dedupe_enriched_cve_demands(cves: list[dict]) -> list[dict]:
    """Keep one analyzer record per CVE, preferring Threat Intel output."""
    by_cve: dict[str, dict] = {}
    for record in cves:
        source_cve = record.get("source_cve")
        if not source_cve:
            by_cve[f"__anonymous_{len(by_cve)}"] = record
            continue
        existing = by_cve.get(source_cve)
        if existing is None or (
            record.get("enriched_by") == "threat_intel"
            and existing.get("enriched_by") != "threat_intel"
        ):
            by_cve[source_cve] = record
    return list(by_cve.values())


def aggregate_demands_node(state: MasterState):
    grouped_demands = defaultdict(list)
    try:
        graph_data = get_cached_graph_data(settings.graph)
        callers_map = build_caller_map(graph_data)
        node_map = {n.get("id"): n for n in graph_data.get("nodes", []) if n.get("id")}
        sub_nodes_index = build_sub_nodes_index(graph_data)
        node_imports_map = build_import_map(graph_data)

        logging.info(f"Loaded graph data: {len(callers_map)} caller entries, {len(node_imports_map)} import entries.")
        notes = state.get("notes", [])
        cves = _dedupe_enriched_cve_demands(state.get("cve_demands", []))
        cves = filter_cve_demands_by_keywords(cves)
        logging.info(f"Processing {len(notes)} notes and {len(cves)} CVE demands.")

        # Process Explorer Notes
        updated_notes = _route_explorer_notes(notes, graph_data, callers_map, grouped_demands)

        # Process CVE Analyzer Outputs (demands + upgrade-only hypotheses)
        cve_hypotheses = _process_cve_demands(cves, node_imports_map, grouped_demands, node_map, sub_nodes_index)
        logging.info(f"Emitted {len(cve_hypotheses)} upgrade-only CVE hypothesis(es) directly into the vulnerabilities channel.")

        # Log the accurate total by summing the lengths of the lists
        total_demands = sum(len(d) for d in grouped_demands.values())
        logging.info(f"Summary: Grouped {total_demands} total demands across {len(grouped_demands)} target nodes.")

        return {
            "grouped_demands": dict(grouped_demands),
            "notes": updated_notes,
            "vulnerabilities": cve_hypotheses,
        }
    finally:
        # Deterministic pass: release the memoized source text / extracted
        # results so they never persist into the downstream LLM stages.
        clear_aggregate_caches()

# ==========================================
# Contract verifier node
# ==========================================

def dispatch_verifiers(state: MasterState):
    grouped_demands = state.get("grouped_demands", {})

    # If no demands were found across the whole codebase, skip straight to the end
    if not grouped_demands:
        return END

    commands: list[Send] = []
    progress_id = uuid.uuid4().hex

    for target_node_id, demands_list in grouped_demands.items():
        target_code = get_node_code(target_node_id)

        # If we don't have code (e.g., it's a 3rd party library), skip it
        if not target_code:
            continue

        payload = {
            "target_node_id": target_node_id,
            "target_code": target_code,
            "incoming_demands": demands_list,
            "progress_id": progress_id,
        }

        commands.append(Send("contract_verifier", payload))

    # If all targets were 3rd party libraries and we generated 0 commands
    if not commands:
        return "synchronization"

    with _agent_progress_lock:
        _agent_progress[progress_id] = {"total": len(commands), "completed": 0}
    logging.info(
        "Starting contract verifier scan: 0/%d complete, %d remaining.",
        len(commands),
        len(commands),
    )
    return commands


def _contract_verifier_node(state: VerifierState) -> dict:
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

    structured_llm = fast_llm.with_structured_output(VerifierOutput, method="json_schema", strict=True)
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


def contract_verifier_node(state: VerifierState) -> dict:
    result = _contract_verifier_node(state)
    _log_agent_completion(
        state.get("progress_id", ""),
        "Contract verifier",
        f"node={state.get('target_node_id', 'unknown')}",
    )
    return result

# ==========================================
# Reviewer agent
# ==========================================

# Hypotheses are routed to one of two reviewer tracks based on
# vulnerability_type (see _reviewer_mode_for). Each track binds only its own
# tool subset; the ToolNode in graph.py registers the union so both can run
# through the same compiled subgraph.
CODE_LEVEL_REVIEWER_TOOLS = [
    tools.read_source_code,
    tools.read_file,
    tools.get_node_connections,
    tools.search_codebase,
    tools.get_definition,
    tools.submit_evaluation,
]
FRAMEWORK_DEPENDENCY_REVIEWER_TOOLS = [
    tools.read_file,
    tools.get_node_connections,
    tools.search_codebase,
    tools.list_container_artifacts,
    tools.read_container_artifact,
    tools.submit_evaluation,
]


def _reviewer_mode_for(hypothesis: dict) -> str:
    """Route a hypothesis to its reviewer track.

    'framework_dependency' (Known Dependency Vulnerability) or 'code_level'.
    """
    if hypothesis.get("vulnerability_type") == "Known Dependency Vulnerability":
        return "framework_dependency"
    return "code_level"


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
            mode=_reviewer_mode_for(hypothesis),
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
    mode = state.get("mode", "code_level")

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

    # Bind the tool subset for the hypothesis's reviewer track
    if mode == "framework_dependency":
        reviewer_tools = FRAMEWORK_DEPENDENCY_REVIEWER_TOOLS
        mode_prompt = REVIEWER_AGENT.get("framework_dependency", "")
    else:
        reviewer_tools = CODE_LEVEL_REVIEWER_TOOLS
        mode_prompt = REVIEWER_AGENT.get("code_level", "")

    llm_with_tools = smart_llm.bind_tools(
        reviewer_tools,
        parallel_tool_calls=True
    )

    if not state.get("messages"):
        # Compose the targeted system prompt: shared directives + mode-specific
        # reachability standard.
        sys_prompt = REVIEWER_AGENT.get('prompt', '')
        if mode_prompt:
            sys_prompt = f"{sys_prompt}\n\n{mode_prompt}"
        sys_msg = SystemMessage(content=sys_prompt)

        report = state.get("expert_report", {})
        node_id = state.get("node_id")

        formatted_vuln = (
            f"Target: {state['node_id']}\n\n"
            f"Potential Issue to Investigate:\n"
            f"- Type: {report.get('vulnerability_type', 'Code Defect')}\n"
            f"- CWE ID: {report.get('cwe_id', 'Unknown')}\n"
            f"- Component: {report.get('vulnerable_component', 'Unknown')}\n"
            f"- Description: {report.get('description', '')}\n"
        )
        if report.get("source_cve"):
            formatted_vuln += f"- Source CVE: {report.get('source_cve')}\n"

        if node_id:
            target_node_source = get_node_code(node_id, reviewer_mode=True)
            if target_node_source:
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
        # With parallel tool calls the model may submit a final evaluation
        # alongside other reads; end if any tool in the latest batch submitted.
        for msg in reversed(state["messages"]):
            if msg.type != "tool":
                break
            if getattr(msg, "name", "") == "submit_evaluation":
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
            sandbox_url=state.get("sandbox_url"),
            container_name=state.get("container_name"),
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
        # tools.read_sandbox_file,
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
