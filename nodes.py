from pathlib import Path
import json
import os
import re
import subprocess
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from llm_debug import build_debug_http_client
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
import browser_tools
import attacker_tools
from state import MasterState, ExplorerState, CVEAnalyzerState, ThreatIntelState, VerifierState, ReviewerState, ValidatorState, IntegrationAuditorState
from schemas import ExpertTask, AnalysisNote, BatchedAnalysisResult, CVEAnalysis, VerifierOutput, MANAGER_AGENT, EXPERT_AGENTS, CVE_ANALYZER_AGENT, THREAT_INTEL_AGENT, VERIFIER_AGENT, REVIEWER_AGENT, VALIDATOR_AGENT, INTEGRATION_AUDITOR_AGENT, cwes
from utils import build_networkx_graph, extract_imports, get_cached_graph_data, get_node_code, index_file, run_osv_scanner, run_osv_scanner_image, deduplicate_cves, cache, safe_cache_filename, resolve_node_id, uses_namespace_in_ast, is_node_worth_scanning, is_path_excluded, format_node_context, find_container_builds, build_images, start_sandbox, extract_container_artifacts, scan_codebase_for_keywords, find_unsupported_code_files, read_file_text, clear_aggregate_caches, is_high_severity, cache_reviewer, reviewer_cache_key
from tool_loop import CompactionConfig, ToolLoopAgent
from dedup import Embeddings, cluster_vulnerabilities


_agent_progress: dict[str, dict[str, int]] = {}
_agent_progress_lock = threading.Lock()

if settings.llm_provider == "openai":
    fast_llm = ChatOpenAI(
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        stream_usage=True,
        temperature=0.2,
        max_completion_tokens=settings.fast_max_completion_tokens,
        reasoning_effort="none",
        http_client=build_debug_http_client(),
    )
    smart_llm = ChatOpenAI(
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        stream_usage=True,
        temperature=0.8,
        max_completion_tokens=settings.smart_max_completion_tokens,
        reasoning_effort="medium",
        http_client=build_debug_http_client(),
    )
    reviewer_llm = ChatOpenAI(
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        stream_usage=True,
        temperature=0.8,
        max_completion_tokens=settings.reviewer_max_completion_tokens,
        reasoning_effort="low",
        http_client=build_debug_http_client(),
    )
elif settings.llm_provider == "ollama":
    fast_llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.2,
        num_predict=settings.fast_max_completion_tokens,
        reasoning=False
    )
    smart_llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.8,
        num_predict=settings.smart_max_completion_tokens,
        reasoning=True
    )
    reviewer_llm = smart_llm
else:
    raise ValueError(
        f"Unknown llm_provider {settings.llm_provider!r}; expected 'openai' or 'ollama'."
    )

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
            tag = settings.docker_image_tag or f"vulnscan-{settings.app_path.name.lower()}:latest"
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
            # Skip files in excluded paths (dependency trees, tests, docs) so
            # the AST symbol index stays focused on scannable application code.
            if is_path_excluded(str(filepath)):
                continue
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
    combined code size stays under settings.explorer_batch_char_threshold characters.

    Skeleton (file-level) nodes are dispatched only when no real sibling of the
    same file is eligible in the same task, so skeleton-only files are analyzed too.
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

        # Eligible-by-file: (node_id, is_skeleton). Skeletons are file/module-level
        # placeholder nodes; a real sibling's scan already sees the whole file
        # (module-level code included) as context.
        eligible: dict[str, list[tuple[str, bool]]] = defaultdict(list)

        for node_id in community_nodes:
            node_data = G.nodes[node_id]
            source_file = node_data.get("source_file", "")
            label = node_data.get("label") or ""
            is_skeleton = bool(source_file) and source_file.endswith(label)

            # Drop inert nodes (pure types, empty skeletons, flat constants) to save LLM budget
            if not is_node_worth_scanning(node_id):
                skipped_nodes += 1
                logging.debug(f"Skipping inert node {node_id} (no executable signals).")
                continue

            # Drop nodes whose source file lives in an excluded path
            # (dependency trees, tests, docs) before spending LLM budget on it.
            if is_path_excluded(source_file):
                skipped_nodes += 1
                logging.debug(f"Skipping excluded-path node {node_id} ({source_file}).")
                continue

            eligible[source_file].append((node_id, is_skeleton))

        # Drop a skeleton node whose file already has a real (non-skeleton)
        # eligible sibling in this task — the sibling's batch carries the whole
        # file as context, so scanning the skeleton too would duplicate coverage.
        files: dict[str, list[str]] = {}
        for file_path, nodes in eligible.items():
            has_real = any(not sk for _, sk in nodes)
            files[file_path] = [nid for nid, sk in nodes if not (sk and has_real)]

        # Pack each file's nodes into batches whose combined code size stays under the threshold,
        # unless batching is disabled, in which case every node is its own dispatch.
        for file_path, file_nodes in files.items():
            if settings.explorer_batching_enabled:
                batches = _pack_node_batches(file_nodes, settings.explorer_batch_char_threshold)
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

    # The aggregate_demands join barrier requires explorer_agent to fire even
    # when nothing was dispatchable; emit a no-op task otherwise.
    if not commands:
        commands.append(Send("explorer_agent", ExplorerState(
            node_ids=[],
            role="explorer",
            task_description="",
            progress_id="",
        )))

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

    if not node_ids:
        # No-op task (empty dispatches): fires the aggregate_demands join
        # barrier without any LLM call.
        return {}

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

    extracted_vulns = []

    # Extract hypotheses from the real 'vulns' key (kept in dict_note for later consumers)
    raw_hypotheses = dict_note.get("vulns", [])
    for hyp in raw_hypotheses:
        extracted_vulns.append({
            "affected_nodes": [node_id],
            "cwe_id": hyp.get("cwe", "OTHER_UNCATEGORIZED"),
            "description": hyp.get("component", ""),
            "vulnerable_component": hyp.get("pattern_label") or None,
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
        f"{EXPERT_AGENTS['batch_prompt']}"
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
                "Evaluate this module-level skeleton for global configuration issues. Omitted child functions "
                "are evaluated separately; do not report vulnerabilities for them under this ID.\n"
                f"{context_block}"
                f"```python\n{source_code}\n```"
            )
        else:
            sections.append(
                f"### Node '{node_id}' ({label})\n"
                f"{context_block}"
                f"Analyze the specific logic inside '{label}'. "
                f"Report vulnerabilities affecting this specific node using ONLY the ID '{node_id}'.\n"
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

        # Extract hypotheses from the real 'vulns' key (kept in dict_note for later consumers)
        raw_hypotheses = dict_note.get("vulns", [])
        for hyp in raw_hypotheses:
            extracted_vulns.append({
                "affected_nodes": [node_id],
                "cwe_id": hyp.get("cwe", "OTHER_UNCATEGORIZED"),
                "description": hyp.get("component", ""),
                "vulnerable_component": hyp.get("pattern_label") or None,
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

    # The aggregate_demands join barrier requires the cve_analyzer chain to
    # fire even with zero SCA findings; emit a no-op task otherwise. The no-op
    # flows through threat_intel_gate -> no-op threat_intel so the barrier sees
    # a write from threat_intel too.
    if not commands:
        commands.append(Send("cve_analyzer", CVEAnalyzerState(cve={}, progress_id="")))

    logging.info(
        "Starting CVE analyzer scan: 0/%d complete, %d remaining.",
        len(commands),
        len(commands),
    )
    return commands


def _normalize_cwe_ids(value) -> list[str]:
    """Normalize an OSV `cwe_ids` value: strip whitespace, drop empties and
    non-string entries, de-duplicate while preserving order. Accepts a list or
    a single bare string."""
    entries = value if isinstance(value, list) else ([value] if isinstance(value, str) else [])
    seen = set()
    normalized = []
    for entry in entries:
        if not isinstance(entry, str):
            continue
        cwe = entry.strip()
        if not cwe:
            continue
        if cwe in seen:
            continue
        seen.add(cwe)
        normalized.append(cwe)
    return normalized


def _backfill_osv_cwe_ids(cached: dict, cve: dict) -> dict:
    """Patch deterministic OSV-suggested CWEs into a cached analyzer output.

    The CVE analyzer / threat-intel caches are keyed by CVE id only, so records
    written before `cwe_ids` existed (or when `deduplicate_cves` dropped the
    classifications) go stale. `cwe_ids` is deterministic metadata sourced from
    the OSV record, so on a cache hit merge it in from the current
    `known_vulns` entry without re-running the LLM."""
    if not isinstance(cached, dict):
        return cached
    cached = dict(cached)
    osv_cwes = _normalize_cwe_ids(cve.get("cwe_ids"))
    cached["cwe_ids"] = osv_cwes or _normalize_cwe_ids(cached.get("cwe_ids"))
    return cached


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
    # Programmatically attach the CWEs the OSV scanner suggested for this CVE
    # (from `known_vulns[i].cwe_ids`, itself extracted from
    # `database_specific.cwe_ids`). This is deterministic — it runs after the
    # LLM call for both the CVE analyzer and Threat Intel paths, so the output
    # record always carries the OSV classifications regardless of what the
    # model emitted.
    dict_analysis["cwe_ids"] = _normalize_cwe_ids(cve.get("cwe_ids"))
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
    if not cve:
        # No-op task (empty dispatches): fires the cve_analyzer -> threat_intel_gate
        # chain so the aggregate_demands join barrier sees a write from
        # threat_intel even when there are no SCA findings.
        return {}
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
        return {"cve_demands": [_backfill_osv_cwe_ids(cached_demand, cve)]}

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


def _noop_threat_intel_send() -> list[Send]:
    """A single stateless Send that fires the aggregate_demands join barrier
    when there is nothing to enrich (no TAVILY key, or no CVEs meet the
    criteria). `threat_intel_node` returns immediately for an empty `cve`,
    so this costs no API calls."""
    return [Send("threat_intel", ThreatIntelState(cve={}, prior_analysis=None, progress_id=""))]


def dispatch_threat_intel(state: MasterState):
    """Dispatch only CVEs whose mechanics need external threat intelligence.

    Always returns at least one `Send` to `threat_intel`: the join barrier into
    `aggregate_demands` requires `threat_intel` to run exactly once even when
    there is nothing to enrich, so the empty case emits a no-op task instead of
    routing `aggregate_demands` directly."""
    if not getattr(settings, "threat_intel_enabled", True):
        logging.info("Threat Intel disabled via settings.threat_intel_enabled=False.")
        return _noop_threat_intel_send()
    if not os.environ.get("TAVILY_API_KEY"):
        logging.warning("Threat Intel disabled: TAVILY_API_KEY is not configured.")
        return _noop_threat_intel_send()

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
        return _noop_threat_intel_send()

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
        return {"cve_demands": [_backfill_osv_cwe_ids(cached, cve)]}

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
    if not state.get("cve"):
        # No-op task: fires the aggregate_demands join barrier so it runs once
        # even when there is nothing to enrich. No external calls.
        return {}
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
    """Map each node to the source nodes of its incoming ``calls`` edges.

    Only genuine call edges qualify. Structural relations (``method`` /
    ``contains`` / ``inherits`` / ``mixes_in`` / ``implements`` / ``imports`` /
    ``references``) do not carry parameter contracts: routing explorer upstream
    demands through them delivers a callee's contracts to its class/file node,
    whose sibling method bodies are pruned by ``get_node_code`` ("Body omitted
    ... evaluated by a peer agent"), leaving the call sites invisible — the
    verifier could only guess there.
    """
    callers_map = defaultdict(list)
    for edge in graph_data.get("links", []):
        if edge.get("relation") == "calls":
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
        if target_node_id == current_node_id:
            # Drop assumptions about the node under analysis
            return
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
        # Expected for methods with no in-graph caller (private helpers,
        # entrypoints); with the calls-only caller map this fires for every
        # never-called method, so it must not be a warning.
        logging.debug(f"[{current_node_id}] UPSTREAM DROP: No callers found in graph for this node.")


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
                # OSV-suggested CWEs, forwarded to the contract verifier as a
                # hint when it picks the cwe_id for a FAILED evaluation.
                "cwe_ids": _normalize_cwe_ids(record.get("cwe_ids")),
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
        "affected_nodes": [f"dependency:{package}"],
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
    ``required_keywords`` field existed). Intended for `application_mitigation`
    records only: `aggregate_demands_node` exempts `upgrade_only` records, whose
    hypotheses live in a synthetic `dependency:<package>` node and must not be
    dropped on the basis of app-source keywords.
    """
    if not cves:
        return cves

    keyword_cves = [r for r in cves if r.get("required_keywords")]
    if not keyword_cves:
        return cves

    all_keywords = sorted({
        kw for r in keyword_cves for kw in (r.get("required_keywords") or [])
    })
    if not all_keywords:
        return cves

    present_keywords, scanned_bytes = scan_codebase_for_keywords(all_keywords)
    if scanned_bytes == 0:
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
            (kw for kw in keywords if kw in present_keywords),
            None
        )
        if match:
            logging.debug(f"{source_cve}: keyword '{match}' present in code — keeping.")
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

        # Upgrade-only records are exempt from the keyword pre-filter: their
        # hypotheses are anchored to synthetic `dependency:<package>` nodes and
        # route directly to the framework/dependency reviewer, which adjudicates
        # exposure via container artifacts/config — not app-source substrings.
        # Import-site keywords (e.g. build-tool internals like 'pip install')
        # rarely appear in application code, so filtering them out would silently
        # drop every upgrade-only hypothesis.
        upgrade_only = [r for r in cves if r.get("fix_category") == "upgrade_only"]
        cves = filter_cve_demands_by_keywords(
            [r for r in cves if r.get("fix_category") != "upgrade_only"]
        )
        cves = upgrade_only + cves
        logging.info(
            f"Processing {len(notes)} notes and {len(cves)} CVE demands "
            f"({len(upgrade_only)} upgrade-only exempted from the keyword pre-filter)."
        )

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
            # `notes` is an operator.add-reduced channel: returning `updated_notes`
            # would re-append the very notes this node just consumed.
            "notes": [],
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

    # No app-level demands to verify, but pending hypotheses (explorer
    # findings, upgrade-only CVE hypotheses) may still await review. Only END
    # when there is nothing at all; otherwise fall through to synchronization
    # so dispatch_reviewers adjudicates them (contract_verifier is correctly
    # skipped — there are no demands to verify).
    if not grouped_demands:
        pending = [
            v.get("status") if isinstance(v, dict) else v.model_dump().get("status")
            for v in state.get("vulnerabilities", [])
        ]
        return "synchronization" if "hypothesis" in pending else END

    commands: list[Send] = []
    progress_id = uuid.uuid4().hex

    graph_data = get_cached_graph_data(settings.graph)
    node_source_map = {n.get("id"): n.get("source_file") for n in graph_data.get("nodes", [])}

    for target_node_id, demands_list in grouped_demands.items():
        # Skip demands whose target node lives in an excluded path (dependency
        # trees, tests, docs) — nothing to verify there.
        if is_path_excluded(node_source_map.get(target_node_id) or ""):
            logging.debug(f"Skipping contract verification of excluded-path node {target_node_id}.")
            continue

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

    # Build a lookup map of demand_id -> original description, plus a map to the
    # full demand dict so FAILED evaluations can be traced back to their source
    # (e.g. an application_mitigation CVE demand, which tags the output record).
    demand_lookup = {}
    demand_meta = {}
    formatted_demands = []

    for d in demands:
        # Determine the ID you assigned in the prompt formatting
        # (e.g., matching whatever you put inside the [ID: ...] tag)
        d_id = d.get("parameter_name") or d.get("source") or "unknown"
        demand_lookup[d_id] = d.get("description")
        demand_meta[d_id] = d

        dtype = d.get("type")
        source = d.get("source")
        desc = d.get("description")

        if dtype == "explorer_upstream_assumption":
            param = d.get("parameter_name", "unknown")
            formatted_demands.append(
                f"- [ID: {param}] [DEMAND FROM CALLEE] You call '{source}'. It demands: '{desc}'"
            )
        elif dtype == "cve_assumption":
            suggested_cwes = _normalize_cwe_ids(d.get("cwe_ids"))
            suffix = f" [SUGGESTED CWE: {', '.join(suggested_cwes)}]" if suggested_cwes else ""
            formatted_demands.append(
                f"- [ID: {source}] [LIBRARY CVE MITIGATION] Known constraint: '{desc}'{suffix}"
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

            demand_source = demand_meta.get(eval.get("demand_id"), {})
            verifier_cwe = eval.get("cwe")
            if not verifier_cwe and demand_source.get("type") == "cve_assumption":
                # Deterministic fallback for an omitted CWE: seed the FAILED
                # evaluation from the OSV-suggested classification (first one the
                # record schema accepts). Only fires when the model returned no
                # CWE at all — a deliberate OTHER_UNCATEGORIZED is respected.
                verifier_cwe = next(
                    (c for c in _normalize_cwe_ids(demand_source.get("cwe_ids")) if c in cwes),
                    None,
                )

            new_vuln = {
                "affected_nodes": [target_node_id],
                "cwe_id": verifier_cwe,
                "description": f"Fails to satisfy demand: '{original_desc}'. Reasoning: {eval.get("reasoning")}",
                "status": "hypothesis",
                "demand_id": eval.get("demand_id"),
                "vulnerable_component": eval.get("demand_id")
            }
            # application_mitigation CVE demands (type == "cve_assumption") produce
            # a dependency-mitigation review: the flaw is in a third-party package
            # and the app is expected to mitigate it. Tag the record so the reviewer
            # routes it to the dependency_mitigation track (and so the CVE id is
            # visible in the reviewer prompt); explorer demands stay code-level.
            if demand_source.get("type") == "cve_assumption":
                new_vuln["vulnerability_type"] = "Dependency Mitigation Vulnerability"
                new_vuln["source_cve"] = demand_source.get("source")

            new_vulnerabilities.append(new_vuln)

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
    tools.search_codebase,
    tools.list_container_artifacts,
    tools.find_in_container,
    tools.read_container_artifact,
    tools.submit_evaluation,
]


def _reviewer_mode_for(hypothesis: dict) -> str:
    """Route a hypothesis to its reviewer track.

    'framework_dependency' (Known Dependency Vulnerability), 'dependency_mitigation'
    (Dependency Mitigation Vulnerability — application-mitigable CVEs whose contract
    verification failed), 'systemic' (Systemic Vulnerability) or 'code_level'.
    """
    vuln_type = hypothesis.get("vulnerability_type")
    if vuln_type == "Known Dependency Vulnerability":
        return "framework_dependency"
    if vuln_type == "Dependency Mitigation Vulnerability":
        return "dependency_mitigation"
    if vuln_type == "Systemic Vulnerability":
        return "systemic"
    return "code_level"


def _primary_node(record: dict, default: str = "Unknown") -> str:
    """First affected node of a record — the primary anchor for reviewer
    state/cache keys. The full list rides inside `expert_report`."""
    affected = record.get("affected_nodes") or []
    return affected[0] if affected else default


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

    # Semantic dedup before fan-out. Duplicate hypotheses (same real
    # flaw described differently by different agents) are merged so one reviewer
    # subgraph adjudicates the pattern once. Fails open: exact-key dedup (safe,
    # deterministic) always runs; embedding clustering only when Ollama serves
    # the configured model.
    embedder = None
    if settings.semantic_dedup_enabled:
        _emb = Embeddings(
            settings.embeddings_base_url,
            settings.embeddings_model,
            settings.embeddings_timeout,
        )
        if _emb.available():
            embedder = _emb
        else:
            logging.warning(
                "Semantic dedup: embeddings unavailable (Ollama idle or model %r "
                "not pulled?); using exact-key dedup only.",
                settings.embeddings_model,
            )
    hypotheses = cluster_vulnerabilities(hypotheses, settings.semantic_dedup_threshold, embedder)

    commands = []
    for hypothesis in hypotheses:
        payload = ReviewerState(
            node_id=_primary_node(hypothesis),
            expert_report=hypothesis,
            mode=_reviewer_mode_for(hypothesis),
            iterations=0,
            vulnerabilities=[],
            messages=[]
        )
        commands.append(Send("reviewer_agent", payload))

    logging.info(f"Dispatching {len(commands)} reviewers.")
    return commands


# Each agent's ledger format only differs in the system prompt; the summary
# rendering itself is shared. The generic compaction machinery (CompactionConfig,
# ToolLoopAgent, history splitting/transcript rendering, and the summary
# generator) lives in tool_loop.py; these constants carry the exact per-agent
# prompt text so the ledger flavor is a property of the agent, not a separate
# function.
REVIEWER_SUMMARY_LEDGER = (
    "You are an expert summarizer. The following is the message history of a "
    "security reviewer agent investigating whether a reported vulnerability "
    "hypothesis in a target application is a true positive or a false positive. "
    "Your summary will REPLACE these messages in the model context, so the "
    "reviewer must be able to continue the investigation from it WITHOUT "
    "re-reading the original tool outputs.\n\n"
    "Produce an information-dense summary as a security investigation ledger "
    "with exactly these sections:\n"
    "## Objective\n"
    "One or two sentences restating the exact hypothesis under review and the "
    "target component/node.\n"
    "## Checks & Artifacts Examined\n"
    "Bulleted, deduplicated list of every node, file, container artifact, "
    "search, and request already examined, with the single most important fact "
    "each one revealed. Do NOT include full code or full tool outputs — distill "
    "them into their conclusions.\n"
    "## Confirmed Facts\n"
    "Bulleted list of verified facts established so far, stated in final form.\n"
    "## Ruled-Out Dead Ends\n"
    "Bulleted list of hypotheses or investigation paths already disproven, with "
    "a one-line reason for each.\n"
    "## Active Leads & Next Steps\n"
    "Bulleted list of the most promising remaining checks not yet completed.\n\n"
    "RULES:\n"
    "- If the history contains a prior SUMMARY (a system message containing "
    "'CONTEXT COMPACTION SUMMARY'), treat it as the anchoring summary: extend "
    "and refine it with only the new facts gathered since, rather than "
    "regenerating from scratch.\n"
    "- Write in final form ('the code does X', 'the sink is reachable'), never "
    "'the model checked X'.\n"
    "- Keep it concise, under 1024 words. Preserve exact file paths, node ids, "
    "CVE ids, and tool argument names.\n"
)

VALIDATOR_SUMMARY_LEDGER = (
    "You are an expert summarizer. The following is the message history of a "
    "security validator agent attempting to prove or refute a reported "
    "vulnerability hypothesis against a live sandbox application via HTTP "
    "requests. Your summary will REPLACE these messages in the model context, "
    "so the validator must be able to continue the proof from it WITHOUT "
    "re-reading the original HTTP responses or tool arguments.\n\n"
    "Produce an information-dense summary as a security validation ledger "
    "with exactly these sections:\n"
    "## Objective\n"
    "One or two sentences restating the exact vulnerability hypothesis under "
    "proof and the reviewer-provided reproduction steps (the chronological "
    "action plan to follow).\n"
    "## Requests Performed\n"
    "Bulleted list of every HTTP request already sent (method, path, params, "
    "auth state), with the single most important fact each response revealed. "
    "Do NOT include full request/response bodies — distill them into "
    "conclusions (status codes, key values echoed, observable mitigations).\n"
    "## Confirmed Facts\n"
    "Bulleted list of verified facts established so far (e.g. endpoint reachable "
    "without auth, parameter reflected in response, validation present), stated "
    "in final form.\n"
    "## Ruled-Out Dead Ends\n"
    "Bulleted list of hypotheses or attack paths already disproven, with a "
    "one-line reason for each.\n"
    "## Active Leads & Next Steps\n"
    "Bulleted list of the most promising remaining checks not yet completed.\n\n"
    "RULES:\n"
    "- If the history contains a prior SUMMARY (a system message containing "
    "'CONTEXT COMPACTION SUMMARY'), treat it as the anchoring summary: extend "
    "and refine it with only the new facts gathered since, rather than "
    "regenerating from scratch.\n"
    "- Write in final form ('POST succeeded', 'the sink is reachable', 'auth "
    "was required'), never 'the model checked X'.\n"
    "- Keep it concise, under 1024 words. Preserve exact paths, parameter "
    "names, status codes, and any cookies that were set.\n"
)


class ReviewerAgent(ToolLoopAgent):
    """Reviewer track: mode-dependent toolsets, cache-hit Command, and the
    trailing-batch submit_evaluation end condition."""

    terminal_tool = "submit_evaluation"

    def bind_tools(self, state):
        mode = state.get("mode", "code_level")
        if mode == "framework_dependency":
            reviewer_tools = FRAMEWORK_DEPENDENCY_REVIEWER_TOOLS
        else:
            # code_level, dependency_mitigation, systemic all trace first-party
            # code and share the source-reading toolset.
            reviewer_tools = CODE_LEVEL_REVIEWER_TOOLS
        return reviewer_llm.bind_tools(
            reviewer_tools,
            parallel_tool_calls=True
        )

    def pre_agent(self, state):
        if not state.get("messages"):
            report = state.get("expert_report", {})
            cached_data = cache_reviewer(reviewer_cache_key(report, state.get("node_id", "Unknown")), report)
            if cached_data:
                logging.info("Reviewer cache hit.")
                return Command(
                    update={
                        "vulnerabilities": [cached_data]
                    }
                )
        return None

    def first_turn(self, state, llm_with_tools) -> dict:
        # Compose the targeted system prompt: shared directives + mode-specific
        # reachability standard. Mode names match agents.yaml keys exactly.
        mode = state.get("mode", "code_level")
        mode_prompt = REVIEWER_AGENT.get(mode, "")
        sys_prompt = REVIEWER_AGENT.get('prompt', '')
        if mode_prompt:
            sys_prompt = f"{sys_prompt}\n\n{mode_prompt}"
        sys_msg = SystemMessage(content=sys_prompt)

        report = state.get("expert_report", {})
        node_id = state.get("node_id")

        affected = [n for n in (report.get("affected_nodes") or []) if n]
        affected_str = ", ".join(affected) if affected else node_id
        formatted_vuln = (
            f"Target: {node_id}\n"
            f"Affected Nodes: {affected_str}\n\n"
            f"Potential Issue to Investigate:\n"
            f"- Type: {report.get('vulnerability_type', 'Code Defect')}\n"
            f"- CWE ID: {report.get('cwe_id', 'Unknown')}\n"
            f"- Component: {report.get('vulnerable_component', 'Unknown')}\n"
            f"- Description: {report.get('description', '')}\n"
        )
        if report.get("source_cve"):
            formatted_vuln += f"- Source CVE: {report.get('source_cve')}\n"

        feedback_qs = report.get("open_questions") or []
        if feedback_qs:
            qs_str = "\n".join(f"  {i}. {q}" for i, q in enumerate(feedback_qs, 1))
            formatted_vuln += (
                f"\n--- VALIDATOR FEEDBACK (INSUFFICIENT CONTEXT) ---\n"
                f"The downstream Validator could not confirm this vulnerability because it "
                f"lacked the information below. Resolve EACH question using your tools, then "
                f"re-emit fully self-sufficient reproduction steps (exact HTTP method, path, "
                f"parameters/headers/body, and any session state) with `submit_evaluation`. "
                f"This is feedback round {report.get('review_round', 0)}.\n"
                f"{qs_str}\n"
            )

        # Synthetic `dependency:<package>` nodes (CVE upgrade-only / known
        # dependency vulnerabilities) don't exist in the app graph: there is no
        # source code to attach, so skip the lookup entirely.
        if node_id and not node_id.startswith("dependency:"):
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
        return {"messages": [sys_msg, human_msg, response], "iterations": 1}

    def pre_router(self, state) -> bool:
        return len(state["messages"]) == 0 and state.get("vulnerabilities")

    def fallback(self, state) -> Command:
        """Resolve a review that hit the iteration cap without a submit_evaluation
        verdict. Mirrors submit_evaluation's output shape so the record flows
        through the standard reviewer output/cache path, but marks it review_error
        instead of silently confirming or discarding it."""
        report = dict(state.get("expert_report", {}))

        updated_vuln = dict(report)
        updated_vuln["status"] = "review_error"
        updated_vuln["reviewer_reasoning"] = (
            f"Review terminated after {state.get('iterations', 0)} tool-loop iterations "
            f"without a submit_evaluation verdict (loop budget exceeded)."
        )

        cache_reviewer(reviewer_cache_key(report, state.get("node_id", "Unknown")), report, updated_vuln)

        return Command(
            update={
                "vulnerabilities": [updated_vuln],
            }
        )


reviewer_agent = ReviewerAgent(
    name="reviewer",
    settings_prefix="reviewer",
    compaction=CompactionConfig(prefix="reviewer"),
    summary_ledger=REVIEWER_SUMMARY_LEDGER,
    summary_llm=fast_llm,
)

# Module-level graph node callables (kept so graph.py's imports stay untouched).
reviewer_agent_node = reviewer_agent.agent
reviewer_router = reviewer_agent.router
reviewer_fallback_node = reviewer_agent.fallback
ask_reviewer_for_tool = reviewer_agent.ask

# ==========================================
# Validator agent
# ==========================================

def dispatch_validators(state: MasterState):
    """Creates a parallel validation thread for each vulnerability that survived the reviewer.

    Confirmed records are routed by their `validation_strategy`:
    - `direct_to_validator` (or missing): the vulnerability can be triggered directly
      or its only prerequisites are freely attainable via public endpoints (e.g., open
      self-registration, standard login); sent to the Validator to be proven externally.
    - `requires_integration`: the vulnerability requires privileges that cannot be
      freely registered (e.g., an Admin account) or strictly requires the output of
      another exploit; DEFERRED here — it is not sent to the Integration Auditor
      until the direct-to-validator records above have been proven, so the auditor's
      chain candidates carry real validator `poc_payload`s. `dispatch_integration_audits`
      (run from `integration_audit_dispatch` after the validator superstep) fans them out.
    - `static_finding_only`: no network-reachable path exists; the Reviewer's static
      proof is accepted into the final report and the record is not dispatched.

    Returns Sends to `validator_agent` when at least one direct record must be
    validated, otherwise the `integration_audit_dispatch` marker so the pipeline
    still advances into the audit phase (even with zero validator tasks).
    """
    raw_vulns = state.get("vulnerabilities", [])
    all_vulns = [
        v if isinstance(v, dict) else v.model_dump()
        for v in raw_vulns
    ]
    # Filter for vulnerabilities that were confirmed by the Reviewer
    confirmed_vulns = [v for v in all_vulns if v.get("status") == "confirmed"]
    logging.info(
        f"dispatch_validators sees {len(all_vulns)} records in the parent channel, "
        f"{len(confirmed_vulns)} confirmed "
        f"({[v.get('vuln_id') for v in confirmed_vulns]})."
    )

    commands = []
    for evaluation in confirmed_vulns:
        strategy = evaluation.get("validation_strategy") or "direct_to_validator"
        if strategy == "static_finding_only":
            # Real in source, no network-reachable path. Accept the Reviewer's static
            # proof into the final report without Validator testing.
            logging.info(
                f"{evaluation.get('vuln_id')} marked static_finding_only — "
                f"accepted into report without validation."
            )
            continue
        if strategy == "requires_integration":
            # Deferred to the integration-audit phase: runs only after the
            # direct-to-validator records have been validated, so the auditor's
            # chain candidates (exploitable peers) carry proven poc_payloads.
            logging.info(
                f"{evaluation.get('vuln_id')} requires_integration — deferred to "
                f"the integration audit phase (after direct validation)."
            )
            continue
        payload = ValidatorState(
            report_to_test=evaluation,
            sandbox_url=state.get("sandbox_url"),
            messages=[],
            iterations=0,
            vulnerabilities=[],
            cookies={},
            agent_id=uuid.uuid4().hex,
        )
        commands.append(Send("validator_agent", payload))

    if not commands:
        # Nothing to validate directly: advance to the integration-audit phase
        # (dispatch_integration_audits) instead of ENDing, so deferred
        # `requires_integration` records still reach the auditor.
        return "integration_audit_dispatch"

    return commands


def dispatch_integration_audits(state: MasterState):
    """Runs strictly AFTER all direct-to-validator records have been proven.

    Fans every still-`confirmed` `requires_integration` record to the Integration
    Auditor. Each record's chain candidates (`confirmed_vulns`) are the OTHER
    records that reached `exploitable` — i.e. the direct-to-validator findings the
    Validator already proved in the sandbox, whose `poc_payload` the auditor can
    reason over and whose proven payloads a `chained` verdict will hand back to the
    Validator. Records with no exploitable peers are resolved `unchainable` by the
    auditor's `first_turn` without an LLM call.

    Records re-reviewed into `confirmed` after a reviewer re-answer are picked up
    here again and re-audited; the `review_round` one-ask cap drains that loop.
    """
    raw_vulns = state.get("vulnerabilities", [])
    all_vulns = [
        v if isinstance(v, dict) else v.model_dump()
        for v in raw_vulns
    ]
    pending = [
        v for v in all_vulns
        if v.get("status") == "confirmed"
        and v.get("validation_strategy") == "requires_integration"
    ]
    proven = [v for v in all_vulns if v.get("status") == "exploitable"]
    logging.info(
        f"dispatch_integration_audits sees {len(all_vulns)} records, "
        f"{len(pending)} requires_integration still confirmed, "
        f"{len(proven)} proven exploitable peer(s)."
    )

    if not pending:
        return END

    commands = []
    for evaluation in pending:
        others = [
            v for v in proven
            if v.get("vuln_id") != evaluation.get("vuln_id")
        ]
        logging.info(
            f"Auditing {evaluation.get('vuln_id')}: {len(others)} proven peer(s) "
            f"to chain with "
            f"({[v.get('vuln_id') for v in others]})."
        )
        payload = IntegrationAuditorState(
            report_to_test=evaluation,
            confirmed_vulns=others,
            iterations=0,
            vulnerabilities=[],
            messages=[]
        )
        commands.append(Send("integration_auditor", payload))

    return commands


INTEGRATION_AUDITOR_SUMMARY_LEDGER = (
    "You are an expert summarizer. The following is the message history of a "
    "security integration auditor agent deciding whether a `requires_integration` "
    "vulnerability (real but not exploitable in isolation) can be combined with "
    "other confirmed vulnerabilities into a concrete multi-step external exploit "
    "chain. Your summary will REPLACE these messages in the model context, so the "
    "auditor must be able to continue the decision from it WITHOUT re-reading the "
    "original tool outputs.\n\n"
    "Produce an information-dense summary as a security chaining ledger with "
    "exactly these sections:\n"
    "## Objective\n"
    "One or two sentences restating the exact `requires_integration` vulnerability "
    "under audit (vuln_id, CWE, affected nodes) and the chain decision pending.\n"
    "## Candidate Peers Examined\n"
    "Bulleted, deduplicated list of every other confirmed vulnerability whose "
    "details were fetched, with the single most important fact each revealed "
    "about how it could provide a precondition (privilege, session, state, file) "
    "to the chained path. Do NOT include full records — distill them.\n"
    "## Confirmed Chain Facts\n"
    "Bulleted list of verified facts established for the chain (e.g. 'IDOR "
    "leaks any user id so it can obtain the admin cookie', 'XSS fires only after "
    "authentication') stated in final form.\n"
    "## Ruled-Out Candidates\n"
    "Bulleted list of other vulnerabilities or chaining paths already rejected, "
    "with a one-line reason for each.\n"
    "## Active Leads & Next Steps\n"
    "Bulleted list of the most promising remaining chain checks not yet completed.\n\n"
    "RULES:\n"
    "- If the history contains a prior SUMMARY (a system message containing "
    "'CONTEXT COMPACTION SUMMARY'), treat it as the anchoring summary: extend "
    "and refine it with only the new facts gathered since, rather than "
    "regenerating from scratch.\n"
    "- Write in final form ('the IDOR returns any profile', 'auth is required "
    "before the sink'), never 'the model checked X'.\n"
    "- Keep it concise, under 1024 words. Preserve exact vuln_ids, node ids, "
    "and reproduction-step content.\n"
)


def _one_line(text: str, limit: int = 240) -> str:
    """Flatten free text to a single compact line for peer summaries."""
    if not text:
        return ""
    return " ".join(str(text).split())[:limit]


class IntegrationAuditorAgent(ToolLoopAgent):
    """Integration auditor track: chain-building tools (get_vulnerability_details
    for peer records, get_node_connections to verify code-level links) and the
    single terminal submit_integration_audit verdict (chained/unchainable)."""

    terminal_tool = "submit_integration_audit"

    def _subject(self, state) -> str:
        return state.get("report_to_test", {}).get("vuln_id", "Unknown")

    def bind_tools(self, state):
        return smart_llm.bind_tools([
            tools.get_vulnerability_details,
            tools.get_node_connections,
            tools.get_path,
            tools.submit_integration_audit,
        ])

    def pre_router(self, state) -> bool:
        """End the auditor loop immediately when `first_turn` already resolved the
        record as `unchainable` (no other confirmed vulnerability exists to chain
        with; a 'chained' verdict is impossible by construction). The router is
        then never allowed to index the still-empty `messages` list. Invariant:
        the subgraph's `vulnerabilities` channel only ever holds this one record
        (init `[]`, `first_turn` writes exactly one), so the match-by-id scan is
        safe and can never end a live with-peers loop."""
        report_vid = state.get("report_to_test", {}).get("vuln_id")
        if not report_vid:
            return False
        return any(
            v.get("vuln_id") == report_vid and v.get("status") == "unchainable"
            for v in state.get("vulnerabilities", [])
        )

    def first_turn(self, state, llm_with_tools) -> dict:
        sys_msg = SystemMessage(content=INTEGRATION_AUDITOR_AGENT.get("prompt", ""))

        report = state.get("report_to_test", {})
        affected = [n for n in (report.get("affected_nodes") or []) if n]
        affected_str = (
            ", ".join(affected) if affected else report.get("node_id", "Unknown")
        )
        formatted_vuln = (
            f"--- CORE VULNERABILITY (requires_integration) ---\n"
            f"Vulnerability ID: {report.get('vuln_id', 'Unknown')}\n"
            f"CWE ID: {report.get('cwe_id', 'Unknown')}\n"
            f"Affected Nodes: {affected_str}\n"
            f"Type: {report.get('vulnerability_type', 'Code Defect')}\n"
            f"Description: {report.get('description', '')}\n"
            f"Reviewer Reasoning: {report.get('reviewer_reasoning', 'None')}\n"
        )
        if report.get("source_cve"):
            formatted_vuln += f"Source CVE: {report.get('source_cve')}\n"

        peers = state.get("confirmed_vulns", [])
        if peers:
            peer_lines = []
            for v in peers:
                peer_nodes = [n for n in (v.get("affected_nodes") or []) if n]
                payload = v.get("poc_payload")
                payload_hint = "proven payload available (see details)" if payload else "no proven payload"
                peer_lines.append(
                    f"- vuln_id={v.get('vuln_id', '?')} | CWE={v.get('cwe_id', '?')} | "
                    f"nodes={', '.join(peer_nodes) or '?'} | {payload_hint} | "
                    f"{_one_line(v.get('description', ''))}"
                )
            formatted_vuln += (
                f"\n--- OTHER PROVEN VULNERABILITIES (candidates to chain with) ---\n"
                f"All of these were validated in the sandbox and are exploitable; their "
                f"working `poc_payload`s are available. Call "
                f"`get_vulnerability_details(<vuln_id>)` to fetch the full record "
                f"(including the proven payload) of any of these before relying on it "
                f"in a chain:\n"
                + "\n".join(peer_lines)
            )
        else:
            # No peers to chain with, so a 'chained' verdict is impossible by
            # construction. Resolve deterministically here -- inside the auditor,
            # not as a separate graph node -- without launching an LLM loop on a
            # dead end: mark the record 'unchainable' (terminal, stays in the
            # report) and let `pre_router` end the subgraph before any message
            # indexing. `agent()` returns this `Command` verbatim.
            logging.warning(
                f"{report.get('vuln_id', 'Unknown')} is requires_integration but arrived "
                f"at the auditor with an EMPTY confirmed_vulns peer list; resolving "
                f"'unchainable' without invoking the auditor LLM. Possible cause: "
                f"dispatch_integration_audits saw no 'exploitable' records in the "
                f"parent channel (check the dispatch_integration_audits log lines "
                f"in this run)."
            )
            record = dict(report)
            existing = record.get("integration_audit_reasoning") or ""
            note = (
                "[integration auditor] No other proven vulnerabilities exist to "
                "chain with; resolved 'unchainable' without invoking the LLM (a "
                "'chained' verdict requires at least one other exploitable "
                "vulnerability)."
            )
            record["integration_audit_reasoning"] = (
                f"{existing}\n{note}" if existing else note
            )
            record["status"] = "unchainable"
            return Command(update={"vulnerabilities": [record]})

        steps = report.get("reproduction_steps") or []
        # Strip any leading "N." / "N)" numbering the reviewer already embedded
        # so our prefixed counter does not double-number each step.
        steps_str = (
            "\n".join(
                f"  {i}. {re.sub(r'^\s*\d+[\.\)]\s+', '', str(s))}"
                for i, s in enumerate(steps, 1)
            )
            if steps else "  None provided by reviewer"
        )
        formatted_vuln += (
            f"\n--- REVIEWER'S ISOLATED REPRODUCTION STEPS (this record alone) ---\n"
            f"{steps_str}"
        )

        human_msg = HumanMessage(content=(
            f"Audit the following `requires_integration` vulnerability for a combinable "
            f"multi-step exploit chain.\n\n{formatted_vuln}"
        ))

        response = llm_with_tools.invoke([sys_msg, human_msg])
        return {"messages": [sys_msg, human_msg, response], "iterations": 1}

    def fallback(self, state) -> Command:
        """Resolve an audit that hit the iteration cap without a
        submit_integration_audit verdict. Keeps the record as 'confirmed' (the
        chain was never proven) and records the timeout in the reasoning."""
        updated_vuln = dict(state.get("report_to_test", {}))
        timeout_note = (
            f"[integration audit timeout] No verdict after {state.get('iterations', 0)} "
            f"tool-loop iterations; the chain was not resolved. Keeping the record as "
            f"'confirmed' (chain inconclusive) without a chained/unchainable verdict."
        )
        existing = updated_vuln.get("integration_audit_reasoning") or ""
        updated_vuln["integration_audit_reasoning"] = (
            f"{existing}\n{timeout_note}" if existing else timeout_note
        )
        return Command(update={"vulnerabilities": [updated_vuln]})


integration_auditor_agent = IntegrationAuditorAgent(
    name="integration_auditor",
    settings_prefix="integration_auditor",
    compaction=CompactionConfig(prefix="integration_auditor"),
    summary_ledger=INTEGRATION_AUDITOR_SUMMARY_LEDGER,
    summary_llm=fast_llm,
)

# Module-level graph node callables (kept so graph.py's imports stay untouched).
integration_auditor_node = integration_auditor_agent.agent
integration_auditor_router = integration_auditor_agent.router
integration_auditor_fallback_node = integration_auditor_agent.fallback
ask_integration_auditor_for_tool = integration_auditor_agent.ask


def route_integration_audit(state: MasterState):
    """Conditional router after the integration auditor completes its tasks.

    `chained` records (upgraded in place by submit_integration_audit) are sent to
    the Validator, which builds a PoC from the auditor's complete combined
    reproduction_steps. Each chained Validator also receives the `peer_payloads`
    — the proven `poc_payload`/`execution_logs` of the OTHER vulnerabilities it
    chains with (matched via `chained_with`) — so the final exploit reuses the
    peers' already-proven payloads instead of re-deriving them. `unchainable`
    records are terminal — they stay in the report without further validation.
    """
    raw_vulns = state.get("vulnerabilities", [])
    all_vulns = [
        v if isinstance(v, dict) else v.model_dump()
        for v in raw_vulns
    ]
    chained = [v for v in all_vulns if v.get("status") == "chained"]

    if not chained:
        return END

    by_id = {v.get("vuln_id"): v for v in all_vulns}
    commands = []
    for record in chained:
        peer_ids = (record.get("chained_with") or [])
        peer_payloads = []
        for vid in peer_ids:
            peer = by_id.get(vid)
            if not peer:
                logging.warning(
                    f"Chained record {record.get('vuln_id')} references "
                    f"chained_with vuln '{vid}' not found in state; skipping peer payload."
                )
                continue
            peer_payloads.append({
                "vuln_id": peer.get("vuln_id"),
                "cwe_id": peer.get("cwe_id"),
                "description": peer.get("description"),
                "poc_payload": peer.get("poc_payload"),
                "execution_logs": peer.get("execution_logs"),
            })
        payload = ValidatorState(
            report_to_test=record,
            sandbox_url=state.get("sandbox_url"),
            peer_payloads=peer_payloads,
            messages=[],
            iterations=0,
            vulnerabilities=[],
            cookies={},
            agent_id=uuid.uuid4().hex,
        )
        commands.append(Send("validator_agent", payload))

    logging.info(
        f"Integration auditor chained {len(commands)} vulnerability(ies); "
        f"dispatching to the validator for PoC construction."
    )
    return commands


def route_validator_feedback(state: MasterState):
    """Conditional router from the validator back into the reviewer (or onward).

    When the Validator flags a record 'insufficient_context', send that record
    (with its open questions back to the Reviewer for a re-review. Each record
    may request context only validator_feedback_max_rounds times; records past
    the cap are left in place, so the loop drains.

    When nothing is flagged, return the `integration_audit_dispatch` marker so
    the pipeline continues into the integration-audit phase after the validator
    superstep (that phase ENDs itself when no audit is pending).
    """
    raw_vulns = state.get("vulnerabilities", [])
    all_vulns = [
        v if isinstance(v, dict) else v.model_dump()
        for v in raw_vulns
    ]
    max_rounds = settings.validator_feedback_max_rounds
    flagged = [
        v for v in all_vulns
        if v.get("status") == "insufficient_context"
        and (v.get("review_round") or 0) <= max_rounds
    ]

    if not flagged:
        return "integration_audit_dispatch"

    commands = []
    for record in flagged:
        payload = ReviewerState(
            node_id=_primary_node(record),
            expert_report=record,
            mode=_reviewer_mode_for(record),
            iterations=0,
            vulnerabilities=[],
            messages=[]
        )
        commands.append(Send("reviewer_agent", payload))

    logging.info(
        f"Validator requested more context for {len(commands)} vulnerability(ies); "
        f"dispatching reviewer feedback re-reviews."
    )
    return commands


class ValidatorAgent(ToolLoopAgent):
    """Validator track: fixed HTTP-proof toolset, live cookie tracking, and the
    terminal-tool end conditions. Ends on `ask_for_context` (round 1 only — the
    tool is unbound after a re-review so the agent cannot ask again) or on
    `mark_validation_complete`."""

    terminal_tool = ("ask_for_context", "mark_validation_complete")

    def _subject(self, state) -> str:
        report = state.get("report_to_test", {})
        affected = [n for n in (report.get("affected_nodes") or []) if n]
        if affected:
            return ", ".join(affected)
        return report.get("node_id", "Unknown")

    def bind_tools(self, state):
        validator_tools = [
            tools.send_http_request,
            browser_tools.browser_navigate,
            browser_tools.browser_click,
            browser_tools.browser_fill,
            browser_tools.browser_evaluate,
            browser_tools.browser_console,
            tools.mark_validation_complete
        ]
        if getattr(settings, "attacker_enabled", False):
            validator_tools += [
                attacker_tools.run_command,
                attacker_tools.write_attacker_file,
                attacker_tools.read_attacker_file
            ]
        # ask_for_context is bound ONLY on the first validation pass. Once the
        # Reviewer has re-answered (review_round > 0), it is removed so the
        # agent cannot be tempted to request more context a second time
        if (state.get("report_to_test", {}).get("review_round") or 0) < settings.validator_feedback_max_rounds:
            validator_tools.append(tools.ask_for_context)
        return smart_llm.bind_tools(validator_tools)

    def first_turn(self, state, llm_with_tools) -> dict:
        # Compose the validator system prompt from the capabilities this run
        # actually grants: the attacker shell tools are only described when
        # enabled, and the insufficient-context escape hatch is only described on
        # the first validation pass (the tool is unbound afterwards, mirroring the
        # gating in bind_tools).
        sys_prompt = VALIDATOR_AGENT["prompt"]
        if getattr(settings, "attacker_enabled", False):
            sys_prompt += "\n\n" + VALIDATOR_AGENT.get("attacker_tools", "")
        if (state.get("report_to_test", {}).get("review_round") or 0) < settings.validator_feedback_max_rounds:
            sys_prompt += "\n\n" + VALIDATOR_AGENT.get("insufficient_context", "")
        sys_msg = SystemMessage(content=sys_prompt)
        # Build a structured string for the LLM
        report = state['report_to_test']
        affected = [n for n in (report.get("affected_nodes") or []) if n]
        affected_str = (
            ", ".join(affected) if affected else report.get('node_id', 'Unknown')
        )
        steps = report.get('reproduction_steps') or []
        steps_str = (
            "\n".join(f"  {i}. {s}" for i, s in enumerate(steps, 1))
            if steps else "  None provided by reviewer"
        )
        formatted_report = (
            f"--- CORE VULNERABILITY ---\n"
            f"Vulnerability ID: {report.get('vuln_id', 'Unknown')}\n"
            f"CWE ID: {report.get('cwe_id', 'Unknown')}\n"
            f"Affected Nodes: {affected_str}\n\n"
            f"--- CONTEXT & REASONING ---\n"
            f"Description: {report.get('description', 'None')}\n\n"
            f"Reviewer Reasoning: {report.get('reviewer_reasoning', 'None')}\n\n"
            f"--- REPRODUCTION STEPS (from Reviewer, follow in order) ---\n"
            f"{steps_str}"
        )
        # For a `chained` record the auditor hands the validator the proven
        # poc_payloads of the OTHER vulnerabilities this chain depends on, so the
        # final exploit reuses real proven primitives instead of re-deriving them.
        peer_payloads = state.get("peer_payloads") or []
        if peer_payloads:
            blocks = []
            for pp in peer_payloads:
                blocks.append(
                    f"### {pp.get('vuln_id', '?')}\n"
                    f"CWE: {pp.get('cwe_id', '?')}\n"
                    f"Description: {pp.get('description') or '_none_'}\n"
                    f"Proven poc_payload:\n"
                    f"```\n{(pp.get('poc_payload') or '_none_').rstrip()}\n```\n"
                    f"Execution logs:\n"
                    f"```\n{(pp.get('execution_logs') or '_none_').rstrip()}\n```"
                )
            formatted_report += (
                f"\n\n--- PROVEN CHAIN COMPONENTS (poc_payloads from peer validators) ---\n"
                f"These OTHER vulnerabilities are already proven exploitable in the "
                f"sandbox and their working payloads are below. The chained "
                f"reproduction steps above build on them — execute/adapt these exact "
                f"payloads to complete the chain.\n"
                f"{'\n\n'.join(blocks)}"
            )
        # Attach the source code of every affected node so the validator can
        # reason about the exact code under test without extra lookups.
        code_sections = []
        for node_id in affected:
            node_source = get_node_code(node_id)
            if node_source:
                code_sections.append(
                    f"Node: {node_id}\n```\n{node_source}\n```"
                )
        if code_sections:
            formatted_report += (
                f"\n\n--- AFFECTED NODES SOURCE CODE ---\n"
                f"Source code of nodes affected by the vulnerability (bodies of "
                f"peer nodes are omitted because not relevant).\n"
                f"{'\n\n'.join(code_sections)}"
            )
        human_msg = HumanMessage(content=(
            f"Target Sandbox: {state['sandbox_url']}\n\n"
            f"Vulnerability to Prove:\n{formatted_report}"
        ))
        messages = [sys_msg, human_msg]
        response = llm_with_tools.invoke(messages)
        return {"messages": [sys_msg, human_msg, response], "iterations": 1}

    def session_state(self, state) -> dict:
        # Update cookies from the recent history (scans the raw, pre-compaction
        # history so compacted cookie-bearing responses are not missed).
        # Both send_http_request (plain {name: value} artifact) and the browser
        # tools (nested {"session_id", "cookies"} artifact) carry cookie jars,
        # which lets the two channels stay in sync across the same validators.
        current_cookies = dict(state.get("cookies", {}))
        for msg in reversed(state["messages"]):
            if getattr(msg, "type", "") == "ai":
                break
            if getattr(msg, "type", "") == "tool" and hasattr(msg, "artifact") and msg.artifact:
                name = getattr(msg, "name", "")
                if name == "send_http_request":
                    current_cookies.update(msg.artifact)
                elif name in browser_tools.BROWSER_TOOL_NAMES and isinstance(msg.artifact, dict):
                    jar = msg.artifact.get("cookies") or {}
                    if isinstance(jar, dict):
                        current_cookies.update(jar)
        return {"cookies": current_cookies}

    def tool_batch_done(self, state) -> bool:
        # `ask_for_context` is terminal only on passes where it is actually
        # bound (mirrors the bind_tools/first_turn gating); on the final pass
        # it must not end the loop without a verdict.
        first_pass = (
            state.get("report_to_test", {}).get("review_round") or 0
        ) < settings.validator_feedback_max_rounds
        terminal_names = [
            n for n in self._terminal_names()
            if n != "ask_for_context" or first_pass
        ]
        for msg in reversed(state["messages"]):
            if getattr(msg, "type", "") != "tool":
                break
            if getattr(msg, "name", "") in terminal_names:
                return True
        return False

    def fallback(self, state) -> Command:
        """Resolve a validation that hit the iteration cap without a
        mark_validation_complete verdict. Keeps the reviewer's "confirmed" status
        (it was never proven exploitable, and was never proven a false positive)
        and records the timeout in the execution logs."""
        updated_vuln = dict(state.get("report_to_test", {}))

        timeout_note = (
            f"[validation timeout] No verdict after {state.get('iterations', 0)} "
            f"tool-loop iterations; result on this vulnerability is unproven."
        )
        existing_logs = updated_vuln.get("execution_logs") or ""
        updated_vuln["execution_logs"] = (
            f"{existing_logs}\n{timeout_note}" if existing_logs else timeout_note
        )

        # Close any headless-browser sessions this validator opened (same
        # per-agent cleanup the terminal tool performs).
        browser_tools.manager.close_agent_sessions(state.get("agent_id"))

        return Command(
            update={
                "vulnerabilities": [updated_vuln],
            }
        )


validator_agent = ValidatorAgent(
    name="validator",
    settings_prefix="validator",
    compaction=CompactionConfig(prefix="validator"),
    summary_ledger=VALIDATOR_SUMMARY_LEDGER,
    summary_llm=fast_llm,
)

# Module-level graph node callables (kept so graph.py's imports stay untouched).
validator_agent_node = validator_agent.agent
validator_router = validator_agent.router
validator_fallback_node = validator_agent.fallback
ask_validator_for_tool = validator_agent.ask
