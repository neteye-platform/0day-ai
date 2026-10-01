"""Explorer stage: per-node/batch code analysis fan-out and note extraction."""

import logging
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

import settings
from llms import fast_llm
from schemas import EXPERT_AGENTS, AnalysisNote, BatchedAnalysisResult
from state import ExplorerState, MasterState
from run_stats import _log_agent_completion, _start_agent_progress
from utils import (
    build_networkx_graph,
    get_cached_graph_data,
    get_node_code,
    format_node_context,
    is_node_worth_scanning,
    is_path_excluded,
    safe_cache_filename,
    cache,
)


def _pack_node_batches(file_nodes: list[str], threshold: int) -> list[list[str]]:
    """Greedily pack node ids into batches whose combined code+context size
    stays below `threshold`; an oversized node becomes its own batch."""
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

    Nodes are batched per (community, source_file) while their combined code
    size stays under settings.explorer_batch_char_threshold. A skeleton
    (file-level) node is dispatched only when no real sibling of the same file
    is eligible in the same task, so skeleton-only files are analyzed too.
    """

    G = build_networkx_graph(settings.graph)
    skipped_nodes = 0
    batched_batches = 0
    single_batches = 0

    # Phase 1: decide dispatches (batch -> task) before registering progress,
    # so the ledger total is exact.
    dispatches: list[tuple[list[str], str, str]] = []

    for task in state["expert_tasks"]:
        task = task if isinstance(task, dict) else task.model_dump()
        clean_id = task.get("target_community", "").lower().replace("community ", "").strip()
        community_nodes = [n for n, attr in G.nodes(data=True) if str(attr.get("community")) == clean_id]

        # Eligible-by-file: (node_id, is_skeleton). Skeletons are file/module-level
        # placeholder nodes; a real sibling's scan already sees the whole file.
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

        # A skeleton whose file has a real eligible sibling is dropped: the
        # sibling's batch carries the whole file as context.
        files: dict[str, list[str]] = {}
        for file_path, nodes in eligible.items():
            has_real = any(not sk for _, sk in nodes)
            files[file_path] = [nid for nid, sk in nodes if not (sk and has_real)]

        for file_path, file_nodes in files.items():
            if settings.explorer_batching_enabled:
                batches = _pack_node_batches(file_nodes, settings.explorer_batch_char_threshold)
            else:
                batches = [[node_id] for node_id in file_nodes]
            for batch in batches:
                if len(batch) > 1:
                    batched_batches += 1
                else:
                    single_batches += 1
                logging.debug(
                    f"Dispatching explorer {task.get('agent_role', '')} on batch "
                    f"{batch} (from {file_path})."
                )
                dispatches.append((
                    batch,
                    task.get("agent_role", ""),
                    task.get("task_description", ""),
                ))

    progress_id = _start_agent_progress(len(dispatches))

    logging.info(
        "Starting explorer scan: 0/%d complete, %d remaining "
        "(%d multi-node batches, %d single-node), skipped %d inert nodes.",
        len(dispatches),
        len(dispatches),
        batched_batches,
        single_batches,
        skipped_nodes,
    )

    commands = [
        Send("explorer_agent", ExplorerState(
            node_ids=batch,
            role=role,
            task_description=task_description,
            progress_id=progress_id,
        ))
        for batch, role, task_description in dispatches
    ]

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
    _log_agent_completion(
        state.get("progress_id", ""),
        "Explorer",
        f"role={state.get('role', 'unknown')}, nodes={', '.join(state.get('node_ids', []))}",
    )


def _extract_hypotheses(node_id: str, raw_hypotheses: list) -> list[dict]:
    """Turn an explorer note's `vulns` entries into standard hypotheses."""
    return [{
        "affected_nodes": [node_id],
        "cwe_id": hyp.get("cwe", "OTHER_UNCATEGORIZED"),
        "description": hyp.get("component", ""),
        "vulnerable_component": hyp.get("pattern_label") or None,
        "status": "hypothesis",
    } for hyp in raw_hypotheses]


def expert_explorer_node(state: ExplorerState) -> dict:
    """Explorer fan-out node: analyzes one node or one same-file batch."""
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
    cache_file = settings.cache_dir / "notes" / safe_cache_filename(f"{node_id}-{role_name}.json")
    cached_note = cache(cache_file, "read")
    if cached_note:
        return cached_note

    source_code = get_node_code(node_id)

    sys_msg = SystemMessage(content=(
        f"{EXPERT_AGENTS['explorer_prompt']}\n\n"
        f"{EXPERT_AGENTS[role_name]['prompt']}"
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

    # 'vulns' is kept in dict_note for later consumers (aggregate edge notes).
    extracted_vulns = _extract_hypotheses(node_id, dict_note.get("vulns", []))

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
        f"{EXPERT_AGENTS['explorer_prompt']}\n\n"
        f"{EXPERT_AGENTS[role_name]['prompt']}\n\n"
        f"{EXPERT_AGENTS['batch_prompt']}"
    ))

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

        extracted_vulns.extend(_extract_hypotheses(node_id, dict_note.get("vulns", [])))
        notes.append(dict_note)

    cache(cache_file, "write", {"notes": notes, "vulnerabilities": extracted_vulns})

    return {
        "notes": notes,
        "vulnerabilities": extracted_vulns
    }
