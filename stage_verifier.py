"""Contract-verifier stage: batch-check security demands against target code."""

import hashlib
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

import settings
from llms import get_llm, invoke_structured_capped
from run_stats import (
    _log_agent_completion,
    _record_stat,
    _start_agent_progress,
    add_usage,
    new_usage,
    raise_if_stopping,
    take_cached_usage,
)
from schemas import VERIFIER_AGENT, VerifierOutput, cwes
from stage_cve import _normalize_cwe_ids
from state import MasterState, VerifierState
from utils import (
    cache,
    get_cached_graph_data,
    get_node_code,
    is_path_excluded,
)

logger = logging.getLogger(__name__)


def dispatch_verifiers(state: MasterState):
    """Fan out one verifier task per target node with demands."""
    grouped_demands = state.get("grouped_demands", {})

    # Even with no demands, advance to synchronization so Edge Traversal
    # still analyzes the explorer notes.
    if not grouped_demands:
        return "synchronization"

    targets: list[tuple[str, str, list]] = []

    graph_data = get_cached_graph_data(settings.graph)
    node_source_map = {
        n.get("id"): n.get("source_file") for n in graph_data.get("nodes", [])
    }

    for target_node_id, demands_list in grouped_demands.items():
        # Skip demands in excluded paths (dependency trees, tests, docs).
        if is_path_excluded(node_source_map.get(target_node_id) or ""):
            logger.debug(
                f"Skipping contract verification of excluded-path node {target_node_id}."
            )
            continue

        target_code = get_node_code(target_node_id)

        if not target_code:
            continue

        targets.append((target_node_id, target_code, demands_list))

    if not targets:
        return "synchronization"

    progress_id = _start_agent_progress(len(targets))
    logger.info(
        "Starting contract verifier scan: 0/%d complete, %d remaining.",
        len(targets),
        len(targets),
    )

    commands = [
        Send(
            "contract_verifier",
            {
                "target_node_id": target_node_id,
                "target_code": target_code,
                "incoming_demands": demands_list,
                "progress_id": progress_id,
            },
        )
        for target_node_id, target_code, demands_list in targets
    ]
    return commands


_DISPLAY_NODES: dict | None = None
_NODE_DISPLAY_MEMO: dict[str, str] = {}


def _node_display_name(node_id: str) -> str:
    """Human-readable node label for prompt prose (members as
    ``Class::method()``, fallback to the raw id). Never a matching key."""
    if not node_id:
        return node_id
    if node_id in _NODE_DISPLAY_MEMO:
        return _NODE_DISPLAY_MEMO[node_id]
    global _DISPLAY_NODES
    if _DISPLAY_NODES is None:
        _DISPLAY_NODES = {
            n["id"]: n
            for n in get_cached_graph_data(settings.graph).get("nodes", [])
            if n.get("id")
        }
    node = _DISPLAY_NODES.get(node_id)
    name = node_id
    if node:
        label = node.get("label") or ""
        if label.startswith(".") and label.endswith("()"):
            mname = label[1:-2]
            if node_id.endswith("_" + mname.lower()):
                parent = _DISPLAY_NODES.get(node_id[: len(node_id) - len(mname) - 1])
                if parent and parent.get("source_file") == node.get("source_file"):
                    name = f"{parent.get('label', '')}::{mname}()"
                else:
                    name = f"{mname}()"
            else:
                name = mname
        else:
            name = label or node_id
    _NODE_DISPLAY_MEMO[node_id] = name
    return name


def _contract_verifier_node(state: VerifierState) -> tuple[dict, str]:
    """Verify each incoming demand of one target node; FAILED ones become
    hypotheses. Returns (state_update, cache tag for the progress line)."""
    target_node_id = state.get("target_node_id")
    target_code = state.get("target_code")
    demands = state.get("incoming_demands", [])

    if not demands or not target_code:
        return {"vulnerabilities": []}, ""

    # Hash over demands (already canonically sorted by dedup): content changes
    # bust the cache, order stays stable across runs.
    demands_hash = hashlib.md5(json.dumps(demands, sort_keys=True).encode()).hexdigest()
    cache_file = (
        settings.cache_dir
        / "contract_verifier"
        / f"{target_node_id}_{demands_hash}.json"
    )

    cached_data = cache(cache_file, "read")
    if cached_data:
        take_cached_usage("contract_verifier", cached_data)
        return {"vulnerabilities": cached_data.get("hypothesis", [])}, "HIT"

    # Map the prompt [ID: ...] back to the original demand dict so FAILED
    # evaluations can be traced to their source (e.g. an CVE demand).
    demand_meta = {}
    formatted_demands = []

    for d in demands:
        d_id = d.get("parameter_name") or d.get("source") or "unknown"
        demand_meta[d_id] = d

        dtype = d.get("type")
        source = d.get("source")
        desc = d.get("description")

        if dtype == "explorer_upstream_assumption":
            param = d.get("parameter_name", "unknown")
            formatted_demands.append(
                f"- [ID: {param}] [DEMAND FROM CALLEE] You call '{_node_display_name(source)}'. It demands: '{desc}'"
            )
        elif dtype == "cve_assumption":
            suggested_cwes = _normalize_cwe_ids(d.get("cwe_ids"))
            suffix = (
                f" [SUGGESTED CWE: {', '.join(suggested_cwes)}]"
                if suggested_cwes
                else ""
            )
            formatted_demands.append(
                f"- [ID: {source}] [LIBRARY CVE MITIGATION] Known constraint: '{desc}'{suffix}"
            )
        else:
            formatted_demands.append(f"- [ID: {source}] {desc}")

    # Chunk calls so a hub node's per-demand output never truncates at the
    # token cap; batches cache individually so retries pay only for unfinished ones.
    batch_size = settings.verifier_max_demands_per_call
    batch_starts = range(0, len(formatted_demands), batch_size)

    sys_msg = SystemMessage(content=VERIFIER_AGENT["prompt"])
    structured_llm = get_llm("contract_verifier").with_structured_output(
        VerifierOutput, method="json_schema", strict=True
    )

    evaluations = []
    skipped_demands = 0
    node_usage = new_usage()  # tokens of THIS build: fresh batches + cached-batch spend
    for b_idx in batch_starts:
        batch = demands[b_idx : b_idx + batch_size]
        batch_hash = hashlib.md5(json.dumps(batch, sort_keys=True).encode()).hexdigest()
        batch_cache_file = (
            settings.cache_dir
            / "contract_verifier"
            / f"{target_node_id}_batch{b_idx}_{batch_hash}.json"
        )
        cached_batch = cache(batch_cache_file, "read")
        if cached_batch and isinstance(cached_batch.get("evaluations"), list):
            # The node entry must reproduce the FULL original build cost, so
            # cached-batch spend joins node_usage too (booked live once by
            # take_cached_usage, which pops the key before it can leak on).
            node_usage = add_usage(node_usage, cached_batch.get("token_usage"))
            take_cached_usage("contract_verifier", cached_batch)
            evaluations.extend(cached_batch["evaluations"])
            continue
        batch_demands_string = "\n".join(formatted_demands[b_idx : b_idx + batch_size])
        human_msg = HumanMessage(
            content=f"```python\n{target_code}\n```\n\nSecurity Demands:\n{batch_demands_string}"
        )
        response, usage = invoke_structured_capped(
            structured_llm,
            [sys_msg, human_msg],
            f"Contract verifier {target_node_id} batch{b_idx}",
            "contract_verifier",
        )
        node_usage = add_usage(node_usage, usage)
        if response is None:
            # Output cap exhausted: drop the batch uncached so a later run re-attempts it.
            skipped_demands += len(batch)
            continue
        response = response if isinstance(response, dict) else response.model_dump()
        batch_evals = response.get("evaluations") or []
        cache(
            batch_cache_file,
            "write",
            {"evaluations": batch_evals, "token_usage": usage},
        )
        evaluations.extend(batch_evals)

    if skipped_demands:
        _record_stat("verifier_demands_skipped_output_cap", skipped_demands)

    # Deterministic demand-verdict ledger for the report statistics (skipped on
    # a whole-node cache hit, which returned above).
    _record_stat("verifier_evaluations", len(evaluations))
    for status in ("MET", "FAILED", "DELEGATED", "OUT_OF_SCOPE"):
        count = sum(1 for ev in evaluations if (ev or {}).get("status") == status)
        if count:
            _record_stat(f"verifier_{status}", count)

    new_vulnerabilities = []
    for evaluation in evaluations:
        if evaluation.get("status") == "FAILED":
            entry = demand_meta.get(evaluation.get("demand_id"))
            original_desc = (
                entry.get("description")
                if entry is not None
                else "No description found."
            )
            demand_source = entry or {}
            verifier_cwe = evaluation.get("cwe")
            if not verifier_cwe and demand_source.get("type") == "cve_assumption":
                # Deterministic fallback when the model omitted the CWE (an
                # explicit OTHER_UNCATEGORIZED is respected).
                verifier_cwe = next(
                    (
                        c
                        for c in _normalize_cwe_ids(demand_source.get("cwe_ids"))
                        if c in cwes
                    ),
                    None,
                )

            new_vuln = {
                "affected_nodes": [target_node_id],
                "cwe_id": verifier_cwe,
                "description": f"Fails to satisfy demand: '{original_desc}'. Evidence: {evaluation.get('evidence')}",
                "status": "hypothesis",
                "demand_id": evaluation.get("demand_id"),
                "vulnerable_component": evaluation.get("demand_id"),
            }
            # FAILED application_mitigation CVE demands become
            # dependency-mitigation reviews (routed by vulnerability_type).
            if demand_source.get("type") == "cve_assumption":
                new_vuln["vulnerability_type"] = "Dependency Mitigation Vulnerability"
                new_vuln["source_cve"] = demand_source.get("source")

            new_vulnerabilities.append(new_vuln)

    # An output-cap skip leaves the node under-evaluated: never cache the
    # partial result, so the next run re-attempts the skipped batch.
    if not skipped_demands:
        cache(
            cache_file,
            "write",
            {
                "hypothesis": new_vulnerabilities,
                "token_usage": node_usage,
            },
        )

    return {"vulnerabilities": new_vulnerabilities}, "MISS"


def contract_verifier_node(state: VerifierState) -> dict:
    """Graph node wrapper: runs the verifier and advances its progress ledger."""
    raise_if_stopping()
    result, cache_tag = _contract_verifier_node(state)
    detail = f"node={state.get('target_node_id', 'unknown')}"
    if cache_tag:
        detail += f", {cache_tag}"
    _log_agent_completion(state.get("progress_id", ""), "Contract verifier", detail)
    return result
