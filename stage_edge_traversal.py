"""Edge Traversal stage: composite cross-boundary findings from trust-boundary edges."""

import logging
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, SystemMessage

import settings
from llms import fast_llm, invoke_structured_capped
from run_stats import _record_stat, as_dict, raise_if_stopping
from schemas import EDGE_TRAVERSAL_AGENT, EdgeTraversalOutput
from state import MasterState
from boundary_edges import (
    build_boundary_edges,
    cluster_boundary_edges,
    render_batch_prompt,
    boundary_batch_fingerprint,
    summarize_boundary_edges,
)
from utils import cache, get_cached_graph_data, safe_cache_filename

# Independent batch LLM calls run concurrently; results are reassembled in
# dispatch order, so emitted hypotheses and downstream cache keys are unchanged.
_PARALLEL_BATCHES = 12


def _run_batches_in_order(batches: list[list[dict]]) -> list[dict]:
    total = len(batches)
    if total <= 1:
        results = [_run_edge_traversal_batch(b, i, total) for i, b in enumerate(batches, 1)]
    else:
        with ThreadPoolExecutor(
            max_workers=min(total, _PARALLEL_BATCHES),
            thread_name_prefix="edge-traversal",
        ) as pool:
            results = list(pool.map(
                _run_edge_traversal_batch,
                batches,
                range(1, total + 1),
                [total] * total,
            ))
    return [h for r in results for h in r]


def edge_traversal_node(state: MasterState):
    """Single synchronous node: deterministically synthesize trust-boundary
    edges from the explorer notes, then run one structured LLM call per
    homogeneous batch. Emits hypotheses into `vulnerabilities` for the
    reviewer's `cross_boundary` track."""
    raise_if_stopping()
    if not getattr(settings, "edge_traversal_enabled", True):
        logging.info("Edge Traversal disabled via settings.edge_traversal_enabled=False.")
        return {}

    note_map = {
        str(dict_note["node_id"]): dict_note
        for note in state.get("notes", [])
        if (dict_note := as_dict(note)).get("node_id")
    }

    graph_data = get_cached_graph_data(settings.graph)

    edges = build_boundary_edges(graph_data, note_map)
    if not edges:
        logging.info("Edge Traversal: no boundary edges to analyze.")
        return {}

    batches = cluster_boundary_edges(edges)
    logging.info(
        "Edge Traversal: %d boundary edge(s) across %d batch(es) (%s).",
        len(edges),
        len(batches),
        summarize_boundary_edges(edges),
    )

    hypotheses = _run_batches_in_order(batches)

    logging.info(
        "Edge Traversal finished: %d composite hypothesis(es) from %d batch(es).",
        len(hypotheses),
        len(batches),
    )
    return {"vulnerabilities": hypotheses}


def _run_edge_traversal_batch(batch: list[dict], idx: int, total: int) -> list[dict]:
    # Cached on the serialized batch (edges + note profiles + artifacts).
    digest = boundary_batch_fingerprint(batch)
    cache_file = settings.cache_dir / "edge_traversal" / safe_cache_filename(f"{digest}.json")
    cached = cache(cache_file, "read")
    if cached:
        logging.debug("Edge Traversal cache hit for batch %d/%d.", idx, total)
        return (cached.get("hypotheses") or []) if isinstance(cached, dict) else []

    prompt = render_batch_prompt(batch)
    sys_msg = SystemMessage(content=EDGE_TRAVERSAL_AGENT.get("prompt", ""))
    human_msg = HumanMessage(content=prompt)

    structured_llm = fast_llm.with_structured_output(EdgeTraversalOutput, method="json_schema", strict=True)
    output = invoke_structured_capped(
        structured_llm, [sys_msg, human_msg], f"Edge Traversal batch {idx}/{total}"
    )
    if output is None:
        # Uncached on purpose: the next run re-attempts this batch.
        _record_stat("edge_traversal_batches_skipped_output_cap")
        return []
    output = output if isinstance(output, dict) else output.model_dump()

    # Assertions are invariant/pruning evidence, not findings.
    for assertion in output.get("assertions") or []:
        a = as_dict(assertion)
        logging.info(
            "Edge Traversal invariant (batch %d): %s -> %s satisfied=%s — %s",
            idx,
            a.get("source_node", "?"),
            a.get("target_node", "?"),
            a.get("satisfied"),
            str(a.get("reasoning"))[:200],
        )

    hypotheses = [_edge_traversal_finding_to_record(f) for f in (output.get("findings") or [])]
    cache(cache_file, "write", {"hypotheses": hypotheses})
    logging.info(
        "Edge Traversal batch %d/%d: %d hypothesis(es).",
        idx,
        total,
        len(hypotheses),
    )
    return hypotheses


def _edge_traversal_finding_to_record(finding) -> dict:
    f = as_dict(finding)
    nodes = [str(n) for n in (f.get("affected_nodes") or []) if n]
    vuln_type = f.get("vulnerability_type", "cross_boundary_contract_mismatch")
    return {
        "affected_nodes": nodes,
        "cwe_id": f.get("cwe_id", "OTHER_UNCATEGORIZED"),
        "description": str(f.get("gap_details") or ""),
        "status": "hypothesis",
        "vulnerability_type": vuln_type,
        "validation_strategy": f.get("validation_strategy"),
        # Stable per-(pair, type) anchor: distinct boundary edges never collide.
        "vulnerable_component": f"edge:{vuln_type}" + (":" + "->".join(nodes) if nodes else ""),
    }
