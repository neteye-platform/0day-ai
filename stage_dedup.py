"""LLM dedup stage: true-duplicate equivalence classes over review hypotheses.

The embedding pass (dedup.py) only catches surface paraphrases; reworded
duplicates and the same sink re-anchored under different node ids survive it,
each burning a full reviewer + validator pass. This stage asks an LLM, one
structured call per (CWE, packed batch) group, for the equivalence classes
of records describing the SAME underlying defect. dispatch_reviewers applies
them: one canonical survives per cluster (affected_nodes unioned, duplicates
listed under agent_merged_from) and the duplicates are never Sent — the exact
precedent of the embedding merges. Fails open: any failed/capped group
dispatches its records un-deduplicated.
"""

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from dedup import cluster_vulnerabilities
from langchain_core.messages import HumanMessage, SystemMessage

import settings
import llms
from llms import get_llm, invoke_structured_capped
from run_stats import _record_stat, as_dict, as_dicts, get_embedder, raise_if_stopping, take_cached_usage
from schemas import DEDUP_AGENT, DedupAgentOutput, cwes
from state import MasterState
from utils import cache, get_cached_graph_data, safe_cache_filename

# One hypothesis per CVE is canonical fact (the CVE analyzer emits exactly one);
# merging two distinct CVEs would silently drop one from the report.
_PASSTHROUGH_TYPE = "Known Dependency Vulnerability"
_SYSTEMIC_TYPE = "Systemic Vulnerability"
_MAX_NODES_SHOWN = 8


def _node_file_map() -> dict[str, str]:
    graph_data = get_cached_graph_data(settings.graph)
    return {
        str(node["id"]): str(node["source_file"]).replace("\\", "/")
        for node in graph_data.get("nodes", [])
        if node.get("id") and node.get("source_file")
    }


def _dir_key(record: dict, file_map: dict[str, str], depth: int) -> str:
    """First `depth` DIRECTORY segments (file name excluded) of the source file
    of the first affected node with a resolvable file. Files shallower than the
    requested depth keep their full directory key, so a flat directory's files
    stay together instead of fragmenting per file. Synthetic/infra nodes
    (dependency:, infra:) and files at the app root share "_root"."""
    for node_id in record.get("affected_nodes") or []:
        source_file = file_map.get(str(node_id))
        if source_file:
            dirs = [p for p in source_file.strip("/").split("/") if p][:-1]
            if dirs:
                return "/".join(dirs[:depth])
            return "_root"
    return "_root"


_FULL_DEPTH = 10_000  # _dir_key depth meaning "whole directory path"


def _bucket_label(chunk: list[dict], file_map: dict[str, str]) -> str:
    """Chunk label for logs/prompt: the distinct depth-1 source dirs of the
    chunk in order of appearance, max 3 then a +N suffix."""
    dirs = list(dict.fromkeys(_dir_key(r, file_map, 1) for r in chunk))
    return "+".join(dirs[:3]) + (f"+{len(dirs) - 3}more" if len(dirs) > 3 else "")


def embedding_dedup(records: list[dict]) -> list[dict]:
    """The cheap deterministic semantic-merge layer, shared by the two call
    sites that must agree on it: dedup_agent_node collapses near-verbatim
    twins BEFORE spending LLM calls on grouping, and dispatch_reviewers
    re-derives the identical survivors before reviewer dispatch (same function
    over the same channel order with disk-cached embeddings => same result,
    so no merge state travels between them). Fails open: exact-key dedup
    always runs, embedding clustering only when the embeddings server serves
    the configured model."""
    embedder = get_embedder(
        settings.semantic_dedup_enabled,
        "Semantic dedup",
        "exact-key dedup only",
    )
    return cluster_vulnerabilities(
        records,
        settings.semantic_dedup_threshold,
        embedder,
        cross_threshold=settings.dedup_cross_node_similarity,
        anchor_confirmed_threshold=settings.dedup_anchor_confirmed_similarity,
        anchor_min_jaccard=settings.dedup_anchor_min_jaccard,
        max_merged_cluster=settings.dedup_max_merged_cluster,
        disk_cache_dir=settings.cache_dir / "hypothesis_embeddings",
    )


def build_groups(records: list[dict], file_map: dict[str, str]) -> list[dict]:
    """Group review hypotheses by cwe_id; groups up to
    settings.dedup_agent_group_max go out as ONE call (vuln_id-sorted, unlabelled
    bucket). Larger CWE floods are ordered by (source directory, vuln_id) and cut
    into balanced near-cap chunks (settings.dedup_agent_max_group per call):
    directory locality is a SORT KEY, not a partition, so small sibling
    directories co-pack into one call and cross-directory duplicates compete in
    the same prompt. Groups below 2 records have no possible duplicate and are
    skipped. Member order is deterministic => stable cache keys."""
    by_cwe: dict[str, list[dict]] = {}
    for record in records:
        by_cwe.setdefault(str(record.get("cwe_id") or "UNKNOWN"), []).append(record)

    cap = max(2, int(settings.dedup_agent_max_group))
    groups: list[dict] = []
    for cwe in sorted(by_cwe):
        members = sorted(by_cwe[cwe], key=lambda r: str(r.get("vuln_id")))
        if len(members) <= settings.dedup_agent_group_max:
            buckets = [("", members)]
        else:
            ordered = sorted(members, key=lambda r: (_dir_key(r, file_map, _FULL_DEPTH), str(r.get("vuln_id"))))
            per = -(-len(ordered) // -(-len(ordered) // cap))  # ceil(n / ceil(n/cap))
            buckets = [
                (
                    _bucket_label(ordered[i:i + per], file_map),
                    sorted(ordered[i:i + per], key=lambda r: str(r.get("vuln_id"))),
                )
                for i in range(0, len(ordered), per)
            ]
        groups.extend(
            {"cwe_id": cwe, "bucket": bucket, "records": chunk}
            for bucket, chunk in buckets
            if len(chunk) >= 2
        )
    return groups


def _truncated_description(record: dict) -> str:
    description = " ".join(str(record.get("description") or "").split())
    budget = max(120, int(settings.dedup_agent_desc_chars))
    return description[:budget]


def _render_group_prompt(group: dict) -> str:
    cwe = group["cwe_id"]
    label = cwes.get(cwe)
    lines = [f"CWE GROUP: {cwe}" + (f" — {label}" if label else "")]
    if group["bucket"]:
        lines.append(f"SOURCE SUBTREE: {group['bucket']} (records may span these dirs)")
    lines.append(f"RECORDS ({len(group['records'])}):")
    for record in group["records"]:
        nodes = [str(n) for n in record.get("affected_nodes") or []]
        shown = ", ".join(nodes[:_MAX_NODES_SHOWN])
        if len(nodes) > _MAX_NODES_SHOWN:
            shown += f" (+{len(nodes) - _MAX_NODES_SHOWN} more)"
        full = " ".join(str(record.get("description") or "").split())
        desc = _truncated_description(record)
        lines.append(
            f"- vuln_id: {record.get('vuln_id')}\n"
            f"  affected_nodes: {shown or '—'}\n"
            f"  description: {desc}" + ("…" if len(full) > len(desc) else "")
        )
    return "\n".join(lines)


def _group_fingerprint(group: dict) -> str:
    # The hashed triple mirrors EXACTLY what the prompt renders (ids, nodes,
    # truncated description), plus the judging model's identity — the DEDUP
    # agent's effective registry config, not the global default, so switching
    # its model/effort busts stale verdicts instead of silently reusing them.
    judge_cfg = llms.get_config("dedup_agent")
    payload = {
        "model": judge_cfg["model"],
        "reasoning_effort": judge_cfg["reasoning_effort"],
        "cwe_id": group["cwe_id"],
        "bucket": group["bucket"],
        "records": [
            [
                str(record.get("vuln_id")),
                [str(n) for n in record.get("affected_nodes") or []],
                _truncated_description(record),
            ]
            for record in group["records"]
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _run_group(group: dict, idx: int, total: int) -> list[dict]:
    digest = _group_fingerprint(group)
    cache_file = settings.cache_dir / "dedup_agent" / safe_cache_filename(f"{digest}.json")
    try:
        cached = cache(cache_file, "read")
        if cached is not None:
            take_cached_usage("dedup_agent", cached)
            logging.debug("Dedup agent cache hit for group %d/%d.", idx, total)
            return [cl for cl in (cached.get("clusters") or []) if isinstance(cl, dict)]

        known = {str(record.get("vuln_id")) for record in group["records"]}
        structured_llm = get_llm("dedup_agent").with_structured_output(DedupAgentOutput, method="json_schema", strict=True)
        output, usage = invoke_structured_capped(
            structured_llm,
            [
                SystemMessage(content=DEDUP_AGENT.get("prompt", "")),
                HumanMessage(content=_render_group_prompt(group)),
            ],
            f"Dedup agent group {idx}/{total} ({group['cwe_id']} · {group['bucket'] or 'all'})",
            "dedup_agent",
        )
        if output is None:
            # Uncached on purpose: the next run re-attempts this group.
            _record_stat("dedup_agent_groups_skipped_output_cap")
            return []
        output = output if isinstance(output, dict) else output.model_dump()

        clusters: list[dict] = []
        for cluster in output.get("clusters") or []:
            c = as_dict(cluster)
            members = sorted({str(m) for m in (c.get("member_vuln_ids") or []) if str(m) in known})
            if len(members) >= 2:
                clusters.append({
                    "cwe_id": group["cwe_id"],
                    "bucket": group["bucket"],
                    "members": members,
                    "reason": str(c.get("reason") or ""),
                })
    except Exception as e:
        # Fail open: invoke_structured_capped only absorbs output-cap/context
        # errors; anything else (gateway down, parse failure, cache I/O) must
        # not kill the node — this group dispatches un-deduplicated. Uncached:
        # the next run re-attempts it.
        logging.warning(
            "Dedup agent group %d/%d (%s) failed (%s); dispatching its %d record(s) un-deduplicated.",
            idx, total, group["cwe_id"], e, len(group["records"]),
        )
        _record_stat("dedup_agent_groups_skipped_errors")
        return []
    cache(cache_file, "write", {"clusters": clusters, "token_usage": usage})
    logging.info(
        "Dedup agent group %d/%d (%s · %s, %d records): %d duplicate cluster(s).",
        idx, total, group["cwe_id"], group["bucket"] or "all", len(group["records"]), len(clusters),
    )
    return clusters


def dedup_agent_node(state: MasterState) -> dict:
    """Single synchronous node (edge-traversal pattern): collapses the cheap
    embedding layer first, then deterministic CWE + directory packing, one
    cached structured LLM call per group, run concurrently. Writes the
    clusters into MasterState; dispatch_reviewers re-derives the same
    embedding collapse and applies them. Touches no records itself."""
    raise_if_stopping()
    if not getattr(settings, "dedup_agent_enabled", True):
        logging.info("Dedup agent disabled via settings.dedup_agent_enabled=False.")
        return {}

    all_hypotheses = [
        v for v in as_dicts(state.get("vulnerabilities", []))
        if v.get("status") == "hypothesis"
    ]
    # Cheap layer first: the deterministic embedding merge collapses
    # near-verbatim twins before any LLM token is spent, so floods reach
    # build_groups (and the cap) with far fewer copies. dispatch_reviewers
    # later re-derives this exact survivor set with the same call.
    collapsed = embedding_dedup(all_hypotheses)
    dropped = len(all_hypotheses) - len(collapsed)
    hypotheses = [
        v for v in collapsed
        if v.get("vulnerability_type") != _PASSTHROUGH_TYPE and v.get("vuln_id")
    ]
    groups = build_groups(hypotheses, _node_file_map())
    if not groups:
        logging.info("Dedup agent: nothing to cluster (%d hypothesis record(s)).", len(hypotheses))
        return {"hypothesis_clusters": []}

    total = len(groups)
    logging.info(
        "Dedup agent: %d record(s), %d collapsed by embedding pass -> %d group(s) (group > %d records packs by source dir).",
        len(hypotheses), dropped, total, settings.dedup_agent_group_max,
    )
    if total <= 1:
        results = [_run_group(group, 1, total) for group in groups]
    else:
        with ThreadPoolExecutor(
            max_workers=min(total, max(1, int(settings.dedup_agent_parallel))),
            thread_name_prefix="dedup-agent",
        ) as pool:
            results = list(pool.map(_run_group, groups, range(1, total + 1), [total] * total))

    clusters = [cluster for result in results for cluster in result]
    _record_stat("dedup_agent_groups", total)
    logging.info("Dedup agent finished: %d duplicate cluster(s) across %d group(s).", len(clusters), total)
    return {"hypothesis_clusters": clusters}


def _canonical_rank(by_id: dict[str, dict], vuln_id: str) -> tuple:
    record = by_id[vuln_id]
    return (
        # A systemic record's prompt already enumerates a defense per affected
        # node, so it absorbs code-level twins without weakening any gate.
        0 if record.get("vulnerability_type") == _SYSTEMIC_TYPE else 1,
        -len(record.get("affected_nodes") or []),
        vuln_id,
    )


def _cluster_reason(clusters: list[dict], bucket: list[str]) -> str:
    # The proposed cluster that actually covers (most of) this final bucket —
    # after union-find closure / CVE splits the canonical alone is not a
    # reliable pointer to the provenance text.
    best = ""
    best_overlap = 0
    members = set(bucket)
    for cluster in clusters:
        overlap = len(members & {str(m) for m in cluster.get("members") or []})
        if overlap > best_overlap:
            best, best_overlap = str(cluster.get("reason") or ""), overlap
    return best


def _split_by_cve(members: list[str], by_id: dict[str, dict]) -> list[list[str]]:
    """Hard guardrail the model prompt also forbids: records naming different
    CVEs are distinct findings and must never share a cluster."""
    buckets: dict[Optional[str], list[str]] = {}
    for vuln_id in members:
        buckets.setdefault(by_id[vuln_id].get("source_cve"), []).append(vuln_id)
    return list(buckets.values())


def apply_agent_clusters(
    hypotheses: list[dict], clusters: Optional[list[dict]]
) -> tuple[list[dict], int]:
    """Materialize the dedup agent's equivalence classes on the (already
    embedding-clustered) dispatch list. Overlapping proposed clusters are
    transitively closed (union-find); members already absorbed by the
    embedding pass are ignored; clusters spanning distinct source_cves split.
    The canonical record (systemic preferred, then most affected nodes, then
    lowest vuln_id) gets the ordered affected_nodes union plus an
    agent_merged_from provenance list; its duplicates are removed from the
    returned list. Returns (records_to_dispatch, merged_away_count)."""
    if not clusters:
        return hypotheses, 0

    by_id = {str(h.get("vuln_id")): h for h in hypotheses}
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for cluster in clusters:
        members = sorted({str(m) for m in (cluster or {}).get("members") or [] if str(m) in by_id})
        for other in members[1:]:
            root_a, root_b = find(members[0]), find(other)
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)

    connected: dict[str, list[str]] = {}
    for vuln_id in parent:
        connected.setdefault(find(vuln_id), []).append(vuln_id)

    drop: set[str] = set()
    updates: dict[str, dict] = {}
    for members in connected.values():
        for bucket in _split_by_cve(sorted(members), by_id):
            if len(bucket) < 2:
                continue
            canonical = min(bucket, key=lambda vid: _canonical_rank(by_id, vid))
            merged_nodes = list(by_id[canonical].get("affected_nodes") or [])
            seen = {str(n) for n in merged_nodes}
            for vuln_id in bucket:
                if vuln_id == canonical:
                    continue
                for node in by_id[vuln_id].get("affected_nodes") or []:
                    if str(node) not in seen:
                        seen.add(str(node))
                        merged_nodes.append(node)
            updated = dict(by_id[canonical])
            updated["affected_nodes"] = merged_nodes
            updated["agent_merged_from"] = [v for v in bucket if v != canonical]
            updates[canonical] = updated
            drop.update(updated["agent_merged_from"])
            logging.info(
                "Dedup agent merge: %s absorbs %s (%s)",
                canonical, ", ".join(updated["agent_merged_from"]),
                _cluster_reason(clusters, bucket)[:160],
            )

    if not drop:
        return hypotheses, 0
    kept = [
        updates.get(str(h.get("vuln_id")), h)
        for h in hypotheses
        if str(h.get("vuln_id")) not in drop
    ]
    return kept, len(drop)
