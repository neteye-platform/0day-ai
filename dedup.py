"""Embedding-based semantic dedup of vulnerability hypotheses and demands.

Before ``dispatch_reviewers`` fans a hypothesis out to a reviewer subgraph,
records describing the SAME vulnerability differently are merged so one
reviewer/validator chain adjudicates the pattern once; ``deduplicate_demands``
does the same for demands before the contract verifier.

Hard constraints (thresholds tuned offline against the GLPI run's cached
embeddings; see settings.py):

  * Clustering only within a (vulnerability_type, cwe_id) group.
  * Dependency-origin records (``source_cve``) are never merged.
  * Same-node-set records merge at ``threshold``; cross-node records pass the
    two-tier gate in ``_pair_mergeable_fast`` (high-confidence cosine, or
    cosine plus descriptive-component jaccard; degenerate anchors never carry
    a merge alone), with cross-node cluster growth capped at
    ``max_merged_cluster``.
  * Fixed-representative (non-chaining) clustering prevents A~B~C over-merges.
  * Merged records keep the seed's vuln_id, union affected_nodes, and fold in
    child descriptions, so no information is lost.
  * Embeddings (local Ollama) are batched and disk-cached per (model, text),
    so re-runs are free.
  * Fails open: any embedding error leaves only the exact-identity pre-merge.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path

import requests

try:  # optional: large (vtype, cwe) groups need O(n²) similarity lookups.
    import numpy as _np
except ImportError:  # pragma: no cover - pure-Python fallback stays correct.
    _np = None

log = logging.getLogger("dedup")

# Shared grouping-fallback identities, so buckets can't drift between paths.
_DEFAULT_VTYPE = "Code Defect"
_DEFAULT_CWE = "OTHER_UNCATEGORIZED"


# ---------------------------------------------------------------------------
# Embeddings client
# ---------------------------------------------------------------------------
class Embeddings:
    """Thin wrapper around a local Ollama embeddings endpoint with per-text caching."""

    def __init__(self, base_url: str, model: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._cache: dict[str, list[float]] = {}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def available(self, timeout: int = 5) -> bool:
        """Cheap liveness check: does the local Ollama serve our embed model?
        Used to fail open before any per-group embedding call if it does not."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
            if not resp.ok:
                return False
            names = [m.get("name", "") for m in (resp.json().get("models") or [])]
            return any(n.split(":", 1)[0] == self.model for n in names)
        except Exception:
            return False

    def embed(self, text: str) -> list[float]:
        key = self._key(text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        resp = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        vec = resp.json().get("embedding")
        if not vec:
            raise ValueError(f"Ollama returned no embedding for model {self.model!r}: {resp.text[:200]}")
        self._cache[key] = vec
        return vec

    # Chunk /api/embed requests: one monolithic multi-thousand request can
    # stall the server (and the pipeline) past any useful timeout.
    BATCH_CHUNK = 128

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts, using Ollama's batch endpoint when available.

        Tries ``/api/embed`` in chunks and falls back to sequential single-text
        ``embed()`` calls on older servers. The per-text cache is consulted and
        filled either way, so repeated runs only pay for unseen texts.
        """
        results: list[list[float] | None] = [None] * len(texts)
        missing: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            cached = self._cache.get(self._key(t))
            if cached is not None:
                results[i] = cached
            else:
                missing.append((i, t))
        try:
            for start in range(0, len(missing), self.BATCH_CHUNK):
                chunk = missing[start:start + self.BATCH_CHUNK]
                resp = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": [t for _, t in chunk]},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                vecs = resp.json().get("embeddings") or []
                if len(vecs) != len(chunk):
                    raise ValueError(
                        f"batch size mismatch: {len(vecs)} embeddings for {len(chunk)} texts"
                    )
                for (i, t), vec in zip(chunk, vecs):
                    self._cache[self._key(t)] = vec
                    results[i] = vec
        except Exception:
            # Older Ollama without /api/embed (or transient error): one-by-one
            # fallback, raising on real unavailability so callers fail open.
            for i, t in missing:
                if results[i] is None:
                    results[i] = self.embed(t)
        return results


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _normalize_text(vulnerability_type, cwe_id, description) -> str:
    """Embedding input: rich text (not the bare label) carries the signal."""
    parts = [vulnerability_type or "", cwe_id or "", (description or "").lower()]
    return " ".join(p for p in parts if p).strip()


def _hypothesis_text(vulnerability_type, cwe_id, vulnerable_component, description) -> str:
    """Cross-node gate text: the component anchor leads, then the description."""
    comp = (vulnerable_component or "").strip()
    desc = (description or "").strip()
    body = f"{comp}. {desc}" if comp else desc
    return _normalize_text(vulnerability_type, cwe_id, body.lower())


def component_tokens(component) -> frozenset[str]:
    """Lowercased alphanumeric token set of a vulnerable_component string."""
    return frozenset(re.sub(r"[^a-z0-9$_%.'\"\[\] ]+", " ", _norm(component)).split())


# Structural punctuation: what a real selector/call-site carries but a bare
# word label never does. Shared convention with
# stage_validator._validation_group_key.
_STRUCTURAL_RE = re.compile(r"[^\w$\s]")


def is_degenerate_anchor(component) -> bool:
    """True when a component string cannot serve as a cross-node identity.

    Bare labels ('$str', 'uid', a node id) recur across hundreds of unrelated
    callers, so identity between them proves nothing: only structural
    punctuation (e.g. ``.prepare()``, ``request_params['filter']``) or a 3+
    word descriptive phrase counts as identifying. Mirrors the convention of
    stage_validator._validation_group_key.
    """
    raw = _norm(component)
    if not raw.strip():
        return True
    if _STRUCTURAL_RE.search(raw):
        return False
    if len(component_tokens(component)) >= 3:
        return False
    return True


def _token_jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _exact_key(record: dict) -> tuple:
    """Deterministic identity for the cheap exact pre-merge: same type, nodes,
    CWE, and semantic anchor. Catches duplicates that differ only in case or
    whitespace without spending any embedding call. ``vulnerability_type`` is
    part of the identity so the exact path honors the same grouping invariant
    as the embedding path (a systemic record can never merge with a code-level
    one and silently change the reviewer track)."""
    nodes = tuple(sorted({n for n in (record.get("affected_nodes") or []) if n}))
    vtype = record.get("vulnerability_type") or _DEFAULT_VTYPE
    cwe = record.get("cwe_id") or ""
    anchor = record.get("vulnerable_component") or record.get("description") or ""
    anchor = re.sub(r"\s+", " ", anchor.strip().lower())
    return (vtype, nodes, cwe, anchor)


def _sort_key(r: dict):
    """Deterministic processing order so the greedy clustering is reproducible."""
    return (
        r.get("vulnerability_type") or _DEFAULT_VTYPE,
        r.get("cwe_id") or _DEFAULT_CWE,
        tuple(sorted(n for n in r.get("affected_nodes") or [])),
        r.get("vuln_id") or "",
        r.get("description") or "",
    )


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def _merge_cluster(members: list[dict]) -> dict:
    """Merge a cluster into one record: the seed supplies identity/fields,
    members union in affected_nodes and description context
    (merge_vulnerabilities same-stage style).
    """
    base = members[0]
    merged = dict(base)
    affected: list[str] = []
    seen: set[str] = set()
    for member in members:
        for node in member.get("affected_nodes") or []:
            if node and node not in seen:
                seen.add(node)
                affected.append(node)
    merged["affected_nodes"] = affected

    desc = base.get("description") or ""
    for member in members[1:]:
        other = member.get("description") or ""
        if other and other not in desc:
            desc = f"{desc}\n\nAdditional context: {other}"
    merged["description"] = desc
    return merged


def _greedy_cluster_indices(vectors: list[list[float]], threshold: float) -> list[list[int]]:
    """Greedy fixed-representative clustering over precomputed vectors.

    Returns index clusters in input order; the first member of each cluster is
    its representative. Comparing each item against cluster representatives
    only (not all members) prevents A~B~C transitive over-merges.
    """
    clusters: list[list[int]] = []
    reps: list[list[float]] = []
    for i, vec in enumerate(vectors):
        best = -1.0
        best_c = -1
        for c, rep in enumerate(reps):
            sim = _cosine(vec, rep)
            if sim > best:
                best, best_c = sim, c
        if best_c >= 0 and best >= threshold:
            clusters[best_c].append(i)
        else:
            clusters.append([i])
            reps.append(vec)
    return clusters


# In the 0.93..0.95 band two descriptive components also need this jaccard
# floor (barely-overlapping anchors = related-but-distinct defects); at
# HIGH_CONFIDENCE_COSINE the embedding decides alone.
CONFIDENT_BAND_MIN_JACCARD = 0.4
HIGH_CONFIDENCE_COSINE = 0.95


def _pair_mergeable_fast(
    nodes_a: tuple, nodes_b: tuple, sim: float,
    ta: frozenset[str], tb: frozenset[str],
    degen_a: bool, degen_b: bool,
    threshold: float, cross_threshold: float,
    anchor_confirmed_threshold: float, anchor_min_jaccard: float,
) -> bool:
    """Two-tier cross-node merge gate over precomputed per-record features."""
    if nodes_a == nodes_b:
        return sim >= threshold

    degen = degen_a or degen_b
    if sim >= cross_threshold:
        return (degen or sim >= HIGH_CONFIDENCE_COSINE
                or _token_jaccard(ta, tb) >= CONFIDENT_BAND_MIN_JACCARD)
    if sim >= anchor_confirmed_threshold:
        return (not degen) and _token_jaccard(ta, tb) >= anchor_min_jaccard
    return False


def _sim_matrix(vectors: list[list[float]]):
    """Pairwise cosine lookup for a group: numpy-boosted when available."""
    if _np is not None:
        M = _np.asarray(vectors, dtype=_np.float64)
        norms = _np.linalg.norm(M, axis=1, keepdims=True)
        M = M / _np.maximum(norms, 1e-9)
        S = M @ M.T

        def lookup(i: int, j: int) -> float:
            return float(S[i, j])
        return lookup

    def lookup(i: int, j: int) -> float:
        return _cosine(vectors[i], vectors[j])
    return lookup


def _cluster_by_similarity(
    records: list[dict], vectors: list[list[float]],
    threshold: float, cross_threshold: float,
    anchor_confirmed_threshold: float, anchor_min_jaccard: float,
    max_merged_cluster: int,
) -> list[list[int]]:
    """Greedy fixed-representative clustering with the two-tier merge gate.

    Candidates join their best-matching passing representative, never crossing
    the (vulnerability_type, cwe_id) boundary. Cross-node growth is capped at
    ``max_merged_cluster`` members (same-node-set members are exempt) so a
    hub-parameter flood can't coalesce into one oversized review.
    """
    nodesets = [tuple(sorted(n for n in r.get("affected_nodes") or [])) for r in records]
    comps = [r.get("vulnerable_component") for r in records]
    tokens = [component_tokens(c) for c in comps]
    degenerate = [is_degenerate_anchor(c) for c in comps]
    sim = _sim_matrix(vectors)

    seeds: list[int] = []
    clusters: list[list[int]] = []
    cross_members: dict[int, int] = {}  # cluster -> members with a different nodeset
    for i in range(len(records)):
        best, best_c = -1.0, -1
        for c, seed in enumerate(seeds):
            s = sim(i, seed)
            if s <= best:
                continue
            if not _pair_mergeable_fast(
                nodesets[i], nodesets[seed], s,
                tokens[i], tokens[seed],
                degenerate[i], degenerate[seed],
                threshold, cross_threshold,
                anchor_confirmed_threshold, anchor_min_jaccard,
            ):
                continue
            cross = nodesets[i] != nodesets[seed]
            if cross and max_merged_cluster and cross_members.get(c, 0) + 1 > max_merged_cluster:
                continue
            best, best_c = s, c
        if best_c >= 0:
            clusters[best_c].append(i)
            if nodesets[i] != nodesets[seeds[best_c]]:
                cross_members[best_c] = cross_members.get(best_c, 0) + 1
        else:
            seeds.append(i)
            clusters.append([i])
    return clusters


def cluster_vulnerabilities(
    hypotheses: list[dict], threshold: float = 0.80,
    embedder: Embeddings | None = None, *,
    cross_threshold: float = 0.93,
    anchor_confirmed_threshold: float = 0.85,
    anchor_min_jaccard: float = 0.6,
    max_merged_cluster: int = 25,
    disk_cache_dir=None,
) -> list[dict]:
    """Merge duplicate hypotheses and return the (possibly reduced) dispatch list.

    ``hypotheses``: list of hypothesis dicts (status == "hypothesis") as they
    arrive at ``dispatch_reviewers``. Order is preserved; records are never
    dropped, only merged. Dependency-origin records (source_cve) pass through
    untouched. Same-node-set duplicates cluster at ``threshold``; cross-node
    duplicates (the same defect reported from different caller nodes) go
    through the two-tier gate in ``_pair_mergeable_fast``. Embeddings are batched
    and served from the per-(model, text) ``disk_cache_dir`` when given.
    Fails open: on any embedding error only the exact-identity pre-merge runs.
    """
    if not hypotheses:
        return hypotheses

    # Dependency records pass through untouched; the rest get the cheap exact
    # pre-merge: identical (type, nodes, cwe, anchor) but different
    # casing/phrasing collapse without spending an embedding call, through the
    # same _merge_cluster the embedding path uses.
    outcome: list[dict] = []
    exact: dict[tuple, list[dict]] = defaultdict(list)
    for h in hypotheses:
        if h.get("source_cve"):
            outcome.append(h)
        else:
            exact[_exact_key(h)].append(h)
    pool = sorted(
        (_merge_cluster(members) if len(members) > 1 else members[0]
         for members in exact.values()),
        key=_sort_key,
    )

    if embedder is None or len(pool) <= 1:
        outcome.extend(pool)
    else:
        try:
            texts = [
                _hypothesis_text(
                    h.get("vulnerability_type"), h.get("cwe_id"),
                    h.get("vulnerable_component"), h.get("description"),
                )
                for h in pool
            ]
            vectors = _embed_with_disk_cache(embedder, texts, disk_cache_dir)
            if len(vectors) != len(pool) or any(v is None for v in vectors):
                raise ValueError("embedding count mismatch in hypothesis dedup")
        except Exception as e:
            log.warning(
                "Semantic dedup: embedding failed (%s); dispatching %d "
                "hypotheses with exact-identity merging only.", e, len(pool),
            )
            outcome.extend(pool)
        else:
            groups: dict[tuple, list[int]] = {}
            for i, h in enumerate(pool):
                key = (
                    h.get("vulnerability_type") or _DEFAULT_VTYPE,
                    h.get("cwe_id") or _DEFAULT_CWE,
                )
                groups.setdefault(key, []).append(i)
            for idxs in groups.values():
                sub = [pool[i] for i in idxs]
                sub_vecs = [vectors[i] for i in idxs]
                clusters = _cluster_by_similarity(
                    sub, sub_vecs, threshold, cross_threshold,
                    anchor_confirmed_threshold, anchor_min_jaccard,
                    max_merged_cluster,
                )
                for cl in clusters:
                    outcome.append(_merge_cluster([sub[i] for i in cl]) if len(cl) > 1 else sub[cl[0]])

    log.info(
        "Semantic dedup: %d hypotheses -> %d unique dispatch records (%d merged).",
        len(hypotheses), len(outcome), len(hypotheses) - len(outcome),
    )
    return outcome


# ---------------------------------------------------------------------------
# Demand dedup (contract-verifier input)
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _canonical_demand_key(d: dict) -> tuple:
    """Total order over demands, independent of channel arrival order. The final
    component breaks full-content ties, so sort output is run-invariant."""
    return (
        str(d.get("type") or ""),
        str(d.get("source") or ""),
        str(d.get("parameter_name") or ""),
        _norm(str(d.get("description") or "")),
        json.dumps(d, sort_keys=True),
    )


def _embed_with_disk_cache(
    embedder: Embeddings, texts: list[str], disk_cache_dir
) -> list[list[float]]:
    """``embed_batch`` with an optional per-text on-disk cache keyed by
    (model, text) hash, so re-runs with unchanged notes never re-pay the
    embedding pass. Cache read/write errors are treated as misses."""
    if not disk_cache_dir:
        return embedder.embed_batch(texts)
    cache_dir = Path(disk_cache_dir)
    results: list[list[float] | None] = [None] * len(texts)
    missing: list[tuple[int, str, Path]] = []
    model = embedder.model
    for i, t in enumerate(texts):
        f = cache_dir / f"{hashlib.sha256(f'{model}:{t}'.encode('utf-8')).hexdigest()}.json"
        try:
            results[i] = json.loads(f.read_text())["embedding"]
            continue
        except Exception:
            missing.append((i, t, f))
    if missing:
        vecs = embedder.embed_batch([t for _, t, _ in missing])
        for (i, t, f), vec in zip(missing, vecs):
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps({"embedding": vec}))
            except Exception:
                pass
            results[i] = vec
    return results


def deduplicate_demands(
    grouped_demands: dict,
    embedder: Embeddings | None,
    threshold: float,
    disk_cache_dir=None,
) -> dict:
    """Merge near-duplicate demands per target node before contract verification.

    Every caller of a hub callee restates the same contract in its own words,
    and the verifier emits one evaluation per demand, so paraphrase floods
    multiply LLM spend without adding checkable information.

    Merge rules per target node:

      * ``cve_assumption`` demands are never merged (distinct CVE contracts).
      * Upstream (callee-parameter) demands merge only on exact normalized
        identity within the same ``(source, parameter_name)`` pair — the
        embedder is never consulted for them.
      * All other explorer demands merge on exact normalized identity first,
        then embedding similarity at ``threshold``; the cluster seed (first
        demand in canonical order) is kept as-is — nothing is concatenated, as
        the paraphrases carry no extra checkable information.
      * Fails open: embedding errors keep the exact-merged demands for that
        target; with ``embedder=None`` only exact merging runs.

    Each target's demands are canonically sorted on entry: the arrival order on
    the demands channel follows parallel explorer completion and varies across
    runs, which would make both the clustering seeds and the downstream
    order-sensitive contract-verifier cache keys (md5 of the ordered list, and
    positional batch slices) flip between runs — recomputing the hub nodes'
    biggest prompts every single run.
    """
    total_in = sum(len(v) for v in grouped_demands.values())
    for target, demands in grouped_demands.items():
        demands = grouped_demands[target] = sorted(demands, key=_canonical_demand_key)
        if len(demands) <= 1:
            continue

        by_type: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(demands):
            by_type[d.get("type") or ""].append(i)

        keep: list[int] = []
        for dtype, idxs in by_type.items():
            if dtype == "cve_assumption" or len(idxs) == 1:
                keep.extend(idxs)
                continue

            if dtype == "explorer_upstream_assumption":
                exact: dict[tuple, int] = {}
                for i in idxs:
                    d = demands[i]
                    key = (d.get("source"), d.get("parameter_name"), _norm(d.get("description")))
                    if key not in exact:
                        exact[key] = i
                keep.extend(exact.values())
                continue

            # Downstream caller assumptions (and any other explorer type):
            # exact-normalized pre-merge, then embedding clustering; each
            # cluster keeps its seed demand, in canonical order.
            exact_desc: dict[str, int] = {}
            for i in idxs:
                key = _norm(demands[i].get("description"))
                if key not in exact_desc:
                    exact_desc[key] = i
            seeds = list(exact_desc.values())
            if len(seeds) == 1 or embedder is None:
                keep.extend(seeds)
                continue
            try:
                texts = [_norm(demands[i].get("description")) for i in seeds]
                vectors = _embed_with_disk_cache(embedder, texts, disk_cache_dir)
                clusters = _greedy_cluster_indices(vectors, threshold)
            except Exception as e:
                log.warning(
                    "Demand dedup: embedding failed for target %s; keeping %d "
                    "exact-unique demands (%s)",
                    target, len(seeds), e,
                )
                keep.extend(seeds)
                continue
            keep.extend(seeds[cl[0]] for cl in clusters)

        keep.sort()
        grouped_demands[target] = [demands[i] for i in keep]

    total_out = sum(len(v) for v in grouped_demands.values())
    if total_out != total_in:
        log.info(
            "Demand dedup: %d -> %d demands before contract verification (%d merged).",
            total_in, total_out, total_in - total_out,
        )
    return grouped_demands
