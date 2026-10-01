"""Embedding-based semantic dedup of vulnerability hypotheses.

Before ``dispatch_reviewers`` fans a hypothesis out to a full
reviewer subgraph, hypotheses that are the SAME real vulnerability described
differently by different agents (e.g. two explorer roles calling the same
plaintext-logging flaw "plaintext password logging" vs "plaintext credential
logging") are merged into one record, so a single reviewer + validator chain
adjudicates the pattern instead of N duplicate chains.

Design constraints (threshold 0.80 was validated against real cached
hypotheses):

  * Clustering happens ONLY within a (vulnerability_type, cwe_id) group — the
    same label can never cross a CWE boundary.
  * Dependency-origin records (those carrying ``source_cve``) are NEVER merged:
    each is a distinct, already-canonical known CVE (deduplicate_cves collapses
    to one record per CVE id).
  * Cross-node merging is allowed only for "Systemic Vulnerability" records
    (the same insecure pattern repeated across nodes). Non-systemic
    (code-level) records are only merged within the SAME affected-node set, so
    distinct localized defects in different functions are never collapsed.
  * Fixed-representative (non-chaining) clustering prevents A~B~C transitive
    over-merges.
  * The merged record keeps its seed's vuln_id, unions all affected_nodes, and
    concatenates the child descriptions, so no information is lost and the
    reviewer/validator see every affected location.
  * Embeddings are served by a local Ollama endpoint and cached per text hash,
    so re-runs are ~free and the whole feature costs $0 of LLM spend.
  * Fails open: any embedding error or an unreachable endpoint returns the
    hypotheses unchanged (plain dedup by exact identity still happens).
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

log = logging.getLogger("dedup")


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

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts, using Ollama's batch endpoint when available.

        Tries ``/api/embed`` (one request for the whole list) and falls back to
        sequential single-text ``embed()`` calls on older servers. The
        per-text cache is consulted and filled either way, so repeated runs
        only pay for unseen texts.
        """
        results: list[list[float] | None] = [None] * len(texts)
        missing: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            cached = self._cache.get(self._key(t))
            if cached is not None:
                results[i] = cached
            else:
                missing.append((i, t))
        if missing:
            try:
                resp = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": [t for _, t in missing]},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                vecs = resp.json().get("embeddings") or []
                if len(vecs) != len(missing):
                    raise ValueError(
                        f"batch size mismatch: {len(vecs)} embeddings for {len(missing)} texts"
                    )
                for (i, t), vec in zip(missing, vecs):
                    self._cache[self._key(t)] = vec
                    results[i] = vec
            except Exception:
                # Older Ollama without /api/embed (or a transient batch error):
                # fall back to proven one-by-one requests, which raise on real
                # unavailability so callers fail open.
                for i, t in missing:
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
    """Embedding input: rich text (not the bare label) carries the semantic
    signal; matches the text the validated threshold was tuned on."""
    parts = [vulnerability_type or "", cwe_id or "", (description or "").lower()]
    return " ".join(p for p in parts if p).strip()


def _exact_key(record: dict) -> tuple:
    """Deterministic identity for the cheap exact pre-merge: same type, nodes,
    CWE, and semantic anchor. Catches duplicates that differ only in case or
    whitespace without spending any embedding call. ``vulnerability_type`` is
    part of the identity so the exact path honors the same grouping invariant
    as the embedding path (a systemic record can never merge with a code-level
    one and silently change the reviewer track)."""
    nodes = tuple(sorted({n for n in (record.get("affected_nodes") or []) if n}))
    vtype = record.get("vulnerability_type") or "Code Defect"
    cwe = record.get("cwe_id") or ""
    anchor = record.get("vulnerable_component") or record.get("description") or ""
    anchor = re.sub(r"\s+", " ", anchor.strip().lower())
    return (vtype, nodes, cwe, anchor)


def _sort_key(r: dict):
    """Deterministic processing order so the greedy clustering is reproducible."""
    return (
        r.get("vulnerability_type") or "Code Defect",
        r.get("cwe_id") or "OTHER_UNCATEGORIZED",
        tuple(sorted(n for n in r.get("affected_nodes") or [])),
        r.get("vuln_id") or "",
        r.get("description") or "",
    )


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def _merge_cluster(members: list[dict]) -> dict:
    """Merge a cluster of hypothesis dicts into one record.

    The seed (cluster representative) supplies the identity/fields; remaining
    members add their affected_nodes and description context. Mirrors
    merge_vulnerabilities' same-stage text combine.
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


def _cluster_group(records: list[dict], embedder: Embeddings, threshold: float) -> list[dict]:
    """Greedy fixed-representative clustering of one (type, cwe[, nodes]) group."""
    if len(records) <= 1:
        return records
    try:
        vectors = [embedder.embed(_normalize_text(r.get("vulnerability_type"), r.get("cwe_id"), r.get("description")))
                   for r in records]
    except Exception:
        log.warning("Semantic dedup: embedding failed for a group; dispatching %d hypotheses unchanged.", len(records))
        return records

    out = []
    for cl in _greedy_cluster_indices(vectors, threshold):
        out.append(_merge_cluster([records[i] for i in cl]) if len(cl) > 1 else records[cl[0]])
    return out


def cluster_vulnerabilities(hypotheses: list[dict], threshold: float = 0.80,
                            embedder: Embeddings | None = None) -> list[dict]:
    """Merge duplicate hypotheses and return the (possibly reduced) dispatch list.

    ``hypotheses``: list of hypothesis dicts (status == "hypothesis") as they
    arrive at ``dispatch_reviewers``. Order is preserved; records are never
    dropped, only merged. Dependency-origin records (source_cve) pass through
    untouched.
    """
    if not hypotheses:
        return hypotheses

    outcome = [h for h in hypotheses if h.get("source_cve")]

    # Cheap exact pre-merge: identical (type, nodes, cwe, anchor) but different
    # casing/phrasing collapse without spending any embedding call. Each group
    # goes through the same _merge_cluster the embedding path uses, so both
    # paths share one merge semantics.
    exact: dict[tuple, list[dict]] = defaultdict(list)
    for h in hypotheses:
        if not h.get("source_cve"):
            exact[_exact_key(h)].append(h)
    pool = sorted(
        (_merge_cluster(members) if len(members) > 1 else members[0]
         for members in exact.values()),
        key=_sort_key,
    )

    if embedder is None:
        outcome.extend(pool)
    else:
        groups: dict[tuple, list[dict]] = {}
        for h in pool:
            vtype = h.get("vulnerability_type") or "Code Defect"
            cwe = h.get("cwe_id") or "OTHER_UNCATEGORIZED"
            if vtype == "Systemic Vulnerability":
                key = (vtype, cwe)
            else:
                # code-level: only merge duplicates within the same node set.
                key = (vtype, cwe, tuple(sorted(n for n in h.get("affected_nodes") or [])))
            groups.setdefault(key, []).append(h)

        for members in groups.values():
            outcome.extend(_cluster_group(members, embedder, threshold))

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
    for i, t in enumerate(texts):
        model = getattr(embedder, "model", "")
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
    and the verifier emits one evaluation per demand — so paraphrase floods
    multiply LLM spend without adding checkable information (one GLPI renderer
    gathered 225 near-identical downstream assumptions from 225 callers;
    clustering at the hypothesis-dedup threshold reduces them to 68).

    Merging semantics per target node:

      * ``cve_assumption`` demands are NEVER merged: each is a distinct CVE
        contract whose ``source_cve`` tags downstream records.
      * Upstream (callee-parameter) demands merge only on exact normalized
        identity within the same ``(source, parameter_name)`` pair — merging
        paraphrases across parameters could drop a distinct argument check,
        so the embedder is never consulted for them.
      * All other explorer demands (downstream caller assumptions) merge on
        exact normalized identity first, then embedding similarity at
        ``threshold``. The cluster seed (first demand in order) is kept
        as-is: unlike hypothesis merges nothing is concatenated, since the
        paraphrase variants carry no extra checkable information and the
        demands feed a size-sensitive prompt.
      * Fails open: embedding errors keep the exact-merged demands for that
        target; with ``embedder=None`` only exact merging runs.
    """
    total_in = sum(len(v) for v in grouped_demands.values())
    for target, demands in grouped_demands.items():
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
            # exact-normalized pre-merge, then embedding clustering on the
            # survivors. The kept demand is each cluster's seed, in original
            # order.
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
