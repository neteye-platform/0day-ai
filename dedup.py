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
import logging
import math
import re
from collections import defaultdict

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

    out = []
    for cl in clusters:
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
