"""Embedding-based semantic dedup of vulnerability hypotheses and demands.

Before ``dispatch_reviewers`` fans a hypothesis out to a reviewer subgraph,
records describing the SAME vulnerability differently are merged so one
reviewer/validator chain adjudicates the pattern once; ``deduplicate_demands``
does the same for demands before the contract verifier.

Hard constraints (thresholds tuned offline against embeddings from
representative real-world scans; see settings.py):

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
  * Embeddings (local Ollama) are requested in configurable chunks; a failed
    chunk is halved and retried (down to one text) instead of aborting the
    pass, requests are serialized against the single llama.cpp server, the
    model is pre-warmed with a dedicated cold-load timeout only when uncached
    texts actually exist, and keep_alive pins it resident across the dedup
    passes. Every success is disk-cached per (model, text) incrementally, so
    re-runs — and interrupted ones — only pay for unseen texts.
  * Fails open: a hung server trips the stall watchdog (abort after N seconds
    with zero progress, no total-size cap; 0 disables it) and leaves only the
    exact-identity pre-merge; individual permanently failing texts stay
    unembedded and degrade only their own cluster/group.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    """Local Ollama embeddings client with per-text caching.

    Request policy tuned for a single (possibly cold) llama.cpp server and
    10k+-text passes: one pre-warm call with its own long timeout absorbs the
    model load that would otherwise silently eat the first chunk's request
    timeout; the rest proceeds in ``batch_size`` chunks against ``/api/embed``
    (serial by default — concurrent chunks only multiply the queue and the
    resident-model copies on the one server), pinned in memory via
    ``keep_alive``. A failing chunk (timeout / HTTP error / batch-count
    mismatch, e.g. a too-large batch hitting context pressure) is HALVED and
    retried down to a single text instead of aborting the whole pass; only a
    single text failing after one retry is dropped (returned as None), which
    degrades its own cluster/group, never the stage. Progress (not total
    time) is what the pass is bounded by: ``stall_budget_sec`` is the maximum
    tolerated period WITHOUT a single embedding landing (0 = unbounded), so a
    hung server trips after ~3 request timeouts while a slow-but-progressing
    pass of any size runs to completion.
    """

    PROGRESS_EVERY = 200

    def __init__(self, base_url: str, model: str, timeout: int = 60, *,
                 batch_size: int = 200, parallel_chunks: int = 1,
                 prewarm_timeout: int = 600, keep_alive: str = "6h",
                 stall_budget_sec: float = 540.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.batch_size = max(1, batch_size)
        self.parallel_chunks = max(1, parallel_chunks)
        self.prewarm_timeout = prewarm_timeout
        self.keep_alive = keep_alive
        self.stall_budget_sec = stall_budget_sec
        self._cache: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def available(self, timeout: int = 5) -> bool:
        """Cheap liveness check: does the local Ollama serve our embed model?
        Used to fail open before any per-group embedding call if it does not.
        N.B. this only hits /api/tags — it does NOT load the model, so a
        passing check does NOT mean the first embed is cold-load free."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
            if not resp.ok:
                return False
            names = [m.get("name", "") for m in (resp.json().get("models") or [])]
            return any(n.split(":", 1)[0] == self.model for n in names)
        except Exception:
            return False

    def prewarm(self) -> bool:
        """Force the embedding model into Ollama's memory with ONE trivial
        /api/embed call carrying its own long timeout, so the cold load never
        counts against a chunk request's ``timeout`` (that's what made every
        request of a healthy-but-cold server read-time out). Also sets
        ``keep_alive`` so the model stays resident for the whole pass (and
        the later dedup stages). Returns False (warns) on any error; callers
        continue either way — the stall watchdog still guards the pass."""
        started = time.monotonic()
        try:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": ["warm up"],
                      "keep_alive": self.keep_alive, "truncate": True},
                timeout=self.prewarm_timeout,
            )
            resp.raise_for_status()
        except Exception as e:
            log.warning("Embeddings: pre-warm failed (%s); continuing anyway.",
                        str(e)[:200])
            return False
        log.info("Embeddings: pre-warmed model %r in %.1fs (keep_alive=%s).",
                 self.model, time.monotonic() - started, self.keep_alive)
        return True

    def embed_batch(
        self,
        texts: list[str],
        on_result=None,
    ) -> list[list[float] | None]:
        """Embed many texts through Ollama's ``/api/embed`` endpoint.

        Cache hits short-circuit; if ANYTHING is uncached the model is
        pre-warmed first, then the remainder is fetched in chunks (see the
        class docstring for the chunk/halving/watchdog policy). Every text
        that succeeds is cached and reported through ``on_result`` the moment
        it lands, so an interrupted run resumes from disk instead of
        re-paying the whole pass. Results are written back at the original
        index, so the returned order matches the input. Raises only when the
        stall watchdog declares the server hung (or the pool surfaced such
        an error); texts that merely failed individually stay None.
        """
        results: list[list[float] | None] = [None] * len(texts)
        missing: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            cached = self._cache.get(self._key(t))
            if cached is not None:
                results[i] = cached
            else:
                missing.append((i, t))
        if not missing:
            return results

        self.prewarm()

        watchdog = self.stall_budget_sec
        state = {"last_progress": time.monotonic(), "done": 0, "failed": 0}

        def _check_stall() -> None:
            if watchdog and time.monotonic() - state["last_progress"] > watchdog:
                raise RuntimeError(
                    f"embeddings stalled: no text embedded for {watchdog:.0f}s "
                    f"({state['done']}/{len(missing)} done, "
                    f"{state['failed']} failed)"
                )

        def _publish(i: int, t: str, vec: list[float]) -> None:
            with self._lock:
                results[i] = vec
                self._cache[self._key(t)] = vec
                state["last_progress"] = time.monotonic()
                state["done"] += 1
                done = state["done"]
            if done % self.PROGRESS_EVERY == 0:
                log.info("Embeddings: %d/%d uncached text(s) embedded.",
                         done, len(missing))
            if on_result is not None:
                try:
                    on_result(t, vec)
                except Exception:  # disk-cache write errors are mere misses
                    pass

        def _note_failure() -> None:
            with self._lock:
                state["failed"] += 1
                failed, done = state["failed"], state["done"]
            # Cascade guard: the stall watchdog covers HUNG servers, but a
            # dead one fast-fails every halved chunk and single retry in
            # milliseconds, so no stall would ever trip. A dead server must
            # not grind through the whole text list — abort once enough texts
            # failed while NOTHING ever succeeded. Sporadic per-text failures
            # (done > 0) are always tolerated.
            if failed >= 25 and done == 0:
                raise RuntimeError(
                    f"embeddings aborted: {failed} texts failed with zero "
                    f"successes — server unreachable/down?"
                )

        def _post(inputs: list[str]) -> list[list[float]]:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": inputs,
                      "keep_alive": self.keep_alive, "truncate": True},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            vecs = resp.json().get("embeddings") or []
            if len(vecs) != len(inputs):
                raise ValueError(
                    f"batch size mismatch: {len(vecs)} embeddings for {len(inputs)} texts"
                )
            return vecs

        def _fetch(chunk: list[tuple[int, str]]) -> None:
            """Embed one chunk into results, halving on failure. Only the
            stall watchdog may raise (killing a doomed pass early)."""
            _check_stall()
            try:
                vecs = _post([t for _, t in chunk])
            except Exception as err:
                if len(chunk) > 1:
                    log.warning(
                        "Embeddings: %d-text chunk failed (%s); halving.",
                        len(chunk), str(err)[:200],
                    )
                    mid = len(chunk) // 2
                    _fetch(chunk[:mid])
                    _fetch(chunk[mid:])
                    return
                # Single text: one retry for a transient blip, then drop
                # ONLY this text (fail-open is per-text, not per-pass).
                _check_stall()
                i, t = chunk[0]
                try:
                    vecs = _post([t])
                except Exception as err2:
                    log.warning("Embeddings: dropping text idx %d after retry (%s).",
                                i, str(err2)[:200])
                    _note_failure()
                    return
            for (i, t), vec in zip(chunk, vecs):
                _publish(i, t, vec)

        chunks = [
            missing[s:s + self.batch_size]
            for s in range(0, len(missing), self.batch_size)
        ]
        log.info("Embeddings: %d uncached text(s) in %d chunk(s) of <=%d.",
                 len(missing), len(chunks), self.batch_size)

        if self.parallel_chunks > 1 and len(chunks) > 1:
            pool = ThreadPoolExecutor(
                max_workers=min(len(chunks), self.parallel_chunks),
                thread_name_prefix="embed",
            )
            errors: list[Exception] = []
            futures = [pool.submit(_fetch, chunk) for chunk in chunks]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    errors.append(e)
            # Queue nothing else on a stall; in-flight requests die on their
            # own request timeout (already-embedded texts are disk-cached).
            pool.shutdown(wait=False, cancel_futures=True)
            if errors:
                raise errors[0]
        else:
            for chunk in chunks:
                _fetch(chunk)

        if state["failed"]:
            log.warning(
                "Embeddings: embedded %d/%d uncached text(s); %d dropped — "
                "clusters/groups touching a dropped text fall back to exact "
                "-only merging.", state["done"], len(missing), state["failed"],
            )
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
    Fails open: on a wholesale embedding error only the exact-identity
    pre-merge runs; individually dropped texts (post-halving failures)
    dispatch unmerged instead of aborting the pass.
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
            if len(vectors) != len(pool):
                raise ValueError("embedding count mismatch in hypothesis dedup")
        except Exception as e:
            log.warning(
                "Semantic dedup: embedding failed (%s); dispatching %d "
                "hypotheses with exact-identity merging only.", e, len(pool),
            )
            outcome.extend(pool)
        else:
            # Per-text failures (embed_batch halving exhausted → None hole)
            # must not forfeit the whole pass: those records simply never
            # enter a similarity group and dispatch unmerged.
            if all(v is None for v in vectors):
                log.warning(
                    "Semantic dedup: every text failed to embed; dispatching "
                    "%d hypotheses with exact-identity merging only.", len(pool),
                )
                outcome.extend(pool)
            else:
                dropped = {i for i, v in enumerate(vectors) if v is None}
                if dropped:
                    log.warning(
                        "Semantic dedup: %d/%d hypothesis texts failed to embed; "
                        "those records dispatch unmerged.", len(dropped), len(pool),
                    )
                groups: dict[tuple, list[int]] = {}
                for i, v in enumerate(vectors):
                    if i in dropped:
                        continue
                    h = pool[i]
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
                outcome.extend(pool[i] for i in sorted(dropped))

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
    embedding pass. Vectors hit disk incrementally (via on_result) as they
    arrive, so an interrupted pass resumes the remainder. Cache read/write
    errors are treated as misses."""
    if not disk_cache_dir:
        return embedder.embed_batch(texts)
    cache_dir = Path(disk_cache_dir)
    results: list[list[float] | None] = [None] * len(texts)
    missing: list[tuple[int, str]] = []
    model = embedder.model

    def _cache_path(t: str) -> Path:
        return cache_dir / f"{hashlib.sha256(f'{model}:{t}'.encode('utf-8')).hexdigest()}.json"

    for i, t in enumerate(texts):
        try:
            results[i] = json.loads(_cache_path(t).read_text())["embedding"]
            continue
        except Exception:
            missing.append((i, t))
    if missing:
        def _cache_one(t: str, vec: list[float]) -> None:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                _cache_path(t).write_text(json.dumps({"embedding": vec}))
            except Exception:
                pass
        vecs = embedder.embed_batch([t for _, t in missing], on_result=_cache_one)
        for (i, _t), vec in zip(missing, vecs):
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

    Embeddings are fetched with ONE global call over the deduped union of all
    pending group texts, not one call per target: a demand flood means
    thousands of targets, and per-target calls serialized thousands of tiny
    HTTP round-trips (minutes of wall time). Per-group clustering inputs are
    unchanged (the embedder caches per text), so results are identical. On a
    global embedding failure every pending group keeps its exact-merged seeds
    (fail open, as before, but at whole-stage rather than per-target
    granularity).
    """
    total_in = sum(len(v) for v in grouped_demands.values())
    # Pass 1: canonical sort + exact-identity merge per target/type; embedding
    # candidates are collected instead of fetched inline. Each pending entry is
    # (keep list of the target, seed indices, embedding texts per seed).
    pending: list[tuple[list[int], list[int], list[str]]] = []
    targets_with_keeps: list[tuple[object, list[int]]] = []
    for target, demands in grouped_demands.items():
        demands = grouped_demands[target] = sorted(demands, key=_canonical_demand_key)
        if len(demands) <= 1:
            continue

        by_type: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(demands):
            by_type[d.get("type") or ""].append(i)

        keep: list[int] = []
        targets_with_keeps.append((target, keep))
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
            pending.append((keep, seeds, [_norm(demands[i].get("description")) for i in seeds]))

    # Pass 2: one global batched embed over the unique texts, then per-group
    # greedy clustering with the precomputed vectors.
    vec_map: dict[str, list[float]] = {}
    if pending and embedder is not None:
        unique_texts = list(dict.fromkeys(t for _, _, texts in pending for t in texts))
        try:
            vectors = _embed_with_disk_cache(embedder, unique_texts, disk_cache_dir)
            if len(vectors) != len(unique_texts):
                raise ValueError("embedding count mismatch in demand dedup")
            vec_map = dict(zip(unique_texts, vectors))
        except Exception as e:
            log.warning(
                "Demand dedup: embedding failed (%s); keeping exact-unique "
                "demands for %d multi-demand group(s).",
                e, len(pending),
            )
    for keep, seeds, texts in pending:
        try:
            clusters = _greedy_cluster_indices([vec_map[t] for t in texts], threshold)
        except Exception as e:  # missing vector for this group only
            log.warning(
                "Demand dedup: keeping %d exact-unique demands (%s)", len(seeds), e
            )
            keep.extend(seeds)
            continue
        keep.extend(seeds[cl[0]] for cl in clusters)

    for target, keep in targets_with_keeps:
        keep.sort()
        grouped_demands[target] = [grouped_demands[target][i] for i in keep]

    total_out = sum(len(v) for v in grouped_demands.values())
    if total_out != total_in:
        log.info(
            "Demand dedup: %d -> %d demands before contract verification (%d merged).",
            total_in, total_out, total_in - total_out,
        )
    return grouped_demands
