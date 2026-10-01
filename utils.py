from collections import defaultdict
import fnmatch
import hashlib
import logging
import re
import shutil
from pathlib import Path
import tree_sitter
import subprocess
import networkx as nx
import json
import tarfile
import time
import requests
from typing import Optional
from langchain_core.messages import AnyMessage
from schemas import VulnerabilityRecord
import settings
from functools import lru_cache


from languages import (
    LANGUAGE_MAP, AST_GRAMMAR_MAP, SYMBOL_QUERIES, MANIFEST_NAMES, GUARD_SPEC, guard_usages,
    SCAN_SIGNAL_TYPES, IMPORT_TYPES, DEFINITION_TYPES, NAME_NODE_TYPES, MAGIC_METHODS,
    PURE_TYPE_CONSTRUCTS, PURE_TYPE_FORBIDDEN_TYPES, BEHAVIORAL_NODE_TYPES,
    TYPE_ALIAS_NODE_TYPES, PHP_INTERFACE_TYPES, PHP_PROPERTY_TYPES, PHP_METHOD_TYPES,
    FRAGMENT_WRAP,
)


def _is_manifest_node(node: dict) -> bool:
    """True if a graph node belongs to a dependency manifest/lockfile.

    Such files are already handled by the SCA layer (osv-scanner), so the LLM
    stages (manager, explorers, reviewers, verifiers) should never spend budget
    re-analyzing them. Matches on the node's ``source_file`` basename against
    ``MANIFEST_NAMES``.
    """
    source_file = node.get("source_file")
    if not source_file:
        return False
    return Path(source_file).name in MANIFEST_NAMES


def _strip_links_to_dropped(graph_data: dict, dropped_ids: set) -> list:
    """Return links whose source AND target both survive the graph filter."""
    links = []
    for edge in graph_data.get("links", []):
        if edge.get("source") in dropped_ids or edge.get("target") in dropped_ids:
            continue
        links.append(edge)
    return links


@lru_cache(maxsize=1)
def get_cached_graph_data(graph_path: Path):
    """Caches the graph JSON in memory to prevent disk I/O bottlenecks.

    Dependency manifest/lockfile nodes (``MANIFEST_NAMES``) are dropped from the
    returned graph entirely: they are already handled by the SCA layer, and the
    LLM stages must not see them (they could mislead the manager, explorers, or
    the reviewer). Their incident links are stripped too.
    """
    try:
        with open(graph_path, "r") as f:
            graph_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"'{graph_path}' not found.")
        return {}

    if not graph_data.get("nodes"):
        return graph_data

    dropped_ids = {
        node.get("id")
        for node in graph_data.get("nodes", [])
        if node.get("id") and _is_manifest_node(node)
    }
    if dropped_ids:
        filtered = {
            "nodes": [
                node for node in graph_data.get("nodes", [])
                if node.get("id") not in dropped_ids
            ],
        }
        filtered["links"] = _strip_links_to_dropped(graph_data, dropped_ids)
        graph_data = filtered

    if getattr(settings, "repair_call_edges", False):
        _repair_call_edges(graph_data)
    return graph_data


# Per-run memoization for the aggregate_demands pass only. Everything here is
# deliberately process-scoped so ``clear_aggregate_caches`` can drop it when the
# node returns: the reviewer/validator LLM stages never see this memory. Only
# small extracted results are cached (alias sets, folded-code strings); parsed
# tree-sitter trees are always transient.
AGGREGATE_MEMO_ALIASES: dict[tuple[str, str], Optional[set[str]]] = {}
AGGREGATE_MEMO_FOLDED: dict[str, Optional[str]] = {}

# Leading dot excludes top-level functions so file nodes are not containers.
_MEMBER_LABEL_RE = re.compile(r"^\.([A-Za-z_]\w*)\(\)$")
_CALL_SITE_RE = re.compile(
    r"(?P<recv>\$this|[A-Za-z_]\w*)\s*(?P<op>::|->|\.)\s*(?P<meth>[A-Za-z_]\w*)\s*\("
)


def _repair_call_edges(graph_data: dict) -> None:
    """Retarget ``calls`` edges graphify mis-bound to a class container.

    The correct member is recovered from the edge's own call line (one regex
    pass, no parser): one match retargets, several fan out, none keeps the
    original edge. In memory only — node ids never change, caches never bust.
    """
    nodes = {
        n["id"]: n
        for n in graph_data.get("nodes", [])
        if n.get("id")
    }
    members: dict[str, dict[str, str]] = {}
    for nid, node in nodes.items():
        match = _MEMBER_LABEL_RE.match(node.get("label") or "")
        if not match:
            continue
        # Member id == f"{container}_{method.lower()}"; strip the exact name.
        mname = match.group(1).lower()
        if not nid.endswith("_" + mname):
            continue
        parent_id = nid[: len(nid) - len(mname) - 1]
        parent = nodes.get(parent_id)
        if (
            not parent
            or parent_id == nid
            or parent.get("source_file") != node.get("source_file")
            or _MEMBER_LABEL_RE.match(parent.get("label") or "")
        ):
            continue
        members.setdefault(parent_id, {})[match.group(1).lower()] = nid
    if not members:
        return

    links = graph_data.setdefault("links", [])
    known_pairs = {(e.get("source"), e.get("target")) for e in links}
    line_cache: dict[str, list[str]] = {}
    remapped = fanned = kept = 0

    for idx in range(len(links)):
        edge = links[idx]
        if edge.get("relation") != "calls":
            continue
        container_id = edge.get("target")
        member_map = members.get(container_id or "")
        if not member_map:
            continue
        source_file = edge.get("source_file")
        loc = edge.get("source_location") or ""
        if not source_file or not loc.startswith("L"):
            kept += 1
            continue
        lines = line_cache.get(source_file)
        if lines is None:
            lines = (read_file_text(source_file) or "").splitlines()
            line_cache[source_file] = lines
        try:
            line_no = int(loc[1:]) - 1
        except ValueError:
            kept += 1
            continue
        if not 0 <= line_no < len(lines):
            kept += 1
            continue

        container_label = (nodes.get(container_id) or {}).get("label") or ""
        hits: set[str] = set()
        for call in _CALL_SITE_RE.finditer(lines[line_no]):
            recv = call["recv"].lstrip("$").lower()
            if not (
                call["meth"].lower() in member_map
                and (
                    recv == container_label.lower()
                    or (call["op"] == "::" and recv in {"self", "static", "parent"})
                    or (call["op"] == "->" and recv in {"this", "self"})
                )
            ):
                continue
            hits.add(member_map[call["meth"].lower()])
        if not hits:
            kept += 1
            continue

        ordered = sorted(hits)
        # Dedup per target so a known sibling never blocks the other's fan-out.
        new_targets = [t for t in ordered if (edge.get("source"), t) not in known_pairs]
        if not new_targets:
            kept += 1
            continue
        edge["target"] = new_targets[0]
        edge["repaired_from"] = container_id
        known_pairs.add((edge.get("source"), new_targets[0]))
        remapped += 1
        for extra in new_targets[1:]:
            links.append({**edge, "target": extra})
            known_pairs.add((edge.get("source"), extra))
            fanned += 1

    logging.info(
        f"call-edge repair: {remapped} retargeted to member nodes, "
        f"{fanned} fanned out (multi-member lines), {kept} left verbatim."
    )


def clear_aggregate_caches() -> None:
    """Release the aggregate_demands memoization (source-text cache + extracted
    results) once the node completes, so the ~codebase-size corpus does not
    persist into the downstream LLM stages."""
    AGGREGATE_MEMO_ALIASES.clear()
    AGGREGATE_MEMO_FOLDED.clear()
    read_file_text.cache_clear()


@lru_cache(maxsize=8192)
def read_file_text(source_file: str) -> Optional[str]:
    """Read a source file's text exactly once and cache it in memory.

    Keys are the raw ``source_file`` strings found on graph nodes (absolute or
    app-relative); the file is resolved against ``settings.app_path``. Every
    phase of the aggregate pass (keyword corpus, import extraction, folded-code
    reads) shares this single copy, so no file text is ever duplicated.
    Returns ``None`` for missing, binary, or unreadable files so the result is
    cacheable. Call ``clear_aggregate_caches`` to release the cached text.
    """
    path = (settings.app_path / Path(source_file)).resolve()
    if not path.exists():
        logging.debug(f"read_file_text: skipping missing file '{path}'.")
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logging.debug(f"read_file_text: skipping binary file '{path}'.")
        return None
    except OSError as e:
        logging.debug(f"read_file_text: skipping unreadable file '{path}': {e}")
        return None


# Matches an entire `<script …>…</script>` block (open tag with any attributes,
# verbatim body, close tag).
_VUE_SCRIPT_RE = re.compile(
    r"""(<script\b(?:"[^"]*"|'[^']*'|[^>"'])*>)([\s\S]*?)(</script\s*>)""",
    re.IGNORECASE,
)


def masked_source_for_parsing(source_text: str, file_path: str | Path | None) -> str:
    """Return source text that is safe / useful to hand a tree-sitter parser.

    For a ``.vue`` single-file component, blank everything outside the
    ``<script>`` bodies (keeping ``\\r``/``\\n``) so the TypeScript grammar sees
    only the script logic while line numbers stay SFC-accurate — the same
    convention the upstream graphify extractor uses, so ``source_location``
    values on graph nodes map 1:1 onto the parsed text. Idempotent for text that
    is already masked. Every other extension is returned unchanged.
    """
    if not source_text:
        return source_text
    try:
        suffix = Path(file_path).suffix.lower() if file_path else ""
    except TypeError:
        suffix = ""
    if suffix != ".vue":
        return source_text

    def _blank(s: str) -> str:
        return re.sub(r"[^\r\n]", " ", s)

    # Idempotency guard: an already-masked text has no literal <script> tags
    # left, so bail without touching it. Also covers template-only SFCs.
    if not _VUE_SCRIPT_RE.search(source_text):
        return source_text

    parts: list[str] = []
    pos = 0
    for m in _VUE_SCRIPT_RE.finditer(source_text):
        parts.append(_blank(source_text[pos:m.start()]))
        parts.append(_blank(m.group(1)))  # <script …> open tag
        parts.append(m.group(2))          # script body, verbatim
        parts.append(_blank(m.group(3)))  # </script> close tag
        pos = m.end()
    parts.append(_blank(source_text[pos:]))
    return "".join(parts)


def iter_code_files():
    """    Yield every unique, non-excluded source file whose graph nodes carry
    ``file_type == "code"``, in first-seen order.

    Streaming: callers can scan the whole repo exactly once without
    materializing every file in memory (no O(files) RAM corpus).
    """
    graph_data = get_cached_graph_data(settings.graph)
    seen: set[str] = set()
    for node in graph_data.get("nodes", []):
        source_file = node.get("source_file")
        if node.get("file_type") != "code" or not source_file or source_file in seen:
            continue
        if is_path_excluded(source_file):
            continue
        seen.add(source_file)
        yield source_file


def scan_codebase_for_keywords(all_keywords: list[str]):
    """Scan all code files once and return the set of keywords present in any.

    Streaming: files are read one at a time via ``read_file_text`` and searched
    with plain per-keyword substring checks (exact, so overlapping keywords
    like ``yaml.load`` / ``yaml.load_all`` are each reported); scanning stops
    early once every keyword has been found. Returns ``(present,
    scanned_bytes)`` so callers can distinguish 'no matches' from 'nothing to
    scan'.
    """
    kws = sorted({k for k in all_keywords if k})
    if not kws:
        return set(), 0
    present: set[str] = set()
    scanned_bytes = 0
    for source_file in iter_code_files():
        content = read_file_text(source_file)
        if content is None:
            continue
        scanned_bytes += len(content)
        present.update(kw for kw in kws if kw in content)
        if len(present) >= len(kws):
            break
    return present, scanned_bytes


def find_unsupported_code_files(graph_data: dict) -> dict[str, list[str]]:
    """Map unsupported-language code files to their source paths.

    A graph node counts as an unsupported code file when it carries
    ``file_type == "code"`` and a ``source_file`` whose (lowercased) extension is
    not present in both ``LANGUAGE_MAP`` and ``SYMBOL_QUERIES`` (the same gate
    ``index_file`` enforces). Nodes lacking a ``source_file`` or with an empty
    extension are skipped. Returns ``{".java": ["a.java", "b.java"], ...}``.
    """
    unsupported: dict[str, list[str]] = defaultdict(list)
    for node in graph_data.get("nodes", []):
        if node.get("file_type") != "code":
            continue
        source_file = node.get("source_file")
        if not source_file:
            continue
        ext = Path(source_file).suffix.lower()
        if not ext or (ext in LANGUAGE_MAP and ext in SYMBOL_QUERIES):
            continue
        if source_file not in unsupported[ext]:
            unsupported[ext].append(source_file)
    return dict(unsupported)


@lru_cache(maxsize=512)
def guard_map_for_node(node_id: str) -> dict[str, tuple[str, ...]]:
    """Map callee base names to their decision-guard usage contexts in a node's code.

    Analyzes the node's raw (unpruned) source slice with tree-sitter; languages
    without a GUARD_SPEC entry yield an empty map. Returns
    {base_name: (context, ...)} with unique contexts in source order.
    """
    node = next(
        (n for n in get_cached_graph_data(settings.graph).get("nodes", []) if n.get("id") == node_id),
        None,
    )
    if not node:
        return {}
    source_file = node.get("source_file")
    ext = Path(source_file).suffix.lower() if source_file else ""
    if ext not in GUARD_SPEC:
        return {}
    code = get_node_code(node_id, raw=True)
    if not code:
        return {}
    guards = guard_usages(code, ext)
    if not guards:
        return {}

    guard_map: dict[str, list[str]] = {}
    for guard in guards:
        base = guard["callee"]
        context = guard["context"]
        contexts = guard_map.setdefault(base, [])
        if context not in contexts:
            contexts.append(context)
    return {k: tuple(v) for k, v in guard_map.items()}


def _name_base(name: str) -> str:
    """Normalize a graph node label to a comparable lowercase function name.

    '.login()' / 'isAPI()' / 'Auth::check()' / 'login_required()' all reduce to
    their bare function name so connection labels match the base names produced
    by guard analysis.
    """
    return (
        name.strip().rstrip("()").split("->")[-1].split("::")[-1]
        .lstrip("$").lstrip(".").lower()
    )


def _guard_annotation(source_node_id: Optional[str], target_base: str) -> str:
    """Render the '(participates in: ...)' suffix for a connection.

    Looks up target_base among the guard usages found in source_node_id's code
    (the analyzed node for outgoing connections, the caller for incoming ones).
    """
    if not source_node_id or not target_base:
        return ""
    contexts = guard_map_for_node(source_node_id).get(target_base)
    if not contexts:
        return ""
    return f" (participates in: {', '.join(contexts[:3])})"


def format_node_context(graph_data: dict, node_id: str) -> str:
    """Render 'graphify explain' style context for a node: its summary plus connections.

    Shows the node's label, id, source file/location and community, followed by
    all incoming ('<--') and outgoing ('-->') links with their relation.
    Connections to functions sharing a language with GUARD_SPEC are additionally
    annotated with how the connected function is used in decision guards —
    outgoing links from the analyzed node's own guard analysis, incoming links
    from the caller's ('(participates in: $check_mfa)').
    Returns an empty string if the target node is not found.
    """
    nodes = graph_data.get("nodes", [])
    node_map = {n.get("id"): n for n in nodes}
    target_node = node_map.get(node_id)
    if target_node is None:
        return ""

    self_base = _name_base(target_node.get("label", node_id))

    # Collect connections touching this node, resolved to neighbor labels.
    connections = []
    for edge in graph_data.get("links", []):
        relation = edge.get("relation")
        if edge.get("target") == node_id:
            neighbor = node_map.get(edge.get("source"))
            arrow = "<--"
        elif edge.get("source") == node_id:
            neighbor = node_map.get(edge.get("target"))
            arrow = "-->"
        else:
            continue
        neighbor_label = neighbor.get("label", edge.get("source") or edge.get("target")) if neighbor else (edge.get("source") or edge.get("target"))
        neighbor_id = edge.get("source") if arrow == "<--" else edge.get("target")
        connections.append((arrow, neighbor_id, neighbor_label, relation, str(edge.get("source_location", ""))))

    # Stable ordering by source line number.
    connections.sort(key=lambda c: c[4])

    label = target_node.get("label", node_id)
    lines = [
        f"Node: {label}",
        f"  ID:        {node_id}",
        f"  Source:    {target_node.get('source_file', '')} {(target_node.get('source_location') or '').strip()}",
        f"  Community: {target_node.get('community', '')}",
        "",
        f"Connections ({len(connections)}):",
    ]
    for arrow, neighbor_id, neighbor_label, relation, _ in connections:
        line = f"  {arrow} {neighbor_label} [{relation}]"
        if arrow == "-->":
            line += _guard_annotation(node_id, _name_base(neighbor_label))
        else:
            line += _guard_annotation(neighbor_id, self_base)
        lines.append(line)

    return "\n".join(lines)


@lru_cache(maxsize=1)
def get_cached_symbol_index(index_path: Path) -> list[dict]:
    """Caches the AST symbol index in memory to prevent disk I/O bottlenecks."""
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"AST symbol index not found at '{index_path}'.")
        return []


def build_networkx_graph(graph_path: Path, allowed_communities: Optional[list[int]] = None) -> nx.DiGraph:
    """
    Reads the Graphify JSON output and builds a NetworkX Directed Graph.
    Optionally filters the graph to only include specific communities.
    """
    graph_data = get_cached_graph_data(graph_path)

    # Initialize a Directed Graph
    G = nx.DiGraph()

    # Convert list to set for faster lookups
    allowed_set = set(allowed_communities) if allowed_communities is not None else None

    # Add Nodes with their attributes (community, type, file_path, etc.)
    for node in graph_data.get('nodes', []):
        node_id = node.get('id')
        if not node_id:
            continue

        # FILTERING LOGIC: Skip node if it doesn't belong to the allowed communities
        if allowed_set is not None:
            community_id = node.get('community')
            if community_id not in allowed_set:
                continue

        # Copy all other key-value pairs as node attributes
        attributes = {k: v for k, v in node.items() if k != 'id'}
        G.add_node(node_id, **attributes)

    # Add Edges with their attributes (e.g., relationship type like 'calls')
    for edge in graph_data.get('links', []):
        source = edge.get('source')
        target = edge.get('target')
        if not source or not target:
            continue

        # CRITICAL: Only add edges if BOTH nodes survived the community filter.
        # Otherwise, NetworkX will silently re-create the deleted nodes.
        if source in G and target in G:
            # Copy all other key-value pairs as edge attributes
            attributes = {k: v for k, v in edge.items() if k not in ['source', 'target']}
            G.add_edge(source, target, **attributes)

    return G


def _merge_affected_nodes(target: dict, *sources: dict) -> None:
    """Union the `affected_nodes` of the given records into `target`, deduped
    and in first-seen order (earlier sources win ordering). Systemic records
    share one node-independent vuln_id, so every collision accumulates nodes."""
    merged = []
    seen = set()
    for source in sources:
        for node in source.get("affected_nodes") or []:
            if node and node not in seen:
                seen.add(node)
                merged.append(node)
    target["affected_nodes"] = merged


def reviewer_cache_key(report: Optional[dict], default: str = "Unknown") -> str:
    """Stable reviewer-cache key prefix derived from a report's affected nodes.

    Replaces the old single `node_id` prefix; the report content hash (which
    already includes `affected_nodes`) keeps entries distinct per node set."""
    affected = sorted({n for n in (report or {}).get("affected_nodes") or [] if n})
    return "+".join(affected) if affected else default


def is_feedback_review(report: Optional[dict]) -> bool:
    """True when the report is a Validator->Reviewer feedback re-review rather
    than a first-pass hypothesis review.

    `ask_for_context` bumps `review_round` and fills `open_questions`, and
    `route_validator_feedback` re-dispatches exactly those flagged records to the
    reviewer. Feedback re-reviews must NEVER be served from (or written to) the
    reviewer cache: their whole purpose is to genuinely re-answer the Validator's
    questions and emit self-sufficient reproduction steps for re-validation.
    Round-0 hypothesis reviews are pure/idempotent analyses whose caching is
    safe, but a round-N feedback report is byte-identical across every replay of
    the same dispatch (e.g. resuming a checkpoint inside the feedback cycle), so
    a content-hash cache would silently collapse each replay into the earlier
    verdict with zero LLM turns — the reviewer appears to "not run again"."""
    if not report:
        return False
    return (report.get("review_round") or 0) > 0 or bool(report.get("open_questions"))


# Merge ladder: higher rank wins on vuln_id collision. "chained" ranks above
# "confirmed" (auditor proved the record joins a multi-step exploit) but below
# "exploitable" (a validator PoC outranks the auditor's static chain proof).
# "unchainable" is a terminal auditor verdict that stays in the report, like
# "false_positive".
_STATUS_PRIORITY = {
    "hypothesis": 0,
    "review_error": 1,
    "confirmed": 2,
    "chained": 3,
    "insufficient_context": 3,
    "exploitable": 4,
    "false_positive": 5,
    "unchainable": 5,
    "proven": 5,
}


def merge_vulnerabilities(existing: list[dict], updates: list[dict]) -> list[dict]:
    vuln_map = {}

    # Map existing records
    for vuln in existing:
        vid = vuln.get("vuln_id")
        if vid:
            vuln_map[vid] = vuln

    # Process new incoming updates
    for update in updates:
        # If it's a Pydantic model, convert to dict
        if not isinstance(update, dict):
            update = update.model_dump() if hasattr(update, "model_dump") else dict(update)

        # If vuln_id wasn't pre-computed, instantiate the model to trigger the validator logic
        if not update.get("vuln_id"):
            record = VulnerabilityRecord(**update)
            update = record.model_dump()

        vid = update.get("vuln_id")
        new_status = update.get("status", "hypothesis")

        if vid in vuln_map:
            current_status = vuln_map[vid].get("status", "hypothesis")

            # --- FEEDBACK LOOP: REVIEWER RE-ANSWER REPLACES INSUFFICIENT CONTEXT ---
            # When the Validator flagged a record insufficient_context and the Reviewer has
            # re-reviewed it, the reviewer verdict must replace the flag unconditionally
            # (confirmed and review_error have LOWER priority than insufficient_context, so
            # the plain ladder would wrongly keep the flag).
            re_review_statuses = {"confirmed", "false_positive", "exploitable", "review_error"}
            if current_status == "insufficient_context" and new_status in re_review_statuses:
                _merge_affected_nodes(update, update, vuln_map[vid])
                vuln_map[vid] = update
                continue

            # --- STATUS UPGRADE: COMPLETELY REPLACE ---
            if _STATUS_PRIORITY.get(new_status, 0) > _STATUS_PRIORITY.get(current_status, 0):
                _merge_affected_nodes(update, update, vuln_map[vid])
                vuln_map[vid] = update

            # --- SAME STAGE: MERGE CONTEXT ---
            # If two parallel agents at the same stage find the same issue
            # (e.g., two explorers finding the same hypothesis), merge the text.
            # Exception: records from two DISTINCT known CVEs never text-merge,
            # even if their vuln_ids ever collide — each CVE is a canonical,
            # separately-fixed flaw, so only affected_nodes are unioned.
            elif _STATUS_PRIORITY.get(new_status, 0) == _STATUS_PRIORITY.get(current_status, 0):
                current = vuln_map[vid]
                distinct_cves = (
                    bool(current.get("source_cve"))
                    and bool(update.get("source_cve"))
                    and current["source_cve"] != update["source_cve"]
                )

                if not distinct_cves:
                    curr_desc = current.get("description", "")
                    upd_desc = update.get("description", "")

                    if upd_desc and upd_desc not in curr_desc:
                        current["description"] = f"{curr_desc}\n\nAdditional context: {upd_desc}"

                _merge_affected_nodes(current, current, update)
                vuln_map[vid] = current
        else:
            # --- NEW UNIQUE VULNERABILITY ---
            vuln_map[vid] = update

    return list(vuln_map.values())


def estimate_message_tokens(messages: list[AnyMessage]) -> int:
    """
    Conservative token estimate for a list of messages. Starts from a ~2
    chars/token base, then adds per-message metadata overhead (8 tokens each)
    and a 15% fudge factor. Measured on real reviewer histories, deepseek-v4-
    flash tokenizes prose at ~5 chars/token (so the 2 chars/token base alone
    already over-estimates prose ~2.5x), while dense code can run below the 2
    chars/token rate — the overhead + fudge keeps the estimate above the model's
    real token count in both regimes so context compaction never races the hard
    input limit.
    """
    total_chars = 0
    message_count = 0
    for msg in messages:
        if msg is None:
            continue
        message_count += 1
        content = getattr(msg, "content", None)
        if content:
            total_chars += len(str(content))
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
                if args:
                    total_chars += len(str(args))
    return int((total_chars / 2) * 1.15) + 8 * message_count


def resolve_node_id(module, symbol):
    graph_data = get_cached_graph_data(settings.graph)
    nodes_list = graph_data.get("nodes", [])

    # Sanitize Inputs
    # Handle cases where the LLM returns None, empty string, or "global"
    if not module or module.lower() == "self":
        module = ""

    if not symbol:
        logging.warning("No symbol provided to resolve_node_id.")
        return None

    # Handle LLM concatenating multiple symbols (e.g., "User::dropdown/Group::dropdown")
    if "/" in symbol:
        symbol = symbol.split("/")[0].strip()

    # Extract Class/Method from Symbol (e.g., "User::dropdown" -> "User", "dropdown")
    class_name = ""
    if "::" in symbol:
        class_name, symbol = symbol.split("::", 1)
    elif "." in symbol:
        class_name, symbol = symbol.split(".", 1)

    # Normalize for matching
    normalized_module = module.replace(".", "/").replace("\\", "/").lower()
    base_module_name = normalized_module.split("/")[-1] if normalized_module else ""

    lower_symbol = symbol.lower()
    lower_class = class_name.lower()

    for n in nodes_list:
        source_file = n.get("source_file", "")
        if not source_file:
            continue

        lower_file = source_file.replace("\\", "/").lower()
        label = str(n.get("label", "")).lower()

        # Evaluate File/Scope Match
        # If we have a module, use the original logic.
        # If we have a class name, look for the class name in the file path (e.g., User.php) or label.
        # If neither exist, allow file_match to be True and search globally.
        file_match = True
        if base_module_name:
            file_match = (
                base_module_name in lower_file or
                normalized_module in lower_file
            )
        elif lower_class:
            file_match = (lower_class in lower_file or lower_class in label)

        # Evaluate Label Match
        label_match = (
            label == lower_symbol or
            label == f"{lower_symbol}()" or
            label.endswith(f"::{lower_symbol}") or
            label.endswith(f"->{lower_symbol}") or
            label.endswith(f".{lower_symbol}") or
            lower_symbol in label
        )

        if file_match and label_match:
            return n.get("id")

    # Better logging to help debug what was actually searched
    search_mod = module if module else "global"
    logging.warning(f"Failed to find graph node for [{search_mod}] '{symbol}' (Class: {class_name}).")
    return None


def _run_osv(cmd: list, label: str, skip_os: bool = False) -> list[dict]:
    """Run an osv-scanner command and collect raw vulnerability records.

    Exit code 1 simply means "vulnerabilities found" - the JSON on stdout is
    still valid; only an empty stdout indicates no results. ``skip_os`` drops
    OS-package scan results (image scans).
    """
    raw_vulnerabilities = []
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if not result.stdout.strip():
            if result.stderr:
                logging.error(f"Error running osv-scanner on {label}: {result.stderr}")
            return []

        data = json.loads(result.stdout)

        # Extract the vulnerability objects from the osv-scanner JSON schema
        for scan_result in data.get("results", []):
            if skip_os and scan_result.get("source", {}).get("type") == "os":
                logging.debug("Skipping OS-package vulnerabilities from image scan.")
                continue
            for package in scan_result.get("packages", []):
                raw_vulnerabilities.extend(package.get("vulnerabilities", []))

    except FileNotFoundError:
        logging.error("osv-scanner is not installed or not in PATH.")
    except json.JSONDecodeError:
        logging.error("Could not parse osv-scanner output.")

    return raw_vulnerabilities


def run_osv_scanner(repo_path: Path) -> list[dict]:
    """Runs osv-scanner on a directory and extracts raw vulnerability records."""
    if not repo_path.exists():
        logging.error(f"Input report does not exist: {repo_path}")
        return []
    return _run_osv(["osv-scanner", "-r", "--format", "json", repo_path], str(repo_path))


COMPOSE_FILENAMES = ("compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml")
DOCKERFILE_NAMES = ("Dockerfile",)
_BUILD_IGNORED_DIRS = {".git", "node_modules", "vendor", ".cache", ".next", "graphify-out"}


def _is_build_ignored(path: Path) -> bool:
    return any(part in _BUILD_IGNORED_DIRS for part in path.parts)


def find_container_builds(app_path: Path) -> list[tuple[str, Path]]:
    """Locate container build definitions in the app repo.

    Compose files take precedence over raw Dockerfiles. Returns a list of
    ``(kind, path)`` tuples where kind is ``"compose"`` or ``"dockerfile"``.
    """
    compose = sorted(
        p for p in app_path.rglob("*")
        if p.is_file() and p.name in COMPOSE_FILENAMES and not _is_build_ignored(p)
    )
    if compose:
        return [("compose", compose[0])]

    dockerfiles = sorted(
        p for p in app_path.rglob("*")
        if p.is_file()
        and (p.name in DOCKERFILE_NAMES or p.name.startswith("Dockerfile.") or p.name.endswith(".dockerfile"))
        and not _is_build_ignored(p)
    )
    return [("dockerfile", d) for d in dockerfiles]


def _docker(*args: str, timeout: int | None = None) -> subprocess.CompletedProcess | None:
    """Run a docker CLI command; returns the CompletedProcess, or None (with a
    logged error) when the docker binary is missing."""
    try:
        return subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError:
        logging.error("docker is not installed or not in PATH.")
        return None


def _image_exists(image: str) -> bool:
    """True if a local docker image with the given tag exists."""
    result = _docker("image", "inspect", image)
    return result is not None and result.returncode == 0


def _build_definition_hash(kind: str, path: Path) -> str:
    digest = hashlib.md5()
    digest.update(path.read_bytes())
    if kind == "compose":
        # Fold in any Dockerfiles under the compose directory so edits to the
        # build context bust the reuse check too.
        for candidate in sorted(p for p in path.parent.rglob("Dockerfile*") if p.is_file()):
            if not _is_build_ignored(candidate):
                digest.update(candidate.read_bytes())
    return digest.hexdigest()


def _build_hash_file(path: Path) -> Path:
    return settings.cache_dir / "container_builds" / f"{_slugify_image(str(path))}.hash"


def _build_definition_unchanged(kind: str, path: Path) -> bool:
    try:
        digest = _build_definition_hash(kind, path)
    except OSError:
        return False
    stamp = _build_hash_file(path)
    if not stamp.exists():
        return False
    try:
        return stamp.read_text().strip() == digest
    except OSError:
        return False


def _record_build_hash(kind: str, path: Path) -> None:
    try:
        stamp = _build_hash_file(path)
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(_build_definition_hash(kind, path))
    except OSError:
        pass


def _compose_images(path: Path) -> list[str]:
    """Derive the image names a compose file builds without building."""
    result = _docker("compose", "-f", str(path), "config", "--images")
    if result is None:
        return []
    if result.returncode != 0:
        logging.error(f"docker compose config failed: {result.stderr}")
        return []
    images = [img.strip() for img in result.stdout.splitlines() if img.strip()]
    if not images:
        logging.error("docker compose config returned no images.")
    # Compose emits build-only services (and unqualified `image:` refs)
    # without a tag (e.g. "<project>-<service>"); osv-scanner rejects
    # untagged references, so normalize them to :latest.
    return [img if ":" in img else f"{img}:latest" for img in images]


def build_images(kind: str, path: Path, tag: str) -> list[str]:
    """Build the container image(s) described by a compose file or Dockerfile.

    Returns the built image tag(s). On failure logs the error and returns [].
    Unless ``settings.force_rebuild`` is set, images whose build definition
    content is unchanged since the last recorded build are reused as-is (the
    caller still scans/extracts artifacts from them).
    """
    if kind == "compose":
        images = _compose_images(path)
        if (
            not settings.force_rebuild
            and images
            and all(_image_exists(img) for img in images)
            and _build_definition_unchanged(kind, path)
        ):
            logging.info(f"Reusing existing compose image(s) {images} (build definition unchanged).")
            return images

        build = _docker("compose", "-f", str(path), "build")
        if build is None:
            return []
        if build.returncode != 0:
            logging.error(f"docker compose build failed: {build.stderr}")
            return []
        _record_build_hash(kind, path)
        return _compose_images(path)

    # Single Dockerfile build
    if (
        not settings.force_rebuild
        and _image_exists(tag)
        and _build_definition_unchanged(kind, path)
    ):
        logging.info(f"Reusing existing image {tag} (build definition unchanged).")
        return [tag]

    build = _docker("build", "-t", tag, "-f", str(path), str(path.parent))
    if build is None:
        return []
    if build.returncode != 0:
        logging.error(f"docker build failed: {build.stderr}")
        return []
    _record_build_hash(kind, path)
    return [tag]


SANDBOX_START_TIMEOUT = 30  # seconds to wait for a sandbox HTTP port to come up


def _published_ports(container_name: str) -> list[int]:
    """Return the host ports published by a container via `docker port`."""
    result = _docker("port", container_name)
    if result is None:
        return []

    ports = []
    for line in result.stdout.splitlines():
        # Format: "8080/tcp -> 0.0.0.0:32768" (or IPv6 "[::]:32768")
        if "->" in line:
            host = line.split("->", 1)[1].strip()
            port = host.rsplit(":", 1)[-1]
            if port.isdigit():
                ports.append(int(port))
    return ports


def docker_bridge_gateway() -> str | None:
    """Discover the docker default-bridge gateway IP (e.g. 172.17.0.1).

    The sandbox is published on ``0.0.0.0``, so it is reachable from both the
    host and any bridge-networked container (the attacker box) via this gateway
    address. Returning it lets every validator path (HTTP request, browser,
    attacker shell) share one target URL. Returns ``None`` if it cannot be
    resolved, so callers can fall back to ``127.0.0.1``."""
    try:
        r = _docker(
            "network", "inspect", "bridge",
            "--format", "{{(index .IPAM.Config 0).Gateway}}",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
        logging.warning("Failed to resolve bridge gateway: %s", e)
        return None
    if r is None or r.returncode != 0:
        return None
    gw = r.stdout.strip()
    return gw or None


def _probe_http_ports(ports: list[int]) -> int | None:
    """Wait up to ``SANDBOX_START_TIMEOUT`` for any candidate host port to
    answer an HTTP request. Probes on the host loopback (``127.0.0.1``) and
    returns the first responsive port, or ``None`` if none ever responds.

    The returned host is the loopback only; callers should prefer
    ``docker_bridge_gateway()`` to build the externally-reachable ``sandbox_url``
    so the host and the attacker container agree on one target address."""
    deadline = time.monotonic() + SANDBOX_START_TIMEOUT
    remaining_ports = list(ports)

    while remaining_ports:
        if time.monotonic() >= deadline:
            break
        next_ports = []
        for port in remaining_ports:
            try:
                requests.get(f"http://127.0.0.1:{port}", timeout=1)
                return port
            except requests.RequestException:
                next_ports.append(port)
        remaining_ports = next_ports
        time.sleep(1)
    return None


def _sandbox_url_for_port(port: int) -> str:
    """Build the externally-reachable sandbox URL for a published ``port``.

    Prefers the docker bridge gateway (reachable from both the host and the
    attacker container); falls back to ``127.0.0.1`` when the gateway cannot be
    resolved. This keeps every validator tool pointed at the same target IP."""
    gw = docker_bridge_gateway()
    host = gw if gw else "127.0.0.1"
    return f"http://{host}:{port}"


def _remove_stale_compose_containers(path: Path) -> None:
    """Remove any existing container holding one of the compose file's static
    container names, so ``docker compose up`` can reuse it.

    The preprocessor restarts its own disposable sandbox each run; stale
    containers from an earlier project with the same pinned name would make
    ``compose up`` fail with a "name already in use" Conflict.
    """
    result = _docker("compose", "-f", str(path), "config", "--format", "json")
    if result is None:
        return
    if result.returncode != 0:
        logging.warning(f"docker compose config failed: {result.stderr}")
        return
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return

    for service in data.get("services", {}).values():
        name = service.get("container_name")
        if not name:
            continue
        rm = _docker("rm", "-f", name)
        if rm is not None and rm.returncode == 0:
            logging.info(f"Removed stale container '{name}' before compose up.")


def start_sandbox(kind: str, path: Path, tag: str, app_name: str) -> dict | None:
    """Start the built container image(s) in the background and return runtime data.

    Detaches the container(s) (compose: ``up -d``; Dockerfile: ``docker run -d -P``),
    discovers the published host ports, and waits for one of them to answer HTTP.

    Returns ``{"container_name": str, "sandbox_url": str}`` pointing at the first
    HTTP-responsive container (for compose that is the app service, not a DB sidecar),
    or ``None`` with a logged warning on any failure. Never raises.
    """
    if kind == "compose":
        # The compose file may pin static `container_name:` values still held
        # by leftover containers from another compose project (e.g. a prior
        # run of this scanner on the same app, or a different checkout).
        # Compose refuses to reuse a name owned by a differently-labelled
        # container, so pre-emptively remove anything holding those names.
        # These are throwaway scanner sandboxes - never a production service.
        _remove_stale_compose_containers(path)
        up = _docker("compose", "-f", str(path), "up", "-d")
        if up is None:
            return None
        if up.returncode != 0:
            logging.error(f"docker compose up failed: {up.stderr}")
            return None

        ps = _docker("compose", "-f", str(path), "ps", "--format", "json")
        if ps is None:
            return None
        if ps.returncode != 0:
            logging.error(f"docker compose ps failed: {ps.stderr}")
            return None

        containers = []
        for line in ps.stdout.splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if entry.get("State") == "running":
                    containers.append(entry.get("Name", ""))
            except json.JSONDecodeError:
                continue
        containers = [c for c in containers if c]
        if not containers:
            logging.error("docker compose up started no running containers.")
            return None
    else:
        name = f"vulnscan-{app_name}"
        # Remove any stale container from a previous run
        _docker("rm", "-f", name)
        run = _docker("run", "-d", "-P", "--name", name, tag)
        if run is None:
            return None
        if run.returncode != 0:
            logging.error(f"docker run failed: {run.stderr}")
            return None
        containers = [name]

    # Prefer the container that publishes a responsive HTTP port
    for container in containers:
        port = _probe_http_ports(_published_ports(container))
        if port:
            url = _sandbox_url_for_port(port)
            logging.info(f"Sandbox running: container={container} url={url}")
            return {"container_name": container, "sandbox_url": url}

    logging.warning(
        "Sandbox container(s) started but none published a responsive HTTP port "
        f"({containers}). Validator tools will report no sandbox configured."
    )
    return None


def run_osv_scanner_image(image: str) -> list[dict]:
    """Runs `osv-scanner scan image` against a built container image and
    extracts raw vulnerability records (same JSON shape as a source scan)."""
    return _run_osv(
        ["osv-scanner", "scan", "image", "--format", "json", image],
        f"image '{image}'",
        skip_os=True,
    )


# ==========================================
# Container artifact snapshot
# ==========================================
#
# During preprocessing each built container image is snapshotted (deterministic,
# decoupled from the live sandbox): a throwaway container is created but NEVER
# started, its filesystem is exported, security-relevant files are extracted
# into <target_app>/.cache/container_artifacts/<image-slug>/, and a full
# filesystem index plus image metadata are written for the reviewer tools.

# Size/count guards that keep the extracted artifacts directory small.
ARTIFACT_MAX_FILE_BYTES = 512 * 1024      # per extracted file
ARTIFACT_MAX_TOTAL_BYTES = 20 * 1024 * 1024  # total across one image
ARTIFACT_MAX_FILES = 300                  # max extracted files per image
ARTIFACT_MAX_INDEX_ENTRIES = 100_000      # max lines in filesystem_index.txt per image

# Catch-all pass: alongside the curated patterns we also pull small, text-like
# files under the app's WORKDIR that the patterns missed (e.g. an arbitrary
# named ``next.config.mjs``, ``.eslintrc.json``). This closes the "regex blind
# spot" for generated configs with nonstandard names, without needing to diff
# against the base image. Source-code files are excluded (the reviewer already
# sees the repo via the graph and read_file), as are dependency install trees.
ARTIFACT_CATCHALL_MAX_BYTES = 256 * 1024
_ARTIFACT_SOURCE_EXTS = tuple(sorted(set(LANGUAGE_MAP)))
_ARTIFACT_BINARY_EXTS = (".so", ".o", ".a", ".bin", ".class", ".jar", ".woff",
                         ".ttf", ".png", ".jpg", ".jpeg", ".gif", ".ico",
                         ".webp", ".pdf", ".gz", ".xz", ".zip", ".whl", ".tgz",
                         ".tar")
# Pure build-noise files/extensions that are neither config nor source and
# would flood the artifacts (framework build output: maps, RSC payloads,
# bundled media, generated manifests deep in framework internals).
_ARTIFACT_CATCHALL_NOISE_EXTS = (".map", ".rsc", ".meta", ".html", ".htm",
                                 ".css", ".svg", ".nft.json", ".trace")
_ARTIFACT_CATCHALL_NOISE_DIRS = ("/.next/", "/build/", "/server/", "/static/",
                                 "/cache/", "/diagnostics/", "/media/")

# Regexes matched (case-insensitive) against the FULL path inside the container.
# These deliberately capture security-relevant configuration, entrypoints, and
# generated build artifacts (e.g. a resolved Next.js config) while ignoring the
# bulk of the base-image filesystem (libraries, node_modules, binaries...).
ARTIFACT_PATTERNS = [
    # Web / reverse proxy / app server configuration
    r"(^|/)nginx[^/]*\.conf$",
    r"(^|/)nginx/conf\.d/.*\.conf$",
    r"(^|/)httpd[^/]*\.conf$",
    r"(^|/)apache2/.*\.conf$",
    r"(^|/)\.htaccess$",
    r"(^|/)Caddyfile$",
    r"(^|/)haproxy\.cfg$",
    r"(^|/)lighttpd\.conf$",
    r"(^|/)traefik\.(yml|yaml|toml)$",
    r"(^|/)envoy\.ya?ml$",
    r"(^|/)supervisord[^/]*\.conf$",
    # Runtime interpreters / databases
    r"(^|/)gunicorn[^/]*\.(conf|py)$",
    r"(^|/)uwsgi[^/]*\.(ini|ya?ml|yml|json)$",
    r"(^|/)php(-fpm)?[^/]*\.(ini|conf)$",
    r"(^|/)my\.cnf$",
    r"(^|/)redis\.conf$",
    r"(^|/)mongod\.conf$",
    # Entrypoint / startup manifests
    r"(^|/)docker-entrypoint[^/]*$",
    r"(^|/)entrypoint[^/]*\.sh$",
    r"(^|/)docker-entrypoint\.d/.*",
    r"(^|/)start\.sh$",
    # Environment / secret-like files baked into the image
    r"(^|/)\.env($|\.)",
    # Generated build artifacts & installed-dependency manifests
    r"(^|/)\.next/[^/]*\.json$",
    r"(^|/)package\.json$",
    r"(^|/)composer\.json$",
    r"(^|/)Gemfile$",
    r"(^|/)requirements[^/]*\.txt$",
    # Catch-all for config-like files directly under /etc
    r"(^|/)etc/[^/]*\.(conf|ini|cfg|ya?ml|yml|toml|json)$",
]

_ARTIFACT_RE = re.compile("|".join(f"(?:{p})" for p in ARTIFACT_PATTERNS), re.IGNORECASE)

# Paths inside these directories are never extracted: dependency install trees
# (already handled by the SCA layer) and VCS metadata are pure noise.
_ARTIFACT_EXCLUDED_DIRS = ("node_modules", "vendor", ".git")


def _slugify_image(image: str) -> str:
    """Turn a docker image reference into a safe directory name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", image.split("@")[0].replace("/", "_").replace(":", "_"))


def _docker_image_metadata(image: str) -> dict:
    """Return the interesting subset of `docker inspect` Config for an image.

    Never raises: returns {} on any failure (logged)."""
    try:
        result = subprocess.run(
            ["docker", "inspect", image],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0 or not result.stdout.strip():
            logging.warning(f"docker inspect failed for '{image}': {result.stderr.strip()}")
            return {}
        data = json.loads(result.stdout)
        config = (data or [{}])[0].get("Config", {})
        keys = ["Env", "WorkingDir", "Entrypoint", "Cmd", "ExposedPorts", "User", "Labels"]
        return {k: config.get(k) for k in keys if config.get(k) is not None}
    except (json.JSONDecodeError, subprocess.SubprocessError, TimeoutError) as e:
        logging.warning(f"Failed to inspect image '{image}': {e}")
        return {}


def _in_excluded_dir(path: str) -> bool:
    """True if a container path lives inside a dependency/VCS tree that we
    never extract nor index (node_modules, vendor, .git). Those are already
    covered by the SCA layer and would dominate both the artifacts and the
    filesystem index without adding signal."""
    return any(f"/{d}/" in f"/{path}" for d in _ARTIFACT_EXCLUDED_DIRS)


def _should_extract(member_path: str) -> bool:
    """True if a container path matches the curated artifact patterns.

    Skips dependency install trees (``node_modules``, ``vendor``) entirely:
    those manifests are already covered by the SCA layer, and extracting them
    would flood the artifacts directory with hundreds of near-identical files.
    """
    if _in_excluded_dir(member_path):
        return False
    return bool(_ARTIFACT_RE.search(member_path))


def _is_catchall_candidate(path: str, size: int, workdir: str) -> bool:
    """True for a small text-like config file under WORKDIR that the curated
    patterns did not match, e.g. an app config with a nonstandard name.

    Deliberately conservative: bounded size, non-source extension, outside
    dependency/VCS trees, and confined to the app's working directory (guarded
    so a root or empty WORKDIR disables the pass instead of scanning /).
    """
    if size <= 0 or size > ARTIFACT_CATCHALL_MAX_BYTES:
        return False
    if not workdir or workdir.strip("/") == "":
        return False
    if _in_excluded_dir(path):
        return False
    suffix = Path(path).suffix.lower()
    if suffix in _ARTIFACT_SOURCE_EXTS or suffix in _ARTIFACT_BINARY_EXTS:
        return False
    lower = path.lower()
    if lower.endswith(_ARTIFACT_CATCHALL_NOISE_EXTS):
        return False
    if any(seg in f"/{lower}" for seg in _ARTIFACT_CATCHALL_NOISE_DIRS):
        return False
    root = workdir.rstrip("/").lstrip("/")
    p = path.lstrip("/")
    return p == root or p.startswith(root + "/")


def extract_container_artifacts(images: list[str]) -> dict:
    """Create an ephemeral snapshot of each built container image.

    For every image a throwaway container is created (never started), its
    filesystem is exported and streamed: matching files are extracted to
    ``<target>/.cache/container_artifacts/<image-slug>/rootfs/``, non-dependency
    members are recorded (sorted, bounded) in ``filesystem_index.txt``, and
    image metadata
    (ENV / WORKDIR / ENTRYPOINT / CMD / EXPOSE / USER / LABELS) is written to
    ``image_metadata.json`` alongside an ``extraction_summary.json``.

    Deliberately never raises: failures are logged and the image is skipped.
    Returns a summary dict ``{image: {"extracted": n, "files": [...]}}``.
    """
    artifacts_root = settings.cache_dir / "container_artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}

    for image in images:
        slug = _slugify_image(image)
        image_dir = artifacts_root / slug
        rootfs_dir = image_dir / "rootfs"

        try:
            # Remove any stale snapshot from a previous run so it always
            # reflects the current build.
            subprocess.run(["docker", "rm", "-f", slug], capture_output=True, text=True)
        except FileNotFoundError:
            logging.warning("docker is not installed or not in PATH. Skipping container artifact extraction.")
            return summary
        except Exception as e:
            logging.warning(f"Failed to remove stale container '{slug}': {e}")

        create = subprocess.run(
            ["docker", "create", "--name", slug, image],
            capture_output=True,
            text=True,
        )
        if create.returncode != 0:
            logging.warning(f"docker create failed for image '{image}': {create.stderr.strip()}")
            continue

        try:
            metadata = _docker_image_metadata(image)
            workdir = (metadata or {}).get("WorkingDir")
            _extract_container_export(slug, image_dir, rootfs_dir, workdir)
            image_dir.mkdir(parents=True, exist_ok=True)
            if metadata:
                with open(image_dir / "image_metadata.json", "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
            summary[image] = {"slug": slug, "dir": str(image_dir)}
            logging.info(
                f"Container artifacts for '{image}' written to {image_dir} "
                f"(metadata keys: {sorted(metadata.keys())})."
            )
        finally:
            try:
                subprocess.run(["docker", "rm", "-f", slug], capture_output=True, text=True)
            except Exception:
                pass

    # Remove snapshots for images that are no longer built (e.g. after a
    # compose service is removed) so the tool never serves stale configs.
    current_slugs = {info["slug"] for info in summary.values()}
    for existing in artifacts_root.glob("*"):
        if existing.is_dir() and existing.name not in current_slugs:
            logging.info(f"Removing stale artifact snapshot '{existing.name}'.")
            shutil.rmtree(existing)

    return summary


def _extract_container_export(cid: str, image_dir: Path, rootfs_dir: Path,
                             workdir: str | None = None) -> dict:
    """Stream `docker export <cid>` and write extracted files + fs index.

    ``workdir`` enables the catch-all pass for small text config files under the
    app's working directory that the curated patterns miss.
    Returns ``{"extracted": n, "skipped": [paths...]}``. Never raises; on
    failure returns empty stats with a logged warning.
    """
    extracted: list[str] = []
    catchall: list[str] = []
    skipped: list[str] = []
    total_bytes = 0
    index_entries: list[str] = []

    try:
        proc = subprocess.Popen(
            ["docker", "export", cid],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        logging.warning("docker is not installed or not in PATH.")
        return {"extracted": 0, "skipped": []}
    except Exception as e:
        logging.warning(f"Failed to export container '{cid}': {e}")
        return {"extracted": 0, "skipped": []}

    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for member in tar:
                # Normalize to a root-relative path (docker export emits
                # absolute member names) so stored/indexed/displayed paths are
                # all uniform, e.g. `app/next.config.mjs`.
                path = member.name.lstrip("/")
                # Guard against malicious tar members escaping the rootfs dir.
                if ".." in path.split("/"):
                    continue

                # Record interesting entries for the fs index. Dependency
                # install trees (node_modules/vendor) and VCS metadata are
                # skipped: they would dominate the index (and the read cost)
                # without adding signal (already covered by the SCA layer).
                indexed = not _in_excluded_dir(path)

                if member.isdir():
                    if indexed:
                        index_entries.append(f"d\t0\t{path.rstrip('/')}")
                    continue
                if member.issym():
                    if indexed:
                        index_entries.append(f"l\t{member.size}\t{path}")
                    continue
                if not member.isfile():
                    if indexed:
                        index_entries.append(f"?\t0\t{path}")
                    continue

                if indexed:
                    index_entries.append(f"f\t{member.size}\t{path}")

                # Only exact-match files ever count toward the caps below, so
                # oversized irrelevant binaries do not bloat the skipped list.
                catchall_hit = False
                if not _should_extract(path):
                    if _is_catchall_candidate(path, member.size, workdir):
                        catchall_hit = True
                    else:
                        continue
                if (
                    len(extracted) >= ARTIFACT_MAX_FILES
                    or total_bytes + member.size > ARTIFACT_MAX_TOTAL_BYTES
                    or member.size > ARTIFACT_MAX_FILE_BYTES
                ):
                    skipped.append(path)
                    continue

                target = rootfs_dir / path
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    extracted.append(path)
                    if catchall_hit:
                        catchall.append(path)
                    total_bytes += member.size
                except (OSError, EOFError, tarfile.TarError) as e:
                    skipped.append(path)
                    logging.debug(f"Failed to extract '{path}': {e}")
    except (tarfile.TarError, OSError) as e:
        logging.warning(f"Failed to stream export of container '{cid}': {e}")
    finally:
        if proc.stdout:
            proc.stdout.close()
        try:
            proc.wait(timeout=30)
        except Exception:
            # If the stream was aborted mid-read, docker blocks writing to a
            # full, unread pipe and never exits on its own: kill it so we do
            # not leak an orphaned `docker export` process.
            try:
                proc.kill()
            except Exception:
                pass

    image_dir.mkdir(parents=True, exist_ok=True)

    # Bound the index size: sort, then cap (record whether we truncated so the
    # listing tool can tell the reviewer the index is partial).
    index_sorted = sorted(index_entries)
    index_truncated = len(index_sorted) > ARTIFACT_MAX_INDEX_ENTRIES
    if index_truncated:
        index_sorted = index_sorted[:ARTIFACT_MAX_INDEX_ENTRIES]

    try:
        with open(image_dir / "filesystem_index.txt", "w", encoding="utf-8") as f:
            f.write("# type\tsize\tpath (container filesystem index)\n")
            f.write("\n".join(index_sorted))
    except OSError as e:
        logging.warning(f"Failed to write filesystem index: {e}")

    try:
        with open(image_dir / "extraction_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {"extracted": extracted, "catchall": catchall, "skipped": skipped,
                 "total_extracted_bytes": total_bytes,
                 "index_entries": len(index_sorted), "index_truncated": index_truncated},
                f, indent=2,
            )
    except OSError as e:
        logging.warning(f"Failed to write extraction summary: {e}")

    if skipped or catchall:
        logging.info(
            f"Container '{cid}': extracted {len(extracted)} files "
            f"({len(catchall)} via WORKDIR catch-all), skipped "
            f"{len(skipped)} ({','.join(skipped[:5])}{'...' if len(skipped) > 5 else ''})."
        )
    return {"extracted": len(extracted), "skipped": skipped}


def get_container_artifacts_root() -> Path:
    """Root directory holding per-image container artifact snapshots."""
    return settings.cache_dir / "container_artifacts"


def get_canonical_id(record):
    """Extracts the underlying CVE ID from OSV record's metadata."""

    # Check standard OSV 'aliases' or 'upstream' fields
    for field in ["aliases", "upstream"]:
        for alias in record.get(field, []):
            if alias.startswith("CVE-"):
                return alias

    # Check if the ID itself embeds the CVE; fall back to the record ID.
    m = re.search(r"CVE-20\d{2}-\d+", record.get("id", ""))
    if m:
        return m.group(0)
    return record.get("id", "UNKNOWN")


def cvss_v3_base_score(vector: str | None) -> float | None:
    """Calculate a CVSS v3.x base score (0.0-10.0) from a CVSS vector.

    OSV commonly stores the CVSS vector rather than a numeric score, and the
    reporter uses this to score each finding deterministically instead of
    trusting model arithmetic. Returns None when the vector is missing,
    malformed, or from another CVSS generation.
    """
    if not isinstance(vector, str) or not vector.startswith("CVSS:3."):
        return None

    metrics = {}
    for part in vector.split("/")[1:]:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        metrics[key] = value

    required = ("AV", "AC", "PR", "UI", "S", "C", "I", "A")
    if any(key not in metrics for key in required):
        return None

    weights = {
        "AV": {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2},
        "AC": {"L": 0.77, "H": 0.44},
        "UI": {"N": 0.85, "R": 0.62},
        "C": {"N": 0.0, "L": 0.22, "H": 0.56},
        "I": {"N": 0.0, "L": 0.22, "H": 0.56},
        "A": {"N": 0.0, "L": 0.22, "H": 0.56},
    }
    try:
        av = weights["AV"][metrics["AV"]]
        ac = weights["AC"][metrics["AC"]]
        ui = weights["UI"][metrics["UI"]]
        confidentiality = weights["C"][metrics["C"]]
        integrity = weights["I"][metrics["I"]]
        availability = weights["A"][metrics["A"]]
        scope_changed = metrics["S"] == "C"
        pr = {
            "U": {"N": 0.85, "L": 0.62, "H": 0.27},
            "C": {"N": 0.85, "L": 0.68, "H": 0.5},
        }["C" if scope_changed else "U"][metrics["PR"]]
    except KeyError:
        return None

    impact_subscore = 1 - ((1 - confidentiality) * (1 - integrity) * (1 - availability))
    if impact_subscore <= 0:
        return 0.0

    if not scope_changed:
        impact = 6.42 * impact_subscore
    else:
        impact = 7.52 * (impact_subscore - 0.029) - 3.25 * (impact_subscore - 0.02) ** 15
    exploitability = 8.22 * av * ac * pr * ui
    raw_score = min(impact + exploitability, 10.0)
    if scope_changed:
        raw_score = min(1.08 * (impact + exploitability), 10.0)
    return min(10.0, (int(raw_score * 10 + 0.999999) / 10))


def is_high_severity(record: dict) -> bool:
    """Return whether an OSV-normalized record meets the HIGH threshold."""
    label = str(record.get("severity_label") or "").upper()
    if label:
        return label in {"HIGH", "CRITICAL"}
    return (cvss_v3_base_score(record.get("cvss_vector")) or 0.0) >= 7.0


def cvss_severity_label(score: float | None) -> str:
    """Map a CVSS base score to its v3 qualitative severity rating."""
    if score is None:
        return "Not assessed"
    if score >= 9.0:
        return "Critical"
    if score >= 7.0:
        return "High"
    if score >= 4.0:
        return "Medium"
    if score > 0.0:
        return "Low"
    return "None"


def deduplicate_cves(vulns: list[dict]) -> list[dict]:
    """
    Extracts unique vulnerabilities by canonical ID and keeps up to 3 distinct
    descriptions available for each.

    Descriptions differing only trivially (whitespace, casing, or one fully
    contained in the other) are collapsed to the longest representative, so the
    analyzer never sees near-identical copies. The `details` field remains the
    single longest/best description (backward compatible); the new
    `descriptions` list carries up to 3 distinct descriptions, longest-first,
    as input for the CVE analyzer.

    Besides {id, details, descriptions, package}, best-effort enrichment fields
    are carried forward when present in the OSV record:
    - fixed_version: first `fixed` event across affected version ranges.
    - cwe_ids: database_specific.cwe_ids (list) when the OSV entry classifies them.
    - severity_label: database_specific.severity when available.
    - cvss_vector: the first CVSS v3 vector when available.
    """
    best_records = {}

    def normalize(text: str) -> str:
        return re.sub(r"[^0-9a-z]+", "", (text or "").lower())

    def collapse_descriptions(descriptions: list[str]) -> list[str]:
        """Dedupe near-identical texts: sorted longest-first, a description is
        kept only if no already-kept (longer or equal) one fully contains it."""
        kept: list[str] = []
        for d in sorted(descriptions, key=len, reverse=True):
            if not d:
                continue
            norm = normalize(d)
            if not any(norm == normalize(k) or norm in normalize(k) for k in kept):
                kept.append(d)
        return kept[:3]

    def extract_fixed_version(record: dict) -> Optional[str]:
        for affected in record.get("affected", []):
            for rng in affected.get("ranges", []):
                for event in rng.get("events", []):
                    if event.get("fixed"):
                        return event["fixed"]
        return None

    def extract_cwe_ids(record: dict) -> list[str]:
        return record.get("database_specific", {}).get("cwe_ids", []) or []

    def extract_severity_label(record: dict) -> Optional[str]:
        severity = record.get("database_specific", {}).get("severity")
        if isinstance(severity, str) and severity.strip():
            return severity.strip().upper()
        return None

    def extract_cvss_vector(record: dict) -> Optional[str]:
        vectors = record.get("severity", []) or []
        for entry in vectors:
            if not isinstance(entry, dict):
                continue
            vector = entry.get("score")
            if entry.get("type", "").upper().startswith("CVSS_V3") and isinstance(vector, str):
                return vector
        return None

    for vuln in vulns:
        if not vuln.get("details"):
            continue

        canonical_id = get_canonical_id(vuln)
        current_details = vuln.get("details", "")
        packages = [
            affected.get("package", {}).get("name", "unknown")
            for affected in vuln.get("affected", [])
        ]

        if canonical_id not in best_records:
            best_records[canonical_id] = {
                "id": canonical_id,
                "details": current_details,
                "descriptions": [current_details] if current_details else [],
                "package": packages[0] if packages else "unknown",
                "fixed_version": extract_fixed_version(vuln),
                "cwe_ids": extract_cwe_ids(vuln),
                "severity_label": extract_severity_label(vuln),
                "cvss_vector": extract_cvss_vector(vuln),
            }
            continue

        record = best_records[canonical_id]
        if current_details:
            record["descriptions"].append(current_details)
            record["descriptions"] = collapse_descriptions(record["descriptions"])

        # `details` always mirrors the longest kept description (backward compat).
        best = record["descriptions"][0] if record["descriptions"] else ""
        if best and best != record["details"] and len(best) > len(record["details"]):
            record["details"] = best
            record["original_osv_id"] = vuln.get("id")

        # Enrichment is best-effort: backfill any missing field from ANY
        # advisory for this canonical CVE — never gate it on the `details`
        # replacement above, so a field carried only by a shorter advisory
        # (e.g. a GHSA entry's cwe_ids) is not silently dropped.
        if not record.get("fixed_version"):
            record["fixed_version"] = extract_fixed_version(vuln)
        if not record.get("cwe_ids"):
            record["cwe_ids"] = extract_cwe_ids(vuln)
        if not record.get("severity_label"):
            record["severity_label"] = extract_severity_label(vuln)
        if not record.get("cvss_vector"):
            record["cvss_vector"] = extract_cvss_vector(vuln)

    return list(best_records.values())


def safe_cache_filename(filename: str, max_bytes: int = 240) -> str:
    """Keep a cache filename within filesystem component length limits.

    Path separators are never legal inside a single filesystem component: a
    dependency node label containing '/' (e.g. 'dependency:golang.org/x/text')
    would otherwise make cache() silently create nested directories instead of
    one file.
    """
    filename = filename.replace("/", "_").replace("\\", "_").replace("\x00", "_")
    if len(filename.encode("utf-8")) <= max_bytes:
        return filename

    suffix = Path(filename).suffix
    digest = hashlib.md5(filename.encode("utf-8")).hexdigest()
    prefix_bytes = max_bytes - len(suffix.encode("utf-8")) - len(digest) - 1
    prefix = filename.encode("utf-8")[:max(0, prefix_bytes)].decode("utf-8", "ignore")
    return f"{prefix}-{digest}{suffix}"


def cache(file: Path, action: str, content: dict = {}) -> Optional[dict]:
    if action == "read":
        if not file.exists():
            return

        try:
            with open(file, "r") as f:
                cached_data = json.load(f)
            logging.debug(f"Loaded note from cache ({file}).")
            return cached_data
        except json.JSONDecodeError:
            logging.warning(f"Cache file {file} corrupted. Re-generating...")

    elif action == "write":
        if not file.parent.exists():
            file.parent.mkdir(parents=True)

        try:
            with open(file, "w") as f:
                json.dump(content, f, indent=2)
            logging.debug(f"Saved cache file {file}.")
        except Exception as e:
            logging.warning(f"Failed to write cache file {file}: {e}")

    else:
        logging.error(f"Unknown action: {action}")


def _content_hash_cache(subdir: str, prefix: str, content: dict, result: Optional[dict] = None) -> Optional[dict]:
    """Read (``result`` is None) or write one content-hash cache entry under
    ``.cache/<subdir>/<prefix>_<md5(content)>.json``. Callers must keep the
    hashed ``content`` payload byte-identical across stages so entries neither
    miss spuriously nor collide."""
    digest = hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()
    cache_file = settings.cache_dir / subdir / safe_cache_filename(f"{prefix}_{digest}.json")
    if result is None:
        return cache(cache_file, "read")
    cache(cache_file, "write", result)
    return None


def cache_reviewer(node_id: str, report: dict, updated_vuln: Optional[dict] = None) -> Optional[dict]:
    """Read (``updated_vuln`` is None) or write a reviewer outcome cache entry.

    Keyed by (node_id, report content hash); shared by the normal
    ``submit_evaluation`` path and the loop-fallback path so both land in the
    same ``.cache/reviewer/`` namespace.

    Prompt/schema changes do NOT bust entries: selective re-adjudication is
    done by deleting the individual ``<prefix>_<hash>.json`` files (grep the
    directory for the record's ``vuln_id``). Delete the whole directory only
    for a full re-review pass."""
    return _content_hash_cache("reviewer", node_id, report, updated_vuln)


def cache_validator(report: dict, peer_payloads: Optional[list] = None, updated_vuln: Optional[dict] = None) -> Optional[dict]:
    """Read (``updated_vuln`` is None) or write a validator outcome cache entry.

    Keyed by (vuln_id, content hash of the report plus the injected
    ``peer_payloads`` — for a ``chained`` record the validator's verdict depends
    on the proven peer payloads, which are NOT part of the report itself). The
    live ``sandbox_url`` is deliberately excluded so results survive across runs
    despite docker reassigning the port. Shared by ``mark_validation_complete``,
    ``ask_for_context``, and the loop-fallback path so all three land in the
    same ``.cache/validator/`` namespace.
    """
    vuln_id = (report or {}).get("vuln_id") or "Unknown"
    peers = sorted(
        (p for p in (peer_payloads or []) if isinstance(p, dict)),
        key=lambda p: p.get("vuln_id", ""),
    )
    return _content_hash_cache(
        "validator", vuln_id, {"report": report, "peer_payloads": peers}, updated_vuln
    )


def cache_integration_auditor(report: dict, peers: Optional[list] = None, updated_vuln: Optional[dict] = None) -> Optional[dict]:
    """Read (``updated_vuln`` is None) or write an integration-auditor outcome cache entry.

    Keyed by (vuln_id, content hash of the report plus the ``confirmed_vulns``
    peer list the auditor may chain with). Peers are sorted by vuln_id so the
    hash is order-independent. Shared by ``submit_integration_audit``, the
    deterministic no-peers ``unchainable`` resolution, and the loop-fallback
    path so all land in the same ``.cache/integration_auditor/`` namespace.
    """
    vuln_id = (report or {}).get("vuln_id") or "Unknown"
    peers_sorted = sorted(
        (p for p in (peers or []) if isinstance(p, dict)),
        key=lambda p: p.get("vuln_id", ""),
    )
    return _content_hash_cache(
        "integration_auditor", vuln_id, {"report": report, "confirmed_vulns": peers_sorted}, updated_vuln
    )


def cache_reporter(report: dict, finding: Optional[dict] = None) -> Optional[dict]:
    """Read (``finding`` is None) or write a per-vulnerability reporter outcome.

    Keyed on the content hash of the single reportable record (including its
    ``poc_payload``/``execution_logs``, so a changed validator proof busts the
    entry). The reporter runs once per vulnerability, so one cache file per
    record. A hit returns the stored ``ReporterFinding`` dict.
    """
    vuln_id = (report or {}).get("vuln_id") or "Unknown"
    if finding is None:
        cached = _content_hash_cache("reporter", vuln_id, report or {}, None)
        if isinstance(cached, dict) and isinstance(cached.get("finding"), dict):
            return cached["finding"]
        return None
    _content_hash_cache("reporter", vuln_id, report or {}, {"finding": finding})
    return None


def _extract_namespace_aliases(source_file: str, target_namespace: str) -> Optional[set[str]]:
    """Discover the identifiers aliased from a namespace in the import/use
    statements of ``source_file`` (plus the namespace itself).

    Parses the file once per (file, namespace); the resulting tree is discarded
    immediately — only the small alias set is kept, and only for the duration of
    the aggregate pass (see ``AGGREGATE_MEMO_ALIASES``). Returns ``None`` when
    the file cannot be read or its language is unsupported.
    """
    full_text = read_file_text(source_file)
    if full_text is None:
        return None
    ext = Path(source_file).suffix.lower()
    if ext not in LANGUAGE_MAP:
        return None

    aliases = {target_namespace}
    parser = tree_sitter.Parser(LANGUAGE_MAP[ext])
    full_code_bytes = masked_source_for_parsing(full_text, source_file).encode("utf-8")
    try:
        full_tree = parser.parse(full_code_bytes)

        def extract_aliases(node: tree_sitter.Node):
            node_type = node.type.lower()
            # Check if this node is an import/use statement
            if any(kw in node_type for kw in ["import", "use", "require", "include"]):
                text = full_code_bytes[node.start_byte:node.end_byte].decode("utf-8")

                # If this import statement pulls from our target namespace
                if target_namespace in text:
                    # Extract all identifiers within this statement as potential aliases
                    def get_identifiers(n: tree_sitter.Node):
                        if len(n.children) == 0:
                            n_type = n.type.lower()
                            if "identifier" in n_type or "name" in n_type:
                                val = full_code_bytes[n.start_byte:n.end_byte].decode("utf-8")
                                if val != target_namespace:
                                    aliases.add(val)
                        for c in n.children:
                            get_identifiers(c)

                    get_identifiers(node)
            else:
                for child in node.children:
                    extract_aliases(child)

        extract_aliases(full_tree.root_node)
    except Exception as e:
        logging.warning(f"Could not extract aliases from {source_file}: {e}")

    return aliases


def _namespace_aliases(source_file: str, target_namespace: str) -> Optional[set[str]]:
    """Memoized per-run access to ``_extract_namespace_aliases`` so a single
    (file, namespace) is parsed at most once across the whole aggregate pass."""
    key = (source_file, target_namespace)
    if key not in AGGREGATE_MEMO_ALIASES:
        AGGREGATE_MEMO_ALIASES[key] = _extract_namespace_aliases(source_file, target_namespace)
    return AGGREGATE_MEMO_ALIASES[key]


def _folded_node_code(node_id: str, node_map=None, sub_nodes_index=None) -> Optional[str]:
    """Memoized per-run default-mode folded code for a node (via
    ``get_node_code``). Avoids re-reading/re-parsing the same node's file for
    every (node, namespace) usage check."""
    if node_id not in AGGREGATE_MEMO_FOLDED:
        AGGREGATE_MEMO_FOLDED[node_id] = get_node_code(
            node_id, node_map=node_map, sub_nodes_index=sub_nodes_index
        )
    return AGGREGATE_MEMO_FOLDED[node_id]


def uses_namespace_in_ast(node_id: str, target_namespace: str,
                          node_map: Optional[dict] = None,
                          sub_nodes_index: Optional[dict] = None) -> bool:
    """
    Checks if a specific namespace (or its imported symbols) is used within a node's AST.
    """
    source_code = _folded_node_code(node_id, node_map, sub_nodes_index)
    if not source_code:
        return False

    target_node = (node_map or {}).get(node_id)
    if target_node is None:
        graph = settings.graph
        graph_data = get_cached_graph_data(graph)
        if not graph_data:
            return False
        target_node = next((node for node in graph_data.get("nodes", []) if node.get("id") == node_id), None)
    if not target_node or not target_node.get("source_file"):
        logging.error(f"Node '{node_id}' does not have a valid source file mapped.")
        return False

    source_file = target_node.get("source_file")
    ext = Path(source_file).suffix.lower()

    if ext not in LANGUAGE_MAP:
        logging.warning(f"Unsupported extension '{ext}' for AST parsing on node '{node_id}'.")
        return False

    # Track the namespace and any symbols imported from it (e.g., 'g', 'request', 'Blueprint')
    aliases = _namespace_aliases(source_file, target_namespace)
    if aliases is None:
        aliases = {target_namespace}

    # Parse the specific node's folded code to check for actual usage
    parser = tree_sitter.Parser(LANGUAGE_MAP[ext])
    source_bytes = source_code.encode("utf-8")
    tree = parser.parse(source_bytes)

    def walk(node: tree_sitter.Node) -> bool:
        # Ignore import statements in the folded snippet to strictly verify actual usage
        if any(keyword in node.type.lower() for keyword in ["import", "include", "use_declaration"]):
            return False

        # If it's a leaf node, check if its text matches the namespace OR any of its extracted aliases
        if len(node.children) == 0:
            if "comment" not in node.type.lower() and "string" not in node.type.lower():
                token_text = source_bytes[node.start_byte:node.end_byte].decode("utf-8")
                if token_text in aliases:
                    return True

        for child in node.children:
            if walk(child):
                return True

        return False

    return walk(tree.root_node)


def get_node_code(node_id: str, raw: bool = False, reviewer_mode: bool = False,
                  node_map: Optional[dict] = None,
                  sub_nodes_index: Optional[dict] = None) -> str | None:
    graph = settings.graph
    graph_data = get_cached_graph_data(graph)

    # Find the target node (precomputed map when available, else linear scan)
    target_node = (node_map or {}).get(node_id)
    if target_node is None:
        target_node = next((node for node in graph_data.get("nodes", []) if node.get("id") == node_id), None)
    if not target_node:
        logging.debug(f"Node ID '{node_id}' not found in graph.")
        return None

    source_file_path = target_node.get("source_file")
    if not source_file_path:
        logging.error(f"Node '{node_id}' does not have a source file mapped.")
        return None

    source_file = graph.parent.parent / Path(source_file_path)
    source_location = target_node.get("source_location")
    file_type = target_node.get("file_type")

    if not source_file.exists():
        logging.error(f"Source file '{source_file}' not found on disk.")
        return None

    # Handle standard text/document files
    if file_type == "document" or not source_location:
        content = read_file_text(source_file_path)
        if content is None:
            if source_file.exists():
                return f"Error: '{source_file}' is binary."
            logging.error(f"Source file '{source_file}' not found on disk.")
            return None
        return content

    source_content = read_file_text(source_file_path)
    if source_content is None:
        logging.error(f"Source file '{source_file}' not found on disk or unreadable.")
        return None
    source_bytes = masked_source_for_parsing(source_content, source_file_path).encode("utf-8")

    # Get target start line
    try:
        target_start_line = int(source_location.replace("L", ""))
    except ValueError:
        return None

    is_file_node = (target_start_line == 1 and target_node.get("label", "") == source_file.name)

    lang = LANGUAGE_MAP.get(source_file.suffix)
    if not lang:
        return source_content if is_file_node else None

    try:
        parser = tree_sitter.Parser(lang)
        tree = parser.parse(source_bytes)

        grammar = AST_GRAMMAR_MAP.get(source_file.suffix, {})
        body_node_types = grammar.get("body_node", ["block", "compound_statement", "declaration_list", "statement_block", "class_body"])
        if isinstance(body_node_types, str):
            body_node_types = [body_node_types]

        # Helper to find AST node containing a body starting on a specific line
        def find_ast_node_with_body(line_idx):
            candidates = []
            def walk(n):
                if n.start_point[0] == line_idx:
                    candidates.append(n)
                for c in n.children:
                    if c.start_point[0] <= line_idx <= c.end_point[0]:
                        walk(c)
            walk(tree.root_node)

            for cand in candidates:
                for c in cand.children:
                    if c.type in body_node_types:
                        return cand, c
            return candidates[0] if candidates else None, None

        target_line_idx = target_start_line - 1

        if is_file_node:
            target_ast_node = tree.root_node
            target_body_node = None
        else:
            target_ast_node, target_body_node = find_ast_node_with_body(target_line_idx)
            if not target_ast_node:
                return source_content

        if raw:
            return source_bytes[target_ast_node.start_byte:target_ast_node.end_byte].decode("utf-8")

        # Find all sub-nodes in the graph mapped to this file (precomputed
        # index when available, otherwise a full graph scan).
        sub_nodes = []
        file_index = (sub_nodes_index or {}).get(source_file_path)
        if file_index is not None:
            target_node_id = target_node.get("id")
            sub_nodes = [(line_, nid) for line_, nid in file_index if nid != target_node_id]
        else:
            for n in graph_data.get("nodes", []):
                if n.get("id") == target_node.get("id"):
                    continue

                if n.get("source_file") == source_file_path and n.get("source_location"):
                    try:
                        n_start_line = int(n["source_location"].replace("L", ""))
                        sub_nodes.append((n_start_line, n.get("id")))
                    except ValueError:
                        continue

        # Map sub-nodes to their AST bodies and filter based on your new rule
        sub_nodes.sort(key=lambda x: x[0])
        ranges_to_prune = []

        for n_start_line, child_id in sub_nodes:
            child_ast_node, child_body_node = find_ast_node_with_body(n_start_line - 1)
            if not child_body_node or not child_ast_node:
                continue

            sub_start = child_body_node.start_byte
            sub_end = child_body_node.end_byte

            # Rule logic: If querying a specific node, DO NOT prune the target itself
            # or any ancestor/container enclosing the target (e.g. keep the parent class open).
            if not is_file_node:
                # Another graph node aliasing the same start line resolves to the
                # target's own AST body; compare body-to-body so it is never pruned.
                is_target = (
                    target_body_node is not None
                    and sub_start == target_body_node.start_byte
                    and sub_end == target_body_node.end_byte
                )
                is_ancestor = (sub_start <= target_ast_node.start_byte and sub_end >= target_ast_node.end_byte)

                # Check for exact target match
                if is_target:
                    continue
                # Check if it's an ancestor (like the enclosing Class)
                if is_ancestor:
                    continue

            ranges_to_prune.append((child_body_node.start_byte, child_body_node.end_byte, child_id))

        # Filter out nested overlapping ranges
        ranges_to_prune.sort(key=lambda x: x[0])
        filtered_ranges = []
        last_end_byte = -1

        for start_b, end_b, child_id in ranges_to_prune:
            if start_b >= last_end_byte:
                filtered_ranges.append((start_b, end_b, child_id))
                last_end_byte = end_b

        # Reconstruction: Always return the ENTIRE file content now!
        start_boundary = 0
        end_boundary = len(source_bytes)
        comment = grammar.get("comment", "//")

        result_chunks = []
        last_idx = start_boundary

        for start_byte, end_byte, child_id in filtered_ranges:
            if start_byte < start_boundary or end_byte > end_boundary:
                continue

            result_chunks.append(source_bytes[last_idx:start_byte].decode("utf-8"))
            if reviewer_mode:
                stripped_note = f"[Body omitted: use read_source_code with node_id '{child_id}' to read this content]"
            else:
                stripped_note = "[Body omitted: This function is out of scope for the current target and is evaluated by a peer agent. Assume its implementation is secure.]"
            result_chunks.append(f"\n    {comment} {stripped_note}\n")
            last_idx = end_byte

        result_chunks.append(source_bytes[last_idx:end_boundary].decode("utf-8"))

        return "".join(result_chunks)

    except Exception as e:
        logging.warning(f"Tree-sitter failed on '{source_file}': {e}")
        return source_content


# --- Node triage: decide if a graph node deserves LLM-based vulnerability scanning ---
#
# All per-language node-type data this triage consumes (scan signals, import /
# definition / name node types, magic methods, pure-type detection sets, and the
# PHP fragment wrap) lives in languages.py; what follows is the generic
# algorithm that reads those constants.

_SECURITY_KEYWORDS: tuple[str, ...] = (
    "verify", "auth", "authenticate", "authorize", "permission", "secret", "token",
    "password", "passwd", "credential", "tls", "ssl", "private_key", "privatekey",
    "api_key", "apikey", "csrf", "jwt", "session", "cookie", "role", "admin",
    "sudo", "root", "privilege", "debug", "trust", "allow", "bypass", "skip", "disable",
)
_CRITICAL_SUBSTRINGS: tuple[str, ...] = (
    "secret", "password", "passwd", "token", "credential", "privatekey", "apikey", "csrf",
)


def _walk(node: tree_sitter.Node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


def _has_node_type(root: tree_sitter.Node, types: set[str]) -> bool:
    for n in _walk(root):
        if n.type in types:
            return True
    return False


def _is_security_name(name: str) -> bool:
    n = name.lower().strip().strip("()")
    if not n:
        return False

    # Separator-delimited keywords and prefix matches (covers snake_case and camelCase)
    for kw in _SECURITY_KEYWORDS:
        if re.search(rf"(^|[_\-./]){re.escape(kw)}([_\-./]|$)", n) or n.startswith(kw):
            return True

    # Substring fallback for the most dangerous identifiers (e.g. mySecret, apiKey)
    for kw in _CRITICAL_SUBSTRINGS:
        if kw in n:
            return True

    return False


def _has_security_names(root: tree_sitter.Node, label: str) -> bool:
    if _is_security_name(label):
        return True

    for n in _walk(root):
        if n.type in NAME_NODE_TYPES:
            text = n.text.decode()
            text = re.split(r"[=:]", text, 1)[0]
            for tok in re.split(r"[^A-Za-z0-9_]+", text):
                tok = tok.strip().strip("$")
                if tok and _is_security_name(tok):
                    return True
    return False


def _body_of(def_node: tree_sitter.Node):
    for c in def_node.children:
        if c.type in ("block", "statement_block", "class_body", "compound_statement", "declaration_list"):
            return c
    return None


def _is_docstring(stmt: tree_sitter.Node) -> bool:
    return bool(stmt.children) and all(c.type == "string" for c in stmt.children)


def _is_empty_body(body: tree_sitter.Node) -> bool:
    if body is None:
        return False
    if body.type == "block":  # Python: `pass` and docstrings do not count as logic
        for stmt in body.children:
            if stmt.type == "pass_statement":
                continue
            if stmt.type == "expression_statement" and _is_docstring(stmt):
                continue
            return False
        return True
    # Brace-based bodies: only named children count (`{`, `}` are anonymous tokens)
    return not any(c.is_named for c in body.children)


def _unwrap_root(root: tree_sitter.Node) -> tree_sitter.Node:
    """Returns the single top-level definition node if the parsed unit contains exactly one.

    When tree-sitter parses a raw fragment (e.g. a method body), it wraps it in a
    module/program. If that unit holds exactly one definition, unwrap to it so skeleton
    detection applies. Files with many top-level definitions are left intact.
    """
    defs = [n for n in _walk(root)
            if n.type in DEFINITION_TYPES and n.type != "decorated_definition"]
    return defs[0] if len(defs) == 1 else root


def _is_empty_skeleton(root: tree_sitter.Node, ext: str) -> bool:
    node = _unwrap_root(root)
    if node.type not in DEFINITION_TYPES:
        return False
    if node.type in ("interface_declaration", "type_alias_declaration", "enum_declaration", "type_alias_statement"):
        return False
    body = _body_of(node)
    return _is_empty_body(body)


def _has_substantive_docstring(root: tree_sitter.Node) -> bool:
    for n in _walk(root):
        if n.type == "string":
            text = n.text.decode().strip()
            if len(text) >= 20 or n.start_point[0] != n.end_point[0]:
                return True
    return False


def _is_magic_method(label: str, ext: str) -> bool:
    n = label.lower().strip().strip("()")
    return n in MAGIC_METHODS.get(ext, set())


def _has_field_defaults(root: tree_sitter.Node) -> bool:
    for n in _walk(root):
        if n.type in ("assignment", "property_element", "public_field_definition", "property_signature", "enum_assignment"):
            if "=" in n.text.decode():
                return True
    return False


def _pydantic_base(class_node: tree_sitter.Node) -> bool:
    for c in class_node.children:
        if c.type in ("argument_list", "base_clause"):
            for b in _walk(c):
                if b.type in ("identifier", "attribute"):
                    text = b.text.decode().lower()
                    if "basemodel" in text or "pydantic" in text or text.endswith("model"):
                        return True
    return False


def _is_pydantic_like(root: tree_sitter.Node) -> bool:
    for n in _walk(root):
        if n.type == "class_definition" and _pydantic_base(n):
            return True
        if n.type == "decorator" and "validator" in n.text.decode().lower():
            return True
        if n.type == "assignment" and n.text.decode().split("=")[0].strip() == "model_config":
            return True
    return False


def _class_is_field_only(class_node: tree_sitter.Node) -> bool:
    body = _body_of(class_node)
    if body is None:
        return False

    has_field = False
    for stmt in body.children:
        if stmt.type == "pass_statement":
            continue
        if stmt.type in ("expression_statement", "assignment"):
            has_field = True
            continue
        return False

    if not has_field:
        return False
    if _has_node_type(class_node, {"call", "function_definition", "lambda", "if_statement",
                                   "for_statement", "while_statement", "try_statement",
                                   "with_statement", "match_statement"}):
        return False
    return True


def _is_pure_type(root: tree_sitter.Node, ext: str) -> bool:
    signals = SCAN_SIGNAL_TYPES.get(ext, set())
    imports = IMPORT_TYPES.get(ext, set())

    # Type-declaration branch (js/ts family): the node must declare types and
    # hold nothing executable/definitional beyond imports.
    type_constructs = PURE_TYPE_CONSTRUCTS.get(ext)
    if type_constructs is not None:
        if not _has_node_type(root, type_constructs):
            return False
        forbidden = (signals - imports) | PURE_TYPE_FORBIDDEN_TYPES.get(ext, set())
        return not _has_node_type(root, forbidden)

    # Python branch: no behavioral node at all, plus a type alias statement or
    # a field-only (dataclass/pydantic shape) class.
    behavioral = BEHAVIORAL_NODE_TYPES.get(ext)
    if behavioral is not None:
        if _has_node_type(root, behavioral):
            return False
        if _has_node_type(root, TYPE_ALIAS_NODE_TYPES.get(ext, set())):
            return True
        for n in _walk(root):
            if n.type == "class_definition" and _class_is_field_only(n):
                return True
            if n.type == "decorated_definition":
                if any(c.type == "class_definition" and _class_is_field_only(c) for c in n.children):
                    return True
        return False

    # PHP branch: interfaces without runtime signals, or property-only classes.
    if ext == ".php":
        # `signals - imports` === the old `signals - imports -
        # {"namespace_use_declaration"}`: IMPORT_TYPES[".php"] is exactly that
        # one node type.
        runtime = signals - imports
        if _has_node_type(root, PHP_INTERFACE_TYPES):
            return not _has_node_type(root, runtime)
        for n in _walk(root):
            if n.type == "class_declaration":
                body = _body_of(n)
                if body is not None and _has_node_type(n, PHP_PROPERTY_TYPES):
                    if not _has_node_type(n, PHP_METHOD_TYPES):
                        return True
        return False

    return False


def _is_config_only(root: tree_sitter.Node, ext: str) -> bool:
    if _has_node_type(root, DEFINITION_TYPES):
        return False
    imports = IMPORT_TYPES.get(ext, set())
    forbidden = SCAN_SIGNAL_TYPES.get(ext, set()) - imports
    if _has_node_type(root, forbidden):
        return False
    return len(root.children) > 0


def _has_regex_literal(root: tree_sitter.Node) -> bool:
    if _has_node_type(root, {"regex"}):
        return True
    for n in _walk(root):
        if n.type in ("identifier", "name", "property_identifier", "variable_name"):
            text = n.text.decode().lower()
            if "regex" in text or text.endswith("_re") or text.endswith("pattern"):
                return True
    return False


def _has_object_literal(root: tree_sitter.Node) -> bool:
    return _has_node_type(root, {"dictionary", "object", "array_creation_expression"})


def _count_signal_nodes(root: tree_sitter.Node, ext: str, limit: int) -> int:
    signal_types = SCAN_SIGNAL_TYPES.get(ext, set())
    count = 0
    for n in _walk(root):
        if n.type in signal_types:
            count += 1
            if count >= limit:
                return count
    return count


def _node_code_is_worth_scanning(source_code: str, label: str, ext: str, min_signals: int = 1) -> bool:
    """Core triage on raw source. Returns True when the node cannot be inspected."""
    lang = LANGUAGE_MAP.get(ext)
    if not lang or ext not in SCAN_SIGNAL_TYPES:
        return True

    # Raw fragments of wrapped languages are re-prefixed before parsing (see
    # languages.FRAGMENT_WRAP): PHP method/class fragments (as returned by
    # get_node_code) omit the `<?php` tag, which tree-sitter needs to avoid
    # parsing everything as plain text.
    insert, detect = FRAGMENT_WRAP.get(ext, ("", ""))
    if insert and not source_code.lstrip().startswith(detect):
        source_code = insert + source_code

    try:
        tree = tree_sitter.Parser(lang).parse(source_code.encode("utf-8"))
    except Exception as e:
        logging.warning(f"tree-sitter failed on node '{label}': {e}")
        return True

    root = tree.root_node

    # Rule 1: Pure Types & Interfaces -> keep only with defaults, validation, or security names
    if _is_pure_type(root, ext):
        return (_has_field_defaults(root)
                or _is_pydantic_like(root)
                or _has_security_names(root, label))

    # Rule 2: Empty Skeletons & No-Ops -> keep only with security/magic names or docstrings
    if _is_empty_skeleton(root, ext):
        return (_is_security_name(label)
                or _is_magic_method(label, ext)
                or _has_substantive_docstring(root))

    # Rule 3: Primitive Constants & Configs -> keep only with security names, regex, or objects
    if _is_config_only(root, ext):
        return (_has_security_names(root, label)
                or _has_regex_literal(root)
                or _has_object_literal(root))

    # Default: executable signal count (or a runtime validation wrapper)
    return _is_pydantic_like(root) or _count_signal_nodes(root, ext, min_signals) >= min_signals


# --------------------------------------------------------------------------
# File/path-level scan exclusion
#
# Complements the per-node code-signal filter ``is_node_worth_scanning`` with a
# whole-path relevance gate: nodes/code whose ``source_file`` matches any of
# these patterns are dropped before the expensive LLM stages (explorer fan-out,
# CVE keyword corpus, contract verifier) and blocked from reviewer file reads.
# This lets a full repo (including third-party trees, tests, docs) be scanned
# without manually pruning non-relevant paths first.
# --------------------------------------------------------------------------

# Directory fragments dropped by default (matched as any path component).
_DEFAULT_EXCLUDE_DIRS = {
    # dependency / third-party install trees
    "vendor", "node_modules", "third_party", "thirdparty", "external",
    "site-packages", "bower_components",
    # build / cache / VCS / tooling noise
    ".git", ".cache", ".next", "dist", "build", "__pycache__", ".venv",
    "venv", "env", "graphify-out", ".idea", ".vscode", ".gradle", "target",
    # non-app-source trees the user typically trims by hand
    "tests", "__tests__", "spec", "specs", "docs", ".github", ".gitlab",
}

# Basename globs dropped by default (docs + test files).
_DEFAULT_EXCLUDE_NAME_GLOBS = (
    "*.md", "*.markdown", "*.txt", "*.rst", "*.adoc", "*.rdoc",
    "*.test.*", "*.spec.*", "test_*", "*_test.*", "*_spec.*",
)


def _parse_exclude_patterns(patterns: list[str]) -> tuple[set[str], list[str]]:
    """Split ``scan_exclude_paths`` into (dir fragments, full-path globs).

    A bare token (no '/' and no '*') is treated as a directory fragment to
    match against any path component. Anything else is a fnmatch glob matched
    against the relative path (trailing '/' expands to '/**').
    """
    dirs: set[str] = set()
    globs: list[str] = []
    for raw in patterns or []:
        pat = raw.strip()
        if not pat:
            continue
        if "/" not in pat and "*" not in pat:
            dirs.add(pat)
        else:
            if pat.endswith("/"):
                globs.append(pat + "**")
            else:
                globs.append(pat)
    return dirs, globs


def is_path_excluded(source_file: str) -> bool:
    """Return True if ``source_file`` should be skipped during scanning.

    Matching is relative to ``settings.app_path``. A path is excluded when any
    of its components is a configured directory fragment, or its relative path
    fnmatch-matches a configured glob, or its basename fnmatch-matches a
    configured name glob. Honors the built-in defaults unless
    ``settings.scan_exclude_defaults`` is False.
    """
    if not source_file:
        return False

    try:
        rel = str(Path(source_file).resolve().relative_to(settings.app_path.resolve()))
    except (ValueError, OSError):
        rel = str(Path(source_file).as_posix())
    rel_posix = rel
    parts = Path(rel).parts
    name = parts[-1] if parts else ""

    user_dirs, user_globs = _parse_exclude_patterns(getattr(settings, "scan_exclude_paths", []))

    dir_fragments = _DEFAULT_EXCLUDE_DIRS | user_dirs if getattr(settings, "scan_exclude_defaults", True) else user_dirs
    if any(d in parts for d in dir_fragments):
        return True

    for pat in user_globs:
        if fnmatch.fnmatch(rel_posix, pat):
            return True

    name_globs = _DEFAULT_EXCLUDE_NAME_GLOBS if getattr(settings, "scan_exclude_defaults", True) else ()
    for pat in name_globs:
        if fnmatch.fnmatch(name, pat):
            return True

    return False


def is_node_worth_scanning(node_id: str, min_signals: int = 1) -> bool:
    """Decides whether a graph node deserves LLM-based vulnerability scanning.

    Uses tree-sitter to classify the node:
    - Pure types/interfaces are dropped unless they set default values, use runtime
      validation (Pydantic), or carry security-sensitive names.
    - Empty skeletons/no-ops are dropped unless security-named, magic methods, or
      (Python) carrying substantive docstrings.
    - Flat primitive-constant configs are dropped unless security-sensitive names,
      regex literals, or (nested) object literals are present.
    - Remaining nodes are kept when they contain >= `min_signals` executable signals
      (function calls, imports, string interpolation, control flow).

    Unparseable nodes (unsupported extension, parse failure) are kept.
    Dependency manifest/lockfile nodes are always dropped: they are already
    handled by the SCA layer (osv-scanner).
    """
    graph_data = get_cached_graph_data(settings.graph)
    target_node = next((node for node in graph_data.get("nodes", []) if node.get("id") == node_id), None)
    if not target_node or not target_node.get("source_file"):
        return False

    if Path(target_node["source_file"]).name in MANIFEST_NAMES:
        logging.debug(f"Skipping dependency manifest node {node_id} ({target_node['source_file']}).")
        return False

    source_code = get_node_code(node_id, raw=True)
    if not source_code:
        return False

    ext = Path(target_node["source_file"]).suffix.lower()
    label = target_node.get("label", "")
    return _node_code_is_worth_scanning(source_code, label, ext, min_signals)


def extract_imports(source_code: str, file_path: str) -> list[str]:
    """Uses tree-sitter and AST_GRAMMAR_MAP to reliably extract base module imports."""
    imports = set()

    suffix = Path(file_path).suffix
    lang = LANGUAGE_MAP.get(suffix)
    grammar = AST_GRAMMAR_MAP.get(suffix)

    if not lang or not grammar or "import_query" not in grammar:
        return []

    # Callers may pass the raw SFC text; mask non-<script> regions first so the
    # TS grammar (and its import_query) only sees the script body.
    source_code = masked_source_for_parsing(source_code, file_path)

    parser = tree_sitter.Parser(lang)
    tree = parser.parse(bytes(source_code, "utf8"))

    query = lang.query(grammar["import_query"])

    if hasattr(query, "captures"):
        raw_captures = query.captures(tree.root_node)
    else:
        cursor = tree_sitter.QueryCursor(query)
        raw_captures = cursor.captures(tree.root_node)

    # Extract nodes safely
    if isinstance(raw_captures, dict):
        import_nodes = raw_captures.get("import", [])
    else:
        import_nodes = [node for node, name in raw_captures if name == "import"]

    # Process the nodes
    for node in import_nodes:
        full_module = node.text.decode("utf8").strip()

        # Clean up PHP modifiers (function and const)
        if full_module.startswith("function "):
            full_module = full_module[9:]
        elif full_module.startswith("const "):
            full_module = full_module[6:]

        base_module = full_module.split(grammar["import_separator"])[0]
        base_module = base_module.strip("'\" ")

        if base_module:
            imports.add(base_module)

    return list(imports)


def index_file(filepath: str | Path) -> list[dict]:
    # Convert to a Path object and get the extension
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext not in LANGUAGE_MAP or ext not in SYMBOL_QUERIES:
        return [] # Unsupported language

    language = LANGUAGE_MAP[ext]
    query_code = SYMBOL_QUERIES[ext]

    # Parse the file (read_bytes() replaces the 'with open()' block). For .vue
    # SFCs, mask non-<script> regions first so the TS symbol queries only see
    # the script body; newlines are preserved so start/end lines stay
    # SFC-accurate and get_definition can read them back from the real file.
    parser = tree_sitter.Parser(language)
    if ext == ".vue":
        tree = parser.parse(
            masked_source_for_parsing(
                path.read_text(encoding="utf-8", errors="replace"), path
            ).encode("utf-8")
        )
    else:
        tree = parser.parse(path.read_bytes())

    # Execute the language-specific query
    query = tree_sitter.Query(language, query_code)
    cursor = tree_sitter.QueryCursor(query)
    matches = cursor.matches(tree.root_node)

    symbol_index = {}

    def _as_list(nodes):
        return [nodes] if nodes and not isinstance(nodes, list) else nodes

    # Standardized extraction loop
    for match in matches:
        captures = match[1] 

        # Method captures
        class_nodes = _as_list(captures.get("class_name"))
        parent_nodes = _as_list(captures.get("parent_class"))
        method_name_nodes = _as_list(captures.get("method_name"))
        method_body_nodes = _as_list(captures.get("method_body"))

        # Function captures
        function_name_nodes = _as_list(captures.get("function_name"))
        function_body_nodes = _as_list(captures.get("function_body"))

        # Scenario A: It's a class method
        if class_nodes and method_name_nodes and method_body_nodes:
            raw_class = class_nodes[0].text
            raw_method = method_name_nodes[0].text
            class_name = raw_class.decode('utf8') if raw_class else "Unknown"
            method_name = raw_method.decode('utf8') if raw_method else "Unknown"

            # Safely extract the parent class if it exists
            parent_name = None
            if parent_nodes:
                raw_parent = parent_nodes[0].text
                parent_name = raw_parent.decode('utf8') if raw_parent else None

            symbol_name = f"{class_name}::{method_name}"
            if symbol_name not in symbol_index or (parent_name and not symbol_index[symbol_name]["parent"]):
                symbol_index[symbol_name] = {
                    "type": "method",
                    "name": symbol_name,
                    "class": class_name,
                    "parent": parent_name,
                    "method": method_name,
                    "start_line": method_body_nodes[0].start_point[0] + 1,
                    "end_line": method_body_nodes[0].end_point[0] + 1,
                    "filepath": str(path)
                }

        # Scenario B: It's a standalone function
        elif function_name_nodes and function_body_nodes:
            raw_func = function_name_nodes[0].text
            func_name = raw_func.decode('utf8') if raw_func else "Unknown"

            if func_name not in symbol_index:
                symbol_index[func_name] = {
                    "type": "function",
                    "name": func_name,
                    "start_line": function_body_nodes[0].start_point[0] + 1,
                    "end_line": function_body_nodes[0].end_point[0] + 1,
                    "filepath": str(path)
                }

    return list(symbol_index.values())
