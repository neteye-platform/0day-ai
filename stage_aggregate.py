"""Aggregate-demands stage: route explorer/CVE demands to target nodes.

Pure Python (no LLM) except the optional embedding dedup: explorer notes become
upstream/downstream demands routed to callers/callees, CVE analyses become
cve_assumption demands or upgrade-only hypotheses, then near-duplicates are
merged before the contract-verifier fan-out."""

import logging
import re
from collections import defaultdict
from pathlib import Path

import settings
from dedup import deduplicate_demands
from run_stats import as_dict, get_embedder
from stage_cve import _normalize_cwe_ids
from state import MasterState
from utils import (
    clear_aggregate_caches,
    extract_imports,
    get_cached_graph_data,
    get_node_code,
    read_file_text,
    resolve_node_id,
    scan_codebase_for_keywords,
    uses_namespace_in_ast,
)


def build_caller_map(graph_data: dict):
    """Map each node to the source nodes of its incoming ``calls`` edges.

    Only genuine call edges qualify: structural relations carry no parameter
    contracts, so routing demands through them delivers contracts to a
    class/file node whose sibling call sites are invisible to get_node_code."""
    callers_map = defaultdict(list)
    for edge in graph_data.get("links", []):
        if edge.get("relation") == "calls":
            callers_map[edge.get("target")].append(edge.get("source"))
    return callers_map


def build_import_map(graph_data: dict):
    """Map every graph node to the import namespace set of its source file.

    Each unique file is read and parsed exactly once and the set is shared by
    all its nodes."""
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
    """Precompute ``{source_file: [(start_line, node_id), ...]}`` once so
    get_node_code folds don't rescan the graph for same-file siblings."""
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
    """Normalize an AnalysisNote's upstream/downstream entries into flat
    ``{"direction", "target", "description"}`` demand dicts."""
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


_TARGET_ARGS_RE = re.compile(r'\(.*?\)')


def _route_downstream(demand: dict, current_node_id: str, graph_data: dict, grouped_demands: defaultdict) -> None:
    """Route a downstream demand to its resolved target node (callee)."""
    target_str = demand.get("target", "")
    desc = demand.get("description")

    # STRIP LLM HALLUCINATIONS: Remove backticks, parentheses, and arguments
    clean_target = _TARGET_ARGS_RE.sub('', target_str).replace('`', '').strip()

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


_CLASS_PROP_WINDOW_LINES = 400
# Requires the `$` so type segments (`private ?string $name`) cannot masquerade as names.
_CLASS_PROP_DECL_RE = re.compile(
    r"\b(?:public|protected|private|var)\b[^;{(=]*\$([A-Za-z_]\w*)"
)
_MEMBER_PARAM_RE = re.compile(r"\$([A-Za-z_]\w*)")


def _build_container_members(graph_data: dict) -> dict[str, dict[str, str]]:
    """``container_id -> {method_name_lower: member_node_id}`` for class-like
    nodes. Same id-derivation as ``utils._repair_call_edges``."""
    from utils import _MEMBER_LABEL_RE
    nodes = {n["id"]: n for n in graph_data.get("nodes", []) if n.get("id")}
    members: dict[str, dict[str, str]] = {}
    for nid, node in nodes.items():
        match = _MEMBER_LABEL_RE.match(node.get("label") or "")
        if not match:
            continue
        mname = match.group(1).lower()
        if not nid.endswith("_" + mname):
            continue
        base = nid[: len(nid) - len(mname) - 1]
        parent = nodes.get(base)
        if (
            not parent
            or parent.get("source_file") != node.get("source_file")
            or _MEMBER_LABEL_RE.match(parent.get("label") or "")
        ):
            continue
        members.setdefault(base, {})[mname] = nid
    return members


def _member_param_names(member_id: str, node_map: dict, memo: dict) -> set | None:
    """Lowercased ``$names`` in a member's signature; ``None`` if unreadable."""
    if member_id in memo:
        return memo[member_id]
    node = node_map.get(member_id) or {}
    src = node.get("source_file")
    loc = str(node.get("source_location") or "")
    names: set | None = None
    if src and loc.startswith("L") and loc[1:].isdigit():
        lines = (read_file_text(src) or "").splitlines()
        start = int(loc[1:]) - 1
        if 0 <= start < len(lines):
            # Cut at `{` or `;`: abstract decls have no brace and would bleed.
            chunk = re.split(r"[{;]", "\n".join(lines[start:start + 12]), 1)[0]
            names = {m.lower() for m in _MEMBER_PARAM_RE.findall(chunk)}
    memo[member_id] = names
    return names


def _class_property_names(container_id: str, node_map: dict, memo: dict) -> set | None:
    """Lowercased class property names; ``None`` if the source is unreadable."""
    if container_id in memo:
        return memo[container_id]
    node = node_map.get(container_id) or {}
    src = node.get("source_file")
    loc = str(node.get("source_location") or "")
    props: set | None = None
    if src and loc.startswith("L") and loc[1:].isdigit():
        lines = (read_file_text(src) or "").splitlines()
        start = int(loc[1:]) - 1
        props = set()
        for line in lines[start:start + _CLASS_PROP_WINDOW_LINES]:
            if re.search(r"\bfunction\s", line):
                break
            props.update(name.lower() for name in _CLASS_PROP_DECL_RE.findall(line))
    memo[container_id] = props
    return props


def _scope_bare_container_demand(current_node_id: str, target_str: str,
                                 container_members: dict, node_map: dict,
                                 param_memo: dict, prop_memo: dict,
                                 callers_map: dict, callers: list):
    """Scope a bare-target demand from a CONTAINER node.

    ``None`` = keep the class-wide broadcast (not a container / unreadable,
    fail open); a list = qualifying callers (empty -> drop; property target ->
    all callers, since the constructor caller owns field population)."""
    member_map = container_members.get(current_node_id)
    if not member_map:
        return None
    name = re.split(r"[:.]|\->", (target_str or "").strip())[-1].lstrip("$").lower()
    if not name:
        return None
    hit_members: list[str] = []
    unreadable = False
    for member_id in member_map.values():
        pnames = _member_param_names(member_id, node_map, param_memo)
        if pnames is None:
            unreadable = True
        elif name in pnames:
            hit_members.append(member_id)
    if hit_members:
        receivers: list[str] = []
        for member_id in hit_members:
            receivers.extend(callers_map.get(member_id, []))
        if not receivers and not getattr(settings, "repair_call_edges", False):
            # Repair off: an empty member-caller set is a graph artifact.
            return list(callers)
        return sorted(set(receivers))
    if unreadable:
        return None
    props = _class_property_names(current_node_id, node_map, prop_memo)
    if props is None or name in props:
        return list(callers)  # property (or unscannable class body): class-wide
    # Defensive: `saveNew($items)` reaching the bare path names a member.
    call_like = re.match(r"[a-z_][a-z0-9_]*(?=\s*\()", name)
    if call_like and call_like.group(0) in member_map:
        return sorted(set(callers_map.get(member_map[call_like.group(0)], [])))
    return []  # declared nowhere on the class: no addressee, drop


def _caller_invokes_symbol(caller_id: str, symbol: str, code_memo: dict,
                           pattern_memo: dict, node_map: dict | None,
                           sub_nodes_index: dict | None) -> bool:
    """True when the caller's own source contains a call to ``symbol``
    (word-boundary ``symbol(``, so ``show`` never matches ``showForm(``).
    The container's call edge proves the caller touches *some* member; this
    proves it touches *this* one."""
    if caller_id not in code_memo:
        code_memo[caller_id] = get_node_code(
            caller_id, node_map=node_map, sub_nodes_index=sub_nodes_index
        ) or ""
    if symbol not in pattern_memo:
        pattern_memo[symbol] = re.compile(rf"(?<![\w$\\]){re.escape(symbol)}\s*\(")
    return bool(pattern_memo[symbol].search(code_memo[caller_id]))


def _route_upstream(demand: dict, current_node_id: str, callers_map: dict, grouped_demands: defaultdict,
                    node_map: dict | None = None, sub_nodes_index: dict | None = None,
                    code_memo: dict | None = None, pattern_memo: dict | None = None,
                    container_members: dict | None = None, param_memo: dict | None = None,
                    prop_memo: dict | None = None) -> None:
    """Route an upstream demand to the callers responsible for its target.

    Bare targets (``$id``) fan out to all callers, scoped to the declaring
    member on class containers (see ``_scope_bare_container_demand``).
    Call-shaped targets reach only callers that invoke that member; a demand
    with no qualifying caller is dropped."""
    target_str = demand.get("target", "")
    desc = demand.get("description")

    callers = callers_map.get(current_node_id, [])

    # Detection runs on the RAW target: args-stripping erases the parens the
    # dot/bare call shapes are keyed on.
    clean_target = _TARGET_ARGS_RE.sub('', target_str or '').replace('`', '').strip()
    head = (target_str or "").replace("`", "").strip().split("->")[-1].strip()
    module, symbol = "", None
    if "::" in head.split("(", 1)[0]:
        module, symbol = head.split("::", 1)
    elif "(" in head:
        call_head = head.split("(", 1)[0].strip()
        if "." in call_head:
            module, symbol = call_head.rsplit(".", 1)
        else:
            module, symbol = "", call_head
    symbol = symbol.strip().split("(", 1)[0].strip() if symbol else None
    module = module.strip() if module else ""
    if symbol and symbol.startswith("$"):
        # Variable / context reference (e.g. `Session::$current`), not a call.
        symbol = None

    if not symbol:
        receivers = callers
        if container_members:
            scoped = _scope_bare_container_demand(
                current_node_id, target_str, container_members, node_map or {},
                param_memo if param_memo is not None else {},
                prop_memo if prop_memo is not None else {},
                callers_map, callers)
            if scoped is not None:
                if not scoped:
                    if callers:
                        logging.info(
                            f"[{current_node_id}] UPSTREAM SCOPE DROP: bare target "
                            f"'{target_str}' resolves to no qualifying caller of the "
                            f"container (of {len(callers)} class caller(s)) — "
                            f"demand dropped: {str(desc)[:120] if desc else ''}"
                        )
                    return
                receivers = scoped
        if receivers:
            for caller_id in receivers:
                grouped_demands[caller_id].append({
                    "source": current_node_id,
                    "type": "explorer_upstream_assumption",
                    "description": desc,
                    "parameter_name": target_str
                })
        else:
            # Expected for never-called methods; not a warning with the
            # calls-only caller map.
            logging.debug(f"[{current_node_id}] UPSTREAM DROP: No callers found in graph for this node.")
        return

    current_node = (node_map or {}).get(current_node_id) or {}
    current_label = str(current_node.get("label", "")).rstrip("()").lower()
    if current_label == symbol.lower() or current_label.endswith("::" + symbol.lower()):
        # The target names the analyzed node itself: its caller edges are
        # already member-precise — plain broadcast, no text filter needed.
        qualified = list(callers)
    else:
        if not module and container_members and symbol.lower() in container_members.get(current_node_id, {}):
            # Bare member call on a container: deliver via the member's edges.
            qualified = list(callers_map.get(container_members[current_node_id][symbol.lower()], []))
        else:
            resolved = resolve_node_id(module, symbol)
            precise_callers = []
            if resolved and resolved != current_node_id:
                resolved_node = (node_map or {}).get(resolved) or {}
                if resolved_node.get("source_file") == current_node.get("source_file"):
                    precise_callers = callers_map.get(resolved, [])
            if precise_callers:
                qualified = list(precise_callers)
            else:
                qualified = [
                    caller_id for caller_id in callers
                    if _caller_invokes_symbol(caller_id, symbol, code_memo, pattern_memo,
                                              node_map, sub_nodes_index)
                ]

    if not qualified:
        # A member contract no caller exercises has no addressee: drop it
        # instead of burning one verifier evaluation per unrelated caller.
        logging.info(
            f"[{current_node_id}] UPSTREAM SCOPE DROP: no caller invokes '{clean_target}' "
            f"(of {len(callers)} caller(s)) — demand dropped: {str(desc)[:120] if desc else ''}"
        )
        return

    for caller_id in qualified:
        grouped_demands[caller_id].append({
            "source": current_node_id,
            "type": "explorer_upstream_assumption",
            "description": desc,
            "parameter_name": target_str
        })


def _route_explorer_notes(notes: list, graph_data: dict, callers_map: dict, grouped_demands: defaultdict,
                          node_map: dict | None = None, sub_nodes_index: dict | None = None) -> list[dict]:
    """Process explorer notes into grouped demands. Returns the updated notes.

    The per-run memos bound the member-scoped lookups: caller code folded at
    most once per node, one regex compiled per symbol."""
    updated_notes = []
    code_memo: dict = {}
    pattern_memo: dict = {}
    # Container bare-target scoping: member index + lazy signature/property scans.
    container_members = (
        _build_container_members(graph_data)
        if getattr(settings, "container_demands_scope_to_members", False)
        else None
    )
    param_memo: dict = {}
    prop_memo: dict = {}

    for note in notes:
        dict_note = as_dict(note)
        current_node_id = dict_note.get("node_id")

        for demand in _note_demands(dict_note):
            direction = demand.get("direction")
            if direction == "downstream":
                _route_downstream(demand, current_node_id, graph_data, grouped_demands)
            elif direction == "upstream":
                _route_upstream(demand, current_node_id, callers_map, grouped_demands,
                                node_map, sub_nodes_index, code_memo, pattern_memo,
                                container_members, param_memo, prop_memo)
            else:
                logging.warning(f"[{current_node_id}] UNKNOWN DIRECTION: '{direction}'. Demand dropped.")

        updated_notes.append(dict_note)

    return updated_notes


def _process_cve_demands(cves: list, node_imports_map: dict, grouped_demands: defaultdict,
                         node_map: dict | None = None,
                         sub_nodes_index: dict | None = None) -> list[dict]:
    """Route CVE analyzer outputs by fix_category.

    application_mitigation -> cve_assumption demands routed to every node that
    imports AND uses the affected namespace; upgrade_only -> one direct
    hypothesis per CVE anchored to a synthetic `dependency:<package>` node
    (returned for the `vulnerabilities` channel). The inverted imports index
    makes the cost scale with matching nodes, not codebase size."""
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
                # OSV-suggested CWEs, forwarded as a verifier hint on FAILED.
                "cwe_ids": _normalize_cwe_ids(record.get("cwe_ids")),
            })
            matched_any = True

        if not matched_any:
            logging.warning(f"[CVE DROP] {source_cve} for '{target_import}' matched 0 nodes in the graph.")

    return hypotheses


def _build_cve_hypothesis(record: dict, imports_by_namespace: dict, node_map: dict | None = None,
                          sub_nodes_index: dict | None = None) -> dict | None:
    """Build one hypothesis for an upgrade-only CVE, anchored to a synthetic
    `dependency:<package>` node with usage-site hints appended."""
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

    # Usage hints: nodes importing AND using the namespace, capped for prompt size.
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
    """Drop CVE records whose ``required_keywords`` are all absent from the
    codebase (exact, case-sensitive substring, any code file).

    Empty/missing keyword lists are kept (fail-open for pre-field caches).
    For application_mitigation records only: aggregate_demands_node exempts
    upgrade_only records, whose synthetic dependency nodes are not locatable
    via app-source substrings."""
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
    """AND-join barrier output: convert all notes + CVE records into grouped
    demands, upgrade-only hypotheses and the merge-ready notes."""
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
        # hypotheses live on synthetic dependency nodes and are adjudicated by
        # the dependency reviewer via container artifacts, not app source.
        upgrade_only = [r for r in cves if r.get("fix_category") == "upgrade_only"]
        cves = filter_cve_demands_by_keywords(
            [r for r in cves if r.get("fix_category") != "upgrade_only"]
        )
        cves = upgrade_only + cves
        logging.info(
            f"Processing {len(notes)} notes and {len(cves)} CVE demands "
            f"({len(upgrade_only)} upgrade-only exempted from the keyword pre-filter)."
        )

        updated_notes = _route_explorer_notes(notes, graph_data, callers_map, grouped_demands,
                                              node_map, sub_nodes_index)

        cve_hypotheses = _process_cve_demands(cves, node_imports_map, grouped_demands, node_map, sub_nodes_index)
        logging.info(f"Emitted {len(cve_hypotheses)} upgrade-only CVE hypothesis(es) directly into the vulnerabilities channel.")

        # Merge near-duplicate demands per target before the verifier fan-out
        # (one verifier evaluation per demand). cve_assumption demands are
        # never merged; upstream demands merge only on exact identity within
        # the same (callee, parameter). Fails open to exact-key dedup.
        embedder = get_embedder(
            settings.demand_dedup_enabled and settings.semantic_dedup_enabled,
            "Demand dedup",
            "exact-normalized dedup only",
        )
        grouped_demands = deduplicate_demands(
            grouped_demands,
            embedder,
            settings.semantic_dedup_threshold,
            disk_cache_dir=settings.cache_dir / "demand_embeddings",
        )

        total_demands = sum(len(d) for d in grouped_demands.values())
        logging.info(f"Summary: Grouped {total_demands} total demands across {len(grouped_demands)} target nodes.")

        return {
            "grouped_demands": dict(grouped_demands),
            # `notes` is an operator.add-reduced channel: returning
            # updated_notes would re-append the notes just consumed.
            "notes": [],
            "vulnerabilities": cve_hypotheses,
        }
    finally:
        # Release memoized source text so it never persists into the LLM stages.
        clear_aggregate_caches()
