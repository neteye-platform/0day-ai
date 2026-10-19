"""Deterministic extraction of trust-boundary edges for the Edge Traversal Agent.

The graphify AST graph only captures syntactic relations (``calls``, ``imports``,
``references``, ``contains``) and is blind to message queues, IPC, reverse-proxy
routing, and network calls. Two mechanisms recover the real trust boundaries
without rewriting graphify (see ``boundary_edges.md``):

1. **Direct cross-community AST edges** — ``{(u, v) | (u, v) in E_AST and
   community(u) != community(v) and type in (calls, references, imports)}``,
   pruned of low-signal reference targets (logging/utility/pure-type helpers)
   and scoped (optionally) to at least one endpoint carrying an explorer
   interface note.
2. **Synthesized virtual edges** —
   * *async messaging*: join queue/event dispatch call sites (``send_task``,
     ``.delay()``, ``.publish()``, literal event names) to worker-registration
     entry points (``@app.task(name=..)``, ``@receiver(..)``) on the task/event
     name.
   * *network/IPC*: join HTTP client call sites (``requests.get(...)``,
     ``fetch("...")``, ``axios...``) to web route registrations
     (``@app.route(...)``, express ``app.get(...)``) via URL-pattern matching.
   * *infra-to-app*: join reverse-proxy config locations parsed from the
     container artifacts (nginx ``location {... proxy_pass ...}``, Apache
     ``<Location>``/``ProxyPass``) to app routes by prefix matching.

Every mechanism emits a uniform edge dict
``{source_node, target_node, boundary_type, match_key, source_note, target_note,
transport_artifact, source_label, target_label, source_file, target_file}`` that
the Edge Traversal Agent renders into a batch prompt. Labels/files/notes are
attached once, in ``build_boundary_edges``, after the synthesizers emit bare
``(source, target, boundary_type, match_key, transport_artifact)`` edges.

Everything is deterministic and fails open: unsupported languages / missing
artifacts simply contribute no edges, and the caller never sees an exception.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import settings
from utils import (
    get_container_artifacts_root,
    get_node_code,
    is_path_excluded,
    read_file_text,
)

log = logging.getLogger("boundary_edges")

# Boundary categories, ordered by signal-to-noise (rarest/most architectural
# first). This order drives batch clustering and per-category capping.
BOUNDARY_CATEGORY_ORDER = ("infra", "async_messaging", "network_ipc", "in_process")

# Relations that denote real coupling across an AST edge (as opposed to purely
# structural relations like contains/method/inherits). `imports` is also
# included: in PHP monoliths (front-controller / AJAX entrypoint scripts
# reaching into a shared `src/` class layer) the cross-community import from an unauthenticated
# entrypoint into a privileged service is exactly the composition surface the
# Edge Traversal agent must reason about (see boundary_edges.md). Low-signal
# import/reference targets (exceptions, loggers, utilities, pure types) are
# pruned in _direct_cross_community_edges.
_DIRECT_RELATIONS = ("calls", "references", "imports")

# -----------------------------------------------
# Direct cross-community AST edges
# -----------------------------------------------

# Reference targets whose snippet-level signature is a shared utility / pure
# type rather than an architectural peer. Cross-community references to these
# are noisy (every controller statically references the CMS base classes), so
# they are pruned from the direct edge set.
_LOW_SIGNAL_TARGET_FRAGMENTS = (
    "exception",
    "exceptions",
    "logger",
    "logfile",
    "logging",
    "constant",
    "constants",
    "enum",
    "enums",
    "interface",
    "interfaces",
    "util",
    "utils",
    "helper",
    "helpers",
    "test",
    "tests",
    "spec",
    "fixture",
    "fixtures",
    "migration",
    "migrations",
    "dto",
    "dtos",
)


def _is_low_signal_target(node: dict) -> bool:
    label = str(node.get("label") or "").lower()
    return any(frag in label for frag in _LOW_SIGNAL_TARGET_FRAGMENTS)


def _direct_cross_community_edges(
    graph_data: dict, nodes_by_id: dict, note_ids: set
) -> list[dict]:
    require_note = getattr(settings, "edge_traversal_require_note", True)
    edges: list[dict] = []
    for link in graph_data.get("links", []):
        if link.get("relation") not in _DIRECT_RELATIONS:
            continue
        source_id, target_id = link.get("source"), link.get("target")
        source_node = nodes_by_id.get(source_id)
        target_node = nodes_by_id.get(target_id)
        if not source_node or not target_node:
            continue
        if (
            source_node.get("file_type") != "code"
            or target_node.get("file_type") != "code"
        ):
            continue
        src_comm, tgt_comm = source_node.get("community"), target_node.get("community")
        if src_comm is None or tgt_comm is None or src_comm == tgt_comm:
            continue
        src_file = source_node.get("source_file") or ""
        tgt_file = target_node.get("source_file") or ""
        if is_path_excluded(src_file) or is_path_excluded(tgt_file):
            continue
        if link.get("relation") in ("references", "imports") and _is_low_signal_target(
            target_node
        ):
            continue
        if require_note and source_id not in note_ids and target_id not in note_ids:
            continue
        edges.append(
            _make_edge(
                boundary_type="in_process",
                source_node_id=source_id,
                target_node_id=target_id,
                source_node=source_node,
                target_node=target_node,
                match_key=f"community {src_comm}->{tgt_comm} via {link.get('relation')}",
                transport_artifact=None,
            )
        )
    return edges


def _make_edge(
    *,
    boundary_type,
    source_node_id,
    target_node_id,
    source_node,
    target_node,
    match_key,
    transport_artifact,
) -> dict:
    return {
        "boundary_type": boundary_type,
        "source_node": source_node_id,
        "target_node": target_node_id,
        "match_key": match_key,
        "transport_artifact": transport_artifact,
        "source_label": str(source_node.get("label") or source_node_id),
        "target_label": str(target_node.get("label") or target_node_id),
        "source_file": source_node.get("source_file") or "",
        "target_file": target_node.get("source_file") or "",
    }


def _attach_notes(edges: list[dict], note_map: dict) -> list[dict]:
    """Attach the explorer exit/ingress profiles to each edge (single pass)."""
    for edge in edges:
        edge["source_note"] = (note_map or {}).get(edge["source_node"])
        edge["target_note"] = (note_map or {}).get(edge["target_node"])
    return edges


# -----------------------------------------------
# Entry-point discovery (shared by all virtual-edge synthesizers)
# -----------------------------------------------

# Directory fragments (and interesting file basenames) that host routing /
# worker entry points. The full-codebase entry scan is gated on these so
# massive repos are not fully re-read just to find endpoint declarations.
_ENTRY_GATE_DIR_FRAGMENTS = (
    "ajax",
    "front",
    "api",
    "apiv1",
    "grpc",
    "internal",
    "controller",
    "controllers",
    "route",
    "routes",
    "router",
    "web",
    "handler",
    "handlers",
    "worker",
    "workers",
    "tasks",
    "jobs",
    "consumer",
    "consumers",
    "listener",
    "listeners",
    "middleware",
)
_ENTRY_GATE_BASENAMES = (
    "routes.php",
    "web.php",
    "routing.py",
    "urls.py",
    "api.php",
    "routes.ts",
    "routes.js",
    "app.ts",
    "app.js",
    "server.ts",
    "server.js",
)


def _is_entry_file(file_path: str) -> bool:
    parts = Path(file_path).parts
    if any(frag in parts for frag in _ENTRY_GATE_DIR_FRAGMENTS):
        return True
    return Path(file_path).name in _ENTRY_GATE_BASENAMES


def _function_name_after(text: str, position: int) -> str | None:
    """Resolve the function name defined just after a decorator/statement."""
    window = text[position : position + 700]
    m = re.search(r"\b(?:def|function)\s+([A-Za-z_]\w*)", window)
    return m.group(1) if m else None


def _file_level_node(file_nodes: list[dict]) -> dict | None:
    """Fallback anchor: the module/file skeleton node of a source file."""
    basename = None
    for node in file_nodes:
        label = str(node.get("label") or "")
        source_file = node.get("source_file") or ""
        if basename is None:
            basename = Path(source_file).name
        if label == basename:
            return node
    return file_nodes[0] if file_nodes else None


def _node_for_name(file_nodes: list[dict], name: str | None) -> dict | None:
    if name:
        low = name.lower()
        for node in file_nodes:
            label = str(node.get("label") or "").lower()
            if label == low or label.endswith((f"::{low}", f".{low}", f"->{low}")):
                return node
    return _file_level_node(file_nodes)


def _node_after_declaration(
    file_nodes: list[dict], text: str, position: int, known_name: str | None = None
) -> dict | None:
    """Resolve the graph node defined right after a decorator/statement.

    Falls back to the file-level skeleton node when the decorated function
    cannot be matched (e.g. an unnamed module-scope registration).
    """
    name = known_name or _function_name_after(text, position)
    if name:
        node = _node_for_name(file_nodes, name)
        if node:
            return node
    return _file_level_node(file_nodes)


def _group_nodes_by_file(nodes_by_id: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for node in nodes_by_id.values():
        file_path = node.get("source_file")
        if file_path:
            grouped[file_path].append(node)
    return dict(grouped)


# ---- Async messaging -------------------------------------------------------

_QUEUE_DISPATCH_NAMED_RE = re.compile(
    r"\b(?:send_task|enqueue|publish|produce|basic_publish|publish_to_queue)\s*\(\s*[\"']([^\"']+)[\"']"
)
_QUEUE_DISPATCH_OBJ_RE = re.compile(
    r"\b([A-Za-z_]\w*)\s*\.\s*(?:delay|apply_async)\s*\("
)
# Worker/board registration decorators:  @app.task(name="x")  |  @shared_task
# @receiver(signal)  |  bare  @app.task  followed by  def <name>
_QUEUE_ENTRY_NAMED_RE = re.compile(
    r"@\s*[\w.]+\s*\.\s*(?:task|shared_task|job|actor|receiver)\b"
    r"\(\s*(?:name|queue|signal)\s*=\s*[\"']([^\"']+)[\"']"
)
_QUEUE_ENTRY_BARE_RE = re.compile(
    r"@\s*[\w.]+\s*\.\s*(?:task|shared_task|job|actor)\b[^\n]*\n\s*def\s+([A-Za-z_]\w*)",
    re.MULTILINE,
)


# ---- Internal HTTP / microservice calls -------------------------------------

_HTTP_CLIENT_API_RE = re.compile(
    r"\b(?:requests|httpx)\s*\.\s*(?:get|post|put|patch|delete|options|head)\s*"
    r"\(\s*[fFrRbBuU]?[\"']([^\"' ]{4,})[\"']",
)
_HTTP_CLIENT_REQUEST_RE = re.compile(
    r"\b(?:requests|httpx)\s*\.\s*request\s*\(\s*[\"'][A-Za-z]+[\"']\s*,\s*"
    r"[fFrRbBuU]?[\"']([^\"' ]{4,})[\"']",
)
_HTTP_CLIENT_FETCH_RE = re.compile(
    r"\b(?:axios|fetch)\s*[.(]\s*[fFrRbBuU]?[\"'`]([^\"'`]{4,})[\"'`]",
)
_PY_ROUTE_RE = re.compile(
    r"@\s*(?:app|router|bp|blueprint)\s*\.\s*(?:route|get|post|put|patch|delete|head|options)"
    r"\s*\(\s*[\"']([^\"'][^\"']*)[\"']",
)
_JS_ROUTE_RE = re.compile(
    r"\b(?:app|router|route|server)\s*\.\s*(?:get|post|put|patch|delete|head|options|all)"
    r"\s*\(\s*[\"']([^\"'][^\"']*)[\"']",
)
_DJANGO_PATH_RE = re.compile(r"\bpath\s*\(\s*r?[\"']([^\"'][^\"']*)[\"']")

# Route-parameter tokens split out of a route pattern BEFORE escaping, so each
# parameter class is recognized on the raw pattern text (re.escape would escape
# `{`/`}`/`*`, making a later substitution on the escaped string a no-op).
_PARAM_TOKEN_RE = re.compile(r"(<[^>]+>|\{[^}]*\}|:[A-Za-z_][A-Za-z0-9_]*|\*)")


def _route_to_regex(route: str) -> str:
    """Convert a route pattern with parameter tokens into an anchored regex.

    Delimiters split off the raw route are replaced by ``[^/]+`` (or ``.*`` for
    a bare ``*`` wildcard); every literal segment is ``re.escape``d. Handles
    Flask/Django ``<int:id>``, FastAPI/Starlette/Laravel ``{param}``, Express
    ``:id``, and ``*`` wildcards.
    """
    out: list[str] = []
    for part in _PARAM_TOKEN_RE.split(route):
        if not part:
            continue
        if part == "*":
            out.append(r".*")
        elif _PARAM_TOKEN_RE.fullmatch(part):
            out.append(r"[^/]+")
        else:
            out.append(re.escape(part))
    return "".join(out)


def _url_matches(route: str, path: str) -> bool:
    """Match a client request URL/path against a registered route pattern."""
    if not path.startswith("/"):
        scheme = re.match(r"^[a-z][a-z0-9+.\-]*://[^/]+(/.*)$", path, re.IGNORECASE)
        if not scheme:
            return False
        path = scheme.group(1)
    path = path.split("?")[0].rstrip("/") or "/"
    route = route.rstrip("/") or "/"

    if ":" not in route and "{" not in route and "*" not in route and "<" not in route:
        if path == route:
            return True
        return path.startswith(route + "/")

    return re.fullmatch(_route_to_regex(route), path) is not None


def _scan_entry_declarations(
    nodes_by_id: dict,
) -> tuple[list[tuple[str, str, str]], dict[str, list[str]]]:
    """One pass over the (dir-gated) endpoint files discovering both web routes
    and queue/event entry points.

    Returns ``(routes, queue_entries)`` where ``routes`` is
    ``[(node_id, route_pattern, source_file), ...]`` and ``queue_entries`` maps
    a normalized task/event name to the list of worker node ids registering it.
    Worker/route nodes are resolved to the decorated function when possible.
    """
    grouped = _group_nodes_by_file(nodes_by_id)
    routes: list[tuple[str, str, str]] = []
    queue_entries: dict[str, list[str]] = defaultdict(list)
    for file_path, file_nodes in grouped.items():
        if not _is_entry_file(file_path):
            continue
        text = read_file_text(file_path)
        if not text:
            continue
        for regex in (_PY_ROUTE_RE, _JS_ROUTE_RE, _DJANGO_PATH_RE):
            for match in regex.finditer(text):
                route = match.group(1)
                if len(route.strip("/")) < 2:
                    continue
                node = _node_after_declaration(file_nodes, text, match.end())
                if node and node.get("id"):
                    routes.append((node["id"], route, file_path))
        for match in _QUEUE_ENTRY_NAMED_RE.finditer(text):
            node = _node_after_declaration(file_nodes, text, match.end())
            if node and node.get("id"):
                queue_entries[match.group(1)].append(node["id"])
        for match in _QUEUE_ENTRY_BARE_RE.finditer(text):
            node = _node_after_declaration(
                file_nodes, text, match.end(), known_name=match.group(1)
            )
            if node and node.get("id"):
                queue_entries[match.group(1)].append(node["id"])
    return _dedupe_routes(routes), {k: v for k, v in queue_entries.items()}


def _dedupe_routes(entries: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    seen: set[tuple[str, str]] = set()
    out = []
    for node_id, route, file_path in entries:
        key = (node_id, route)
        if key in seen:
            continue
        seen.add(key)
        out.append((node_id, route, file_path))
    return out


def _dispatch_surface(nodes_by_id: dict, note_ids: set) -> list[tuple[str, dict]]:
    """Nodes whose queue/HTTP client call sites are scanned: everything when
    note-scoping is off, else the noted subset. Computed ONCE and shared by
    both virtual-edge passes (they walked the identical surface)."""
    require_note = getattr(settings, "edge_traversal_require_note", True)
    return [
        (node_id, node)
        for node_id, node in nodes_by_id.items()
        if not require_note or node_id in note_ids
    ]


def _queue_virtual_edges(
    nodes_by_id: dict,
    note_ids: set,
    queue_entries: dict,
    surface: list[tuple[str, dict]] | None = None,
) -> list[dict]:
    edges: list[dict] = []
    for node_id, node in (
        surface if surface is not None else _dispatch_surface(nodes_by_id, note_ids)
    ):
        code = get_node_code(node_id, raw=True)
        if not code:
            continue
        keys: set[str] = set()
        for match in _QUEUE_DISPATCH_NAMED_RE.finditer(code):
            keys.add(match.group(1))
        for match in _QUEUE_DISPATCH_OBJ_RE.finditer(code):
            keys.add(match.group(1))
        if not keys:
            continue
        for key in keys:
            for target_id in queue_entries.get(key, []):
                if target_id == node_id:
                    continue
                target_node = nodes_by_id.get(target_id)
                if not target_node:
                    continue
                edges.append(
                    _make_edge(
                        boundary_type="async_messaging",
                        source_node_id=node_id,
                        target_node_id=target_id,
                        source_node=node,
                        target_node=target_node,
                        match_key=f"event/task '{key}'",
                        transport_artifact=None,
                    )
                )
    return edges


def _http_virtual_edges(
    nodes_by_id: dict,
    note_ids: set,
    route_entries: list[tuple],
    surface: list[tuple[str, dict]] | None = None,
) -> list[dict]:
    edges: list[dict] = []
    for node_id, node in (
        surface if surface is not None else _dispatch_surface(nodes_by_id, note_ids)
    ):
        code = get_node_code(node_id, raw=True)
        if not code:
            continue
        paths: set[str] = set()
        for pattern in (
            _HTTP_CLIENT_API_RE,
            _HTTP_CLIENT_REQUEST_RE,
            _HTTP_CLIENT_FETCH_RE,
        ):
            for match in pattern.finditer(code):
                paths.add(match.group(1))
        if not paths:
            continue
        for path in paths:
            for route_node_id, route, _file in route_entries:
                if route_node_id == node_id or not _url_matches(route, path):
                    continue
                target_node = nodes_by_id.get(route_node_id)
                if not target_node:
                    continue
                edges.append(
                    _make_edge(
                        boundary_type="network_ipc",
                        source_node_id=node_id,
                        target_node_id=route_node_id,
                        source_node=node,
                        target_node=target_node,
                        match_key=f"HTTP {path} -> route {route}",
                        transport_artifact=None,
                    )
                )
    return edges


# ---- Infra-to-app (reverse proxy configuration) -----------------------------

_PROXY_CONFIG_PATTERNS = (
    "nginx.conf",
    "nginx*.conf",
    "*.vhost",
    "httpd.conf",
    "apache2.conf",
    ".htaccess",
    "Caddyfile",
    "traefik.yml",
    "traefik.yaml",
    "traefik.toml",
    "haproxy.cfg",
    "site.conf",
)
_NGINX_LOCATION_RE = re.compile(r"location\s+([^\s{]+)\s*\{(.*?)\}", re.DOTALL)
_APACHE_LOCATION_RE = re.compile(
    r"<Location\s*\"?([^\">]+)\"?>(.*?)</Location>", re.DOTALL
)
_APACHE_PROXYPASS_RE = re.compile(r"ProxyPass\s+([^\s]+)\s+([^\s]+)")
_PROXY_PASS_RE = re.compile(r"proxy_pass\s+https?://[^/]+([^;\s]*);")


def _find_proxy_config_files() -> list[tuple[str, str]]:
    """Return ``(name, text)`` pairs for proxy config files in the artifacts."""
    root = get_container_artifacts_root()
    if not root.exists():
        return []
    results: list[tuple[str, str]] = []
    for rootfs in sorted(p for p in root.rglob("rootfs") if p.is_dir()):
        for file_path in rootfs.rglob("*"):
            if not file_path.is_file():
                continue
            name = file_path.name
            if any(
                fnmatch.fnmatch(name, pattern) for pattern in _PROXY_CONFIG_PATTERNS
            ):
                results.append(
                    (str(file_path.relative_to(rootfs)), _bounded_read(file_path))
                )
    return [r for r in results if r[1]]


def _bounded_read(path: Path, max_bytes: int = 200_000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[:max_bytes]


def _extract_proxy_directives(text: str) -> list[tuple[str, str]]:
    directives: list[tuple[str, str]] = []
    for match in _NGINX_LOCATION_RE.finditer(text):
        prefix = match.group(1).strip()
        body = " ".join(match.group(2).split())
        if prefix and body:
            directives.append((prefix, body))
    for match in _APACHE_LOCATION_RE.finditer(text):
        prefix = match.group(1).strip()
        body = match.group(2).strip()
        if prefix and body:
            directives.append((prefix, body))
    return directives


def _proxy_upstream_path(prefix: str, body: str) -> str | None:
    match = _PROXY_PASS_RE.search(body)
    if match:
        upstream = match.group(1)
        return upstream if upstream.startswith("/") else "/"
    apache = _APACHE_PROXYPASS_RE.search(body)
    if apache:
        return apache.group(1)
    return prefix if prefix.startswith("/") else None


def _infra_edges(
    nodes_by_id: dict, note_ids: set, route_entries: list[tuple]
) -> list[dict]:
    require_note = getattr(settings, "edge_traversal_require_note", True)
    route_entries = [r for r in route_entries if not require_note or r[0] in note_ids]
    if not route_entries:
        return []
    configs = _find_proxy_config_files()
    if not configs:
        log.debug("Edge traversal: no proxy config files found in container artifacts.")
        return []
    edges: list[dict] = []
    for name, text in configs:
        directives = _extract_proxy_directives(text)
        for idx, (prefix, body) in enumerate(directives):
            upstream = _proxy_upstream_path(prefix, body)
            candidates = (
                [prefix, upstream] if upstream and upstream != prefix else [prefix]
            )
            for route_node_id, route, _file in route_entries:
                if not any(_url_matches(route, cand) for cand in candidates):
                    continue
                target_node = nodes_by_id.get(route_node_id)
                if not target_node:
                    continue
                src_id = f"infra:{_slug(name)}#{idx}"
                pseudo = {
                    "label": f"{name} :: {prefix}",
                    "source_file": f"container:{name}",
                }
                edges.append(
                    _make_edge(
                        boundary_type="infra",
                        source_node_id=src_id,
                        target_node_id=route_node_id,
                        source_node=pseudo,
                        target_node=target_node,
                        match_key=f"proxy {prefix} -> route {route}",
                        transport_artifact=f"{prefix}: {body}",
                    )
                )
    return edges


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


# -----------------------------------------------
# Assembly, dedup, capping, clustering
# -----------------------------------------------

_EDGE_SYNTHESIS_SETTING_KEYS = (
    "edge_traversal_direct_enabled",
    "edge_traversal_queue_enabled",
    "edge_traversal_http_enabled",
    "edge_traversal_infra_enabled",
    "edge_traversal_require_note",
    "edge_traversal_max_edges_per_category",
    "scan_exclude_paths",
    "scan_exclude_defaults",
    "repair_call_edges",
)


def _artifacts_fingerprint() -> list[str]:
    """Digest of everything the infra synthesizer reads: each image snapshot's
    extraction summary plus the (path, size, mtime) of every proxy-config
    candidate in its rootfs. Cheap (extracted rootfs dirs are curated, small)."""
    root = get_container_artifacts_root()
    if not root.exists():
        return []
    lines: list[str] = []
    for image_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        summary = image_dir / "extraction_summary.json"
        try:
            lines.append(
                f"{image_dir.name}:summary:{hashlib.md5(summary.read_bytes()).hexdigest()}"
            )
        except OSError:
            lines.append(f"{image_dir.name}:summary:none")
        rootfs = image_dir / "rootfs"
        if not rootfs.is_dir():
            continue
        for file_path in sorted(rootfs.rglob("*")):
            if not file_path.is_file():
                continue
            if not any(
                fnmatch.fnmatch(file_path.name, pat) for pat in _PROXY_CONFIG_PATTERNS
            ):
                continue
            try:
                st = file_path.stat()
                lines.append(
                    f"{file_path.relative_to(rootfs)}:{st.st_size}:{st.st_mtime_ns}"
                )
            except OSError:
                lines.append(f"{file_path.relative_to(rootfs)}:stale")
    return lines


def _synthesis_fingerprint(note_map: dict) -> str:
    """Digest of the deterministic inputs to the edge synthesis: the graph
    file (identity + mtime + size), the edge-traversal and exclusion settings,
    the full note map (it gates and annotates edges), and the container-
    artifact state the infra pass reads. Target source files are read by the
    queue/HTTP passes but deliberately NOT fingerprinted — they are only
    meaningful in sync with graph.json (regenerated together by graphify), and
    this matches the pipeline-wide "source is immutable while its graph is
    current" cache assumption. A digest match replays byte-identical edges for
    a given graph; hand-edited target sources stay invisible to the cache
    until the graph is re-extracted.
    """
    payload: dict = {
        "app_path": str(settings.app_path),
        "settings": {
            k: getattr(settings, k, None) for k in _EDGE_SYNTHESIS_SETTING_KEYS
        },
    }
    try:
        st = settings.graph.stat()
        payload["graph"] = [str(settings.graph), st.st_size, st.st_mtime_ns]
    except OSError:
        payload["graph"] = str(settings.graph)
    payload["notes_md5"] = hashlib.md5(
        json.dumps(note_map or {}, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    payload["artifacts"] = _artifacts_fingerprint()
    return hashlib.md5(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def build_boundary_edges(graph_data: dict, note_map: dict) -> list[dict]:
    """Assemble the full candidate boundary-edge set (deterministic, no LLM).

    The synthesis is pure (graph + notes + settings + artifacts), so its output
    is disk-cached under ``.cache/edge_traversal/synthesis_<fingerprint>.json``
    — previously it was re-paid in full (~minutes on a large repo) on every run
    and even on checkpoint resumes that re-entered the node. ``graph_data`` is
    consumed on the miss path only; the fingerprint pins ``settings.graph``, so
    callers must pass that same process-cached graph (the edge-traversal node
    does)."""
    from utils import cache, safe_cache_filename

    fingerprint = _synthesis_fingerprint(note_map)
    cache_file = (
        settings.cache_dir
        / "edge_traversal"
        / safe_cache_filename(f"synthesis_{fingerprint}.json")
    )
    hit = cache(cache_file, "read")
    if isinstance(hit, dict) and isinstance(hit.get("edges"), list):
        edges = hit["edges"]
        log.info(
            "Edge traversal: reusing cached boundary-edge synthesis: %d edge(s) (%s).",
            len(edges),
            fingerprint[:12],
        )
        return edges

    started = time.monotonic()
    edges = _build_boundary_edges(graph_data, note_map)
    cache(cache_file, "write", {"fingerprint": fingerprint, "edges": edges})
    log.info(
        "Edge traversal: boundary-edge synthesis done in %.1fs (%d edge(s) after capping); cached.",
        time.monotonic() - started,
        len(edges),
    )
    return edges


def _build_boundary_edges(graph_data: dict, note_map: dict) -> list[dict]:
    nodes_by_id = {
        n.get("id"): n
        for n in graph_data.get("nodes", [])
        if n.get("id") and n.get("file_type") == "code"
    }
    note_ids = {str(nid) for nid in (note_map or {})}
    edges: list[dict] = []

    if getattr(settings, "edge_traversal_direct_enabled", True):
        direct = _direct_cross_community_edges(graph_data, nodes_by_id, note_ids)
        edges.extend(direct)
        log.info("Edge traversal: %d direct cross-community AST edge(s).", len(direct))

    route_entries: list[tuple] = []
    queue_entries: dict[str, list[str]] = {}
    if any(
        getattr(settings, f"edge_traversal_{kind}_enabled", True)
        for kind in ("queue", "http", "infra")
    ):
        route_entries, queue_entries = _scan_entry_declarations(nodes_by_id)
        log.info(
            "Edge traversal: discovered %d web route(s) and %d queue/event registration(s).",
            len(route_entries),
            len(queue_entries),
        )

    # Queue and HTTP both regex-scan client call sites over the identical
    # note-scoped node surface; walk it once. (Code extraction stays inside
    # each pass — the per-file tree parse behind it is memoized in utils.)
    surface: list[tuple[str, dict]] | None = None
    if getattr(settings, "edge_traversal_queue_enabled", True) or getattr(
        settings, "edge_traversal_http_enabled", True
    ):
        surface = _dispatch_surface(nodes_by_id, note_ids)

    if getattr(settings, "edge_traversal_queue_enabled", True):
        queued = _queue_virtual_edges(nodes_by_id, note_ids, queue_entries, surface)
        edges.extend(queued)
        log.info("Edge traversal: %d async-messaging virtual edge(s).", len(queued))

    if getattr(settings, "edge_traversal_http_enabled", True):
        http = _http_virtual_edges(nodes_by_id, note_ids, route_entries, surface)
        edges.extend(http)
        log.info("Edge traversal: %d HTTP virtual edge(s).", len(http))

    if getattr(settings, "edge_traversal_infra_enabled", True):
        infra = _infra_edges(nodes_by_id, note_ids, route_entries)
        edges.extend(infra)
        log.info("Edge traversal: %d infra-to-app virtual edge(s).", len(infra))

    _attach_notes(edges, note_map)
    edges = _dedupe_edges(edges)
    edges = _cap_per_category(edges)
    return edges


def _dedupe_edges(edges: list[dict]) -> list[dict]:
    seen: set[tuple[str, str, str, str]] = set()
    out = []
    for edge in edges:
        key = (
            edge.get("boundary_type", ""),
            edge.get("source_node", ""),
            edge.get("target_node", ""),
            edge.get("match_key", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(edge)
    return out


def _cap_per_category(edges: list[dict]) -> list[dict]:
    """Bound the candidate set per category, preferring edges that carry notes
    on both endpoints (strongest exit/ingress profiles), then source-side notes."""
    cap = getattr(settings, "edge_traversal_max_edges_per_category", 150)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for edge in edges:
        grouped[edge.get("boundary_type", "in_process")].append(edge)

    def rank(edge: dict) -> tuple:
        both = int(bool(edge.get("source_note")) and bool(edge.get("target_note")))
        source = int(bool(edge.get("source_note")))
        return (
            -both,
            -source,
            edge.get("source_node", ""),
            edge.get("target_node", ""),
        )

    kept: list[dict] = []
    for category in BOUNDARY_CATEGORY_ORDER:
        members = grouped.get(category, [])
        limited = sorted(members, key=rank)[:cap]
        kept.extend(limited)
        if len(members) > len(limited):
            log.info(
                "Edge traversal: capped '%s' edges at %d (from %d).",
                category,
                len(limited),
                len(members),
            )
    return kept


def cluster_boundary_edges(edges: list[dict]) -> list[list[dict]]:
    """Group surviving edges into homogeneous batches of related crossings.

    One batch per (category, chunk). Batching is strictly deterministic so the
    cache keys and any re-run reproduce identical prompts.
    """
    batch_size = max(1, getattr(settings, "edge_traversal_batch_size", 8))
    max_batches = max(1, getattr(settings, "edge_traversal_max_batches", 12))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for edge in edges:
        grouped[edge.get("boundary_type", "in_process")].append(edge)

    batches: list[list[dict]] = []
    for category in BOUNDARY_CATEGORY_ORDER:
        members = sorted(
            grouped.get(category, []),
            key=lambda e: (
                e.get("source_node", ""),
                e.get("target_node", ""),
                e.get("match_key", ""),
            ),
        )
        for i in range(0, len(members), batch_size):
            if len(batches) >= max_batches:
                break
            batches.append(members[i : i + batch_size])
        if len(batches) >= max_batches:
            break
    return batches


def summarize_boundary_edges(edges: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for edge in edges:
        counts[edge.get("boundary_type", "in_process")] += 1
    return dict(counts)


def boundary_batch_fingerprint(batch: list[dict]) -> str:
    """Stable cache key for a batch: includes the note profiles, labels, files,
    match keys, and transport/artifact text, so any change re-runs the batch."""
    payload = json.dumps(batch, sort_keys=True, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


# -----------------------------------------------
# Prompt rendering for the batch LLM call
# -----------------------------------------------

_NOTE_FIELDS = ("sources", "sinks", "vulns")


def render_note_profile(note: dict) -> str:
    """Render an explorer AnalysisNote as the node's exit/ingress profile."""
    if not note:
        return "(no explorer interface note recorded for this node)"
    note = note if isinstance(note, dict) else getattr(note, "model_dump", dict)()
    lines: list[str] = []
    for key in _NOTE_FIELDS:
        values = note.get(key) or []
        if values:
            compact = " | ".join(str(v)[:160] for v in values)
            lines.append(f"{key}: {compact}")
    for key in ("upstream", "downstream"):
        values = note.get(key) or []
        if values:
            pairs = []
            for entry in values:
                if not isinstance(entry, dict):
                    pairs.append(str(entry)[:160])
                    continue
                target = entry.get("target", "?")
                desc = str(entry.get("description") or "")[:200]
                pairs.append(f"{target} :: {desc}")
            if pairs:
                lines.append(f"{key}: " + " | ".join(pairs))
    return "\n".join(lines) if lines else "(no business interfaces recorded)"


def render_edge_code(node_id: str, limit: int = 1800) -> str:
    """Bounded raw-code excerpt for a boundary endpoint, for the batch prompt."""
    if not node_id or str(node_id).startswith("infra:"):
        return "(no graph source — infra/pseudo endpoint)"
    code = get_node_code(node_id, raw=True)
    if not code:
        return "(no source code excerpt available)"
    if len(code) > limit:
        return code[:limit] + "\n...[truncated]..."
    return code


def render_batch_prompt(batch: list[dict]) -> str:
    """Render one boundary-edge batch into the human prompt for the LLM."""
    sections = []
    for idx, edge in enumerate(batch, 1):
        boundary_type = edge.get("boundary_type", "in_process")
        header = (
            f"### Boundary edge {idx} — {boundary_type}  (match: {edge.get('match_key', '')})\n"
            f"SOURCE (u): {edge.get('source_node')} ({edge.get('source_label')}) "
            f"file={edge.get('source_file')}\n"
            f"TARGET (v): {edge.get('target_node')} ({edge.get('target_label')}) "
            f"file={edge.get('target_file')}"
        )
        source_profile = render_note_profile(edge.get("source_note"))
        target_profile = render_note_profile(edge.get("target_note"))
        artifact = edge.get("transport_artifact")
        artifact_block = (
            f"\nTRANSPORT / INFRASTRUCTURE ARTIFACTS:\n{artifact}" if artifact else ""
        )
        sections.append(
            f"{header}\n"
            f"\nSOURCE EXIT PROFILE:\n{source_profile}\n"
            f"\nSOURCE CODE EXCERPT:\n{render_edge_code(edge.get('source_node'))}\n"
            f"\nTARGET INGRESS PROFILE:\n{target_profile}\n"
            f"\nTARGET CODE EXCERPT:\n{render_edge_code(edge.get('target_node'))}"
            f"{artifact_block}\n"
        )
    return "\n\n".join(sections)
