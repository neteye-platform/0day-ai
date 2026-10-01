from collections import defaultdict
import logging
import re
from pathlib import Path
import tree_sitter
import subprocess
import networkx as nx
import json
import tarfile
import time
import requests
from typing import Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AnyMessage, AIMessage
from schemas import VulnerabilityRecord
import settings
from functools import lru_cache

from languages import LANGUAGE_MAP, AST_GRAMMAR_MAP, SYMBOL_QUERIES, MANIFEST_NAMES


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
    if not dropped_ids:
        return graph_data

    filtered = {
        "nodes": [
            node for node in graph_data.get("nodes", [])
            if node.get("id") not in dropped_ids
        ],
        "links": [],
    }
    filtered["links"] = _strip_links_to_dropped(graph_data, dropped_ids)
    return filtered


def load_code_corpus() -> dict[str, str]:
    """Read the contents of every unique source file whose graph nodes carry
    ``file_type == "code"``.

    Returns ``{source_file: content}`` so callers can search the whole
    codebase in a single pass (the deterministic pre-filter uses this instead
    of re-reading files per CVE). Unreadable, missing, or binary files are
    skipped with a debug log.
    """
    graph_data = get_cached_graph_data(settings.graph)
    code_files = sorted({
        node.get("source_file")
        for node in graph_data.get("nodes", [])
        if node.get("file_type") == "code" and node.get("source_file")
    })

    corpus: dict[str, str] = {}
    for source_file in code_files:
        path = settings.app_path / Path(source_file)
        if not path.exists():
            logging.debug(f"Code corpus: skipping missing file '{source_file}'.")
            continue
        try:
            corpus[source_file] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logging.debug(f"Code corpus: skipping binary file '{source_file}'.")
        except OSError as e:
            logging.debug(f"Code corpus: skipping unreadable file '{source_file}': {e}")

    logging.debug(f"Code corpus: indexed {len(corpus)}/{len(code_files)} code files.")
    return corpus


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
        if not ext or ext in LANGUAGE_MAP and ext in SYMBOL_QUERIES:
            continue
        if source_file not in unsupported[ext]:
            unsupported[ext].append(source_file)
    return dict(unsupported)


def format_node_context(graph_data: dict, node_id: str) -> str:
    """Render 'graphify explain' style context for a node: its summary plus connections.

    Shows the node's label, id, source file/location and community, followed by
    all incoming ('<--') and outgoing ('-->') links with their relation and confidence.
    Returns an empty string if the target node is not found.
    """
    nodes = graph_data.get("nodes", [])
    node_map = {n.get("id"): n for n in nodes}
    target_node = node_map.get(node_id)
    if target_node is None:
        return ""

    # Collect connections touching this node, resolved to neighbor labels.
    connections = []
    for edge in graph_data.get("links", []):
        relation = edge.get("relation")
        confidence = edge.get("confidence")
        if edge.get("target") == node_id:
            neighbor = node_map.get(edge.get("source"))
            arrow = "<--"
        elif edge.get("source") == node_id:
            neighbor = node_map.get(edge.get("target"))
            arrow = "-->"
        else:
            continue
        neighbor_label = neighbor.get("label", edge.get("source") or edge.get("target")) if neighbor else (edge.get("source") or edge.get("target"))
        connections.append((arrow, neighbor_label, relation, confidence, str(edge.get("source_location", ""))))

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
    for arrow, neighbor_label, relation, confidence, _ in connections:
        lines.append(f"  {arrow} {neighbor_label} [{relation}] [{confidence}]")

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

    print(f"Loaded Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def merge_vulnerabilities(existing: list[dict], updates: list[dict]) -> list[dict]:
    vuln_map = {}

    # Map existing records
    for vuln in existing:
        vid = vuln.get("vuln_id")
        if vid:
            vuln_map[vid] = vuln

    status_priority = {
        "hypothesis": 0,
        "unreachable": 1,
        "confirmed": 2,
        "false_positive": 3,
        "proven": 4
    }

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

            # --- STATUS UPGRADE: COMPLETELY REPLACE ---
            if status_priority.get(new_status, 0) > status_priority.get(current_status, 0):
                vuln_map[vid] = update

            # --- SAME STAGE: MERGE CONTEXT ---
            # If two parallel agents at the same stage find the same issue
            # (e.g., two explorers finding the same hypothesis), merge the text.
            elif status_priority.get(new_status, 0) == status_priority.get(current_status, 0):
                current = vuln_map[vid]
                curr_desc = current.get("description", "")
                upd_desc = update.get("description", "")

                if upd_desc and upd_desc not in curr_desc:
                    current["description"] = f"{curr_desc}\n\nAdditional context: {upd_desc}"

                vuln_map[vid] = current
        else:
            # --- NEW UNIQUE VULNERABILITY ---
            vuln_map[vid] = update

    return list(vuln_map.values())


def extract_subgraph(G: nx.DiGraph, target_communities: list) -> nx.DiGraph:
    """
    Creates a subgraph containing only the nodes in the target communities,
    plus a 1-hop perimeter of incoming/outgoing connections.
    """
    target_nodes = set()

    # Find all nodes belonging to the assigned communities
    for node_id, data in G.nodes(data=True):
        if str(data.get('community')) in target_communities:
            target_nodes.add(node_id)

    # Include 1-hop neighbors to provide boundary context
    perimeter_nodes = set(target_nodes)
    for node in target_nodes:
        # Add nodes that call into our target community
        perimeter_nodes.update(G.predecessors(node))
        # Add nodes that our target community calls
        perimeter_nodes.update(G.successors(node))

    # Create and return the isolated subgraph
    subgraph = G.subgraph(perimeter_nodes).copy()
    return subgraph


# Tools that takes a lot of context
heavy_tools = ["read_source_code", "read_file", "send_http_request", "search_codebase", "get_definition", "read_container_artifact"]

def compact_tool_history(messages: list[AnyMessage], safe_window: int = 4, threshold: int = 300) -> list[AnyMessage]:
    """
    Compresses heavy tool outputs
    """
    compacted_messages = []

    for i, msg in enumerate(messages):
        # The agent's reasoning remains as its memory
        if isinstance(msg, AIMessage) or isinstance(msg, SystemMessage) or isinstance(msg, HumanMessage):
            compacted_messages.append(msg)
            continue

        # Prune bulky ToolMessages, but leave the most recent turn intact.
        # A safe_window of 4 preserves the last ~2 AI/Tool interaction pairs.
        is_older_message = i < len(messages) - safe_window

        if is_older_message and isinstance(msg, ToolMessage):

            if msg.name in heavy_tools and len(str(msg.content)) > threshold:
                crushed_msg = msg.model_copy(
                    update={"content": f"[PRUNED] Raw {msg.name} data removed to save context. Rely on your subsequent reasoning in the chat history to remember what you found here."}
                )
                compacted_messages.append(crushed_msg)
                continue

        # Keep everything else as-is
        compacted_messages.append(msg)

    return compacted_messages


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


def run_osv_scanner(repo_path: Path) -> list[dict]:
    """Runs osv-scanner on a directory and extracts raw vulnerability records."""
    raw_vulnerabilities = []

    if not repo_path.exists():
        logging.error(f"Input report does not exist.")

    try:
        # Run the scanner recursively (-r) and output as JSON
        result = subprocess.run(
            ["osv-scanner", "-r", "--format", "json", repo_path],
            capture_output=True,
            text=True
        )

        # If stdout is empty, either no vulns were found or it failed before JSON output
        if not result.stdout.strip():
            error = result.stderr
            if error:
                logging.error(f"Error running osv-scanner: {error}")
            return []

        data = json.loads(result.stdout)

        # Extract the vulnerability objects from the osv-scanner JSON schema
        for scan_result in data.get("results", []):
            for package in scan_result.get("packages", []):
                for vuln in package.get("vulnerabilities", []):
                    raw_vulnerabilities.append(vuln)

    except FileNotFoundError:
        print("Error: osv-scanner is not installed or not in PATH.")
    except json.JSONDecodeError:
        print("Error: Could not parse osv-scanner output.")

    return raw_vulnerabilities


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


def build_images(kind: str, path: Path, tag: str) -> list[str]:
    """Build the container image(s) described by a compose file or Dockerfile.

    Returns the built image tag(s). On failure logs the error and returns [].
    """
    try:
        if kind == "compose":
            build = subprocess.run(
                ["docker", "compose", "-f", str(path), "build"],
                capture_output=True,
                text=True
            )
            if build.returncode != 0:
                logging.error(f"docker compose build failed: {build.stderr}")
                return []
            result = subprocess.run(
                ["docker", "compose", "-f", str(path), "config", "--images"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logging.error(f"docker compose config failed: {result.stderr}")
                return []
            images = [img.strip() for img in result.stdout.splitlines() if img.strip()]
            if not images:
                logging.error("docker compose config returned no images.")
            return images

        # Single Dockerfile build
        build = subprocess.run(
            ["docker", "build", "-t", tag, "-f", str(path), str(path.parent)],
            capture_output=True,
            text=True
        )
        if build.returncode != 0:
            logging.error(f"docker build failed: {build.stderr}")
            return []
        return [tag]

    except FileNotFoundError:
        print("Error: docker is not installed or not in PATH.")
        return []


SANDBOX_START_TIMEOUT = 30  # seconds to wait for a sandbox HTTP port to come up


def _published_ports(container_name: str) -> list[int]:
    """Return the host ports published by a container via `docker port`."""
    try:
        result = subprocess.run(
            ["docker", "port", container_name],
            capture_output=True,
            text=True
        )
    except FileNotFoundError:
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


def _probe_http_ports(ports: list[int]) -> str | None:
    """Wait up to ``SANDBOX_START_TIMEOUT`` for any candidate host port to
    answer an HTTP request. Returns the first responsive ``http://127.0.0.1:<port>``
    URL or ``None`` if none ever responds."""
    deadline = time.monotonic() + SANDBOX_START_TIMEOUT
    remaining_ports = list(ports)

    while remaining_ports:
        if time.monotonic() >= deadline:
            break
        next_ports = []
        for port in remaining_ports:
            try:
                requests.get(f"http://127.0.0.1:{port}", timeout=1)
                return f"http://127.0.0.1:{port}"
            except requests.RequestException:
                next_ports.append(port)
        remaining_ports = next_ports
        time.sleep(1)
    return None


def start_sandbox(kind: str, path: Path, tag: str, app_name: str) -> dict | None:
    """Start the built container image(s) in the background and return runtime data.

    Detaches the container(s) (compose: ``up -d``; Dockerfile: ``docker run -d -P``),
    discovers the published host ports, and waits for one of them to answer HTTP.

    Returns ``{"container_name": str, "sandbox_url": str}`` pointing at the first
    HTTP-responsive container (for compose that is the app service, not a DB sidecar),
    or ``None`` with a logged warning on any failure. Never raises.
    """
    try:
        if kind == "compose":
            up = subprocess.run(
                ["docker", "compose", "-f", str(path), "up", "-d"],
                capture_output=True,
                text=True
            )
            if up.returncode != 0:
                logging.error(f"docker compose up failed: {up.stderr}")
                return None

            ps = subprocess.run(
                ["docker", "compose", "-f", str(path), "ps", "--format", "json"],
                capture_output=True,
                text=True
            )
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
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True)
            run = subprocess.run(
                ["docker", "run", "-d", "-P", "--name", name, tag],
                capture_output=True,
                text=True
            )
            if run.returncode != 0:
                logging.error(f"docker run failed: {run.stderr}")
                return None
            containers = [name]

        # Prefer the container that publishes a responsive HTTP port
        for container in containers:
            url = _probe_http_ports(_published_ports(container))
            if url:
                logging.info(f"Sandbox running: container={container} url={url}")
                return {"container_name": container, "sandbox_url": url}

        logging.warning(
            "Sandbox container(s) started but none published a responsive HTTP port "
            f"({containers}). Validator tools will report no sandbox configured."
        )
        return None

    except FileNotFoundError:
        print("Error: docker is not installed or not in PATH.")
        return None


def run_osv_scanner_image(image: str) -> list[dict]:
    """Runs `osv-scanner scan image` against a built container image and
    extracts raw vulnerability records (same JSON shape as a source scan)."""
    raw_vulnerabilities = []

    try:
        result = subprocess.run(
            ["osv-scanner", "scan", "image", "--format", "json", image],
            capture_output=True,
            text=True
        )

        # Exit code 1 simply means "vulnerabilities found" - the JSON on
        # stdout is still valid. Only an empty stdout indicates no results.
        if not result.stdout.strip():
            error = result.stderr
            if error:
                logging.error(f"Error running osv-scanner on image '{image}': {error}")
            return []

        data = json.loads(result.stdout)

        for scan_result in data.get("results", []):
            for package in scan_result.get("packages", []):
                for vuln in package.get("vulnerabilities", []):
                    raw_vulnerabilities.append(vuln)

    except FileNotFoundError:
        print("Error: osv-scanner is not installed or not in PATH.")
    except json.JSONDecodeError:
        print("Error: Could not parse osv-scanner output.")

    return raw_vulnerabilities


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
        container_id = create.stdout.strip()

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
    current_slugs = {summary[img]["slug"] for img in summary}
    for existing in artifacts_root.glob("*"):
        if existing.is_dir() and existing.name not in current_slugs:
            logging.info(f"Removing stale artifact snapshot '{existing.name}'.")
            for f in existing.rglob("*"):
                if f.is_file():
                    f.unlink()
            for d in sorted((p for p in existing.rglob("*") if p.is_dir()), reverse=True):
                d.rmdir()
            existing.rmdir()

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
                if len(extracted) >= ARTIFACT_MAX_FILES:
                    skipped.append(path)
                    continue
                if total_bytes + member.size > ARTIFACT_MAX_TOTAL_BYTES:
                    skipped.append(path)
                    continue
                if member.size > ARTIFACT_MAX_FILE_BYTES:
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

    # Check if the ID itself embeds the CVE
    m = re.match(".*(CVE-20[0-9]{2}-[0-9]+).*", record["id"])
    if m:
        return m.group(1)

    # Fallback to the record ID if no CVE is found
    return record.get("id", "UNKNOWN")


def deduplicate_cves(vulns: list[dict]) -> list[dict]:
    """
    Extracts unique vulnerabilities by canonical ID and selects
    the most detailed description available for each.

    Besides {id, details, package}, best-effort enrichment fields are carried
    forward when present in the OSV record:
    - fixed_version: first `fixed` event across affected version ranges.
    - cwe_ids: database_specific.cwe_ids (list) when the OSV entry classifies them.
    """
    best_records = {}

    def extract_fixed_version(record: dict) -> Optional[str]:
        for affected in record.get("affected", []):
            for rng in affected.get("ranges", []):
                for event in rng.get("events", []):
                    if event.get("fixed"):
                        return event["fixed"]
        return None

    def extract_cwe_ids(record: dict) -> list[str]:
        return record.get("database_specific", {}).get("cwe_ids", []) or []

    for vuln in vulns:
        canonical_id = get_canonical_id(vuln)
        current_details = vuln.get("details", "")
        affected_packages = [affected.get("package", {}) for affected in vuln.get("affected", [])]
        packages = [pkg.get("name", pkg.get("name", "unknown")) for pkg in affected_packages]

        # If we haven't seen this CVE yet, or if the new record has a longer description
        if canonical_id not in best_records:
            best_records[canonical_id] = {
                "id": canonical_id,
                "details": current_details,
                "package": packages[0] if len(packages) >= 1 else "unknown",
                "fixed_version": extract_fixed_version(vuln),
                "cwe_ids": extract_cwe_ids(vuln),
            }
        else:
            # Compare the length of the details to keep the most comprehensive one
            existing_details = best_records[canonical_id]["details"]
            if len(current_details) > len(existing_details):
                best_records[canonical_id]["original_osv_id"] = vuln.get("id")
                best_records[canonical_id]["details"] = current_details
                # Enrichment is best-effort: backfill any missing fields.
                if not best_records[canonical_id].get("fixed_version"):
                    best_records[canonical_id]["fixed_version"] = extract_fixed_version(vuln)
                if not best_records[canonical_id].get("cwe_ids"):
                    best_records[canonical_id]["cwe_ids"] = extract_cwe_ids(vuln)

    return list(best_records.values())


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


def uses_namespace_in_ast(node_id: str, target_namespace: str) -> bool:
    """
    Checks if a specific namespace (or its imported symbols) is used within a node's AST.
    """
    source_code = get_node_code(node_id)
    if not source_code:
        return False

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

    parser = tree_sitter.Parser(LANGUAGE_MAP[ext])

    # Track the namespace and any symbols imported from it (e.g., 'g', 'request', 'Blueprint')
    aliases = set([target_namespace])

    # Parse the full file to discover aliases / imported components
    try:
        full_file_path = settings.app_path / Path(source_file)
        if full_file_path.exists():
            with open(full_file_path, "r", encoding="utf-8") as f:
                full_code_bytes = f.read().encode("utf-8")

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

    # Parse the specific node's folded code to check for actual usage
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


def get_node_code(node_id: str, raw: bool = False, reviewer_mode: bool = False) -> str | None:
    graph = settings.graph
    graph_data = get_cached_graph_data(graph)

    # Find the target node
    target_node = next((node for node in graph_data.get("nodes", []) if node.get("id") == node_id), None)
    if not target_node:
        logging.error(f"Error: Node ID '{node_id}' not found in graph.")
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
        try:
            with open(source_file, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            return f"Error: '{source_file}' is binary."
        except Exception as e:
            return None

    with open(source_file, "r", encoding="utf-8") as f:
        source_content = f.read()
    source_bytes = source_content.encode("utf-8")

    # Get target start line
    try:
        target_start_line = int(source_location.replace("L", ""))
    except ValueError:
        return None

    is_file_node = (target_start_line == 1 and target_node.get("label", "") == source_file.name)

    lang = LANGUAGE_MAP.get(source_file.suffix)
    if not lang:
        return source_content

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

        # Find all sub-nodes in the graph mapped to this file
        sub_nodes = []
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

        result_chunks = []
        last_idx = start_boundary

        for start_byte, end_byte, child_id in filtered_ranges:
            if start_byte < start_boundary or end_byte > end_boundary:
                continue

            comment = AST_GRAMMAR_MAP.get(source_file.suffix, {}).get("comment", "//")
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

# tree-sitter node types indicating executable logic (function calls, imports,
# string interpolation, control flow). Nodes exposing none of these are inert.
SCAN_SIGNAL_TYPES: dict[str, set[str]] = {
    ".py": {
        "call", "import_statement", "import_from_statement",
        "if_statement", "for_statement", "while_statement", "try_statement",
        "with_statement", "match_statement", "interpolation",
    },
    ".js": {
        "call_expression", "new_expression", "import_statement",
        "if_statement", "for_statement", "while_statement", "switch_statement",
        "try_statement", "template_substitution",
    },
    ".jsx": {
        "call_expression", "new_expression", "import_statement",
        "if_statement", "for_statement", "while_statement", "switch_statement",
        "try_statement", "template_substitution",
    },
    ".ts": {
        "call_expression", "new_expression", "import_statement",
        "if_statement", "for_statement", "while_statement", "switch_statement",
        "try_statement", "template_substitution",
    },
    ".tsx": {
        "call_expression", "new_expression", "import_statement",
        "if_statement", "for_statement", "while_statement", "switch_statement",
        "try_statement", "template_substitution",
    },
    ".php": {
        "function_call_expression", "member_call_expression", "scoped_call_expression",
        "object_creation_expression", "namespace_use_declaration", "include_expression",
        "include_once_expression", "require_expression", "require_once_expression",
        "echo_statement", "if_statement", "for_statement", "foreach_statement",
        "while_statement", "switch_statement", "try_statement", "encapsed_string",
    },
}

# Import-like declarations are tolerated inside pure type/interface/config nodes
# (they only bring names into scope and do not execute anything by themselves).
_IMPORT_TYPES: dict[str, set[str]] = {
    ".py": {"import_statement", "import_from_statement"},
    ".js": {"import_statement"},
    ".jsx": {"import_statement"},
    ".ts": {"import_statement"},
    ".tsx": {"import_statement"},
    ".php": {"namespace_use_declaration"},
}

# Nodes that introduce callable/structured definitions (bodies, classes, types).
_DEFINITION_TYPES: set[str] = {
    "function_definition", "class_definition", "decorated_definition", "method_declaration",
    "function_declaration", "class_declaration", "arrow_function", "method_definition",
    "function_expression", "lambda", "interface_declaration", "type_alias_declaration",
    "enum_declaration", "type_alias_statement",
}

# Node types whose names are security-relevant when used as assignment targets.
_NAME_NODE_TYPES: set[str] = {
    "assignment", "variable_declarator", "assignment_expression", "property_declaration",
    "property_element", "public_field_definition", "property_signature", "pair",
    "array_element_initializer",
}

_SECURITY_KEYWORDS: tuple[str, ...] = (
    "verify", "auth", "authenticate", "authorize", "permission", "secret", "token",
    "password", "passwd", "credential", "tls", "ssl", "private_key", "privatekey",
    "api_key", "apikey", "csrf", "jwt", "session", "cookie", "role", "admin",
    "sudo", "root", "privilege", "debug", "trust", "allow", "bypass", "skip", "disable",
)
_CRITICAL_SUBSTRINGS: tuple[str, ...] = (
    "secret", "password", "passwd", "token", "credential", "privatekey", "apikey", "csrf",
)

_MAGIC_METHODS: dict[str, set[str]] = {
    ".py": {
        "__reduce__", "__reduce_ex__", "__setstate__", "__getstate__", "__getattr__",
        "__setattr__", "__getattribute__", "__del__", "__delattr__", "__enter__",
        "__exit__", "__new__", "__init__", "__call__", "__getitem__", "__setitem__",
        "__repr__", "__str__",
    },
    ".php": {
        "__construct", "__destruct", "__wakeup", "__sleep", "__call", "__callstatic",
        "__get", "__set", "__isset", "__unset", "__tostring", "__invoke", "__set_state",
        "__clone", "__debuginfo", "__serialize", "__unserialize",
    },
    ".js": set(), ".jsx": set(), ".ts": set(), ".tsx": set(),
}


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
        if n.type in _NAME_NODE_TYPES:
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
            if n.type in _DEFINITION_TYPES and n.type != "decorated_definition"]
    return defs[0] if len(defs) == 1 else root


def _is_empty_skeleton(root: tree_sitter.Node, ext: str) -> bool:
    node = _unwrap_root(root)
    if node.type not in _DEFINITION_TYPES:
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
    return n in _MAGIC_METHODS.get(ext, set())


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
    imports = _IMPORT_TYPES.get(ext, set())

    if ext in (".ts", ".tsx", ".js", ".jsx"):
        type_constructs = {"type_alias_declaration", "interface_declaration", "enum_declaration"}
        if not _has_node_type(root, type_constructs):
            return False
        forbidden = (signals - imports) | {
            "function_declaration", "class_declaration", "arrow_function",
            "method_definition", "function_expression",
            "assignment", "variable_declarator", "public_field_definition", "pair",
        }
        return not _has_node_type(root, forbidden)

    if ext == ".py":
        behavioral = {"call", "if_statement", "for_statement", "while_statement",
                      "try_statement", "with_statement", "match_statement",
                      "interpolation", "function_definition", "lambda"}
        if _has_node_type(root, behavioral):
            return False
        if _has_node_type(root, {"type_alias_statement"}):
            return True
        for n in _walk(root):
            if n.type == "class_definition" and _class_is_field_only(n):
                return True
            if n.type == "decorated_definition":
                if any(c.type == "class_definition" and _class_is_field_only(c) for c in n.children):
                    return True
        return False

    if ext == ".php":
        runtime = signals - imports - {"namespace_use_declaration"}
        if _has_node_type(root, {"interface_declaration"}):
            return not _has_node_type(root, runtime)
        for n in _walk(root):
            if n.type == "class_declaration":
                body = _body_of(n)
                if body is not None and _has_node_type(n, {"property_declaration"}):
                    if not _has_node_type(n, {"method_declaration", "function_definition"}):
                        return True
        return False

    return False


def _is_config_only(root: tree_sitter.Node, ext: str) -> bool:
    if _has_node_type(root, _DEFINITION_TYPES):
        return False
    imports = _IMPORT_TYPES.get(ext, set())
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

    # PHP method/class raw fragments (as returned by get_node_code) omit the `<?php`
    # tag, which tree-sitter needs to avoid parsing everything as plain text.
    if ext == ".php" and not source_code.lstrip().startswith("<?"):
        source_code = "<?php\n" + source_code

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

    # Parse the file (read_bytes() replaces the 'with open()' block)
    parser = tree_sitter.Parser(language)
    tree = parser.parse(path.read_bytes())

    # Execute the language-specific query
    query = tree_sitter.Query(language, query_code)
    cursor = tree_sitter.QueryCursor(query)
    matches = cursor.matches(tree.root_node)

    symbol_index = {}

    # Standardized extraction loop
    for match in matches:
        captures = match[1] 

        # Method captures
        class_nodes = captures.get("class_name")
        parent_nodes = captures.get("parent_class")
        method_name_nodes = captures.get("method_name")
        method_body_nodes = captures.get("method_body")

        # Function captures
        function_name_nodes = captures.get("function_name")
        function_body_nodes = captures.get("function_body")

        # Normalize to lists
        if class_nodes and not isinstance(class_nodes, list): class_nodes = [class_nodes]
        if method_name_nodes and not isinstance(method_name_nodes, list): method_name_nodes = [method_name_nodes]
        if method_body_nodes and not isinstance(method_body_nodes, list): method_body_nodes = [method_body_nodes]
        if parent_nodes and not isinstance(parent_nodes, list): parent_nodes = [parent_nodes]
        if function_name_nodes and not isinstance(function_name_nodes, list): function_name_nodes = [function_name_nodes]
        if function_body_nodes and not isinstance(function_body_nodes, list): function_body_nodes = [function_body_nodes]

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

