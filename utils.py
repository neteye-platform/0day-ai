from collections import defaultdict
import logging
import re
from pathlib import Path
import tree_sitter
import subprocess
import networkx as nx
import json
from typing import Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AnyMessage, AIMessage
from schemas import VulnerabilityRecord
import settings
from functools import lru_cache

from languages import LANGUAGE_MAP, AST_GRAMMAR_MAP, SYMBOL_QUERIES


@lru_cache(maxsize=1)
def get_cached_graph_data(graph_path: Path):
    """Caches the graph JSON in memory to prevent disk I/O bottlenecks."""
    try:
        with open(graph_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"'{graph_path}' not found.")
        return {}


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
        "confirmed": 1,
        "false_positive": 2,
        "proven": 3
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
heavy_tools = ["read_source_code", "send_http_request", "search_codebase", "get_definition"]

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
    with open(settings.graph, "r") as f:
        graph_data = json.load(f)
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


def run_osv_scanner(repo_path: str) -> list[dict]:
    """Runs osv-scanner on a directory and extracts raw vulnerability records."""
    raw_vulnerabilities = []

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
                logging.warning(f"Error running osv-scanner: {error}")
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
    """
    best_records = {}

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
                "package": packages[0] if len(packages) >= 1 else "unknown"
            }
        else:
            # Compare the length of the details to keep the most comprehensive one
            existing_details = best_records[canonical_id]["details"]
            if len(current_details) > len(existing_details):
                best_records[canonical_id]["original_osv_id"] = vuln.get("id")
                best_records[canonical_id]["details"] = current_details

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


def get_node_code(node_id: str, raw: bool = False) -> str | None:
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
        else:
            target_ast_node, _ = find_ast_node_with_body(target_line_idx)
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
                is_target = (sub_start == target_ast_node.start_byte and sub_end == target_ast_node.end_byte)
                is_ancestor = (sub_start <= target_ast_node.start_byte and sub_end >= target_ast_node.end_byte)

                # Check for exact target match
                if sub_start == target_ast_node.start_byte and sub_end == target_ast_node.end_byte:
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
            result_chunks.append(f"\n    {comment} [Body omitted: use read_source_code with node_id '{child_id}' to read this content]\n")
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
    """
    source_code = get_node_code(node_id, raw=True)
    if not source_code:
        return False

    graph_data = get_cached_graph_data(settings.graph)
    target_node = next((node for node in graph_data.get("nodes", []) if node.get("id") == node_id), None)
    if not target_node or not target_node.get("source_file"):
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

