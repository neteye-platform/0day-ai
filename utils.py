import logging
import re
from pathlib import Path
import tree_sitter
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript
import tree_sitter_php
import subprocess
import networkx as nx
import json
from typing import Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AnyMessage, AIMessage
from schemas import VulnerabilityRecord
import settings
from functools import lru_cache


@lru_cache(maxsize=1)
def get_cached_graph_data(graph_path: Path):
    """Caches the graph JSON in memory to prevent disk I/O bottlenecks."""
    try:
        with open(graph_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"'{graph_path}' not found.")
        return {}


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


def merge_vulnerabilities(
    existing: list[dict], 
    updates: list[dict]
) -> list[dict]:
    """Custom reducer to merge vulnerabilities using standard dictionaries."""
    vuln_map = {}

    # Map existing state
    for vuln in existing:
        vid = vuln.get("vuln_id") if isinstance(vuln, dict) else getattr(vuln, "vuln_id", None)
        if vid:
            vuln_map[vid] = vuln

    # Process updates
    for update in updates:
        # Failsafe: if a Pydantic model accidentally slips in, dump it to a dict
        if not isinstance(update, dict):
            update = update.model_dump() if hasattr(update, "model_dump") else dict(update)

        vid = update.get("vuln_id")
        if not vid:
            node_id = update.get("node_id", "Unknown")
            cwe_id = update.get("cwe_id", "OTHER_UNCATEGORIZED")
            vid = f"{node_id}:{cwe_id}"

            # Inject it into the dictionary so it stays tracked forever
            update["vuln_id"] = vid 

        if vid in vuln_map:
            current = vuln_map[vid]

            if isinstance(current, dict):
                # Update status if we are upgrading from a hypothesis
                if update.get("status") and update.get("status") != "hypothesis":
                    current["status"] = update.get("status")

                # Merge descriptions without losing context
                curr_desc = current.get("description", "")
                upd_desc = update.get("description", "")
                if upd_desc and upd_desc not in curr_desc:
                    current["description"] = f"{curr_desc}\n\nAdditional context: {upd_desc}"

                # Merge any new tool fields (e.g., confidence_score) dynamically
                for key, value in update.items():
                    if key not in ["status", "description"] and value is not None:
                        current[key] = value

            vuln_map[vid] = current
        else:
            # It's a completely new vulnerability
            vuln_map[vid] = update

    return list(vuln_map.values())


def get_graph_summary(G: nx.DiGraph) -> str:
    # Group nodes by community
    communities_map = {}
    for node_id, data in G.nodes(data=True):
        comm_id = str(data.get('community', 'unknown'))
        if comm_id not in communities_map:
            communities_map[comm_id] = []
        communities_map[comm_id].append((node_id, data))

    # Generate a summary for the Manager
    summary = (
        f"Application Topology Summary:\n"
        f"- Total Nodes: {G.number_of_nodes()} | Total Edges: {G.number_of_edges()}\n"
        f"- Total Communities: {len(communities_map)}\n\n"
        f"Community Breakdown:\n"
    )

    for comm_id, nodes_data in communities_map.items():
        node_ids = [n[0] for n in nodes_data]

        # Node Types
        node_types = {"functions": 0, "classes/files": 0, "docs/other": 0}
        for _, data in nodes_data:
            f_type = data.get('file_type', 'unknown')
            label = data.get('label', '')
            if f_type == 'code':
                if '()' in label:
                    node_types["functions"] += 1
                else:
                    node_types["classes/files"] += 1
            else:
                node_types["docs/other"] += 1
        type_str = ", ".join([f"{v} {k}" for k, v in node_types.items() if v > 0])

        # Find the "God Node" (the most connected file/function in this community)
        subgraph = G.subgraph(node_ids)
        if len(subgraph.nodes) > 0:
            degrees = dict(subgraph.degree())
            god_node = max(degrees, key=lambda x: (degrees[x], x))
        else:
            god_node = "None"

        summary += (
            f"- Community {comm_id} ({len(node_ids)} nodes): [{type_str}]\n"
            f"  -> Central Hub Node: {god_node}\n"
        )

    return summary


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


def serialize_for_json(obj):
    """
    Recursively parses objects to create a clean, JSON structure.
    Strips heavy LangChain metadata but preserves full content and tool IDs.
    """
    if isinstance(obj, dict):
        # Strip out noisy metadata
        return {
            k: serialize_for_json(v)
            for k, v in obj.items()
            if k not in ["usage_metadata", "response_metadata"]
        }

    elif isinstance(obj, list):
        # Filter out hidden system messages
        parsed_list = [serialize_for_json(item) for item in obj]
        return parsed_list
        return [item for item in parsed_list if item != "[SYSTEM PROMPT HIDDEN]"]

    elif hasattr(obj, "model_dump"):
        # Unpack Pydantic models
        return serialize_for_json(obj.model_dump())

    elif hasattr(obj, "content") and hasattr(obj, "type"):
        if obj.type == "system":
            return "[SYSTEM PROMPT HIDDEN]"

        # Structure the message exactly how you requested
        msg_data = {"type": obj.type}

        if obj.content:
            msg_data["content"] = obj.content

        # If the AI calls a tool, include the call details and ID
        if hasattr(obj, "tool_calls") and obj.tool_calls:
            msg_data["tool_calls"] = obj.tool_calls

        # If this is a tool responding, include its name and the ID it's answering
        if obj.type == "tool":
            if hasattr(obj, "name"):
                msg_data["tool_name"] = obj.name
            if hasattr(obj, "tool_call_id"):
                msg_data["responds_to_id"] = obj.tool_call_id

        return msg_data

    elif type(obj) in (int, float, bool, type(None), str):
        return obj
    else:
        return str(obj)


# Tools that takes a lot of context
heavy_tools = ["read_source_code", "send_http_request", "search_codebase"]

def compact_tool_history(messages: list[AnyMessage], safe_window: int = 10, threshold: int = 300) -> list[AnyMessage]:
    """
    Compresses heavy tool outputs in the message history to save context space,
    while preserving the structural timeline of the conversation.
    """
    compacted_messages = []

    for i, msg in enumerate(messages):

        # Scrub reasoning from AIMessages
        if isinstance(msg, AIMessage):
            update_dict = {}
            new_tool_calls = []
            modified_tools = False

            # Scrub bulky arguments from ALL tool calls (even recent ones)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc["name"] == "read_source_code" and "reason_for_reading" in tc.get("args", {}):
                        scrubbed_tc = tc.copy()
                        scrubbed_tc["args"] = tc["args"].copy()
                        scrubbed_tc["args"]["reason_for_reading"] = "[redacted]"
                        new_tool_calls.append(scrubbed_tc)
                        modified_tools = True
                    else:
                        new_tool_calls.append(tc)

                if modified_tools:
                    update_dict["tool_calls"] = new_tool_calls

            # Scrub heavy Chain-of-Thought content (ONLY for older messages)
            is_older_message = i < len(messages) - safe_window
            has_heavy_content = msg.content and len(str(msg.content)) > threshold

            # We only crush the text if the model actually made a tool call.
            if is_older_message and has_heavy_content and msg.tool_calls:
                update_dict["content"] = "[Internal reasoning redacted to conserve memory.]"

            # Apply updates if we changed anything
            if update_dict:
                crushed_ai_msg = msg.model_copy(update=update_dict)
                compacted_messages.append(crushed_ai_msg)
                continue

        # Crush heavy ToolMessages (Only outside the safe window)
        if i < len(messages) - safe_window and isinstance(msg, ToolMessage):
            content_str = str(getattr(msg, "content", ""))

            # Crush heavy outputs
            if msg.name in heavy_tools and len(str(msg.content)) > threshold:
                crushed_msg = msg.model_copy(
                    update={"content": "[SYSTEM OVERRIDE: Raw data removed to conserve memory. Please refer to your notes for details.]"}
                )
                compacted_messages.append(crushed_msg)
                continue

            # Crush redundant rejection
            if content_str.startswith("SYSTEM REJECTION:"):
                crushed_msg = msg.model_copy(
                    update={"content": "REJECTED"}
                )
                compacted_messages.append(crushed_msg)
                continue

        # If it doesn't meet the criteria, keep the original message
        compacted_messages.append(msg)

    return compacted_messages


def enforce_note_taking(messages: list[AnyMessage], max_consecutive: int = 4) -> Optional[str]:
    """
    Scans recent history to ensure the agent took a note after successfully using a heavy tool.
    Returns a rejection string if the rule is violated, otherwise returns None.
    """
    consecutive_count = 0

    scan_window = max(8, max_consecutive * 3)

    # Scan the recent tool messages in the history
    for msg in reversed(messages[-scan_window:]):
        if getattr(msg, "type", "") == "tool":

            # The agent took a note recently
            if msg.name == "take_notes":
                return None

            elif msg.name in heavy_tools:
                content_str = str(getattr(msg, "content", ""))

                # Check if it was a successful data extraction
                if not content_str.startswith("Error:") and (len(content_str) > 100 or "[SYSTEM OVERRIDE" in content_str):
                    consecutive_count += 1

                    if consecutive_count >= max_consecutive:
                        return (
                            f"SYSTEM REJECTION: Access Denied. You have used heavy tools {consecutive_count} times in a row "
                            "without documenting your findings. You MUST use the `take_notes` tool to summarize "
                            "your current context before you are allowed to execute another heavy action."
                        )

    return None


LANGUAGE_MAP = {
    ".py": tree_sitter.Language(tree_sitter_python.language()),
    ".js": tree_sitter.Language(tree_sitter_javascript.language()),
    ".jsx": tree_sitter.Language(tree_sitter_javascript.language()),
    ".php": tree_sitter.Language(tree_sitter_php.language_php()),
    ".ts": tree_sitter.Language(tree_sitter_typescript.language_typescript()),
    ".tsx": tree_sitter.Language(tree_sitter_typescript.language_tsx()),
}

# Define the AST mappings for the languages your agents will scan
AST_GRAMMAR_MAP = {
    ".py": {
        "keep_whole": ["import_statement", "import_from_statement", "expression_statement"],
        "prune_bodies": ["function_definition", "class_definition", "decorated_definition"],
        "body_node": "block"
    },
    ".js": {
        "keep_whole": ["import_statement", "lexical_declaration", "variable_declaration"],
        "prune_bodies": ["function_declaration", "class_declaration", "arrow_function", "method_definition"],
        "body_node": "statement_block"
    },
    ".go": {
        "keep_whole": ["import_declaration"],
        "prune_bodies": ["function_declaration", "method_declaration"],
        "body_node": "block"
    },
    ".php": {
        "keep_whole": ["namespace_definition", "namespace_use_declaration", "expression_statement"],
        "prune_bodies": ["function_definition", "method_declaration", "class_declaration", "trait_declaration", "interface_declaration"],
        "body_node": ["compound_statement", "declaration_list"]
    },
    ".ts": {
        "keep_whole": ["import_statement", "lexical_declaration", "variable_declaration", "type_alias_declaration"],
        "prune_bodies": ["function_declaration", "class_declaration", "arrow_function", "method_definition", "interface_declaration", "module"],
        "body_node": ["statement_block", "class_body", "object_type", "module_block"]
    },
    ".tsx": {
        "keep_whole": ["import_statement", "lexical_declaration", "variable_declaration", "type_alias_declaration"],
        "prune_bodies": ["function_declaration", "class_declaration", "arrow_function", "method_definition", "interface_declaration", "module"],
        "body_node": ["statement_block", "class_body", "object_type", "module_block"]
    },
}


def resolve_node_id(module, symbol):
    with open(settings.graph, "r") as f:
        graph_data = json.load(f)
        nodes_list = graph_data.get("nodes", [])

        normalized_module = module.replace(".", "/").replace("\\", "/")
        module_with_symbol = f"{normalized_module}/{symbol}" # PHP/Java style path

        node = next(
            (n for n in nodes_list 
             if n.get("source_file") 
             and (
                 # Python/JS style: the module maps directly to the file path
                 str(Path(n.get("source_file")).with_suffix("")).replace("\\", "/").endswith(normalized_module) or
                 # PHP/Java style: the file is named after the class/symbol inside the module folder
                 str(Path(n.get("source_file")).with_suffix("")).replace("\\", "/").endswith(module_with_symbol)
             )
             and n.get("label") in [symbol, f"{symbol}()"]),
            None
        )

        if not node:
            logging.warning(f"Failed to find graph node for local symbol '{module}.{symbol}'.")
            return

    return node.get("id")


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
    Checks if a specific namespace is used within a node's AST, 
    using the semantically folded source code.
    """
    source_code = get_node_code(node_id)
    if not source_code:
        return False

    # Extract node data to determine the file extension
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

    # Map extension to tree-sitter language
    if ext not in LANGUAGE_MAP:
        logging.warning(f"Unsupported extension '{ext}' for AST parsing on node '{node_id}'.")
        return False

    # Initialize parser
    parser = tree_sitter.Parser(LANGUAGE_MAP[ext])

    # Parse the folded code
    source_bytes = source_code.encode("utf-8")
    tree = parser.parse(source_bytes)

    def walk(node: tree_sitter.Node) -> bool:
        # Ignore import/require statements so we only match actual usage
        if any(keyword in node.type for keyword in ["import", "include", "use_declaration"]):
            return False

        # If it's a leaf node, check its text
        if len(node.children) == 0:
            # Safely ignore comments and string literals
            if "comment" not in node.type and "string" not in node.type:
                token_text = source_bytes[node.start_byte:node.end_byte].decode("utf-8")
                if token_text == target_namespace:
                    return True

        # Recurse through children
        for child in node.children:
            if walk(child):
                return True

        return False

    return walk(tree.root_node)


def get_node_code(node_id: str) -> str | None:
    graph = settings.graph
    graph_data = get_cached_graph_data(graph)

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

    # Read source bytes for Tree-sitter
    with open(source_file, "r", encoding="utf-8") as f:
        source_content = f.read()
    source_bytes = source_content.encode("utf-8")

    # Resolve target row
    try:
        start_line = int(source_location.replace("L", ""))
        target_row = start_line - 1  # 0-indexed for Tree-sitter
    except ValueError:
        return None

    # If the node represents the whole file, target_row becomes None (prune everything)
    is_file_node = (start_line == 1 and target_node.get("label", "") == source_file.name)
    active_target_row = None if is_file_node else target_row

    lang = LANGUAGE_MAP.get(source_file.suffix)
    if not lang:
        # Fallback to returning raw code if no parser exists
        return source_content

    try:
        parser = tree_sitter.Parser(lang)
        tree = parser.parse(source_bytes)

        grammar = AST_GRAMMAR_MAP.get(source_file.suffix, {})
        prune_bodies = grammar.get("prune_bodies", [])
        body_node_type = grammar.get("body_node", "block")

        def get_body_node(n):
            for c in n.children:
                if isinstance(body_node_type, (list, tuple)) and c.type in body_node_type:
                    return c
                elif c.type == body_node_type:
                    return c
                if c.type in prune_bodies:
                    res = get_body_node(c)
                    if res: return res
            return None

        # Identify the specific target AST node first
        target_ast_node = None
        if active_target_row is not None:
            def find_target(n):
                deepest = None
                # If this node matches criteria, it is a candidate
                if n.type in prune_bodies and n.start_point[0] <= active_target_row <= n.end_point[0]:
                    deepest = n
                # Check children for a deeper match
                for c in n.children:
                    res = find_target(c)
                    if res:
                        deepest = res
                return deepest

            target_ast_node = find_target(tree.root_node)

        # Walk the AST and collect byte ranges of bodies to prune
        ranges_to_prune = []

        def find_prunable_ranges(node, inside_target=False):
            # Once we hit the target node, flag it and all its children as protected
            is_target_now = inside_target or (node == target_ast_node)

            if node.type in prune_bodies:
                # Only attempt to prune if we are NOT inside the target block
                if not is_target_now:
                    contains_target = False
                    if active_target_row is not None:
                        contains_target = node.start_point[0] <= active_target_row <= node.end_point[0]

                    # If it doesn't contain the target, prune its body and stop traversing this branch
                    if not contains_target:
                        body = get_body_node(node)
                        if body:
                            ranges_to_prune.append((body.start_byte, body.end_byte))
                        return 

            # Recurse into children, passing down the protection flag
            for child in node.children:
                find_prunable_ranges(child, is_target_now)

        find_prunable_ranges(tree.root_node)

        # Reconstruct the source code using byte replacement
        ranges_to_prune.sort(key=lambda x: x[0])
        result_chunks = []
        last_idx = 0

        for start_byte, end_byte in ranges_to_prune:
            # Append code up to the start of the pruned body
            result_chunks.append(source_bytes[last_idx:start_byte].decode("utf-8"))
            # Insert our folded marker
            result_chunks.append("\n    # ... [Body omitted] ...\n")
            last_idx = end_byte

        # Append the remaining code
        result_chunks.append(source_bytes[last_idx:].decode("utf-8"))

        return "".join(result_chunks)

    except Exception as e:
        logging.warning(f"Tree-sitter failed on '{source_file}': {e}")
        return source_content
