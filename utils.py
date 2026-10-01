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
import settings


def build_networkx_graph(graph_path: Path) -> nx.DiGraph:
    """
    Reads the Graphify JSON output and builds a NetworkX Directed Graph.
    """
    with open(graph_path, 'r') as f:
        graph_data = json.load(f)

    # Initialize a Directed Graph
    G = nx.DiGraph()

    # Add Nodes with their attributes (community, type, file_path, etc.)
    for node in graph_data.get('nodes', []):
        node_id = node.get('id')
        if not node_id:
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

        # Copy all other key-value pairs as edge attributes
        attributes = {k: v for k, v in edge.items() if k not in ['source', 'target']}
        G.add_edge(source, target, **attributes)

    print(f"Loaded Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


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


# def run_stream(app, inputs, config=None, output_file="trace.json"):
#     print(f"\n[System] Running Multi-Agent Analysis. Assembling state to {output_file}...")
#
#     assembled_states = {}
#     raw_main_state = {}
#
#     # Token usage stats
#     tracked_msg_ids = set()
#     agent_token_stats = {}
#     token_stats = {"input": 0, "output": 0, "total": 0}
#
#     for event in app.stream(inputs, stream_mode="values", subgraphs=True, config=config):
#         namespace, state = event
#
#         # Format the namespace so it's readable in the JSON
#         if not namespace:
#             graph_name = "Main_Graph"
#             raw_main_state = state
#         else:
#             # Subgraphs/Agents have namespaces like ('manager', 'expert', 'b3f1...')
#             graph_name = ' -> '.join(namespace)
#
#         if hasattr(state.get("task"), "agent_role"):
#             graph_name = graph_name[:17] + f" ({state.get('task').agent_role})"
#
#         # Overwrite the key with the most recent full state.
#         assembled_states[graph_name] = serialize_for_json(state)
#
#         # ==========================================
#         # --- TERMINAL PROGRESS INDICATOR ---
#         # ==========================================
#         last_msg = None
#
#         # 1. Catch Subgraph Agents (they still use the 'messages' array)
#         if namespace and "messages" in state and state["messages"]:
#             last_msg = state["messages"][-1]
#
#         # 2. Catch the Manager (runs on Main Graph, uses 'manager_message' key)
#         elif not namespace and "manager_message" in state and state["manager_message"]:
#             last_msg = state["manager_message"]
#
#         # If we successfully grabbed a message from either source, process it:
#         if last_msg:
#             msg_type = getattr(last_msg, "type", "unknown")
#             msg_id = getattr(last_msg, "id", None)
#
#             # Extract content safely, even if it's nested
#             content = getattr(last_msg, "content", "")
#             if isinstance(content, list):
#                 content = str(content)
#
#             # Create a clean, single-line snippet
#             snippet = (content[:200] + "...") if len(content) > 200 else content
#             snippet = snippet.replace('\n', ' ').strip()
#
#             if msg_type == "ai":
#                 # Track token usage
#                 if msg_id and msg_id not in tracked_msg_ids:
#                     tracked_msg_ids.add(msg_id)
#                     usage = getattr(last_msg, "usage_metadata", {})
#                     if usage:
#                         in_tok = usage.get("input_tokens", 0)
#                         out_tok = usage.get("output_tokens", 0)
#                         tot_tok = usage.get("total_tokens", 0)
#
#                         # Initialize agent in stats dictionary if not present
#                         if graph_name not in agent_token_stats:
#                             agent_token_stats[graph_name] = {"input": 0, "output": 0, "total": 0}
#
#                         agent_token_stats[graph_name]["input"] += in_tok
#                         agent_token_stats[graph_name]["output"] += out_tok
#                         agent_token_stats[graph_name]["total"] += tot_tok
#
#                         token_stats["input"] += in_tok
#                         token_stats["output"] += out_tok
#                         token_stats["total"] += tot_tok
#
#                 # Print the AI's thought process (Chain of Thought)
#                 if snippet:
#                     print(f"[{graph_name}] \033[96m🧠 AI: {snippet}\033[0m", flush=True)
#
#                 # Print the Tool Call (if it decided to act)
#                 if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
#                     tool_strings = []
#                     for tc in last_msg.tool_calls:
#                         name = tc.get("name", "unknown")
#                         args = tc.get("args", {})
#                         args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
#                         tool_strings.append(f"{name}({args_str})")
#
#                     print(f"[{graph_name}] \033[93m🛠️  Calling tool: {' | '.join(tool_strings)}\033[0m", flush=True)
#
#             elif msg_type == "tool":
#                 tool_name = getattr(last_msg, 'name', 'unknown')
#                 print(f"[{graph_name}] \033[92m✅ Tool executed: {tool_name} | Output: {snippet[:50]}\033[0m", flush=True)
#
#             elif msg_type == "human":
#                 print(f"[{graph_name}] \033[94m👤 Human: {snippet}\033[0m", flush=True)
#
#             else:
#                 print(f"[{graph_name}] \033[90m⚙️  {msg_type.capitalize()} message\033[0m", flush=True)
#
#         else:
#             # Tell us exactly WHICH state keys were updated in the background
#             state_keys = ", ".join([k for k in state.keys() if k not in ["messages", "manager_message"]])
#             if state_keys:
#                 print(f"[{graph_name}] \033[90mState updated: [{state_keys}]\033[0m", flush=True)
#         # -----------------------------------
#
#     # Dump the cohesive final states to a JSON file
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(assembled_states, f, indent=2)
#
#     print("[System] Execution Finished.")
#
#     # --- Print Token Usage Summary ---
#     print("\n" + "="*50)
#     print("📊 \033[1mToken Usage Summary by Agent\033[0m")
#     print("-" * 50)
#
#     for agent_name in sorted(agent_token_stats.keys()):
#         stats = agent_token_stats[agent_name]
#         print(f"🔹 \033[96m{agent_name}\033[0m")
#         print(f"   In: {stats['input']:,}  |  Out: {stats['output']:,}  |  Total: {stats['total']:,}")
#
#     print("-" * 50)
#     print("🏆 \033[1mGrand Totals\033[0m")
#     print(f"   Input Tokens:  {token_stats['input']:,}")
#     print(f"   Output Tokens: {token_stats['output']:,}")
#     print(f"   Total Tokens:  \033[95m{token_stats['total']:,}\033[0m")
#     print("="*50 + "\n")
#
#     main_state = assembled_states.get("Main_Graph", {})
#     return {"vulnerability_reports": raw_main_state.get("vulnerability_reports", [])}


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


def format_notes(notes_list: list[dict]) -> str:
    """
    Takes a list of note dictionaries and converts them into a structured 
    Markdown prompt for the Researcher agent.
    """
    notes_str = "### Current Audit Memory\n\n"

    for item in notes_list:
        notes_str += f"#### Node ID: `{item.get('node_id')}`\n"
        notes_str += f"{item.get('role_in_system')}\n\n"

        assumptions = item.get("assumptions_to_verify", [])
        if assumptions:
            notes_str += "**Assumptions to Verify:**\n"
            for asm in assumptions:
                notes_str += f"- {asm.get('description')}\n"

                snippet = asm.get('snippet', '').strip()
                if snippet:
                    notes_str += f"  ```python\n  {snippet}\n  ```\n"

                dep = asm.get("depends_on")
                if dep:
                    resolved_id = dep.get("resolved_node_id", "External / Unresolved")
                    notes_str += f"  - Depends on: `{dep.get('module')}.{dep.get('symbol')}` (Target Node: `{resolved_id}`)\n"
            notes_str += "\n"

        issues = item.get("potential_issues", [])
        if issues:
            notes_str += "**Potential Issues:**\n"
            for issue in issues:
                notes_str += f"- {issue.get('description')}\n"

                snippet = issue.get('snippet', '').strip()
                if snippet:
                    notes_str += f"  ```python\n  {snippet}\n  ```\n"

                dep = issue.get("depends_on")
                if dep:
                    resolved_id = dep.get("resolved_node_id", "External / Unresolved")
                    notes_str += f"  - Flows into: `{dep.get('module')}.{dep.get('symbol')}` (Target Node: `{resolved_id}`)\n"
            notes_str += "\n"

        notes_str += "---\n\n"

    return notes_str


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


def get_node_source_code(node_id: str):
    graph = settings.graph
    try:
        with open(graph, "r") as f:
            graph_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"'{graph}' not found.")
        return None

    target_node = next((node for node in graph_data.get("nodes", []) if node.get("id") == node_id), None)
    if not target_node:
        return f"Error: Node ID '{node_id}' not found in graph."

    if not target_node.get("source_file"):
        logging.error(f"Node '{node_id}' does not have a source file mapped.")
        return None

    source_file = graph.parent.parent / Path(target_node.get("source_file"))
    source_location = target_node.get("source_location")
    file_type = target_node.get("file_type")

    if not source_file.exists():
        logging.error(f"Source file '{source_file}' not found on disk. Ensure paths are correct.")
        return None

    if file_type == "document" or not source_location:
        try:
            # Enforce utf-8 encoding. If it is not a text file it will trigger a UnicodeDecodeError
            with open(source_file, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            return f"Error: '{source_file}' is a binary file and cannot be read as plain text."
        except Exception as e:
            logging.error(f"Error reading '{source_file}': {str(e)}")
            return None

        if len(content) > 20000:
            return (
                f"Warning: File is too large. Showing first {20000}/{len(content)} characters:\n\n"
                f"{content[:20000]}\n\n"
            )
        return content

    elif file_type in ["code", "rationale"]:
        with open(source_file, "r", encoding="utf-8") as f:
            source_content = f.read()

        # Convert line location string "L53" to integer 53
        try:
            start_line = int(source_location.replace("L", ""))
            target_row = start_line - 1  # Tree-sitter rows are 0-indexed
        except ValueError:
            logging.error(f"Invalid source_location format '{source_location}'.")
            return None

        # Determine if this is a file
        is_file_node = start_line == 1 and target_node.get("label", "") == source_file.name

        lines = source_content.splitlines()
        lang = LANGUAGE_MAP.get(source_file.suffix)

        if lang:
            try:
                parser = tree_sitter.Parser(lang)
                source_bytes = source_content.encode("utf-8")
                tree = parser.parse(source_bytes)

                # Generate the skeleton context
                skeleton = [
                    "# [FILE CONTEXT - JUST USE FOR REFERENCE]",
                    f"# File: {source_file.name}",
                    "# Imports and Structure:"
                ]

                # Fetch language-specific grammar rules (fallback to an empty dict to be safe)
                grammar = AST_GRAMMAR_MAP.get(source_file.suffix, {})
                keep_whole = grammar.get("keep_whole", [])
                prune_bodies = grammar.get("prune_bodies", [])
                body_node_type = grammar.get("body_node", "block")

                for child in tree.root_node.children:
                    # Keep imports and top-level expressions (globals) intact
                    if child.type in keep_whole:
                        skeleton.append(source_bytes[child.start_byte:child.end_byte].decode("utf-8"))

                    # Prune the bodies of functions and classes
                    elif child.type in prune_bodies:
                        def get_body_node(n):
                            for c in n.children:
                                # Check if body_node_type is a list/tuple to support multiple block node types
                                if isinstance(body_node_type, (list, tuple)) and c.type in body_node_type:
                                    return c
                                elif c.type == body_node_type:
                                    return c

                                if c.type in prune_bodies:
                                    res = get_body_node(c)
                                    if res: return res
                            return None

                        body_node = get_body_node(child)
                        if body_node:
                            # Extract everything up to the start of the block (e.g., 'def init_db():')
                            signature = source_bytes[child.start_byte:body_node.start_byte].decode("utf-8").strip()
                            skeleton.append(f"{signature}\n    # Body omitted for context limits\n")
                        else:
                            # Fallback if no block is found
                            skeleton.append(source_bytes[child.start_byte:child.end_byte].decode("utf-8"))

                skeleton_text = "\n".join(skeleton) if len(skeleton) > 3 else ""

                # If it's a file node, return skeleton (or full file if no grammar rules applied)
                if is_file_node:
                    if not skeleton_text:
                        return source_content
                    return skeleton_text.replace(
                        "# [FILE CONTEXT - JUST USE FOR REFERENCE]\n", 
                        f"# --- FILE SKELETON: {source_file.name} (function/class definitions omitted) ---\n"
                    )

                # Extract specific function/class node
                def find_node(node, row):
                    if node.start_point[0] == row:
                        return node
                    for child in node.children:
                        if child.start_point[0] <= row <= child.end_point[0]:
                            found = find_node(child, row)
                            if found:
                                return found
                    return None

                target_ast_node = find_node(tree.root_node, target_row)

                if target_ast_node:
                    extracted_bytes = source_bytes[target_ast_node.start_byte:target_ast_node.end_byte]
                    node_code = extracted_bytes.decode("utf-8")

                    return (
                        f"{skeleton_text}\n\n"
                        f"# [NODE TO ANALYZE]\n"
                        f"# Node ID: {node_id}\n"
                        f"# Code:\n"
                        f"{node_code}"
                    )

            except Exception as e:
                logging.warning(f"Tree-sitter failed to parse or walk '{source_file}': {e}")
                pass

        # If the node represents the whole file, return the entire raw text
        if is_file_node:
            return source_content

        # If it represents a specific line inside a non-AST file, return just that line
        if 0 < start_line <= len(lines):
            return lines[start_line - 1].strip()

    return None


def resolve_node_id(module, symbol):
    with open(settings.graph, "r") as f:
        graph_data = json.load(f)
        nodes_list = graph_data.get("nodes", [])

        node = next(
            (n for n in nodes_list 
             if n.get("source_file", "").endswith(module.replace(".", "/") + ".py")
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
