from typing import Annotated
import json
from langchain_core.messages import ToolMessage
import requests
from pathlib import Path
import tree_sitter
import tree_sitter_python
import tree_sitter_javascript
import logging
from bs4 import BeautifulSoup
from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command

from schemas import EvaluationToolInput, ValidationToolInput, VulnerabilityReport
from utils import build_networkx_graph
import settings

LANGUAGE_MAP = {
    ".py": tree_sitter.Language(tree_sitter_python.language()),
    ".js": tree_sitter.Language(tree_sitter_javascript.language()),
    ".jsx": tree_sitter.Language(tree_sitter_javascript.language()),
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
    # Add Java, C++, etc., as needed
}

@tool
def read_source_code(node_id: str, reason_for_reading: str) -> str:
    """
    Fetches the source code for a given Node ID.

    Args:
        node_id: The exact ID of the node to read (e.g., 'src_main_query_db').
        reason_for_reading: Explain exactly why you need to read THIS specific node next, and how you expect it to connect to your current knowledge.
    """
    try:
        with open(settings.graph, "r") as f:
            graph_data = json.load(f)
    except FileNotFoundError:
        return f"Error: {settings.graph} not found."

    target_node = next((node for node in graph_data.get("nodes", []) if node.get("id") == node_id), None)
    if not target_node:
        return f"Error: Node ID '{node_id}' not found in graph."

    if not target_node.get("source_file"):
        return f"Error: Node '{node_id}' does not have a source file mapped."

    source_file = settings.app_path / Path(target_node.get("source_file"))
    source_location = target_node.get("source_location")
    file_type = target_node.get("file_type")

    if file_type == "document" or not source_location:
        try:
            # Enforce utf-8 encoding. If it is not a text file it will trigger a UnicodeDecodeError
            with open(source_file, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            return f"Error: '{source_file}' is a binary file and cannot be read as plain text."
        except Exception as e:
            return f"Error reading '{source_file}': {str(e)}"

        if len(content) > 20000:
            return (
                f"Warning: File is too large. Showing first {20000}/{len(content)} characters:\n\n"
                f"{content[:20000]}\n\n"
            )
        return content

    elif file_type in ["code", "rationale"]:
        if not source_file.exists():
            return f"Error: Source file '{source_file}' not found on disk. Ensure paths are correct."

        with open(source_file, "r") as f:
            source_content = f.read()

        # Convert line location string "L53" to integer 53
        try:
            start_line = int(source_location.replace("L", ""))
            target_row = start_line - 1  # Tree-sitter rows are 0-indexed
        except ValueError:
            return f"Error: Invalid source_location format '{source_location}'."

        lines = source_content.splitlines()
        lang = LANGUAGE_MAP.get(source_file.suffix)

        if lang:
            try:
                parser = tree_sitter.Parser(lang)
                source_bytes = source_content.encode("utf-8")
                tree = parser.parse(source_bytes)

                is_file_node = start_line == 1 and target_node.get("label", "") == source_file.name
                if is_file_node:
                    skeleton = [f"--- FILE SKELETON: {source_file.name} (function/class definition omitted) ---"]

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
                                    # Use the dynamic body_node_type instead of hardcoding "block"
                                    if c.type == body_node_type:
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

                    # If we don't have a grammar map for this file, fallback to full text to avoid breaking
                    if len(skeleton) == 1:
                         return source_content

                    return "\n".join(skeleton)

                # Find the largest node starting on the target row
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
                    return extracted_bytes.decode("utf-8")
            except Exception as e:
                logging.warning(f"Tree-sitter failed to parse or walk '{source_file}': {e}")
                pass

        # Fallback: If AST fails to find a structural node (e.g., a rationale/comment node)
        # Just return the specific text from that exact line.
        if 0 < start_line <= len(lines):
            return lines[start_line - 1].strip()

    return "Error: Could not extract code block."


@tool
def check_package_vulnerability(package_name: str, version: str) -> list:
    """
    Use this tool immediately whenever you parse a dependency manifest (like
    package.json or requirements.txt) to check for known vulnerabilities.
    """
    query = {
        "package": {
            "name": package_name,
        },
        "version": version
    }
    response = requests.post("https://api.osv.dev/v1/query", json=query)
    data = json.loads(response.text)
    vulns = data.get("vulns")

    if not vulns:
        return []

    output = []
    for vuln in vulns:
        details = vuln.get("details")
        if not details:
            continue
        output.append({
            "id": vuln.get("id"),
            "aliases": vuln.get("aliases"),
            "details": vuln.get("details"),
            "severity": [s.get("score") for s in vuln.get("severity", [])]
        })

    return output


@tool
def take_notes(
    note: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Saves crucial information (variables, logic flows, hardcoded secrets) to your persistent memory.

    BEST PRACTICES FOR NOTES:
    - Keep it concise and use markdown.
    - Always include the context (e.g., file name, node ID, or endpoint).
    - Example: "Node src_main_py: Found SQL injection sink, where `query` is concatenated."
    - Example: "Login endpoint /api/auth requires CSRF token: `X-CSRF-TOKEN`."
    """
    return Command(
        update={
            "notes": [note],
            "messages": [
                ToolMessage(
                    content="Note successfully saved to your persistent memory.",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool
def submit_report(
    finding: VulnerabilityReport,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Call this tool whenever you find a unique, actionable vulnerability.
    You can call this tool multiple times if multiple flaws exist.
    """
    report_dict = finding.model_dump()
    report_dict["role"] = state["task"].agent_role

    return Command(
        update={
            "vulnerability_reports": [report_dict],
            "messages": [
                ToolMessage(
                    content="Successfully saved finding. Please continue your audit.",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool
def mark_task_complete(summary: str) -> dict:
    """Call this tool ONLY when you have analyzed EVERY single node assigned to you and are ready to finish."""
    return {"audit_status": "completed", "summary": summary}


@tool
def submit_evaluation(
    evaluation: EvaluationToolInput,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Call this tool when you have finished reviewing the source code and made a final decision."""

    evaluation_result = VulnerabilityEvaluation(
        report_id=state.get("report_id", "Unknown"),
        original_report=state.get("expert_report"),
        **evaluation.model_dump()
    )

    return Command(
        update={"filtered_reports": [evaluation_result]},
        goto="__end__"
    )


@tool(response_format="content_and_artifact")
def send_http_request(
    method: str,
    endpoint: str,
    headers: dict,
    body: str = "",
    reset_session: bool = False,
    extract_mode: str = "clean_html",
    state: Annotated[dict, InjectedState] = None
) -> tuple[str, dict]:
    """
    Sends an HTTP request to the sandboxed application. Use this for testing web endpoints.
    This tool preserve session by default, so that you can register and account and login.
    Use reset_session=True to clear the current session cookies.

    extract_mode options:
    - 'clean_html' (default): Returns HTML with scripts/styles removed to save tokens.
    - 'forms': Returns ONLY the <form> elements on the page.
    - 'links': Returns ONLY the <a> tags.
    - 'text': Returns only the visible text (good for reading error messages).
    - 'raw': Returns the untouched body (use cautiously, may truncate).
    - ANY CUSTOM TAG: Enter any HTML tag (e.g., 'form', 'a', 'script', 'input', 'iframe') to extract only those elements.
    """
    if not endpoint.startswith(settings.sandbox_url):
        return f"You can only make requests to the sandbox application at {settings.sandbox_url}", {}

    session = requests.Session()

    if not reset_session and state and "cookies" in state:
        session.cookies.update(state.get("cookies", {}))

    try:
        response = session.request(
            method=method,
            url=endpoint,
            headers=headers,
            data=body,
            timeout=5
        )

        raw_headers = "\r\n".join(f"{k}: {v}" for k, v in response.headers.items())
        http_response_head = f"HTTP/1.1 {response.status_code} {response.reason}\n{raw_headers}\r\n\r\n"

        # HTML parsing
        body_display = ""

        if "text/html" in response.headers.get("Content-Type", ""):
            soup = BeautifulSoup(response.text, 'html.parser')

            if extract_mode == "forms":
                forms = soup.find_all('form')
                body_display = f"[Found {len(forms)} forms]:\n\n" + "\n\n".join([str(f) for f in forms])
            elif extract_mode == "links":
                links = soup.find_all('a', href=True)
                body_display = f"[Found {len(links)} links]:\n" + "\n".join([str(l) for l in links])
            elif extract_mode == "text":
                body_display = soup.get_text(separator='\n', strip=True)
            elif extract_mode == "clean_html":
                # Destroy noise tags
                for noise in soup(['script', 'style', 'svg', 'noscript', 'canvas']):
                    noise.decompose()
                body_display = str(soup)
            elif extract_mode == "raw":
                body_display = response.text
            else:
                tags = soup.find_all(extract_mode)
                if tags:
                    body_display = f"[Found {len(tags)} <{extract_mode}> tags]:\n\n"
                    body_display += "\n\n".join([str(t) for t in tags[:50]])
                    if len(tags) > 50:
                        body_display += f"\n\n... [{len(tags) - 50} more tags truncated] ..."
                else:
                    body_display = f"[No <{extract_mode}> tags found on this page]"

        else:
            # If it's JSON or something else, just return the raw text
            body_display = response.text

        if len(body_display) > 8000:
            body_display = body_display[:8000] + "\n\n... [TRUNCATED: Try a specific extract_mode like 'forms' or 'text'] ..."

        llm_output = f"{http_response_head}{body_display}"

        return llm_output, session.cookies.get_dict()
    except Exception as e:
        return f"Request failed: {str(e)}", {}


@tool
def mark_validation_complete(
    validation: ValidationToolInput,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> dict:
    """
    Call this when you have definitively proven the vulnerability exists,
    or exhausted all options and believe it to be a false positive.
    """
    report = state.get("report_to_test")
    report_id = getattr(report, "report_id", "unknown")

    result = ValidationResult(
        report_id=report_id,
        **validation.model_dump()
    )

    return Command(
        update={"confirmed_vulnerabilities": [result]},
        goto="__end__"
    )


@tool
def search_codebase(keyword: str) -> str:
    """
    Searches the entire application codebase for a specific string. Use this
    to find where specific libraries, functions, or variables are used.
    """
    app_dir = Path(settings.app_path)

    # Load the graph to map physical files to Node IDs
    with open(settings.graph, "r") as f:
        graph_data = json.load(f)
        # Create a lookup dictionary: {"src/main.py": "node_123"}
        file_to_node = {
            n.get("source_file"): n.get("id")
            for n in graph_data.get("nodes", [])
            if n.get("source_file")
        }

    results = []
    match_count = 0
    MAX_MATCHES = 30 # prevent context window overflow

    # Recursively search all files
    for file_path in app_dir.rglob("*"):
        # Ignore hidden directories (like .git), pycache, and common heavy folders
        if any((part.startswith('.') and not part.startswith('..')) or \
            part in ['venv', '__pycache__', 'node_modules', 'graphify-out'] for part in file_path.parts) or \
            not file_path.is_file():
            continue

        try:
            # Read lines and search for the keyword
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if keyword in line:
                        relative_path = str(file_path.relative_to(app_dir))
                        # Match the file back to its Node ID so the agent can read it
                        node_id = file_to_node.get(relative_path, "Unknown (Not in Graph)")

                        results.append(
                            f"File: {relative_path} | Node ID: {node_id}\n"
                            f"Line {line_num}: {line.strip()}\n"
                        )
                        match_count += 1

                        # Stop if we hit the limit
                        if match_count >= MAX_MATCHES:
                            results.append(f"... [Truncated: found more than {MAX_MATCHES} matches] ...")
                            return "\n".join(results)

        except UnicodeDecodeError:
            # Safely skip binary files (images, compiled files, etc.)
            continue

    if not results:
        return f"No matches found for '{keyword}'."

    return "\n".join(results)


@tool
def get_node_connections(node_id: str) -> str:
    """
    Returns the neighbors of a node in the application graph.
    Use this to identify which functions call the current node (callers)
    or which functions/files the current node calls (callees).
    """
    try:
        G = build_networkx_graph(settings.graph)

        if node_id not in G:
            return f"Node ID '{node_id}' not found in the graph structure."

        successors = list(G.successors(node_id))
        predecessors = list(G.predecessors(node_id))

        return (
            f"Node: {node_id}\n"
            f"Called by (Predecessors): {predecessors}\n"
            f"Calls (Successors): {successors}"
        )
    except Exception as e:
        return f"Error traversing graph: {str(e)}"
