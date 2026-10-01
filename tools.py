from typing import Annotated
from langchain_core.tools import tool
import json
from langgraph.prebuilt import InjectedState
import settings
import requests
from pathlib import Path
import tree_sitter
import tree_sitter_python
import tree_sitter_javascript
import logging

from schemas import VulnerabilityReport

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
def read_source_code(node_id: str, current_analysis: str, reason_for_reading: str) -> str:
    """
    Fetches the source code for a given Node ID.

    Args:
        node_id: The exact ID of the node to read (e.g., 'src_main_query_db').
        current_analysis: A detailed summary of the vulnerabilities, data flows, or logic flaws you have found in the code you have ALREADY read. If this is your first read, state your initial hypothesis. You MUST NOT leave this blank.
        reason_for_reading: Explain exactly why you need to read THIS specific node next, and how you expect it to connect to your current analysis.
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
    Return known vulnerabilities for the give package version
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
def submit_report(finding: VulnerabilityReport, state: Annotated[dict, InjectedState]) -> dict:
    """
    Call this tool whenever you find a unique, actionable vulnerability. 
    You can call this tool multiple times if multiple flaws exist.
    """
    report_dict = finding.model_dump()
    report_dict["role"] = state["task"].agent_role
    return {"vulnerability_reports": [report_dict]}


@tool
def mark_task_complete(summary: str) -> dict:
    """Call this tool ONLY when you have analyzed EVERY single node assigned to you and are ready to finish."""
    return {"audit_status": "completed", "summary": summary}

@tool
def send_http_request(method: str, endpoint: str, headers: dict, body: str = "") -> str:
    """
    Sends an HTTP request to the sandboxed application.
    Use this for testing XSS, SQLi, SSRF, and command injection via web endpoints.
    """
    if not endpoint.startswith(settings.sandbox_url):
        return f"You can only make requests to the sandbox application at {settings.sandbox_url}"

    import requests
    # Ensure endpoint is pointing to the sandbox environment!
    try:
        response = requests.request(
            method=method,
            url=endpoint,
            headers=headers,
            data=body,
            timeout=5
        )
        return f"Status: {response.status_code}\nHeaders: {response.headers}\nBody: {response.text[:2000]}"
    except Exception as e:
        return f"Request failed: {str(e)}"


@tool
def mark_validation_complete(is_confirmed: bool, poc_payload: str, evidence: str) -> dict:
    """
    Call this when you have definitively proven the vulnerability exists, 
    or exhausted all options and believe it to be a false positive.
    """
    return {
        "is_confirmed": is_confirmed,
        "poc_payload": poc_payload,
        "execution_logs": evidence
    }
