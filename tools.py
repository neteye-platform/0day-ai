from langchain_core.tools import tool
import json
import os
import ast
import settings
import requests


@tool
def read_source_code(node_id: str) -> str:
    """
    Fetches the source code for a given Node ID.
    """
    try:
        with open(settings.graph, "r") as f:
            graph_data = json.load(f)
    except FileNotFoundError:
        return f"Error: {settings.graph} not found."

    target_node = next((node for node in graph_data.get("nodes", []) if node.get("id") == node_id), None)
    if not target_node:
        return f"Error: Node ID '{node_id}' not found in graph."

    source_file = target_node.get("source_file")
    source_location = target_node.get("source_location")
    file_type = target_node.get("file_type")

    if not source_file:
        return f"Error: Node '{node_id}' does not have a source file mapped."
    source_file = os.path.join(settings.app_path, source_file)

    if file_type == "document" or not source_location:
        try:
            # Enforce utf-8 encoding. If it is not a text file
            # this will trigger a UnicodeDecodeError
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
        if not os.path.exists(source_file):
            return f"Error: Source file '{source_file}' not found on disk. Ensure paths are correct."
        with open(source_file, "r") as f:
            source_content = f.read()

        # Convert line location string "L53" to integer 53
        try:
            start_line = int(source_location.replace("L", ""))
        except ValueError:
            return f"Error: Invalid source_location format '{source_location}'."

        # Use AST to extract the exact code block
        try:
            parsed_ast = ast.parse(source_content)
        except SyntaxError:
            return f"Error: Source file '{source_file}' contains syntax errors and could not be parsed."

        # Walk through the parsed AST to find the node that starts on our target line
        for ast_node in ast.walk(parsed_ast):
            if hasattr(ast_node, "lineno") and ast_node.lineno == start_line:
                extracted_code = ast.get_source_segment(source_content, ast_node)
                if extracted_code:
                    return extracted_code

        # Fallback: If AST fails to find a structural node (e.g., a rationale/comment node)
        # Just return the specific text from that exact line.
        lines = source_content.splitlines()
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
