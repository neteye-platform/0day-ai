from langchain_core.tools import tool
import json
import settings
import requests
from pathlib import Path
import tree_sitter
import tree_sitter_python
import tree_sitter_javascript
import logging

LANGUAGE_MAP = {
    ".py": tree_sitter.Language(tree_sitter_python.language()),
    ".js": tree_sitter.Language(tree_sitter_javascript.language()),
    ".jsx": tree_sitter.Language(tree_sitter_javascript.language()),
}

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
