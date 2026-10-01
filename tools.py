from langchain_core.tools import tool
import json
import os
import ast
import settings


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




# import os
# from typing import Optional
# import tree_sitter_python as tspython
# from tree_sitter import Language, Parser, Node
# import subprocess
#
# PY_LANGUAGE = Language(tspython.language())
# parser = Parser(PY_LANGUAGE)
# WORKSPACE_BASE = os.path.abspath("./testapp")
#
#
# @tool
# def grep_search(pattern: str, directory: str) -> str:
#     """Use this to find exact text patterns, variable usages, or keywords across the codebase."""
#     print(f"   [Tool Execution] Running grep for: {pattern}")
#     try:
#         # Run a safe grep command
#         result = subprocess.run(
#             ["grep", "-rnw", os.path.join(WORKSPACE_BASE, directory), "-e", pattern],
#             capture_output=True, text=True, timeout=10
#         )
#         # Truncate output to prevent context window overflow
#         return result.stdout[:2000] + "\n...[truncated]" if len(result.stdout) > 2000 else result.stdout
#     except Exception as e:
#         return f"Error running grep: {str(e)}"
#
#
# @tool
# def get_definition(symbol_name: str, file_path: str) -> str:
#     """
#     Parses a local Python file using Tree-Sitter AST and extracts the full
#     source code definition of a specific function or class by its name.
#     """
#     print(f"   [Tool Execution] Searching AST for symbol '{symbol_name}' in: {file_path}")
#     full_path = os.path.join(WORKSPACE_BASE, file_path)
#
#     if not os.path.exists(full_path):
#         return f"Error: Target file not found at {file_path}"
#
#     with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
#         source_code = f.read()
#
#     # Tree-sitter handles parsing via UTF-8 encoded bytes
#     source_bytes = bytes(source_code, "utf8")
#     tree = parser.parse(source_bytes)
#     root_node = tree.root_node
#
#     # Inner helper function to recursively navigate the tree
#     def find_target_node(node: Node, target: str) -> Optional[Node]:
#         # Check if the current node is a function or class definition
#         if node.type in ("function_definition", "class_definition"):
#             # Retrieve the child node mapped to the field name 'name'
#             name_node = node.child_by_field_name("name")
#             if name_node:
#                 # Extract the literal name string from the source bytes
#                 extracted_name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf8")
#                 if extracted_name == target:
#                     return node
#
#         # Recursively process the children of this node
#         for child in node.children:
#             found = find_target_node(child, target)
#             if found:
#                 return found
#         return None
#
#     # Search the AST starting from the root module node
#     target_node = find_target_node(root_node, symbol_name)
#
#     if target_node:
#         # Use the node's exact start and end byte configurations to isolate the source text
#         defined_source = source_bytes[target_node.start_byte:target_node.end_byte].decode("utf8")
#         return defined_source
#
#     return f"Error: Symbol '{symbol_name}' could not be found in the structural layout of {file_path}."
#
#
# @tool
# def get_callers(target_function: str, directory: str) -> str:
#     """
#     Scans a directory for Python files and uses Tree-Sitter AST parsing
#     to find all functions or methods that call the target_function.
#     """
#     print(f"   [Tool Execution] Resolving callers for '{target_function}' in: {directory}")
#
#     if not os.path.exists(directory):
#         return f"Error: Directory not found at {directory}"
#
#     callers_found = []
#
#     # 1. Helper to traverse UP the tree to find the enclosing context
#     def get_enclosing_context(node: Node, source_bytes: bytes) -> str:
#         current = node
#         context_name = "Global Scope"
#
#         while current.parent:
#             current = current.parent
#             if current.type in ("function_definition", "class_definition"):
#                 name_node = current.child_by_field_name("name")
#                 if name_node:
#                     context_name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf8")
#                 break
#         return context_name
#
#     # 2. Walk the directory to parse every Python file
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if not file.endswith(".py"):
#                 continue
#
#             file_path = os.path.join(root, file)
#             try:
#                 with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
#                     source_code = f.read()
#
#                 source_bytes = bytes(source_code, "utf8")
#                 tree = parser.parse(source_bytes)
#
#                 # 3. Recursive helper to find call nodes
#                 def find_calls(node: Node):
#                     if node.type == "call":
#                         # The function being called is mapped to the 'function' field
#                         func_node = node.child_by_field_name("function")
#                         if func_node:
#                             call_text = source_bytes[func_node.start_byte:func_node.end_byte].decode("utf8")
#
#                             # Match standard calls (e.g., target_function) or method calls (e.g., obj.target_function)
#                             if call_text == target_function or call_text.endswith(f".{target_function}"):
#                                 enclosing = get_enclosing_context(node, source_bytes)
#                                 # Record the caller, the file, and the line number (Tree-sitter uses 0-indexed rows)
#                                 callers_found.append(f"- `{enclosing}()` in {file_path} (Line {node.start_point[0] + 1})")
#
#                     # Continue searching children
#                     for child in node.children:
#                         find_calls(child)
#
#                 find_calls(tree.root_node)
#
#             except Exception as e:
#                 print(f"Error parsing {file_path}: {e}")
#
#     # 4. Format the final output for the LLM
#     if not callers_found:
#         return f"No callers found for '{target_function}' in {directory}."
#
#     unique_callers = list(set(callers_found))
#     return "Found the following callers:\n" + "\n".join(unique_callers)
#
#
# @tool
# def ls_directory(relative_path: str = ".") -> str:
#     """
#     Lists the contents of a directory.
#     Use this to explore the repository structure step-by-step.
#     Provide a relative path (e.g., "." for root, or "src/controllers" for subdirectories).
#     """
#     print(f"   [Tool Execution] Running ls on: {relative_path}")
#
#     # Resolve the absolute path and ensure it stays within the workspace
#     target_dir = os.path.abspath(os.path.join(WORKSPACE_BASE, relative_path))
#
#     if not target_dir.startswith(WORKSPACE_BASE):
#         return "Error: Access denied. Cannot navigate outside the repository workspace."
#
#     if not os.path.exists(target_dir):
#         return f"Error: Directory '{relative_path}' does not exist."
#
#     if not os.path.isdir(target_dir):
#         return f"Error: '{relative_path}' is a file, not a directory."
#
#     try:
#         items = os.listdir(target_dir)
#
#         # Sort and format the output: folders first (with a trailing /), then files
#         folders = []
#         files = []
#         for item in items:
#             # Ignore hidden files/folders like .git
#             if item.startswith("."):
#                 continue
#
#             item_path = os.path.join(target_dir, item)
#             if os.path.isdir(item_path):
#                 folders.append(f"{item}/")
#             else:
#                 files.append(item)
#
#         folders.sort()
#         files.sort()
#
#         output = folders + files
#
#         if not output:
#             return f"Directory '{relative_path}' is empty."
#
#         return "\n".join(output)
#
#     except Exception as e:
#         return f"Error reading directory: {str(e)}"
