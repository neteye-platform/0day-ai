import hashlib
from typing import Annotated
import json
from langchain_core.messages import ToolMessage
import requests
from pathlib import Path
import logging
from bs4 import BeautifulSoup
from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
import docker
from docker.errors import NotFound, APIError
import re

from schemas import EvaluationToolInput, AnalysisNote, PackageCheck, ValidationToolInput
from utils import build_networkx_graph, cache, get_cached_symbol_index, get_node_code
import settings


@tool
def read_source_code(node_id: str, reason_for_reading: str, current_state: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Fetches the source code for a given Node ID.

    Args:
        node_id (str): The exact ID of the node to read (e.g., 'src_main_query_db').
        reason_for_reading (str): Explain exactly why you need to read THIS specific node next, and how you expect it to connect to your current knowledge.
        current_state (str): A detailed summary of the your current state and the outcome of your previous command.
    """
    messages = state.get("messages", [])
    for msg in messages[:-1]:
        msg = msg if isinstance(msg, dict) else msg.model_dump()
        if msg.get("type") == "ai":
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                if tc.get("name") == "read_source_code" and tc.get("args", {}).get("node_id") == node_id:
                    return f"System Notice: You have already read the source code for '{node_id}' in a previous step. The code is static and it will not change."

    node_code = get_node_code(node_id)

    if not node_code:
        return "Error: Could not extract code block."

    return node_code


@tool
def check_package_vulnerability(packages: list[PackageCheck]) -> list:
    """
    Use this tool immediately whenever you parse a dependency manifest (like
    package.json or requirements.txt) to check for known vulnerabilities.
    """
    output = []
    for pkg in packages:
        query = {"package": {"name": pkg.name}, "version": pkg.version}
        response = requests.post("https://api.osv.dev/v1/query", json=query)
        data = json.loads(response.text)
        vulns = data.get("vulns", [])

        if not vulns:
            output.append(f"[OK] {pkg.name}@{pkg.version}: No vulnerabilities found.")
            continue

        pkg_out = f"\n\n{pkg.name}\n"
        for vuln in vulns:
            details = vuln.get("details")
            if not details:
                continue
            pkg_out += f"  ID: {vuln.get('id', 'Unknown')}"
            pkg_out += f"  Aliases: {', '.join(vuln.get('aliases', []))}"
            pkg_out += f"  Details: {vuln.get('details', '')}"
            pkg_out += f"  Severity: {', '.join([s.get('score') for s in vuln.get('severity', [])])}"

        return output

    return []


@tool
def mark_task_complete(summary: str = "") -> dict:
    """Call this tool ONLY when you have analyzed EVERY single node assigned to you and are ready to finish."""
    return {"audit_status": "completed", "summary": summary}


@tool(args_schema=EvaluationToolInput)
def submit_evaluation(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    **kwargs
) -> Command:
    """Call this tool when you have finished reviewing the source code and made a final decision."""

    report = state.get("expert_report", {})
    node_id = state.get("node_id", "Unknown")

    # Mutate a copy of the single report
    updated_vuln = dict(report)
    updated_vuln["status"] = "confirmed" if kwargs.get("is_exploitable") else "false_positive"
    updated_vuln["confidence_score"] = kwargs.get("confidence_score")
    updated_vuln["reviewer_reasoning"] = kwargs.get("reasoning")
    updated_vuln["entry_point_url"] = kwargs.get("entry_point_url")
    updated_vuln["http_method"] = kwargs.get("http_method")
    updated_vuln["required_parameters"] = kwargs.get("required_parameters", [])
    updated_vuln["auth_required"] = kwargs.get("auth_required", False)

    tool_msg = ToolMessage(
        content="Evaluation submitted successfully. Ending review.",
        name="submit_evaluation",
        tool_call_id=tool_call_id
    )

    # Save to cache so subsequent runs skip the tool-calling loop
    report_hash = hashlib.md5(json.dumps(report, sort_keys=True).encode()).hexdigest()
    cache_file = settings.cache_dir / "reviewer" / f"{node_id}_{report_hash}.json"
    cache(cache_file, "write", updated_vuln)

    return Command(
        update={
            "vulnerabilities": [updated_vuln],
            "messages": [tool_msg]
        }
    )


@tool(response_format="content_and_artifact")
def send_http_request(
    method: str,
    endpoint: str,
    headers: dict,
    state: Annotated[dict, InjectedState],
    body: str = "",
    reset_session: bool = False,
    extract_mode: str = "clean_html"
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
    - ANY CUSTOM TAG: Enter any HTML tag (e.g., 'script', 'input', 'iframe') to extract only those elements.
    """
    # WARNING: You can only call this a maximum of 4 times before you must use the `take_notes` tool. Plan your batches accordingly.

    # Build the full URL from the endpoint parameter
    endpoint = endpoint.strip()
    if endpoint.startswith(("http://", "https://")):
        url = endpoint
    else:
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        url = f"{settings.sandbox_url}{endpoint}"

    if not url.startswith(settings.sandbox_url):
        return f"Error: You can only make requests to the sandbox application at {settings.sandbox_url}", {}

    session = requests.Session()

    if not reset_session and state and "cookies" in state:
        session.cookies.update(state.get("cookies", {}))

    try:
        response = session.request(
            method=method,
            url=url,
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
        return f"Error: Request failed: {str(e)}", {}


@tool(args_schema=ValidationToolInput)
def mark_validation_complete(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    **kwargs
) -> Command:
    """
    Call this when you have definitively proven the vulnerability exists,
    or exhausted all options and believe it to be a false positive.
    """
    # Get the single vulnerability assigned to this Validator agent
    report = state.get("report_to_test", {})

    # Create a copy to avoid mutating the local dictionary directly
    updated_vuln = dict(report)

    # Update the lifecycle status so the custom reducer merges it correctly
    if kwargs.get("is_exploitable"):
        updated_vuln["status"] = "exploitable"
    else:
        # If the exploit fails, mark it as a false positive
        updated_vuln["status"] = "false_positive"

    # Inject the Validator's findings
    updated_vuln["poc_payload"] = kwargs.get("poc_payload")
    updated_vuln["execution_logs"] = kwargs.get("execution_logs")

    tool_msg = ToolMessage(
        content="Validation complete. Ending validation phase.",
        name="mark_validation_complete",
        tool_call_id=tool_call_id
    )

    return Command(
        update={
            "vulnerabilities": [updated_vuln],
            "messages": [tool_msg]
        }
    )


@tool
def search_codebase(keyword: str, current_state: str, state: Annotated[dict, InjectedState], thought: str, regex: bool = True) -> str:
    """
    Searches the entire application codebase for a specific string or regular expression. 
    Use this to find where specific libraries, functions, variables, or class instantiations are used. 

    Args:
        keyword (str): The string or pattern to search in the codebase.
        regex (bool): Set to True if the keyword parameter is a regular expression, False otherwise (default = True).
        thought (str): Explain explicitly why you are running this search and what specific vulnerability path you are tracking.
        current_state (str): A detailed summary of the your current state and the outcome of your previous command.

    Returns:
        str: List of nodes with a match and the matched line of code.
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
    MAX_MATCHES = 20 # prevent context window overflow
    query = re.compile(keyword) if regex else re.escape(keyword)

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
                    match = re.search(query, line)
                    if match:
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
def get_node_connections(node_id: str, thought: str, current_state: str) -> str:
    """
    Returns the neighbors of a node in the application graph.
    Use this to identify which functions call the current node (callers)
    or which functions/files the current node calls (callees).

    Args:
        node_id (str): The identifier of the node in the graph.
        thought (str): Explain explicitly why examining the neighbors or data-flow edges of this node is necessary for your investigation.
        current_state (str): A detailed summary of the your current state and the outcome of your previous command.
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



@tool
def list_files(path: str = ".") -> str:
    """
    Lists files and directories in the specified path within the sandbox container.
    CRITICAL INSTRUCTION: Use this tool ONLY to verify the success of an exploit.
    DO NOT use this tool for initial reconnaissance, to read the source code, 
    or to understand the application structure. You already have all the context you need.

    Args:
        path (str): The directory path to inspect inside the container. Defaults to the current working directory.

    Returns:
        str: The raw output of the `ls -la` command, or an error message if the path doesn't exist.
    """
    container_name = settings.container_name

    try:
        client = docker.from_env()
        container = client.containers.get(container_name)

        # Execute the 'ls -la' command inside the container
        exit_code, output = container.exec_run(["ls", "-la", path])
        if not isinstance(output, bytes):
            return f"Error: Expected bytes, got {type(output).__name__}"

        decoded_output = output.decode("utf-8")

        if exit_code != 0:
            return f"Error listing files at '{path}':\n{decoded_output}"

        return decoded_output

    except NotFound:
        return f"Error: Container '{container_name}' not found. Ensure the sandbox is running."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


@tool
def read_file(path: str) -> str:
    """
    Reads the content of a file from the sandbox container.
    CRITICAL INSTRUCTION: Use this tool ONLY to verify the success of an exploit.
    DO NOT use this tool for initial reconnaissance, to read the source code, 
    or to understand the application structure. You already have all the context you need.

    Args:
        path: The absolute or relative path to the file inside the sandbox.
    """
    container_name = settings.container_name

    try:
        client = docker.from_env()
        container = client.containers.get(container_name)

        exit_code, output = container.exec_run(["cat", path])
        if not isinstance(output, bytes):
            return f"Error: Expected bytes, got {type(output).__name__}"

        if exit_code == 0:
            return output.decode('utf-8')
        else:
            error_msg = output.decode('utf-8').strip()
            return f"Error reading file '{path}': {error_msg} (Exit code: {exit_code})"

    except NotFound:
        return f"Error: The container '{container_name}' could not be found."
    except APIError as e:
        return f"Error: Docker API issue occurred: {str(e)}"
    except Exception as e:
        return f"Error: An unexpected error occurred: {str(e)}"


@tool
def get_definition(symbol_name: str, thought: str, current_state: str) -> str:
    """
    Retrieves the exact source code for a specific function or class method.
    If investigating a class method, format the input as ClassName::methodName.

    Args:
        symbol_name (str): The function or method name.
        thought (str): Explain explicitly why you are looking up this symbol and how it helps verify your hypothesis.
        current_state (str): A detailed summary of the your current state and the outcome of your previous command.
    """
    index_file_path = Path(settings.app_path) / ".ast_symbol_index.json"
    symbol_index = get_cached_symbol_index(index_file_path)

    if not symbol_index:
        return "Error: The AST symbol index is empty or could not be loaded."

    # Initial direct lookup
    target = next((item for item in symbol_index if item["name"] == symbol_name), None)
    resolution_trail = []

    # Inheritance traversal (if direct lookup fails)
    if not target and "::" in symbol_name:
        current_class, method = symbol_name.split("::", 1)

        while current_class:
            # Find any method belonging to the current class to look up its parent
            class_entry = next((item for item in symbol_index if item.get("class") == current_class), None)

            if not class_entry or not class_entry.get("parent"):
                break  # Reached the top of the chain, or class doesn't exist

            parent_class = class_entry["parent"]
            resolution_trail.append(parent_class)

            # Check if the parent implements the target method
            parent_symbol = f"{parent_class}::{method}"
            target = next((item for item in symbol_index if item["name"] == parent_symbol), None)

            if target:
                break  # We found the inherited method

            current_class = parent_class  # Move up to the next parent

    # Final verification
    if not target:
        return f"Error: The definition for '{symbol_name}' could not be found, even after checking parent classes."

    # Extract and Format
    filepath = target["filepath"]
    start_line = target["start_line"]
    end_line = target["end_line"]

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        code = "".join(lines[start_line - 1 : end_line])

        # Build the response string
        header = f"File: {filepath}\nLines: {start_line}-{end_line}\n"

        # Append the helpful note if we had to walk the inheritance tree
        if resolution_trail:
            original_class = symbol_name.split("::")[0]
            chain = " -> ".join(resolution_trail)
            header += f"\n> Note: Method resolved via inheritance: {original_class} -> {chain}\n"

        return f"{header}\n{code}"

    except Exception as e:
        return f"Error reading file {filepath}: {str(e)}"
