from typing import Annotated, Literal
import json
from langchain_core.messages import ToolMessage
import requests
from pathlib import Path
import logging
from bs4 import BeautifulSoup
from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool, InjectedToolCallId, ToolException
from langgraph.types import Command

from schemas import EvaluationToolInput, TakeNoteInput, ValidationToolInput, VulnerabilityReport, PackageCheck, VulnerabilityEvaluation
from utils import build_networkx_graph, enforce_note_taking, get_node_source_code
import settings

GUARDRAIL_MESSAGES = 8

@tool
def read_source_code(node_id: str, reason_for_reading: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Fetches the source code for a given Node ID.

    Args:
        node_id: The exact ID of the node to read (e.g., 'src_main_query_db').
        reason_for_reading: Explain exactly why you need to read THIS specific node next, and how you expect it to connect to your current knowledge.

    WARNING: You can only call this a maximum of 4 times before you must use the `take_notes` tool. Plan your batches accordingly.
    """
    rejection = enforce_note_taking(state.get("messages", []))
    if rejection:
        raise ToolException(rejection)

    node_code = get_node_source_code(settings.graph, node_id)

    if not node_code:
        raise ToolException("Error: Could not extract code block.")

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


@tool(args_schema=TakeNoteInput)
def take_notes(
    tool_call_id: Annotated[str, InjectedToolCallId],
    **kwargs
) -> Command:
    """
    Use this tool to save a structured summary of a code node IMMEDIATELY after reading it. You must call this tool exactly once for every node you analyze. Keep descriptions extremely brief and focused purely on data flow, access control, and security logic. Your notes will be preserved in your long-term memory for the duration of the audit.

    If you confirm a vulnerability, record it in the potential_issues field of this tool, and THEN immediately call the submit_report tool.
    """
    note = TakeNoteInput(**kwargs)
    note_dict = note.model_dump()

    return Command(
        update={
            "notes": [note_dict],
            "messages": [
                ToolMessage(
                    content="Note updated successfully.",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool(args_schema=VulnerabilityReport)
def submit_report(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    **kwargs
) -> Command:
    """
    Call this tool whenever you find a unique, actionable vulnerability.
    You can call this tool multiple times if multiple flaws exist.
    """
    finding = VulnerabilityReport(**kwargs)

    report_dict = finding.model_dump()
    report_dict["role"] = state["task"].get("agent_role") if isinstance(state["task"], dict) else state["task"].agent_role

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
def mark_task_complete(summary: str = "") -> dict:
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

    WARNING: You can only call this a maximum of 4 times before you must use the `take_notes` tool. Plan your batches accordingly.
    """

    rejection = enforce_note_taking(state.get("messages", []))
    if rejection:
        raise ToolException(rejection)

    if not endpoint.startswith(settings.sandbox_url):
        raise ToolException(f"You can only make requests to the sandbox application at {settings.sandbox_url}")

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
        raise ToolException(f"Request failed: {str(e)}")


@tool
def mark_validation_complete(
    validation: ValidationToolInput,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
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
def search_codebase(keyword: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Searches the entire application codebase for a specific string. Use this
    to find where specific libraries, functions, or variables are used.

    WARNING: You can only call this a maximum of 4 times before you must use the `take_notes` tool. Plan your batches accordingly.
    """

    rejection = enforce_note_taking(state.get("messages", []))
    if rejection:
        return rejection

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
