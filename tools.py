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

from schemas import EvaluationToolInput, AnalysisNote, PackageCheck, ValidationToolInput, AskForContextInput, IntegrationAuditInput, VulnerabilityDetailsInput
from utils import build_networkx_graph, get_cached_graph_data, get_cached_symbol_index, get_node_code, get_container_artifacts_root, cache_reviewer, reviewer_cache_key
from languages import MANIFEST_NAMES
import settings
import browser_tools


@tool
def read_source_code(node_id: str, reason_for_reading: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Fetches the source code for a given Node ID.

    Args:
        node_id (str): The exact ID of the node to read (e.g., 'src_main_query_db').
        reason_for_reading (str): Explain exactly why you need to read THIS specific node next, and how you expect it to connect to your current knowledge.
    """
    messages = state.get("messages", [])
    for msg in messages[:-1]:
        msg = msg if isinstance(msg, dict) else msg.model_dump()
        if msg.get("type") == "ai":
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                if tc.get("name") == "read_source_code" and tc.get("args", {}).get("node_id") == node_id:
                    return f"System Notice: You have already read the source code for '{node_id}' in a previous step. The code is static and it will not change."

    node_code = get_node_code(node_id, reviewer_mode=True)

    if not node_code:
        return "Error: Could not extract code block."

    return node_code


# Maximum number of lines read_file will return in a single call.
READ_FILE_MAX_LINES = 150


def _read_lines_range(file_path: str, target: Path, start_line: int,
                      end_line: int | None, max_lines: int, *, kind: str = "File",
                      header_path: str | None = None, continuation: str = "read_file") -> str:
    """Read a bounded line range of an existing text file.

    Shared by ``read_file`` and ``read_container_artifact`` so that paging,
    validation, and truncation behaviour stay identical between the two.
    """
    try:
        with open(target, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        return f"Error: '{file_path}' appears to be a binary file and cannot be read as text."
    except Exception as e:
        return f"Error reading {kind.lower()} '{file_path}': {e}"

    total_lines = len(lines)
    if total_lines == 0:
        return f"{kind} '{file_path}' is empty (0 lines)."

    requested_start = start_line
    if start_line < 1:
        start_line = 1
    if start_line > total_lines:
        return f"Error: start_line {requested_start} is beyond the end of '{file_path}' (file has {total_lines} lines)."

    requested_end = end_line if end_line is not None else total_lines
    if requested_end < start_line:
        return f"Error: end_line ({requested_end}) is smaller than start_line ({start_line})."

    end = min(requested_end, total_lines)

    truncated = False
    if end - start_line + 1 > max_lines:
        end = start_line + max_lines - 1
        truncated = True

    body = "".join(
        f"{i:>6}: {line}" for i, line in enumerate(lines[start_line - 1:end], start_line)
    )

    header = f"{kind}: {header_path or file_path} (lines {start_line}-{end} of {total_lines})\n"

    if truncated:
        body += (
            f"\n... [TRUNCATED: requested lines {requested_start}-{requested_end} exceeds the "
            f"{max_lines}-line limit. Shown lines {start_line}-{end}. "
            f"Call {continuation} again with start_line={end + 1} to continue reading.] ..."
        )

    return f"{header}\n{body}"


@tool
def read_file(file_path: str, start_line: int = 1, end_line: int | None = None) -> str:
    """
    Reads a specific line range of a file from the application directory by path.
    Use this for files that are NOT in the application graph (e.g., Dockerfile, config files, templates).
    For code that IS in the graph, prefer read_source_code or get_definition.
    Never returns more than 150 lines per call; use start_line/end_line to page through large files.

    Args:
        file_path (str): Path to the file, relative to the application root (e.g., 'Dockerfile', 'config/settings.py').
        start_line (int): First line to read, 1-indexed and inclusive. Defaults to 1.
        end_line (int): Last line to read, 1-indexed and inclusive. Defaults to the end of the file (or the 150-line cap).
    """
    app_dir = Path(settings.app_path).resolve()
    target = (app_dir / file_path).resolve()

    if not target.is_relative_to(app_dir):
        return (
            f"Error: '{file_path}' resolves to '{target}', which is outside the "
            f"application directory '{app_dir}'. Only files within the app are readable."
        )

    if not target.is_file():
        return f"Error: File '{file_path}' not found in the application directory."

    return _read_lines_range(file_path, target, start_line, end_line, READ_FILE_MAX_LINES)


# Maximum number of lines read_container_artifact will return in a single call.
CONTAINER_ARTIFACT_MAX_LINES = 150
CONTAINER_ARTIFACT_MAX_SUMMARY_FILES = 300
# Maximum number of paths find_in_container returns per call.
CONTAINER_ARTIFACT_MAX_SEARCH_MATCHES = 200


@tool
def list_container_artifacts() -> str:
    """
    Lists the config/build artifacts extracted from the BUILT container image(s)
    during pre-processing, plus image metadata (WORKDIR, ENTRYPOINT, CMD, EXPOSE,
    USER, baked-in ENV vars).

    Use this to inspect the EFFECTIVE runtime configuration of the target as it
    exists inside the container (e.g. the resolved Next.js config in
    .next/required-server-files.json, an nginx server config, an entrypoint
    script, or credentials baked into the image ENV) — without touching the
    live sandbox. To locate files NOT extracted here (by name or pattern), use
    find_in_container.

    This tool takes no arguments.
    """
    artifacts_root = get_container_artifacts_root().resolve()

    if not artifacts_root.exists() or not any(p.is_dir() for p in artifacts_root.iterdir()):
        return (
            "No container artifacts are available. The preprocessor did not "
            "build/snapshot any container image for this target (no Dockerfile/"
            "compose found, the build failed, or docker is unavailable)."
        )

    sections = []
    for image_dir in sorted(p for p in artifacts_root.iterdir() if p.is_dir()):
        slug = image_dir.name
        lines = [f"Image snapshot: {slug}", f"  Directory: {image_dir}"]

        # Image metadata (ENV / WORKDIR / ENTRYPOINT / CMD / EXPOSE / USER / LABELS)
        meta_file = image_dir / "image_metadata.json"
        if meta_file.is_file():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                meta = {}
            if meta.get("WorkingDir"):
                lines.append(f"  WORKDIR: {meta['WorkingDir']}")
            for key in ("Entrypoint", "Cmd", "ExposedPorts", "User", "Labels"):
                if meta.get(key):
                    lines.append(f"  {key}: {json.dumps(meta[key])}")
            env = meta.get("Env") or []
            if env:
                lines.append("  ENV:")
                for pair in env:
                    key, _, value = pair.partition("=")
                    shown = value if len(value) <= 500 else value[:500] + "...[truncated]"
                    lines.append(f"    {key}={shown}")

        # Extracted files
        summary_file = image_dir / "extraction_summary.json"
        summary = {}
        extracted = []
        catchall = []
        if summary_file.is_file():
            try:
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
                extracted = summary.get("extracted", [])
                catchall = summary.get("catchall", [])
            except (json.JSONDecodeError, OSError):
                extracted = []
        rootfs_dir = image_dir / "rootfs"
        # Fall back to walking the tree when the summary is missing.
        if not extracted and rootfs_dir.is_dir():
            extracted = sorted(
                str(p.relative_to(rootfs_dir)) for p in rootfs_dir.rglob("*") if p.is_file()
            )

        lines.append(f"  Extracted files ({len(extracted)}):")
        if catchall:
            lines.append(
                f"    ({len(catchall)} via WORKDIR catch-all: small text config "
                f"files the curated patterns did not match.)"
            )
        for rel in extracted[:CONTAINER_ARTIFACT_MAX_SUMMARY_FILES]:
            size = ""
            target = rootfs_dir / rel if rootfs_dir else image_dir / rel
            if target.is_file():
                try:
                    size = f" ({target.stat().st_size} bytes)"
                except OSError:
                    pass
            lines.append(f"    rootfs/{rel}{size}")
        if len(extracted) > CONTAINER_ARTIFACT_MAX_SUMMARY_FILES:
            lines.append(f"    ... [{len(extracted) - CONTAINER_ARTIFACT_MAX_SUMMARY_FILES} more] ...")

        sections.append("\n".join(lines))

    return (
        "CONTAINER ARTIFACTS (built-image config snapshot)\n"
        "Read any extracted file with read_container_artifact, e.g. "
        "read_container_artifact(file_path='<slug>/rootfs/<path>').\n\n"
        + "\n\n".join(sections)
    )


@tool
def read_container_artifact(file_path: str, start_line: int = 1, end_line: int | None = None) -> str:
    """
    Reads a specific line range of a file extracted from the BUILT container
    image snapshot. Use this to inspect effective runtime configuration that is
    NOT visible in the application repo (e.g. resolved framework configs,
    server configs, entrypoint scripts, baked-in ENV files).

    For files that ARE in the application graph, prefer read_source_code or
    get_definition; for files in the application repo, prefer read_file.

    Args:
        file_path (str): Path relative to the artifacts root, as shown by
            list_container_artifacts (e.g. 'vulnscan-web_reactoops_latest/rootfs/app/.next/required-server-files.json'
            or 'vulnscan-web_reactoops_latest/rootfs/bin/sh'). When only
            one image was snapshotted you may omit the '<slug>/' prefix.
        start_line (int): First line to read, 1-indexed and inclusive. Defaults to 1.
        end_line (int): Last line to read, 1-indexed and inclusive. Defaults to the end of the file (or the 150-line cap).
    """
    artifacts_root = get_container_artifacts_root().resolve()

    if not artifacts_root.exists():
        return "Error: No container artifacts are available for this target."

    def _resolve(candidate: Path) -> Path:
        return (artifacts_root / candidate).resolve()

    target = _resolve(file_path)

    # Convenience: when there is exactly one image snapshot, allow omitting the
    # '<slug>/' prefix (try both rootfs/ and top-level files).
    if not target.is_relative_to(artifacts_root):
        return (
            f"Error: '{file_path}' resolves to '{target}', which is outside the "
            f"artifacts directory '{artifacts_root}'. Only extracted container "
            f"artifacts are readable."
        )

    if not target.is_file():
        image_dirs = [p for p in artifacts_root.iterdir() if p.is_dir()]
        if len(image_dirs) == 1:
            slug = image_dirs[0].name
            for candidate in (image_dirs[0] / "rootfs" / file_path, image_dirs[0] / file_path):
                resolved = candidate.resolve()
                if resolved.is_file() and resolved.is_relative_to(artifacts_root):
                    target = resolved
                    break
        if not target.is_file():
            return f"Error: Artifact '{file_path}' not found in the container artifacts directory."

    rel = target.relative_to(artifacts_root)
    return _read_lines_range(
        file_path, target, start_line, end_line,
        CONTAINER_ARTIFACT_MAX_LINES,
        kind="Artifact", header_path=rel, continuation="read_container_artifact",
    )


@tool
def find_in_container(keyword: str, is_regex: bool = False) -> str:
    """
    Searches the BUILT container image filesystem index for paths matching a
    keyword (substring) or regular expression. Use this to locate config files,
    binaries, scripts, or data files that were NOT extracted by the curated
    artifact patterns (e.g. a nonstandard server config, a helper script, or a
    credential/key file baked into the image).

    The index is the curated container filesystem path list produced at image
    snapshot time; dependency install trees (node_modules/vendor) and VCS
    metadata are excluded. Only 'extracted' matches are readable with
    read_container_artifact.

    Args:
        keyword (str): The string or regular expression pattern to match against
            container filesystem paths (e.g. 'nginx', 'private.key', 'entrypoint').
        is_regex (bool): Set to True if keyword is a regular expression,
            False (default) for a literal substring search. When True you can
            search multiple keywords at once with an alternation regex like
            'nginx|apache|traefik'.
    """
    artifacts_root = get_container_artifacts_root().resolve()

    if not artifacts_root.exists() or not any(p.is_dir() for p in artifacts_root.iterdir()):
        return (
            "No container artifacts are available. The preprocessor did not "
            "build/snapshot any container image for this target (no Dockerfile/"
            "compose found, the build failed, or docker is unavailable)."
        )

    prefix = ""
    try:
        pat = re.compile(keyword) if is_regex else re.compile(re.escape(keyword))
    except re.error as e:
        prefix = (
            f"NOTE: {keyword!r} was an invalid regular expression ({e}); "
            f"searched as a literal string instead.\n"
        )
        pat = re.compile(re.escape(keyword))

    matches = []

    for image_dir in sorted(p for p in artifacts_root.iterdir() if p.is_dir()):
        if not image_dir.is_dir():
            continue
        slug = image_dir.name
        index_file = image_dir / "filesystem_index.txt"
        if not index_file.is_file():
            continue
        rootfs_dir = image_dir / "rootfs"
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                index_lines = f.read().splitlines()
        except (OSError, UnicodeDecodeError) as e:
            logging.info(f"Failed to read index for '{slug}': {e}")
            continue

        for line in index_lines:
            # Skip the header (and any malformed line): entries are
            # "type\tsize\tpath".
            parts = line.split("\t")
            if len(parts) != 3 or parts[0] == "#":
                continue
            _type, _size, path = parts
            if pat.search(path):
                extracted = (rootfs_dir / path).is_file()
                matches.append(
                    f"{slug} | {_type}\t{_size}\t{path}\t"
                    f"{'[extracted]' if extracted else '[not extracted]'}"
                )
                if len(matches) >= CONTAINER_ARTIFACT_MAX_SEARCH_MATCHES:
                    matches.append(
                        f"... [Truncated: found more than "
                        f"{CONTAINER_ARTIFACT_MAX_SEARCH_MATCHES} matches] ..."
                    )
                    return prefix + "\n".join(matches)

    if not matches:
        return f"{prefix}No matches for '{keyword}' in the container filesystem index."

    return (
        prefix
        + "CONTAINER FILESYSTEM MATCHES (type, size, path, extracted status)\n"
        + "Read an '[extracted]' match with read_container_artifact, e.g. "
        "read_container_artifact(file_path='<slug>/rootfs/<path>').\n"
        + "\n".join(matches)
    )


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

    # Mutate a copy of the single report
    updated_vuln = dict(report)
    updated_vuln["status"] = "confirmed" if kwargs.get("is_exploitable") else "false_positive"
    updated_vuln["confidence_score"] = kwargs.get("confidence_score")
    updated_vuln["reviewer_reasoning"] = kwargs.get("reasoning")
    updated_vuln["reproduction_steps"] = kwargs.get("reproduction_steps", [])
    updated_vuln["validation_strategy"] = kwargs.get("validation_strategy")

    tool_msg = ToolMessage(
        content="Evaluation submitted successfully. Ending review.",
        name="submit_evaluation",
        tool_call_id=tool_call_id
    )

    # Save to cache so subsequent runs skip the tool-calling loop
    cache_reviewer(reviewer_cache_key(report, state.get("node_id", "Unknown")), report, updated_vuln)

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
    sandbox_url = state.get("sandbox_url")
    if not sandbox_url:
        return "Error: No sandbox is configured. The preprocessor could not start a sandbox container.", {}
    if endpoint.startswith(("http://", "https://")):
        url = endpoint
    else:
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        url = f"{sandbox_url}{endpoint}"

    if not url.startswith(sandbox_url):
        return f"Error: You can only make requests to the sandbox application at {sandbox_url}", {}

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

    # Update the lifecycle status so the custom reducer merges it correctly.
    # is_confirmed drives the verdict (an unconvincing exploit = false positive).
    if kwargs.get("is_confirmed"):
        updated_vuln["status"] = "exploitable"
    else:
        updated_vuln["status"] = "false_positive"

    # Inject the Validator's findings
    updated_vuln["poc_payload"] = kwargs.get("poc_payload")
    updated_vuln["execution_logs"] = kwargs.get("execution_logs")

    # Close this validator's headless-browser sessions (per-agent, never
    # touching other concurrently running validators' sessions).
    browser_tools.manager.close_agent_sessions(state.get("agent_id"))

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


@tool(args_schema=AskForContextInput)
def ask_for_context(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    **kwargs
) -> Command:
    """
    Call this when you CANNOT reach a verdict because the report leaves you
    unable to test the vulnerability — no reachable sandbox/route, ambiguous
    reproduction steps, a missing HTTP method/path/body, or blocking
    authentication/session details. Flag the record as insufficient_context and
    enumerate the specific questions the Reviewer must answer. This tool is
    available only on your FIRST validation pass; after a re-review it is
    removed and you must conclude via mark_validation_complete instead.
    """
    # Get the single vulnerability assigned to this Validator agent
    report = state.get("report_to_test", {})

    # Create a copy to avoid mutating the local dictionary directly
    updated_vuln = dict(report)

    # Flag the record with the concrete questions the Reviewer must resolve and
    # bump the feedback round; route_validator_feedback re-dispatches it to the
    # reviewer (only while review_round is within validator_feedback_max_rounds).
    current_round = report.get("review_round") or 0
    updated_vuln["status"] = "insufficient_context"
    updated_vuln["review_round"] = current_round + 1
    updated_vuln["open_questions"] = list(kwargs.get("open_questions") or [])
    updated_vuln["execution_logs"] = kwargs.get("reasoning")
    updated_vuln["poc_payload"] = None

    # Close this validator's headless-browser sessions (per-agent, never
    # touching other concurrently running validators' sessions).
    browser_tools.manager.close_agent_sessions(state.get("agent_id"))

    tool_msg = ToolMessage(
        content=(
            "Context insufficient. Requesting more context from the Reviewer; "
            "you will not continue validating unless this record is re-dispatched."
        ),
        name="ask_for_context",
        tool_call_id=tool_call_id
    )

    return Command(
        update={
            "vulnerabilities": [updated_vuln],
            "messages": [tool_msg]
        }
    )


@tool(args_schema=VulnerabilityDetailsInput)
def get_vulnerability_details(
    vuln_id: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """
    Fetches the full record of another confirmed vulnerability by its vuln_id, so
    you can reason over its real mechanics (full description, reviewer reasoning,
    reproduction steps) when deciding whether it chains with your assigned
    vulnerability. Only entries from the provided summary of other confirmed
    vulnerabilities are available.
    """
    confirmed = state.get("confirmed_vulns", [])
    for record in confirmed:
        if record.get("vuln_id") == vuln_id:
            return json.dumps(record, indent=2, default=str)
    available = ", ".join(r.get("vuln_id", "?") for r in confirmed) or "none"
    return (
        f"Error: no confirmed vulnerability with vuln_id '{vuln_id}' is available. "
        f"Available vuln_ids: {available}"
    )


@tool(args_schema=IntegrationAuditInput)
def submit_integration_audit(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    **kwargs
) -> Command:
    """Call this tool when you have decided whether the assigned `requires_integration`
    vulnerability combines with other confirmed vulnerabilities into a concrete
    multi-step exploit chain (chained) or cannot be chained (unchainable)."""
    report = state.get("report_to_test", {})

    # Mutate a copy of the single report (same vuln_id, upgrade in place).
    updated_vuln = dict(report)
    updated_vuln["status"] = "chained" if kwargs.get("is_chained") else "unchainable"
    updated_vuln["confidence_score"] = kwargs.get("confidence_score")
    updated_vuln["integration_audit_reasoning"] = kwargs.get("reasoning")
    updated_vuln["chained_with"] = kwargs.get("chained_with")
    if kwargs.get("is_chained"):
        # The auditor's reproduction_steps are the FULL combined chain plan the
        # downstream Validator executes to build the PoC.
        updated_vuln["reproduction_steps"] = kwargs.get("reproduction_steps", [])

    tool_msg = ToolMessage(
        content="Integration audit submitted. Ending chaining review.",
        name="submit_integration_audit",
        tool_call_id=tool_call_id
    )

    return Command(
        update={
            "vulnerabilities": [updated_vuln],
            "messages": [tool_msg]
        }
    )


@tool
def search_codebase(keyword: str, state: Annotated[dict, InjectedState], regex: bool = True) -> str:
    """
    Searches the entire application codebase for a specific string or regular expression. 
    Use this to find where specific libraries, functions, variables, or class instantiations are used. 

    Args:
        keyword (str): The string or pattern to search in the codebase.
        regex (bool): Set to True if the keyword parameter is a regular expression, False otherwise (default = True). When True you can search multiple keywords at once with an alternation regex like 'auth|login|token'.

    Returns:
        str: List of nodes with a match and the matched line of code.
    """
    app_dir = Path(settings.app_path)

    # Load the graph (manifest/dependency nodes already stripped centrally) to
    # map physical files to Node IDs: {"src/main.py": "node_123"}
    graph_data = get_cached_graph_data(settings.graph)
    file_to_node = {
        n.get("source_file"): n.get("id")
        for n in graph_data.get("nodes", [])
        if n.get("source_file")
    }

    results = []
    match_count = 0
    MAX_MATCHES = 20 # prevent context window overflow
    regex_fallbacks = []
    try:
        query = re.compile(keyword) if regex else re.escape(keyword)
    except re.error as e:
        regex_fallbacks.append(
            f"NOTE: {keyword!r} was an invalid regular expression ({e}); searched as a literal string instead."
        )
        query = re.escape(keyword)

    # Recursively search all files
    for file_path in app_dir.rglob("*"):
        # Ignore dependency manifest/lockfiles (handled by the SCA layer) plus
        # hidden directories (like .git), pycache, and common heavy folders.
        if file_path.is_file() and file_path.name in MANIFEST_NAMES:
            continue
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

    prefix = "\n".join(regex_fallbacks) + "\n" if regex_fallbacks else ""

    if not results:
        return f"{prefix}No matches found for '{keyword}'."

    return prefix + "\n".join(results)


@tool
def get_node_connections(node_id: str) -> str:
    """
    Returns the neighbors of a node in the application graph.
    Use this to identify which functions call the current node (callers)
    or which functions/files the current node calls (callees).

    Args:
        node_id (str): The identifier of the node in the graph.
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
def list_files(path: str = ".", state: Annotated[dict, InjectedState] = {}) -> str:
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
    container_name = state.get("container_name")
    if not container_name:
        return "Error: No sandbox container is configured. Ensure the preprocessor started the sandbox."

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
def read_sandbox_file(path: str, state: Annotated[dict, InjectedState] = {}) -> str:
    """
    Reads the content of a file from the sandbox container.
    CRITICAL INSTRUCTION: Use this tool ONLY to verify the success of an exploit.
    DO NOT use this tool for initial reconnaissance, to read the source code, 
    or to understand the application structure. You already have all the context you need.

    Args:
        path: The absolute or relative path to the file inside the sandbox.
    """
    container_name = state.get("container_name")
    if not container_name:
        return "Error: No sandbox container is configured. Ensure the preprocessor started the sandbox."

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
def get_definition(symbol_name: str) -> str:
    """
    Retrieves the exact source code for a specific function or class method.
    If investigating a class method, format the input as ClassName::methodName.

    Args:
        symbol_name (str): The function or method name.
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
