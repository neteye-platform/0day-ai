from typing import Annotated, Optional, Any, Union
import json
from langchain_core.messages import ToolMessage
import requests
from requests.adapters import HTTPAdapter
from pathlib import Path
import logging
import threading
from bs4 import BeautifulSoup
from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
import re
import networkx as nx

from schemas import EvaluationToolInput, ValidationToolInput, AskForContextInput, IntegrationAuditInput, VulnerabilityDetailsInput, cwes
from utils import build_networkx_graph, get_cached_graph_data, get_cached_symbol_index, get_node_code, get_container_artifacts_root, cache_reviewer, cache_validator, cache_integration_auditor, reviewer_cache_key, is_path_excluded
from languages import MANIFEST_NAMES
import settings
import browser_tools
import attacker_tools


@tool
def read_source_code(node_id: str, include_context: bool = False, state: Annotated[Optional[dict], InjectedState] = None) -> str:
    """
    Fetches the source code for a given Node ID.

    Args:
        node_id (str): The exact ID of the node to read (e.g., 'src_main_query_db').
        include_context (bool): If True, return the node's code within its full file
            context (sibling bodies pruned). If False (default), return only the node's
            own code without file context; assume all necessary imports and global
            variables are securely defined elsewhere.
    """
    messages = state.get("messages", [])
    for msg in messages[:-1]:
        msg = msg if isinstance(msg, dict) else msg.model_dump()
        if msg.get("type") == "ai":
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                if (tc.get("name") == "read_source_code"
                        and tc.get("args", {}).get("node_id") == node_id
                        and tc.get("args", {}).get("include_context", False) == include_context):
                    return f"System Notice: You have already read the source code for '{node_id}' in a previous step. The code is static and it will not change."

    node_code = get_node_code(node_id, reviewer_mode=True, raw=not include_context)

    if not node_code:
        return "Error: Could not extract code block."

    return node_code


# Maximum number of lines read_file will return in a single call.
READ_FILE_MAX_LINES = 150

# Bounding for get_path: max number of paths returned and max path length (hops).
MAX_PATHS = 50
MAX_PATH_CUTOFF = 12


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

    if is_path_excluded(file_path):
        return (
            f"Error: '{file_path}' is in an excluded scan path (dependency trees, "
            f"tests, or docs). It is not part of the analyzed application surface."
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


# --- send_http_request: redirect cap, per-session cookies, CSRF handling -----

# Sentinel form-field values that request automatic CSRF resolution: the tool
# GETs a fresh page, extracts the token, and substitutes the real token (and,
# when the agent guessed the wrong field name, the discovered field name) into
# the outgoing request.
CSRF_PLACEHOLDERS = {"__CSRF__", "__CSRF_TOKEN__", "TOKEN"}

# <meta> name attributes that conventionally carry (or name) a CSRF token.
CSRF_META_NAMES = {
    "csrf-token", "csrf_token", "csrftoken", "csrfToken", "csrf", "_csrf",
    "csrf-param", "authenticity_token",
}

# Hidden <input> name attributes that conventionally carry a CSRF token.
CSRF_INPUT_NAMES = {
    "csrf_token", "csrftoken", "csrfmiddlewaretoken", "_token", "_csrf",
    "csrf", "csrfToken", "_csrf_token", "authenticity_token",
    "__RequestVerificationToken",
}


def _extract_csrf_tokens(soup) -> dict:
    """Return {field_name: token} found in common <meta> tags and hidden inputs.

    Hidden ``<input>``s are scanned FIRST and are authoritative: their ``name``
    is the field an agent must echo back in a form POST. Meta tags (Django/
    Laravel ``csrf-token``, Rails ``csrf-param`` + ``csrf-token``) are added
    afterwards only under names not already claimed by a hidden input, so a page
    carrying both a ``csrf-token`` meta and a ``csrfmiddlewaretoken`` hidden
    input resolves to the hidden field's name.
    """
    tokens = {}
    for hidden in soup.find_all("input", {"type": "hidden"}):
        name = (hidden.get("name") or "").strip()
        value = (hidden.get("value") or "").strip()
        if not name or not value:
            continue
        lowered = name.lower()
        if lowered in CSRF_INPUT_NAMES or "csrf" in lowered or "token" in lowered:
            tokens.setdefault(name, value)
    csrf_param_name = None
    for meta in soup.find_all("meta"):
        key = (meta.get("name") or meta.get("property") or "").strip()
        content = (meta.get("content") or "").strip()
        if not key or not content:
            continue
        lowered = key.lower()
        if lowered == "csrf-param":
            csrf_param_name = content
        elif lowered in CSRF_META_NAMES or "csrf" in lowered:
            tokens.setdefault(key, content)
    if csrf_param_name and "csrf-token" in tokens:
        tokens.setdefault(csrf_param_name, tokens["csrf-token"])
    return tokens


class _CappedRedirectAdapter(HTTPAdapter):
    """HTTPAdapter honouring a per-request redirect cap.

    ``requests`` only exposes a fixed class-level default of 30 redirects and
    no per-call ``max_redirects``, so mount an adapter whose ``max_redirects``
    is set on the instance (read back by ``HTTPAdapter.send``).
    """

    def __init__(self, max_redirects: int):
        super().__init__()
        self.max_redirects = max_redirects


class HttpSessionManager:
    """Thread-safe store of per-(agent_id, session_id) HTTP cookie jars.

    Holds plain ``{name: value}`` jars rather than live ``requests.Session``
    objects, so concurrent ToolNode workers never share a mutable session:
    every call builds a fresh ``requests.Session`` seeded from the stored jar
    and writes the jar back afterwards. Mirrors browser_tools' agent_id
    namespacing so two validators can pick the same session label without
    colliding.
    """

    def __init__(self):
        self._jars = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(state, session_id) -> str:
        agent = (state.get("agent_id") if isinstance(state, dict) else None) or "no-agent"
        return f"{agent}:{session_id}"

    def load(self, state, session_id) -> Optional[dict]:
        key = self._key(state, session_id)
        with self._lock:
            jar = self._jars.get(key)
        return dict(jar) if jar else None

    def save(self, state, session_id, cookies) -> None:
        key = self._key(state, session_id)
        with self._lock:
            self._jars[key] = dict(cookies)

    def reset(self, state, session_id) -> None:
        key = self._key(state, session_id)
        with self._lock:
            self._jars.pop(key, None)


# Shared HTTP session store for the validator tool loop.
http_sessions = HttpSessionManager()


def _fetch_csrf_tokens(session, target_url, sandbox_url, allow_redirects: bool) -> dict:
    """GET ``target_url`` (falling back to the sandbox root) with ``session``
    and parse CSRF tokens out of the HTML. Sharing the session means any cookie
    the token page sets carries into the caller's follow-up request. Returns an
    empty dict when no token-bearing page could be fetched."""
    candidates = [target_url]
    root = sandbox_url.rstrip("/") + "/"
    if target_url != root:
        candidates.append(root)
    for candidate in candidates:
        try:
            resp = session.get(candidate, timeout=5, allow_redirects=allow_redirects)
        except Exception:
            continue
        if "text/html" in resp.headers.get("Content-Type", ""):
            tokens = _extract_csrf_tokens(BeautifulSoup(resp.text, 'html.parser'))
            if tokens:
                return tokens
    return {}


def _inject_csrf_tokens(data, session, target_url, sandbox_url,
                        allow_redirects: bool) -> tuple:
    """Replace CSRF sentinel values in a form ``data`` dict with real tokens.

    Drops each placeholder field and writes the discovered token under its real
    field name (e.g. ``csrfmiddlewaretoken``), so an agent that guessed ``_csrf``
    against Django is still served correctly. Returns (data, None) on success or
    (data, error_msg) when no token could be fetched."""
    placeholders = [
        k for k, v in data.items()
        if isinstance(v, str) and v in CSRF_PLACEHOLDERS
    ]
    if not placeholders:
        return data, None
    tokens = _fetch_csrf_tokens(session, target_url, sandbox_url, allow_redirects)
    if not tokens:
        return data, (
            "Error: a form field requested automatic CSRF injection (sentinel "
            f"values in {sorted(placeholders)!r}) but no CSRF token could be "
            "extracted from a GET of the endpoint or the sandbox root. Re-send "
            "without the sentinel, or fetch the token manually and pass it "
            "explicitly."
        )
    new_data = dict(data)
    for name in placeholders:
        del new_data[name]
        token_name, token_value = next(iter(tokens.items()))
        new_data[token_name] = token_value
    return new_data, None


@tool(response_format="content_and_artifact")
def send_http_request(
    method: str,
    endpoint: str,
    headers: Optional[dict[str, str]] = None,
    params: Optional[dict[str, Any]] = None,
    data: Optional[dict[str, Any]] = None,
    json_data: Optional[dict[str, Any]] = None,
    files: Optional[dict[str, Union[str, tuple[str, str, str]]]] = None,
    body: Optional[str] = None,
    follow_redirects: bool = True,
    max_redirects: int = 5,
    session_id: Optional[str] = None,
    reset_session: bool = False,
    extract_mode: str = "clean_html",
    state: Annotated[Optional[dict], InjectedState] = None,
) -> tuple[str, dict]:
    """
    Sends an HTTP request to the sandboxed application and maintains session state. Always returns raw response headers. The response body is parsed according to 'extract_mode'.

    Parameters:
    - method: HTTP method ('GET', 'POST', 'PUT', 'DELETE', etc.).
    - endpoint: Target URL or path.
    - headers: Optional dictionary of HTTP headers.
    - params: Optional query parameters for the URL.
    - data: Form fields sent as 'application/x-www-form-urlencoded' (e.g., {'username': 'u', 'password': 'p'}).
    - json_data: Structured payload serialized automatically as 'application/json'.
    - files: Files for multipart/form-data upload. Paths inside the attacker
             workdir (default /work, i.e. files created with write_attacker_file)
             are read from inside the attacker container; all other paths are
             read from the host.
             Format: {'field_name': '/path/to/file'}
             Or with metadata: {'field_name': ('custom_filename.png', '/path/to/file', 'image/png')}
    - body: Raw string body (used only if neither data, json_data, nor files is provided).
    - follow_redirects: Whether to follow 301/302 redirects automatically (default True).
    - max_redirects: Maximum number of redirects to follow while follow_redirects is True (default 5).
    - session_id: Optional session label. Cookies persist across calls that reuse the SAME
                  session_id and are isolated per (agent, session_id); omit it to use a
                  transient one-shot session seeded from the shared cookie state.
    - reset_session: Clears stored cookies/session state (for the given session_id, or the
                     shared state when session_id is omitted) before executing.
    - extract_mode:
            'clean_html' (default): Returns HTML with scripts/styles removed to save tokens.
            'forms': Returns ONLY the <form> elements on the page.
            'links': Returns ONLY the <a> tags.
            'text': Returns only the visible text (good for reading error messages).
            'raw': Returns the untouched body (use cautiously, may truncate).
            ANY CUSTOM TAG: Enter any HTML tag (e.g., 'script', 'input', 'iframe') to extract only those elements.

    CSRF AUTO-INJECTION: For a state-changing request behind a CSRF token, set the token's form
    field to a sentinel value (__CSRF__, __CSRF_TOKEN__, or TOKEN), e.g.
    data={'username': 'u', 'password': 'p', '_csrf': '__CSRF__'}. The tool then GETs the endpoint
    (falling back to '/') with the SAME session cookies, extracts the CSRF token from a <meta> tag
    or hidden <input>, and substitutes the real token (using the discovered field name, e.g.
    'csrfmiddlewaretoken', when your guess was wrong) before sending the request. CSRF tokens found
    in any HTML response are also listed at the end of the output for manual use.
    """

    # Build the full URL from the endpoint parameter
    endpoint = endpoint.strip()
    sandbox_url = state.get("sandbox_url") if state else None
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

    # Resolve the starting cookie jar: a persistent per-(agent, session_id)
    # jar when a session label is given, otherwise the shared state jar.
    start_jar = {}
    if session_id:
        if reset_session:
            http_sessions.reset(state, session_id)
        start_jar = http_sessions.load(state, session_id)
        if start_jar is None:
            # First use of this session_id: seed from the shared cookie state so
            # cookies set by the browser channel (or earlier transient calls)
            # carry into this session.
            if state and "cookies" in state:
                start_jar = dict(state.get("cookies", {}))
    elif not reset_session and state and "cookies" in state:
        start_jar = dict(state.get("cookies", {}))

    session = requests.Session()
    if follow_redirects:
        redirect_adapter = _CappedRedirectAdapter(max_redirects)
        session.mount("http://", redirect_adapter)
        session.mount("https://", redirect_adapter)
    if start_jar:
        session.cookies.update(start_jar)

    try:
        request_kwargs = {
            "method": method,
            "url": url,
            "headers": headers,
            "params": params,
            "timeout": 5,
            "allow_redirects": follow_redirects,
        }

        user_data = None
        if json_data is not None:
            request_kwargs["json"] = json_data
        elif files is not None:
            uploads = {}
            for field, spec in files.items():
                path = spec if isinstance(spec, (str, Path)) else spec[1]
                container_bytes = attacker_tools.read_attacker_file_bytes(str(path))
                if container_bytes is not None:
                    if isinstance(spec, (str, Path)):
                        uploads[field] = (
                            Path(path).name, container_bytes, "application/octet-stream"
                        )
                    else:
                        filename, _, content_type = spec
                        uploads[field] = (filename, container_bytes, content_type)
                else:
                    uploads[field] = spec
            request_kwargs["files"] = uploads
            if data is not None:
                user_data = data
        elif data is not None:
            user_data = data
        elif body is not None:
            user_data = body

        # Automatic CSRF resolution: replace sentinel form-field values with a
        # freshly fetched token (same session, so the token's cookie applies).
        if isinstance(user_data, dict) and any(
            isinstance(v, str) and v in CSRF_PLACEHOLDERS
            for v in user_data.values()
        ):
            user_data, csrf_error = _inject_csrf_tokens(
                user_data, session, url, sandbox_url, follow_redirects
            )
            if csrf_error:
                return csrf_error, {}

        if user_data is not None:
            request_kwargs["data"] = user_data

        response = session.request(**request_kwargs)

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

        # Surface any CSRF tokens found on the page so the agent can include
        # them, or leave a __CSRF__ sentinel to trigger automatic injection.
        if "text/html" in response.headers.get("Content-Type", ""):
            found = _extract_csrf_tokens(soup)
            if found:
                token_lines = "\n".join(
                    f"{name}={value}" for name, value in found.items()
                )
                llm_output += (
                    "\n\n[CSRF tokens extracted from this page — include the "
                    "matching field in your next state-changing request, or set "
                    "a form field's value to __CSRF__ for automatic injection:]\n"
                    f"{token_lines}"
                )

        cookies = session.cookies.get_dict()
        if session_id:
            http_sessions.save(state, session_id, cookies)

        return llm_output, cookies
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

    # Save to cache so subsequent runs skip the tool-calling loop.
    cache_validator(report, state.get("peer_payloads"), updated_vuln)

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

    # Save to cache so subsequent runs skip the tool-calling loop (the cached
    # insufficient_context record keeps review_round bumped, so a repeat of the
    # same round-0 report re-triggers the reviewer feedback loop exactly).
    cache_validator(report, state.get("peer_payloads"), updated_vuln)

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


def _format_vulnerability_markdown(record: dict) -> str:
    """Render a VulnerabilityRecord dict as a compact markdown report."""
    cwe = record.get("cwe_id", "OTHER_UNCATEGORIZED")
    cwe_desc = cwes.get(cwe, "")
    lines = [
        f"## {record.get('vuln_id', 'Unknown')}",
        "",
        f"- **CWE:** {f'{cwe}, {cwe_desc}' if cwe_desc else cwe}",
        f"- **Status:** {record.get('status', 'hypothesis')}",
        f"- **Type:** {record.get('vulnerability_type', 'Code Defect')}",
    ]
    nodes = [n for n in (record.get("affected_nodes") or []) if n]
    if nodes:
        lines.append(f"- **Affected Nodes:** {', '.join(nodes)}")
    if record.get("source_cve"):
        lines.append(f"- **Source CVE:** {record['source_cve']}")

    lines += [
        "",
        "### Description",
        record.get("description", "") or "_none_",
        "",
        "### Reviewer Reasoning",
        record.get("reviewer_reasoning") or "_none_",
        "",
        "### Reproduction Steps",
    ]
    steps = record.get("reproduction_steps") or []
    if steps:
        # Strip any leading "N." / "N)" numbering the reviewer already embedded
        # so our prefixed counter does not double-number each step.
        lines += [
            f"{i}. {re.sub(r'^\s*\d+[\.\)]\s+', '', str(s))}"
            for i, s in enumerate(steps, 1)
        ]
    else:
        lines.append("_none_")

    if record.get("poc_payload") or record.get("execution_logs"):
        lines += [
            "",
            "### Proven Validator Payload",
        ]
        if record.get("poc_payload"):
            lines += ["```", str(record["poc_payload"]).rstrip(), "```"]
        else:
            lines.append("_no payload_")
        if record.get("execution_logs"):
            lines += [
                "",
                "Execution logs:",
                str(record["execution_logs"]).rstrip(),
            ]

    return "\n".join(lines)


@tool(args_schema=VulnerabilityDetailsInput)
def get_vulnerability_details(
    vuln_id: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """
    Fetches the full record of another confirmed vulnerability by its vuln_id as a
    markdown report, so you can reason over its real mechanics (full description,
    reviewer reasoning, reproduction steps) when deciding whether it chains with
    your assigned vulnerability. Only entries from the provided summary of other
    confirmed vulnerabilities are available.
    """
    confirmed = state.get("confirmed_vulns", [])
    for record in confirmed:
        if record.get("vuln_id") == vuln_id:
            return _format_vulnerability_markdown(record)
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

    # Save to cache so subsequent runs skip the tool-calling loop.
    cache_integration_auditor(report, state.get("confirmed_vulns"), updated_vuln)

    return Command(
        update={
            "vulnerabilities": [updated_vuln],
            "messages": [tool_msg]
        }
    )


def _compile_pattern(keyword: str, is_regex: bool) -> re.Pattern:
    if not is_regex:
        return re.compile(re.escape(keyword))

    try:
        return re.compile(keyword)
    except re.error:
        # Split only on pipes NOT preceded by an odd number of backslashes
        branches = re.split(r'(?<!\\)\|', keyword)
        if len(branches) > 1:
            safe_branches = []
            for branch in branches:
                branch = branch.strip()
                if not branch:
                    continue
                try:
                    re.compile(branch)
                    safe_branches.append(branch)
                except re.error:
                    safe_branches.append(re.escape(branch))
            return re.compile("|".join(safe_branches))

        return re.compile(re.escape(keyword))


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
    query = _compile_pattern(keyword, regex)

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
        # Respect the scan path-exclusion filter so the reviewer never spends
        # tokens roaming into dependency trees, tests, or docs.
        try:
            rel = str(file_path.relative_to(app_dir))
        except ValueError:
            rel = str(file_path)
        if is_path_excluded(rel):
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
def get_node_connections(node_ids: list[str]) -> str:
    """
    Returns the neighbors of one or more nodes in the application graph.
    Use this to identify which functions call the current nodes (callers)
    or which functions/files the current nodes call (callees).

    Args:
        node_ids (list[str]): The identifiers of the nodes to inspect in the graph.
    """
    try:
        G = build_networkx_graph(settings.graph)

        blocks = []
        for node_id in node_ids:
            if node_id not in G:
                blocks.append(f"Node ID '{node_id}' not found in the graph structure.")
                continue

            successors = list(G.successors(node_id))
            predecessors = list(G.predecessors(node_id))

            blocks.append(
                f"Node: {node_id}\n"
                f"Called by (Predecessors): {predecessors}\n"
                f"Calls (Successors): {successors}"
            )

        return "\n\n".join(blocks)
    except Exception as e:
        return f"Error traversing graph: {str(e)}"



@tool
def get_path(source_node: str, target_node: str) -> str:
    """
    Returns all directed paths in the application graph from `source_node` to
    `target_node`. Each path is a chain of edges annotated with their relation
    type (e.g. calls/imports/references), useful for verifying a concrete
    call-graph / data-flow link between two nodes.

    Args:
        source_node (str): The starting node ID.
        target_node (str): The destination node ID.
    """
    try:
        G = build_networkx_graph(settings.graph)

        missing = [n for n in (source_node, target_node) if n not in G]
        if missing:
            return f"Node(s) not found in the graph: {', '.join(missing)}"

        if source_node == target_node:
            return f"Source and target are the same node: {source_node}"

        paths = []
        truncated = False
        for i, path in enumerate(nx.all_simple_paths(
            G, source_node, target_node, cutoff=MAX_PATH_CUTOFF
        )):
            if i >= MAX_PATHS:
                truncated = True
                break
            paths.append(path)

        if not paths:
            return (
                f"No directed path found from '{source_node}' to '{target_node}' "
                f"(within max path length {MAX_PATH_CUTOFF})."
            )

        paths.sort(key=len)
        lines = [f"{len(paths)} path(s) from '{source_node}' to '{target_node}':"]
        for i, path in enumerate(paths, 1):
            hops = []
            for a, b in zip(path, path[1:]):
                rel = G.edges[a, b].get("relation", "")
                hops.append(f"{a} -{rel}-> " if rel else f"{a} -> ")
            hops.append(path[-1])
            lines.append(f"{i}. {''.join(hops)}")

        if truncated:
            lines.append(f"(Only the first {MAX_PATHS} paths shown; more may exist.)")

        return "\n".join(lines)
    except Exception as e:
        return f"Error traversing graph: {str(e)}"


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
