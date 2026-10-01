import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from collections import deque

import networkx as nx

import settings
from state import MasterState
from utils import build_networkx_graph, get_node_code, get_cached_graph_data


# ==========================================
# Entry point detection patterns
# ==========================================

# Web route decorators / handlers per language. Applied to each node's own code.
_ROUTE_PATTERNS = [
    # Python: Flask-style decorators (@app.route, @bp.get) and FastAPI decorators.
    r"@\w+\.(?:route|get|post|put|delete|patch|head|options|errorhandler|before_request|after_request|add_url_rule)\(",
    # JS/TS: Express-style handlers (app.get, router.post, app.use, ...).
    r"\b(?:app|router|bp|blueprint)\.(?:get|post|put|delete|patch|all|use|route|handle)\(",
    # PHP: Laravel Route facade and Symfony #[Route] attribute.
    r"Route::(?:get|post|put|delete|patch|any|match|resource|group)\(",
    r"#\[Route\(",
    r"@app\.(?:get|post|put|delete|patch)\(",
]

# CLI argument / stdin reading signals. File skeletons qualify as entry points
# only when their content shows one of these.
_CLI_PATTERNS = [
    r"if\s+__name__\s*==\s*['\"]__main__['\"]",
    r"\bsys\.argv\b",
    r"\bsys\.stdin\b",
    r"\bprocess\.argv\b",
    r"\bprocess\.stdin\b",
    r"\$argv\b",
    r"\$_SERVER\s*\[\s*['\"]argv['\"]\s*\]",
    r"php://stdin",
    r"\bargparse\.(?:ArgumentParser|parse_args)\(",
    r"\bclick\.(?:command|group|argument|option)\(",
    r"\bflag\.Parse\(",
    r"\bgetopt\(",
]

# AnalysisNote.sources values that prove external data enters the snippet.
_EXTERNAL_SOURCE_INDICATORS = (
    "request.", "req.", "request[",
    "sys.argv", "sys.stdin",
    "process.argv", "process.stdin",
    "$_GET", "$_POST", "$_REQUEST", "$_FILES", "$_COOKIE",
    "argparse", "click.",
    "php://stdin",
)

_ENTRY_RE = re.compile("|".join(_ROUTE_PATTERNS + _CLI_PATTERNS))


# ==========================================
# Subprocess detection via Semgrep
# ==========================================

# AST-level rules (one per language family) that detect subprocess / child-process
# invocation that may run an external script. Replaces the old regex patterns:
# semgrep matches on parsed syntax, so strings/comments are never false positives.
_SEMGREP_RULES = """
rules:
  - id: reachability-subprocess-python
    languages: [python]
    severity: INFO
    message: python subprocess call
    pattern-either:
      - pattern: subprocess.run($ARGS, ...)
      - pattern: subprocess.call($ARGS, ...)
      - pattern: subprocess.Popen($ARGS, ...)
      - pattern: subprocess.check_call($ARGS, ...)
      - pattern: subprocess.check_output($ARGS, ...)
      - pattern: subprocess.getoutput($ARGS, ...)
      - pattern: os.system($ARGS, ...)
      - pattern: os.popen($ARGS, ...)
      - pattern: os.execv($ARGS, ...)
      - pattern: os.execve($ARGS, ...)
      - pattern: os.execl($ARGS, ...)
      - pattern: os.execle($ARGS, ...)
      - pattern: os.execvp($ARGS, ...)
      - pattern: os.execvpe($ARGS, ...)
      - pattern: os.execlp($ARGS, ...)
  - id: reachability-subprocess-js
    languages: [javascript, typescript]
    severity: INFO
    message: child process call
    pattern-either:
      - pattern: child_process.exec($ARGS, ...)
      - pattern: child_process.execSync($ARGS, ...)
      - pattern: child_process.spawn($ARGS, ...)
      - pattern: child_process.spawnSync($ARGS, ...)
      - pattern: child_process.fork($ARGS, ...)
      - pattern: child_process.execFile($ARGS, ...)
      - pattern: child_process.execFileSync($ARGS, ...)
  - id: reachability-subprocess-php
    languages: [php]
    severity: INFO
    message: php command execution
    pattern-either:
      - pattern: exec($ARGS, ...)
      - pattern: shell_exec($ARGS, ...)
      - pattern: system($ARGS, ...)
      - pattern: passthru($ARGS, ...)
      - pattern: proc_open($ARGS, ...)
      - pattern: popen($ARGS, ...)
  - id: reachability-subprocess-go
    languages: [go]
    severity: INFO
    message: go exec command
    pattern: exec.Command($ARGS, ...)
  - id: reachability-subprocess-java
    languages: [java]
    severity: INFO
    message: java runtime exec
    pattern: Runtime.getRuntime().exec($ARGS, ...)
  - id: reachability-subprocess-csharp
    languages: [csharp]
    severity: INFO
    message: csharp process start
    pattern: Process.Start($ARGS, ...)
"""

# Edge types that represent real execution / data flow. Graphify's conceptual
# relations (rationale_for, conceptually_related_to, semantically_similar_to,
# shares_data_with) are deliberately excluded to avoid phantom reachability.
_EXECUTION_EDGE_TYPES = {
    "calls", "contains", "imports", "imports_from",
    "uses", "references", "inherits", "extends",
    "synthetic_subprocess",
}

_SUBPROCESS_RELATION = "synthetic_subprocess"


def _semgrep_binary() -> str:
    """Path to the semgrep binary shipped with the active venv."""
    return str(Path(sys.executable).parent / "semgrep")


def _write_semgrep_rules() -> Path:
    """Materialize the embedded rules to a temp file for the semgrep CLI."""
    path = Path(tempfile.gettempdir()) / "reachability_subprocess_rules.yml"
    path.write_text(_SEMGREP_RULES, encoding="utf-8")
    return path


def _run_semgrep(files: list[Path]) -> list[dict]:
    """Run semgrep once over the given files, returning raw match dicts.

    Returns an empty list when semgrep is unavailable or the scan fails, so a
    missing binary degrades gracefully (no synthetic edges) just like osv-scanner.
    """
    if not files:
        return []
    out_path = Path(tempfile.gettempdir()) / "reachability_semgrep_out.json"
    try:
        out_path.unlink(missing_ok=True)
    except OSError:
        pass
    cmd = [
        _semgrep_binary(), "scan",
        "--config", str(_write_semgrep_rules()),
        "--json",
        "--metrics", "off",
        "--output", str(out_path),
    ] + [str(f) for f in files]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        logging.warning("Reachability: semgrep not found — skipping subprocess edge injection.")
        return []
    except subprocess.TimeoutExpired:
        logging.warning("Reachability: semgrep scan timed out.")
        return []
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logging.warning("Reachability: failed to parse semgrep output.")
        return []
    return data.get("results", [])


def _extract_matched_code(match: dict) -> str | None:
    """Slice the exact matched source out of the file using byte offsets."""
    path = match.get("path")
    start = match.get("start") or {}
    end = match.get("end") or {}
    if not path or "offset" not in start or "offset" not in end:
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
        return data[start["offset"]:end["offset"]].decode("utf-8", errors="replace")
    except OSError:
        return None


def _node_at_line(file_lines: dict, source_file: str, line: int) -> str | None:
    """Innermost graph node owning `line` in `source_file`.

    `file_lines` maps source_file -> sorted [(start_line, node_id)]. Iterating
    descending, the first node starting at or before `line` is the innermost one.
    """
    for start_line, node_id in reversed(file_lines.get(source_file, ())):
        if start_line <= line:
            return node_id
    return None


# ==========================================
# 1. Entry point tagging
# ==========================================

def tag_entry_points(G: nx.DiGraph, notes: list) -> nx.DiGraph:
    """Mark nodes reachable from outside as entry points.

    Two complementary signals:
    - Static scan of each node's source for web route / CLI patterns.
    - Explorer AnalysisNote.sources that reference external data (request args,
      stdin, argv, ...).

    File skeletons only qualify through the static scan (a skeleton is an entry
    point only if its content contains a route/CLI pattern or reads argv/stdin).
    """
    graph_data = get_cached_graph_data(settings.graph)
    node_by_id = {n.get("id"): n for n in graph_data.get("nodes", [])}

    entry_points = set()

    for node_id in G.nodes:
        node_attrs = G.nodes[node_id]
        source_file = node_attrs.get("source_file", "")
        if not source_file:
            continue

        code = get_node_code(node_id, raw=True)
        if code and _ENTRY_RE.search(code):
            entry_points.add(node_id)
            continue

        # File-skeleton signals already covered by the static scan above
        # (the skeleton's code is the whole file), so nothing extra here.

    # Secondary signal: explorer notes describing external data entering a node.
    for note in notes:
        note_dict = note if isinstance(note, dict) else note.model_dump()
        node_id = note_dict.get("node_id")
        if not node_id or node_id in entry_points:
            continue
        sources = note_dict.get("sources") or []
        for source in sources:
            if any(ind in str(source) for ind in _EXTERNAL_SOURCE_INDICATORS):
                entry_points.add(node_id)
                break

    for node_id in entry_points:
        if node_id in G:
            G.nodes[node_id]["is_entry_point"] = True

    logging.info(
        f"Reachability: tagged {len(entry_points)} entry point(s) "
        f"({len([n for n in G.nodes if G.nodes[n].get('is_entry_point')])} in graph)."
    )
    return G


# ==========================================
# 2. Synthetic subprocess edges
# ==========================================

def _extract_first_arg(code: str, call_start: int) -> str | None:
    """Return the first positional argument text of a call at call_start ('(')."""
    depth = 0
    i = call_start
    n = len(code)
    while i < n:
        c = code[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return None
        elif c == "," and depth == 1:
            return code[call_start + 1:i].strip()
        elif c in "\"'":
            quote = c
            i += 1
            while i < n and code[i] != quote:
                if code[i] == "\\":
                    i += 1
                i += 1
        i += 1
    return None


def _split_top_level(text: str) -> list[str]:
    """Split a comma-separated argument list, respecting nesting and strings."""
    parts = []
    depth = 0
    current = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            i += 1
            continue
        elif c in "\"'":
            current.append(c)
            i += 1
            while i < n and text[i] != c:
                if text[i] == "\\":
                    current.append(text[i])
                    i += 1
                current.append(text[i])
                i += 1
            current.append(c)
            i += 1
            continue
        current.append(c)
        i += 1
    parts.append("".join(current).strip())
    return [p for p in parts if p]


_INTERPRETER_TOKENS = {
    "python", "python3", "python2", "python3.8", "python3.9", "python3.10",
    "python3.11", "python3.12", "python3.13", "node", "nodejs", "php",
    "ruby", "perl", "bash", "sh", "zsh", "pwsh", "powershell", "deno", "bun",
    "sys.executable", "process.execPath",
}

# Module-level assignment used to resolve a variable holding a script path:
#   CONVERTER_SCRIPT = Path("convert_job.py") / "..."
#   SCRIPT = "worker.py"
_VAR_ASSIGN_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\n]+?)\s*$",
    re.MULTILINE,
)


def _variable_bindings(source_code: str) -> dict[str, str]:
    """Map module-level variable names to string/Path literals they hold."""
    bindings = {}
    for m in _VAR_ASSIGN_RE.finditer(source_code):
        name, value = m.group(1), m.group(2).rstrip(",")
        literals = re.findall(r"""['"]([^'"]+)['"]""", value)
        if literals:
            # Path("x") / "a" / "b" — the last literal is the target file name.
            bindings[name] = literals[-1]
    return bindings


def _element_value(element: str, bindings: dict) -> str | None:
    """Resolve a single call-argument element to a string value."""
    element = element.strip()
    if not element:
        return None

    # String literal (possibly f-string of a constant path).
    m = re.match(r"""^['\"]([^'\"]+)['\"]$""", element)
    if m:
        return m.group(1)

    # str(VAR) / VAR / Path(VAR) / VAR.name
    cleaned = re.sub(r"^str\(|^Path\(|\.name$|\)$", "", element).strip()
    if cleaned in bindings:
        return bindings[cleaned]

    # Bare assignment target already captured by bindings.
    if element in bindings:
        return bindings[element]

    return None


def _script_candidates(first_arg: str, bindings: dict) -> list[str]:
    """Extract possible target script paths from a subprocess call argument."""
    candidates = []

    # Strip the outer brackets for list invocations:
    #   [interpreter, "script.py", "--flag", ...] or ["script.py"]
    list_body = first_arg.strip()
    if list_body.startswith("[") and list_body.endswith("]"):
        list_body = list_body[1:-1].strip()
        elements = _split_top_level(list_body)
        if len(elements) > 1:
            raw_first = elements[0].strip()
            is_interp = raw_first in _INTERPRETER_TOKENS or raw_first.endswith(
                ("/python", "/python3", "/node", "/php", "/bin/sh", "/bin/bash")
            )
            if is_interp:
                values = [v for v in (_element_value(e, bindings) for e in elements[1:]) if v]
                if values:
                    candidates.append(values[0])
            else:
                values = [v for v in (_element_value(e, bindings) for e in elements) if v]
                if values:
                    candidates.append(values[0])
            return candidates

    # Single argument: plain string, possibly with a command prefix.
    elements = _split_top_level(first_arg)
    value = _element_value(elements[0], bindings) if elements else None
    if value:
        candidates.append(value)

    return candidates


def _resolve_script_node(source_file: str, script_path: str, node_abs: dict) -> str | None:
    """Resolve a script path to a graph node's id, or None.

    `node_abs` maps node_id -> resolved absolute path of its source file,
    precomputed once by the caller.
    """
    if not script_path:
        return None

    # Drop command prefixes and quoted remnants.
    parts = script_path.split()
    if len(parts) > 1:
        script_path = parts[-1]
    script_path = script_path.strip("\"'")

    # Reject obviously dynamic / non-file targets.
    if not script_path or script_path.startswith(("$", "{", "(")):
        return None
    if "://" in script_path or script_path.startswith(("-", "/dev/", "/proc/", "/sys/")):
        return None

    # Build absolute candidate paths resolved under the app root.
    script_p = Path(script_path)
    base_dir = Path(source_file).parent
    candidates = set()
    for cand in (script_p, base_dir / script_p):
        if script_p.is_absolute():
            candidates.add(script_p)
        else:
            candidates.add(settings.app_path / cand)
            candidates.add((settings.app_path / base_dir) / cand)
    try:
        resolved = {str(c.resolve()) for c in candidates}
    except Exception:
        return None

    # Match against graph nodes by resolved absolute path.
    for node_id, abs_path in node_abs.items():
        if abs_path in resolved:
            return node_id

    # Fallback: basename match (covers scripts invoked by bare filename).
    script_name = script_p.name
    for node_id, abs_path in node_abs.items():
        if Path(abs_path).name == script_name:
            return node_id

    return None


def inject_subprocess_edges(G: nx.DiGraph) -> nx.DiGraph:
    """Bridge the AST gap left by subprocess / child-process invocation.

    Runs semgrep once over every file referenced by a graph node, then for each
    match that resolves to a script target part of the repo, adds a synthetic
    directed edge owner -> target with relation 'synthetic_subprocess'.
    Unresolvable targets are skipped.
    """
    graph_data = get_cached_graph_data(settings.graph)
    node_by_id = {n.get("id"): n for n in graph_data.get("nodes", [])}

    source_files = {d.get("source_file") for _, d in G.nodes(data=True) if d.get("source_file")}
    if not source_files:
        return G

    # Precomputed lookups used across all matches.
    node_abs = {}
    file_lines = {}
    for node_id, attrs in node_by_id.items():
        source_file = attrs.get("source_file")
        if not source_file:
            continue
        try:
            node_abs[node_id] = str((settings.app_path / Path(source_file)).resolve())
        except Exception:
            continue
        loc = attrs.get("source_location")
        if loc:
            try:
                start_line = int(str(loc).replace("L", "").strip())
            except ValueError:
                continue
            file_lines.setdefault(source_file, []).append((start_line, node_id))
    for lines in file_lines.values():
        lines.sort()

    # abs-path -> source_file lookup for semgrep matches, and the scan targets.
    abs_to_source = {}
    for sf in source_files:
        try:
            abs_to_source[str((settings.app_path / Path(sf)).resolve())] = sf
        except Exception:
            continue
    file_paths = sorted(abs_to_source)
    if not file_paths:
        return G

    matches = _run_semgrep(file_paths)
    if not matches:
        logging.info("Reachability: semgrep found no subprocess calls.")
        return G

    added = 0
    for match in matches:
        path = match.get("path")
        start_line = (match.get("start") or {}).get("line")
        if not path or not start_line:
            continue
        source_file = abs_to_source.get(str(Path(path).resolve()))
        if not source_file:
            continue

        # Map the call site to the innermost graph node that contains it.
        origin = _node_at_line(file_lines, source_file, int(start_line))
        if not origin or origin not in G:
            continue

        code = _extract_matched_code(match)
        if not code:
            continue

        # Module-level variable bindings (resolves e.g. CONVERTER_SCRIPT) come
        # from the full file — the matched snippet only holds the call itself.
        bindings = {}
        try:
            full_file = settings.app_path / Path(source_file)
            if full_file.exists():
                bindings.update(_variable_bindings(full_file.read_text(encoding="utf-8")))
        except Exception:
            pass

        call_start = code.find("(")
        if call_start == -1:
            continue
        first_arg = _extract_first_arg(code, call_start)
        if not first_arg:
            continue
        for candidate in _script_candidates(first_arg, bindings):
            target = _resolve_script_node(source_file, candidate, node_abs)
            if target and target in G and target != origin:
                if not G.has_edge(origin, target):
                    G.add_edge(origin, target, relation=_SUBPROCESS_RELATION,
                               synthetic=True, confidence="SYNTHETIC")
                    added += 1
                    logging.debug(
                        f"Reachability: synthetic edge {origin} -> {target} "
                        f"({candidate!r})"
                    )
                break

    logging.info(f"Reachability: added {added} synthetic subprocess edge(s).")
    return G


# ==========================================
# 3. Reachability filter
# ==========================================

def filter_by_reachability(G: nx.DiGraph, vulnerabilities: list) -> list:
    """Mark hypotheses whose node is unreachable from any entry point.

    All vulnerabilities are returned. Only hypotheses (status == 'hypothesis')
    are re-tagged: reachable nodes keep 'hypothesis', unreachable nodes become
    'unreachable' so downstream stages (e.g. reviewer dispatch) skip them.
    Already confirmed / exploitable / false-positive records are untouched.
    """
    entry_points = [n for n, d in G.nodes(data=True) if d.get("is_entry_point")]
    if not entry_points:
        logging.warning(
            "Reachability: no entry points tagged — skipping filter "
            "(no hypotheses marked unreachable)."
        )
        return vulnerabilities

    # Single multi-source BFS over execution-relevant edge types.
    reachable = set(entry_points)
    queue = deque(entry_points)
    while queue:
        node = queue.popleft()
        for succ in G.successors(node):
            if succ in reachable:
                continue
            edge_data = G.get_edge_data(node, succ) or {}
            if edge_data.get("relation") in _EXECUTION_EDGE_TYPES:
                reachable.add(succ)
                queue.append(succ)

    updated = []
    unreachable = 0
    for vuln in vulnerabilities:
        vuln_dict = vuln if isinstance(vuln, dict) else vuln.model_dump()
        if vuln_dict.get("status") != "hypothesis":
            updated.append(vuln)
            continue
        # Dependency-CVE hypotheses (e.g. upgrade-only RCE in the HTTP server)
        # anchor to a synthetic `dependency:<package>` node that is never part of
        # the execution graph. Exposure assessment is delegated to the Reviewer,
        # so they are exempt from the static reachability filter.
        if vuln_dict.get("source_cve"):
            updated.append(vuln)
            continue
        if vuln_dict.get("node_id") not in reachable:
            vuln_dict["status"] = "unreachable"
            unreachable += 1
            logging.info(
                f"Reachability: marked {vuln_dict.get('vuln_id', vuln_dict.get('node_id'))} "
                f"(node {vuln_dict.get('node_id')}) unreachable — no path from any entry point."
            )
            updated.append(vuln_dict)
        else:
            updated.append(vuln)

    logging.info(
        f"Reachability: {unreachable}/{len(vulnerabilities)} hypothesis(es) marked "
        f"unreachable; reachable hypotheses keep status 'hypothesis'."
    )
    return updated


# ==========================================
# LangGraph node
# ==========================================

def reachability_filter_node(state: MasterState) -> dict:
    """Mark unreachable vulnerability hypotheses before the Reviewer Agent.

    All hypotheses are kept; those with no path from any entry point get their
    status changed to 'unreachable' so reviewer dispatch skips them.
    """
    G = build_networkx_graph(settings.graph)
    notes = state.get("notes", [])
    G = tag_entry_points(G, notes)
    G = inject_subprocess_edges(G)
    filtered = filter_by_reachability(G, state.get("vulnerabilities", []))
    return {"vulnerabilities": filtered}
