"""Patcher tools: minimal, surgical SOURCE-CODE edits that block the proven
exploit flow of one vulnerability record.

Two tools form the patcher's write surface; everything else it needs
(`read_source_code`, `read_file`, `search_codebase`, `get_definition`,
`get_node_connections`) is reused verbatim from ``tools.py`` — all of those read
the CURRENT on-disk files, so the patcher always sees the real code including
any edit it just applied.

`patch_source_file` replaces an exact, agent-named line range of one app file
(the agent's own numbered `read_file` output is the addressing scheme), under a
process-wide lock (concurrent patchers + same-response batches run through the
SequentialToolNode, but a replayed/resumed run may overlap another patcher's
file). It answers with the replaced hunk + a unified diff so a wrong-range edit
is visible to the agent on the very next turn, and warns about line shifts.
`submit_patch` is the terminal tool: it folds the applied edits into the record
(patch fields + status reset to "confirmed") so the normal reviewer/validator
machinery re-adjudicates the PATCHED code.
"""

from pathlib import Path
import difflib
import logging
import threading
from typing import Annotated, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

import settings
from utils import cache_patcher, is_path_excluded
import schemas


# Serializes every source write: two concurrent patchers (and a resumed run
# overlapping a live one) can otherwise interleave reads/writes on one file and
# land an edit computed against a stale line numbering.
_patch_lock = threading.Lock()

# Lines of the replaced region echoed back into the ToolMessage.
_ECHO_MAX_LINES = 30


def _reject(tool_call_id: str, text: str) -> Command:
    """Bounce a write-tool call with a corrective error ToolMessage (a
    status='error' message never ends the loop — ToolLoopAgent.tool_batch_done
    skips it — so the agent sees the problem and fixes it, mirroring
    tools._reject_submission)."""
    return Command(update={"messages": [ToolMessage(
        content=text, name="patch_tool_rejected", tool_call_id=tool_call_id,
        status="error",
    )]})


def _split(text: str) -> list[str]:
    return text.splitlines()


@tool
def patch_source_file(
    file_path: str,
    start_line: int,
    end_line: int,
    replacement: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Replace an inclusive line range of ONE application source file with new
    content — the patcher's ONLY write capability. Line numbers must come from
    a `read_file` you performed on this SAME content (re-read after ANY earlier
    edit that changed the file's line count — inserted/removed lines shift
    everything below the edit). Keep the replacement MINIMAL: it must block
    exactly the proven exploit flow, preserve all legitimate behavior of the
    code, and never disable or bypass the feature.

    The tool refuses files outside the application directory, excluded paths
    (dependency trees/tests/docs), ranges larger than the configured cap, and a
    range that already equals the replacement (idempotence guard on replay).
    Its reply echoes the replaced lines and a unified diff: CHECK IT — a wrong
    range is fixable by the next edit only if you notice it here.

    Args:
        file_path: Path relative to the application root (e.g. 'src/db.py').
        start_line: First line to replace, 1-indexed inclusive.
        end_line: Last line to replace, 1-indexed inclusive.
        replacement: The exact new content for that range ('' deletes it).
    """
    applied = state.get("patch_log") or []
    if len(applied) >= settings.patcher_max_edits:
        return _reject(
            tool_call_id,
            f"Edit refused: this patch already applied {len(applied)} edits "
            f"(cap: {settings.patcher_max_edits}). A bigger change is refactoring, "
            "not patching — refine the edits you have or call submit_patch.",
        )

    app_dir = Path(settings.app_path).resolve()
    target = (app_dir / file_path).resolve()
    if not target.is_relative_to(app_dir):
        return _reject(
            tool_call_id,
            f"Edit refused: '{file_path}' resolves outside the application "
            f"directory '{app_dir}'. Only app source files can be patched.",
        )
    try:
        rel = str(target.relative_to(app_dir))
    except ValueError:
        rel = file_path
    if is_path_excluded(rel):
        return _reject(
            tool_call_id,
            f"Edit refused: '{rel}' is in an excluded path (dependency trees, "
            "tests, docs). Never patch vendored/library code — fix first-party code.",
        )

    new_lines = _split(replacement)  # '' deletes the range
    if len(new_lines) > settings.patcher_max_edit_lines:
        return _reject(
            tool_call_id,
            f"Edit refused: the replacement itself is {len(new_lines)} lines, "
            f"exceeding the {settings.patcher_max_edit_lines}-line patch cap. "
            "Pasting a rewritten function is refactoring, not patching — shrink "
            "the change to the minimum that blocks the exploit flow.",
        )
    if end_line - start_line + 1 > settings.patcher_max_edit_lines:
        return _reject(
            tool_call_id,
            f"Edit refused: replacing {end_line - start_line + 1} lines exceeds the "
            f"{settings.patcher_max_edit_lines}-line patch cap. Shrink the edit to "
            "the minimum that blocks the exploit flow.",
        )

    with _patch_lock:
        try:
            content = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return _reject(
                tool_call_id,
                f"Edit refused: cannot read '{rel}' as text ({exc}). Re-check the "
                "path with read_file.",
            )
        lines = content.splitlines()
        total = len(lines)
        if not (1 <= start_line <= end_line <= min(total, start_line - 1 + settings.patcher_max_edit_lines)):
            # Bounds AND the size cap fail with the same corrective message; a
            # range past EOF usually means the file shifted since the last read.
            return _reject(
                tool_call_id,
                f"Edit refused: lines {start_line}-{end_line} are not a valid range of "
                f"'{rel}' ({total} lines; cap {settings.patcher_max_edit_lines} per edit). "
                "If the file shifted since your last read, re-read it with read_file "
                "and use its printed numbering; otherwise shrink the edit.",
            )
        old_lines = lines[start_line - 1:end_line]
        if old_lines == new_lines:
            # Idempotence guard: the content already IS the replacement (typical
            # for a resumed run that already applied this edit). Count the entry
            # so submit_patch sees the hunk as applied without rewriting bytes.
            note = f"lines {start_line}-{end_line} of '{rel}' already match the replacement (already applied)"
            logging.info(f"Patcher: no-op edit on {rel} ({note})")
            return Command(update={
                "messages": [ToolMessage(
                    content=(
                        f"Already applied: {note}. Treat this edit as done; if it "
                        "was part of your plan, proceed to submit_patch — otherwise "
                        "fix the range/content of the intended edit."
                    ),
                    name="patch_source_file",
                    tool_call_id=tool_call_id,
                )],
                "patch_log": [{"file": rel, "diff": "", "note": note}],
            })

        updated = lines[:start_line - 1] + new_lines + lines[end_line:]
        trailing = "\n" if content.endswith("\n") else ""
        try:
            target.write_text("\n".join(updated) + trailing, encoding="utf-8")
        except OSError as exc:
            return _reject(tool_call_id, f"Edit refused: cannot write '{rel}' ({exc}).")

    diff = "\n".join(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm="", n=3,
    ))
    echo = "\n".join(f"{i:>6}: {l}" for i, l in enumerate(old_lines[:_ECHO_MAX_LINES], start_line))
    delta = len(new_lines) - len(old_lines)
    shift_note = (
        f"LINE SHIFT: this edit {'added ' + str(delta) if delta > 0 else 'removed ' + str(-delta)} "
        "line(s) — every line number BELOW the edit moved; re-read the file before "
        "the next edit in it." if delta else
        "Same line count: line numbering below the edit is unchanged."
    )
    logging.info(f"Patcher: patched {rel} (lines {start_line}-{end_line}, "
                 f"{len(old_lines)} -> {len(new_lines)} lines).")

    return Command(update={
        "messages": [ToolMessage(
            content=(
                f"EDIT APPLIED: {rel} lines {start_line}-{end_line} replaced with "
                f"{len(new_lines)} line(s).\n"
                f"--- OLD LINES (verify this is what you meant to replace) ---\n{echo}\n"
                f"--- DIFF ---\n{diff}\n"
                f"--- {shift_note} ---"
            ),
            name="patch_source_file",
            tool_call_id=tool_call_id,
        )],
        "patch_log": [{"file": rel, "diff": diff}],
    })


def _attempt_outcome(report: dict) -> str:
    """One-line verdict of a SUPERSEDED patch attempt, distilled from the state
    the record carries when submit_patch archives it (retry rounds only). The
    tail of execution_logs wins: the validator's final fix-adjudication is the
    most recent evidence."""
    status = report.get("status")
    state = report.get("patch_state")
    logs = str(report.get("execution_logs") or "").strip()
    excerpt = (" | evidence: ..." + logs[-800:]) if logs else ""
    if state == "rejected":
        return ("REJECTED — the exploit still fired after the sandbox adopted "
                "this patch" + excerpt)
    if status == "false_positive" and state in ("reviewed", "verified"):
        return "false_positive on the patched code (re-review/validator cleared it)"
    if status == "confirmed":
        return "the re-review still found the flow exploitable"
    return f"undetermined (status={status}, patch_state={state})"


@tool(args_schema=schemas.SubmitPatchInput)
def submit_patch(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    summary: str,
) -> Command:
    """
    Call this ONCE, as your final step, to bank the patch you applied with
    patch_source_file. It rewrites the record (patch summary + unified diff +
    patched files) and resets its status so the Reviewer re-adjudicates the
    PATCHED code — a submission with zero applied edits is rejected.
    """
    report = state.get("report_to_test", {})
    applied = [e for e in (state.get("patch_log") or []) if isinstance(e, dict)]
    if not applied:
        return _reject(
            tool_call_id,
            "submit_patch rejected: no edit was applied via patch_source_file — "
            "there is nothing to bank. Apply the minimal blocking edit first, "
            "then submit again.",
        )

    files: list[str] = []
    for entry in applied:
        f = str(entry.get("file") or "")
        if f and f not in files:
            files.append(f)

    # Pure concatenation of real unified diffs — every report-bundled
    # patches/<vuln_id>.patch stays mechanically apply-able. Idempotence-guard
    # notes only surface when NOTHING changed textually (crash-resume replay).
    real_diffs = [str(e.get("diff") or "").rstrip() for e in applied]
    real_diffs = [d for d in real_diffs if d]
    diff_text = "\n".join(real_diffs) or "\n".join(
        str(e.get("note") or "(no textual change)") for e in applied
    )

    updated = dict(report)
    updated["status"] = "confirmed"  # back into the lifecycle: re-review decides
    # Archive the superseded attempt (retry rounds) before overwriting the
    # current patch_* fields — the next attempt's first turn reads this.
    history = list(report.get("patch_history") or [])
    if report.get("patch_diff") or report.get("patch_summary"):
        history.append({
            "round": report.get("patch_round") or 1,
            "summary": str(report.get("patch_summary") or ""),
            "files": list(report.get("patched_files") or []),
            "diff": str(report.get("patch_diff") or ""),
            "outcome": _attempt_outcome(report),
        })
    updated["patch_history"] = history or None
    updated["patch_summary"] = summary.strip()
    updated["patch_diff"] = diff_text
    updated["patched_files"] = files
    updated["patch_round"] = (report.get("patch_round") or 0) + 1
    # A retry re-enters from 'rejected'; clearing (None) on fresh attempts too —
    # never inherit 'reviewed'/'verified' markers into a new pending edit.
    updated["patch_state"] = "applied"

    # Store the verdict so a replay of the same record short-circuits BEFORE
    # the loop: the files on disk are already patched, so re-running the edits
    # is neither needed nor safe.
    cache_patcher(dict(report), updated, state.get("token_spent"))

    logging.info(
        f"Patcher: submitted fix for {report.get('vuln_id', 'Unknown')} "
        f"touching {files}."
    )
    return Command(update={
        "vulnerabilities": [updated],
        "messages": [ToolMessage(
            content=(
                "Patch banked. The record returns to the Reviewer for "
                "re-adjudication against the patched source."
            ),
            name="submit_patch",
            tool_call_id=tool_call_id,
        )],
    })
