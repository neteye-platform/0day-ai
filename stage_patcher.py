"""Patcher stage: minimal first-party fixes for validator-proven exploits.

Every `exploitable` record (flag-gated by `settings.patcher_enabled`, before the
integration-audit phase) is dispatched to a PatcherAgent tool-loop that authors
the smallest source edit blocking the proven flow (`patch_tools`), then the
sandbox is resynced with the new code (`utils.resync_sandbox`) and the record
goes back to the Reviewer for re-adjudication against the PATCHED source — its
`patch_state` lifecycle ("applied" -> "reviewed") and the existing merge-ladder
rules drive the loop, and `tools.submit_evaluation` flips the marker. When a
re-review re-confirms, the normal validator dispatch re-proves (or refutes, per
the patched-target contract) the fix against the resynced sandbox. With the flag
off, the router never returns `patch_dispatch` and nothing here ever runs.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command, Send

import patch_tools
import settings
import tools
from llms import fast_llm, smart_llm
from run_stats import (
    _record_stat,
    _start_agent_progress,
    affected_nodes_label,
    as_dicts,
    record_llm_usage,
    steps_block,
)
from schemas import PATCHER_AGENT
from stage_reviewer import build_reviewer_payload
from state import MasterState, PatcherState
from tool_loop import CompactionConfig, ToolLoopAgent
from utils import cache_patcher, format_node_context, get_cached_graph_data, get_node_code, resync_sandbox

# Record classes the patcher must never touch: dependency-anchored hypotheses
# have their sink inside the vendor tree (first-party edit impossible), and a
# systemic verdict spans a node cluster no bounded edit can cover completely —
# both would waste a loop the fallback ends as `failed` anyway.
_UNPATCHABLE_VULN_TYPES = {
    "Known Dependency Vulnerability",
    "Systemic Vulnerability",
}


def needs_patch(record: dict) -> bool:
    """True when this record is due a patch attempt right now. A REJECTED fix
    (exploit still fired on the patched build) is retry-eligible while the
    patch_round budget lasts: the Patcher sees every prior attempt and its
    failure evidence in its first turn. `applied`/`reviewed` (an attempt in
    flight) and `failed`/`verified` (terminal) never re-enter."""
    return (
        record.get("status") == "exploitable"
        and record.get("patch_state") in (None, "rejected")
        and (record.get("patch_round") or 0) < settings.patcher_max_attempts
        and record.get("vulnerability_type") not in _UNPATCHABLE_VULN_TYPES
    )


def patchable_records(state: MasterState) -> list[dict]:
    return [v for v in as_dicts(state.get("vulnerabilities", [])) if needs_patch(v)]


def _applied(state: MasterState) -> list[dict]:
    """Records whose patch was just banked and whose reviewer re-check (and the
    sandbox resync preceding it) is still owed. The reviewer's submit flips
    "applied" -> "reviewed", so a re-arrival of this router never re-loops."""
    return [
        v for v in as_dicts(state.get("vulnerabilities", []))
        if v.get("patch_state") == "applied"
    ]


def _patcher_payload(
    state: MasterState, record: dict, progress_id: str
) -> PatcherState:
    return PatcherState(
        pipeline_run_id=state.get("pipeline_run_id"),
        report_to_test=record,
        sandbox_url=state.get("sandbox_url"),
        patch_log=[],
        progress_id=progress_id,
        iterations=0,
        vulnerabilities=[],
        messages=[],
    )


def dispatch_patchers(state: MasterState):
    """Fan exploitable records out to patcher threads (flag-gated upstream:
    route_validator_feedback only routes here when settings.patcher_enabled)."""
    records = patchable_records(state)
    if not records:
        return "integration_audit_dispatch"

    progress_id = _start_agent_progress(len(records))
    logging.info(
        f"Starting patch pass: 0/{len(records)} complete, {len(records)} remaining "
        f"({[r.get('vuln_id') for r in records]})."
    )
    commands = [
        Send("patcher_agent", _patcher_payload(state, record, progress_id))
        for record in records
    ]
    _record_stat("patcher_records", len(commands))
    return commands


def route_patch_reviews(state: MasterState):
    """After a patcher superstep: resync the sandbox + re-review exactly when at
    least one patch was banked this superstep; otherwise advance to the audit
    phase (failed attempts already kept their record `exploitable`)."""
    applied = _applied(state)
    if not applied:
        return "integration_audit_dispatch"
    return "sandbox_resync"


def sandbox_resync_node(state: MasterState) -> dict:
    """One deterministic resync for the whole batch of freshly patched records
    (image rebuild for source-built targets, docker-cp + restart otherwise)."""
    files: list[str] = []
    for record in _applied(state):
        for f in record.get("patched_files") or []:
            if f not in files:
                files.append(f)
    res = resync_sandbox(files, state.get("sandbox_container"))
    update = {"sandbox_resync_note": res["note"]}
    if res.get("sandbox_url"):
        update["sandbox_url"] = res["sandbox_url"]
    if res.get("sandbox_container"):
        update["sandbox_container"] = res["sandbox_container"]
    return update


def dispatch_patch_reviews(state: MasterState):
    """Send every freshly-patched record back to the Reviewer (cache-exempt
    re-review via is_feedback_review; the PATCH APPLIED block teaches the mode
    contract in the reviewer's first turn)."""
    applied = _applied(state)
    if not applied:
        return "integration_audit_dispatch"

    progress_id = _start_agent_progress(len(applied))
    commands = [
        Send(
            "reviewer_agent",
            build_reviewer_payload(record, progress_id, state.get("pipeline_run_id")),
        )
        for record in applied
    ]
    logging.info(
        f"Patched records returning to the Reviewer: 0/{len(applied)} complete, "
        f"{len(applied)} remaining ({[r.get('vuln_id') for r in applied]})."
    )
    _record_stat("reviewer_patch_rechecks", len(commands))
    return commands


PATCHER_SUMMARY_LEDGER = (
    "You are an expert summarizer. The following is the message history of a "
    "patcher agent authoring the minimal source-code fix that blocks one "
    "proven-exploitable vulnerability in a target application. Your summary "
    "will REPLACE these messages in the model context, so the patcher must be "
    "able to continue WITHOUT re-reading the original file contents.\n\n"
    "Produce an information-dense summary as a patching ledger with exactly "
    "these sections:\n"
    "## Objective\n"
    "One or two sentences restating the vulnerability, the exploit flow to "
    "block, and the sink/choke point identified.\n"
    "## Files & Code Examined\n"
    "Bulleted, deduplicated list of every file/function/search already "
    "examined with the single fact each revealed (paths and line numbers "
    "preserved exactly). No code dumps.\n"
    "## Edits Applied\n"
    "For each patch_source_file call already made: file, line range, one-line "
    "content of the replacement, and the tool's line-count delta. 'none' if "
    "no edit applied yet.\n"
    "## Confirmed Facts\n"
    "Bulleted list of verified facts (where user input enters, what guards "
    "exist, what the payload is), stated in final form.\n"
    "## Ruled-Out Options\n"
    "Bulleted list of fix locations rejected so far, with a one-line reason.\n"
    "## Remaining Steps\n"
    "Bulleted list of outstanding edits and the final static self-check.\n\n"
    "RULES:\n"
    "- If the history contains a prior SUMMARY (a system message containing "
    "'CONTEXT COMPACTION SUMMARY'), extend and refine it with only the new "
    "facts instead of regenerating.\n"
    "- Line numbers inside the summary may be STALE after an edit: mark them "
    "as pre-edit snapshots so the patcher re-reads before addressing them.\n"
    "- Keep it concise, under 1024 words. Preserve exact file paths, symbol "
    "names, line ranges, and payload strings.\n"
)


class PatcherAgent(ToolLoopAgent):
    """Patcher track: fixed read+edit toolset, cached-verdict replay safety (the
    disk is already patched — the loop must never re-run), submit_patch end."""

    terminal_tool = "submit_patch"
    cache_hit_label = "Patcher"
    progress_label = "Patcher"

    def _subject(self, state) -> str:
        report = state.get("report_to_test", {})
        return affected_nodes_label(report, report.get("node_id", "Unknown"))

    def bind_tools(self, state):
        return smart_llm.bind_tools(
            [
                tools.read_source_code,   # graph-attached code + its numbering
                tools.read_file,          # paged host reads relative to app root
                tools.search_codebase,    # find related flows / other sinks
                tools.get_definition,     # symbol bodies
                patch_tools.patch_source_file,
                patch_tools.submit_patch,
            ],
            parallel_tool_calls=True,
        )

    def cached_verdict(self, state):
        # Keyed on the full record: the patch fields a previous attempt wrote
        # bust the key, so patch_round>0 records (dispatched only up to
        # patcher_max_attempts) can never replay a stale patch.
        return cache_patcher(state.get("report_to_test", {}))

    def first_turn(self, state, llm_with_tools) -> dict:
        sys_msg = SystemMessage(content=PATCHER_AGENT["prompt"])
        report = state.get("report_to_test", {})
        affected = [n for n in (report.get("affected_nodes") or []) if n]
        affected_str = affected_nodes_label(report, report.get("node_id", "Unknown"))
        attempt = (report.get("patch_round") or 0) + 1

        def clipped(text, limit=5000):
            text = text or "_none_"
            return text if len(text) <= limit else text[:limit] + "\n... [truncated]"

        formatted = (
            f"--- EXPLOITABLE VULNERABILITY TO FIX ---\n"
            f"Vulnerability ID: {report.get('vuln_id', 'Unknown')}\n"
            f"CWE ID: {report.get('cwe_id', 'Unknown')}\n"
            f"Component: {report.get('vulnerable_component', 'Unknown')}\n"
            f"Affected Nodes: {affected_str}\n"
            f"Patch attempt: {attempt} of {settings.patcher_max_attempts}\n\n"
            f"Description: {report.get('description', 'None')}\n\n"
            f"Reviewer Reasoning: {clipped(report.get('reviewer_reasoning'), 3000)}\n\n"
            f"--- REPRODUCTION STEPS (the flow your patch must block) ---\n"
            f"{steps_block(report.get('reproduction_steps') or [])}"
        )
        # Retry rounds: what was already tried and HOW it failed. Without this
        # the model re-invents the edit the sandbox just proved insufficient.
        history = report.get("patch_history") or []
        if history:
            blocks = []
            for h in history:
                if not isinstance(h, dict):
                    continue
                blocks.append(
                    f"### Attempt {h.get('round', '?')}\n"
                    f"Summary: {h.get('summary') or '_none_'}\n"
                    f"Files: {', '.join(h.get('files') or []) or '_none_'}\n"
                    f"Outcome: {h.get('outcome') or '_unknown_'}\n"
                    f"```\n{clipped(h.get('diff'), 2500).rstrip()}\n```"
                )
            if blocks:
                formatted += (
                    "\n\n--- PRIOR PATCH ATTEMPTS (failed — do NOT repeat them verbatim) ---\n"
                    "These edits were applied and REJECTED (or abandoned). Build on "
                    "what they got wrong: aim at a deeper choke point or a missed "
                    "entry into the sink; the current on-disk source already "
                    "reflects the LATEST attempt only.\n"
                    f"{'\n\n'.join(blocks)}"
                )
        if report.get("poc_payload"):
            formatted += (
                "\n\n--- PROVEN PoC (validator) ---\n"
                f"```\n{clipped(report['poc_payload']).rstrip()}\n```"
            )
        if report.get("execution_logs"):
            formatted += (
                "\n\n--- EXECUTION LOGS (validator) ---\n"
                f"```\n{clipped(report['execution_logs'], 2500).rstrip()}\n```"
            )

        code_sections = []
        for node_id in affected[:4]:
            node_source = get_node_code(node_id, reviewer_mode=True)
            if node_source:
                code_sections.append(f"Node: {node_id}\n```\n{node_source}\n```")
            # Graph-position block: file:line + callers, to anchor the first reads.
            node_ctx = format_node_context(
                get_cached_graph_data(settings.graph), node_id) or ""
            if len(node_ctx) > 2500:
                node_ctx = node_ctx[:2500] + "\n... [context truncated: search/read for more]"
            if node_ctx:
                code_sections.append(
                    f"Node position: {node_id}\n{node_ctx}"
                )
        if code_sections:
            formatted += (
                "\n\n--- AFFECTED NODES SOURCE (current on-disk code, "
                "numbering from the source tool) ---\n"
                f"{'\n\n'.join(code_sections)}"
            )

        human_msg = HumanMessage(content=(
            f"The application source root (all relative paths in patch_source_file / "
            f"read_file resolve against it) is: {settings.app_path}\n\n"
            "Author the minimal patch blocking the exploit flow below, then call "
            "submit_patch once.\n\n"
            f"{formatted}"
        ))
        messages = [sys_msg, human_msg]
        response = llm_with_tools.invoke(messages)
        return {
            "messages": [sys_msg, human_msg, response],
            "iterations": 1,
            "token_spent": record_llm_usage(self.name, response),
        }

    def fallback(self, state) -> Command:
        """Iteration cap crossed without submit_patch: record the failed attempt
        (the exploit stays `exploitable` and reportable, carrying a patch_round
        that ends its patch lifecycle)."""
        report = state.get("report_to_test", {})
        updated = dict(report)
        updated["patch_state"] = "failed"
        updated["patch_round"] = (report.get("patch_round") or 0) + 1

        # Cache the failure so a resume/replay short-circuits instead of
        # burning another doomed loop (files this run may have half-edited are
        # NOT reverted — the reviewer re-read and the validator smoke test
        # adjudicate them; the status stays exploitable either way).
        cache_patcher(dict(report), updated, state.get("token_spent"))

        logging.info(
            f"Patcher on {updated.get('vuln_id', 'Unknown')} ended after "
            f"{state.get('iterations', 0)} iterations without submitting a fix."
        )
        return Command(update={"vulnerabilities": [updated]})


patcher_agent = PatcherAgent(
    name="patcher",
    settings_prefix="patcher",
    compaction=CompactionConfig(),
    summary_ledger=PATCHER_SUMMARY_LEDGER,
    summary_llm=fast_llm,
)

# Module-level graph node callables (mirrors the other stage modules).
patcher_agent_node = patcher_agent.agent
patcher_router = patcher_agent.router
patcher_fallback_node = patcher_agent.fallback
ask_patcher_for_tool = patcher_agent.ask
