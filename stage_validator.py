"""Validator stage: external exploit proofing against the live sandbox."""

import logging
import re
import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command, Send

import attacker_tools
import browser_tools
import credential_finder
import settings
import tools
from llms import fast_llm, validator_llm
from run_stats import _record_stat, _start_agent_progress, affected_nodes_label, as_dicts, steps_block
from schemas import VALIDATOR_AGENT
from stage_patcher import patchable_records
from state import MasterState, ValidatorState
from stage_reviewer import build_reviewer_payload
from tool_loop import CompactionConfig, ToolLoopAgent
from utils import cache_validator, get_node_code


_BARE_IDENT_RE = re.compile(r"^[\w$]+$")


def _validation_group_key(record: dict):
    """Batching key for a confirmed record, or None when it must be validated
    alone. Members must match (CWE, normalized vulnerable_component) EXACTLY —
    never a fuzzy similarity, since one run's verdict is written to every
    member. Short bare-identifier anchors ('$str', a bare node id) are never
    shareable (hub-callee parameters collide across unrelated defects);
    descriptive labels (>= 3 words) or structural-selector tokens do identify
    one specific flaw site.
    """
    comp = re.sub(r"\s+", " ", (record.get("vulnerable_component") or "").strip().lower())
    if not comp:
        return None
    tokens = comp.split()
    if not (len(tokens) >= 3 or any(not _BARE_IDENT_RE.match(t) for t in tokens)):
        return None
    return (record.get("cwe_id") or "", comp)


def _validator_payload(
    state: MasterState, seed: dict, variants: list[dict] | None, progress_id: str
) -> ValidatorState:
    return ValidatorState(
        pipeline_run_id=state.get("pipeline_run_id"),
        report_to_test=seed,
        validation_variants=variants,
        sandbox_url=state.get("sandbox_url"),
        sandbox_resync_note=state.get("sandbox_resync_note"),
        messages=[],
        iterations=0,
        vulnerabilities=[],
        cookies={},
        agent_id=uuid.uuid4().hex,
        progress_id=progress_id,
    )


def _is_first_pass(state) -> bool:
    """True while the record is still within the reviewer-feedback budget:
    gates the `ask_for_context` binding, its prompt section, and its terminal
    status (all three MUST stay in sync)."""
    return (
        state.get("report_to_test", {}).get("review_round") or 0
    ) < settings.validator_feedback_max_rounds


def dispatch_validators(state: MasterState):
    """Creates a parallel validation thread for each vulnerability that survived
    the reviewer.

    Confirmed records route by `validation_strategy`:
    - `direct_to_validator` (or missing): sent to be proven externally.
    - `requires_integration`: deferred; dispatched by
      `dispatch_integration_audits` only after the direct records are proven,
      so the auditor's chain candidates carry real `poc_payload`s.
    - `static_finding_only`: accepted into the report, not dispatched.

    Returns Sends, or the `integration_audit_dispatch` marker when none.
    """
    all_vulns = as_dicts(state.get("vulnerabilities", []))
    confirmed_vulns = [v for v in all_vulns if v.get("status") == "confirmed"]
    logging.info(
        f"dispatch_validators sees {len(all_vulns)} records in the parent channel, "
        f"{len(confirmed_vulns)} confirmed "
        f"({[v.get('vuln_id') for v in confirmed_vulns]})."
    )

    direct: list[dict] = []
    for evaluation in confirmed_vulns:
        strategy = evaluation.get("validation_strategy") or "direct_to_validator"
        if strategy == "static_finding_only":
            # Real in source, no network-reachable path: static proof accepted.
            logging.info(
                f"{evaluation.get('vuln_id')} marked static_finding_only — "
                f"accepted into report without validation."
            )
            continue
        if strategy == "requires_integration":
            # Deferred so the auditor sees exploitable peers with proven payloads.
            logging.info(
                f"{evaluation.get('vuln_id')} requires_integration — deferred to "
                f"the integration audit phase (after direct validation)."
            )
            continue
        direct.append(evaluation)

    # Batch confirmed records that describe the SAME flaw (identical
    # (cwe_id, vulnerable_component) — e.g. one contract verdict per victim
    # endpoint) into a single validator run carrying every variant's steps.
    # Non-destructive: each member still flows through the vulnerabilities
    # channel as its own record; only the sandbox work is shared.
    groups: dict[tuple, list[dict]] = {}
    for evaluation in direct:
        # Patched records ride alone: their validation is a two-step fix-check
        # (expired PoC + smoke test) bound to THEIR diff, so sharing one verdict
        # across a (cwe, component) group could mark an unverified fix proven.
        if evaluation.get("patch_state") or evaluation.get("patch_diff"):
            continue
        if key := _validation_group_key(evaluation):
            groups.setdefault(key, []).append(evaluation)
    batched = {k: v for k, v in groups.items() if 1 < len(v) <= settings.validator_variant_max_group}
    shared_ids = {id(r) for members in batched.values() for r in members}

    payloads: list[tuple[dict, list[dict] | None]] = []
    for members in batched.values():
        seed, variants = members[0], members[1:]
        logging.info(
            f"Sharing one validation run across {len(members)} equivalent "
            f"findings ({[m.get('vuln_id') for m in members]})."
        )
        payloads.append((seed, variants))

    for evaluation in direct:
        if id(evaluation) in shared_ids:
            continue
        payloads.append((evaluation, None))

    # Dynamically prove patched records the re-review cleared statically: the
    # reviewer's false_positive on the PATCHED code is not the deliverable proof
    # — the rule-7 re-test (exploit dead AND legit flow healthy) is what flips
    # patch_state to 'verified'; an exploit that still fires flips it to
    # 'rejected' (retry-eligible). These always ride alone (never batched).
    for evaluation in all_vulns:
        if (
            evaluation.get("status") == "false_positive"
            and evaluation.get("patch_state") == "reviewed"
        ):
            logging.info(
                f"{evaluation.get('vuln_id')} is a reviewer-cleared PATCHED fix — "
                f"dispatching dynamic fix-proof."
            )
            payloads.append((evaluation, None))

    if not payloads:
        # Nothing to validate directly: advance to the integration-audit phase
        # (instead of ENDing) so deferred records still reach the auditor.
        return "integration_audit_dispatch"

    # Ledger id shared by the fan-out: the base router advances it (and logs
    # turns=) on every validator terminal route.
    progress_id = _start_agent_progress(len(payloads))
    commands = [
        Send("validator_agent", _validator_payload(state, seed, variants, progress_id))
        for seed, variants in payloads
    ]
    logging.info(
        f"Dispatched validator: 0/{len(payloads)} complete, "
        f"{len(payloads)} remaining."
    )
    _record_stat("validator_records", len(commands))
    return commands


def route_validator_feedback(state: MasterState):
    """Conditional router from the validator back into the reviewer (or onward).

    'insufficient_context' records within validator_feedback_max_rounds go back
    to the Reviewer for a re-review; the review_round cap drains the loop.
    Otherwise, freshly `exploitable` records due a fix attempt (flag-gated) head
    to the patch dispatch; else return the `integration_audit_dispatch` marker.
    """
    all_vulns = as_dicts(state.get("vulnerabilities", []))
    max_rounds = settings.validator_feedback_max_rounds
    flagged = [
        v for v in all_vulns
        if v.get("status") == "insufficient_context"
        and (v.get("review_round") or 0) <= max_rounds
    ]

    if not flagged:
        if settings.patcher_enabled and patchable_records(state):
            return "patch_dispatch"
        return "integration_audit_dispatch"

    progress_id = _start_agent_progress(len(flagged))

    commands = []
    for record in flagged:
        commands.append(Send(
            "reviewer_agent",
            build_reviewer_payload(record, progress_id, state.get("pipeline_run_id")),
        ))

    logging.info(
        f"Validator requested more context for {len(commands)} vulnerability(ies); "
        f"dispatching reviewer feedback re-reviews: 0/{len(flagged)} complete, "
        f"{len(flagged)} remaining."
    )
    _record_stat("reviewer_feedback_reviews", len(commands))
    return commands


VALIDATOR_SUMMARY_LEDGER = (
    "You are an expert summarizer. The following is the message history of a "
    "security validator agent attempting to prove or refute a reported "
    "vulnerability hypothesis against a live sandbox application via HTTP "
    "requests. Your summary will REPLACE these messages in the model context, "
    "so the validator must be able to continue the proof from it WITHOUT "
    "re-reading the original HTTP responses or tool arguments.\n\n"
    "Produce an information-dense summary as a security validation ledger "
    "with exactly these sections:\n"
    "## Objective\n"
    "One or two sentences restating the exact vulnerability hypothesis under "
    "proof and the reviewer-provided reproduction steps (the chronological "
    "action plan to follow).\n"
    "## Requests Performed\n"
    "Bulleted list of every HTTP request already sent (method, path, params, "
    "auth state), with the single most important fact each response revealed. "
    "Do NOT include full request/response bodies — distill them into "
    "conclusions (status codes, key values echoed, observable mitigations).\n"
    "## Confirmed Facts\n"
    "Bulleted list of verified facts established so far (e.g. endpoint reachable "
    "without auth, parameter reflected in response, validation present), stated "
    "in final form.\n"
    "## Ruled-Out Dead Ends\n"
    "Bulleted list of hypotheses or attack paths already disproven, with a "
    "one-line reason for each.\n"
    "## Active Leads & Next Steps\n"
    "Bulleted list of the most promising remaining checks not yet completed.\n\n"
    "RULES:\n"
    "- If the history contains a prior SUMMARY (a system message containing "
    "'CONTEXT COMPACTION SUMMARY'), treat it as the anchoring summary: extend "
    "and refine it with only the new facts gathered since, rather than "
    "regenerating from scratch.\n"
    "- Write in final form ('POST succeeded', 'the sink is reachable', 'auth "
    "was required'), never 'the model checked X'.\n"
    "- Keep it concise, under 1024 words. Preserve exact paths, parameter "
    "names, status codes, and any cookies that were set.\n"
)


class ValidatorAgent(ToolLoopAgent):
    """Validator track: fixed HTTP-proof toolset, live cookie tracking, and the
    terminal-tool end conditions. Ends on `ask_for_context` (round 1 only — the
    tool is unbound after a re-review so the agent cannot ask again) or on
    `mark_validation_complete`."""

    terminal_tool = ("ask_for_context", "mark_validation_complete")
    cache_hit_label = "Validator"
    progress_label = "Validator"

    def _subject(self, state) -> str:
        report = state.get("report_to_test", {})
        return affected_nodes_label(report, report.get("node_id", "Unknown"))

    def bind_tools(self, state):
        validator_tools = [
            tools.send_http_request,
            browser_tools.browser_navigate,
            browser_tools.browser_click,
            browser_tools.browser_fill,
            browser_tools.browser_evaluate,
            browser_tools.browser_console,
            tools.mark_validation_complete
        ]
        if settings.attacker_enabled:
            validator_tools += [
                attacker_tools.run_command,
                attacker_tools.write_attacker_file,
                attacker_tools.read_attacker_file
            ]
        # ask_for_context is bound ONLY on the first validation pass.
        if _is_first_pass(state):
            validator_tools.append(tools.ask_for_context)
        return validator_llm.bind_tools(validator_tools)

    def cached_verdict(self, state):
        return cache_validator(
            state.get("report_to_test", {}), state.get("peer_payloads")
        )

    def pre_agent(self, state):
        # A cache hit stores the SEED's verdict; grouped validation variants
        # are not part of the cache key, so replay them from the current state
        # to give every batched member its own updated record.
        cmd = super().pre_agent(state)
        if cmd is None or not state.get("validation_variants"):
            return cmd
        cached = (cmd.update or {}).get("vulnerabilities") or [None]
        if cached[0] is None:
            return cmd
        return Command(update={
            "vulnerabilities": tools.propagate_validation_update(state, cached[0]),
            "cache_tag": "HIT",
        })

    def first_turn(self, state, llm_with_tools) -> dict:
        # System prompt composed from the capabilities this run actually grants
        # (attacker shell only when enabled; insufficient-context hatch only on
        # the first pass, mirroring bind_tools).
        sys_prompt = VALIDATOR_AGENT["prompt"]
        if settings.attacker_enabled:
            sys_prompt += "\n\n" + VALIDATOR_AGENT.get("attacker_tools", "")
        if _is_first_pass(state):
            sys_prompt += "\n\n" + VALIDATOR_AGENT.get("insufficient_context", "")
        sys_msg = SystemMessage(content=sys_prompt)
        report = state['report_to_test']
        affected = [n for n in (report.get("affected_nodes") or []) if n]
        affected_str = affected_nodes_label(report, report.get('node_id', 'Unknown'))
        steps_str = steps_block(report.get('reproduction_steps') or [])
        formatted_report = (
            f"--- CORE VULNERABILITY ---\n"
            f"Vulnerability ID: {report.get('vuln_id', 'Unknown')}\n"
            f"CWE ID: {report.get('cwe_id', 'Unknown')}\n"
            f"Affected Nodes: {affected_str}\n\n"
            f"--- CONTEXT & REASONING ---\n"
            f"Description: {report.get('description', 'None')}\n\n"
            f"Reviewer Reasoning: {report.get('reviewer_reasoning', 'None')}\n\n"
            f"--- REPRODUCTION STEPS (from Reviewer, follow in order) ---\n"
            f"{steps_str}"
        )
        # Reviewer found the flow real but could not statically settle these points;
        # the sandbox adjudicates (mirrors PROVEN CHAIN COMPONENTS rendering).
        reservations = report.get("reservations") or []
        if reservations:
            res_str = "\n".join(f"  {i}. {r}" for i, r in enumerate(reservations, 1))
            formatted_report += (
                f"\n\n--- REVIEWER RESERVATIONS (resolve every item) ---\n"
                f"Points the Reviewer could not verify from source alone. Prove or refute "
                f"each in the sandbox and log the outcome with evidence in "
                f"execution_logs; an exploit failing exactly on a reservation is "
                f"false-positive evidence.\n"
                f"{res_str}"
            )
        # A source→sink flow the Reviewer saw that the hypothesis may not name — test it too.
        concern = report.get("out_of_scope_concern")
        if concern:
            formatted_report += (
                "\n\n--- REVIEWER OUT-OF-SCOPE CONCERN (test it too) ---\n"
                "Observed source-to-sink flow beyond this record's hypothesis: "
                "if the steps above do not exercise it, test it in the sandbox and "
                f"log the outcome with evidence in execution_logs.\n{concern}"
            )
        # Patched records: the fix lives in PROPOSED PATCH; the sandbox either
        # already runs it or does not (SANDBOX SYNC decides what a replay proves).
        # Two-step contract per validator prompt rule 7.
        if report.get("patch_diff"):
            patch_files = ", ".join(report.get("patched_files") or []) or "_none_"
            sync_note = (
                state.get("sandbox_resync_note")
                or "unknown: no resync outcome was recorded — treat the sandbox's "
                   "patch state as UNVERIFIED."
            )
            formatted_report += (
                f"\n\n--- PROPOSED PATCH (attempt {report.get('patch_round', 1)}; applied "
                f"to source after the last exploit proof) ---\n"
                f"Fix summary: {report.get('patch_summary') or '_none_'}\n"
                f"Files touched: {patch_files}\n"
                f"Unified diff:\n```\n{report['patch_diff']}\n```\n"
                f"--- SANDBOX SYNC ---\n{sync_note}\n"
                f"Adjudicate per the PATCHED TARGET rule: replay the reproduction "
                f"steps above; if the sandbox contains the patch, an exploit that "
                f"still fires means the fix FAILED, and an exploit that stays dead "
                f"still requires the legitimate-flow smoke test before any false "
                f"positive. Log both outcomes with evidence in execution_logs."
            )
        # Chained records carry the proven poc_payloads of the peers they chain
        # with, so the final exploit reuses real proven primitives.
        peer_payloads = state.get("peer_payloads") or []
        if peer_payloads:
            blocks = []
            for pp in peer_payloads:
                blocks.append(
                    f"### {pp.get('vuln_id', '?')}\n"
                    f"CWE: {pp.get('cwe_id', '?')}\n"
                    f"Description: {pp.get('description') or '_none_'}\n"
                    f"Proven poc_payload:\n"
                    f"```\n{(pp.get('poc_payload') or '_none_').rstrip()}\n```\n"
                    f"Execution logs:\n"
                    f"```\n{(pp.get('execution_logs') or '_none_').rstrip()}\n```"
                )
            formatted_report += (
                f"\n\n--- PROVEN CHAIN COMPONENTS (poc_payloads from peer validators) ---\n"
                f"These OTHER vulnerabilities are already proven exploitable in the "
                f"sandbox and their working payloads are below. The chained "
                f"reproduction steps above build on them — execute/adapt these exact "
                f"payloads to complete the chain.\n"
                f"{'\n\n'.join(blocks)}"
            )
        # Grouped equivalent findings (same CWE + same vulnerable component,
        # batched by dispatch_validators): render every member's reproduction
        # steps so the single validation run exercises all of them — the
        # terminal verdict will be recorded on each member record.
        variants = state.get("validation_variants") or []
        if variants:
            blocks = []
            for k, v in enumerate(variants, 2):
                blocks.append(
                    f"### VARIANT {k}: {v.get('vuln_id', 'Unknown')}\n"
                    f"Affected Nodes: {affected_nodes_label(v, 'Unknown')}\n"
                    f"Description: {v.get('description', 'None')}\n"
                    f"--- REPRODUCTION STEPS (from Reviewer, follow in order) ---\n"
                    f"{steps_block(v.get('reproduction_steps') or [])}"
                    + (
                        "\n--- REVIEWER RESERVATIONS (resolve for this variant too) ---\n"
                        + "\n".join(f"  {i}. {r}" for i, r in enumerate(v["reservations"], 1))
                        if v.get("reservations") else ""
                    )
                )
            formatted_report += (
                f"\n\n--- EQUIVALENT VARIANTS OF THE SAME PATTERN "
                f"(validate EVERY record above and below) ---\n"
                f"The findings below are duplicate reports of the SAME underlying "
                f"vulnerability (identical CWE and vulnerable component) at other "
                f"affected locations. Exercise the reproduction steps for EACH "
                f"variant in addition to the core one; a generic proof of the "
                f"shared pattern counts only if you demonstrate it applies to "
                f"every variant's affected nodes. Record per-variant outcomes in "
                f"execution_logs. Your single final verdict will be written to "
                f"the core vulnerability AND to every variant.\n"
                f"{'\n\n'.join(blocks)}"
            )
        # Source of every affected node so the validator can reason about the
        # exact code under test without extra lookups.
        all_nodes = list(affected)
        for v in variants:
            for n in (v.get("affected_nodes") or []):
                if n and n not in all_nodes:
                    all_nodes.append(n)
        code_sections = []
        for node_id in all_nodes:
            node_source = get_node_code(node_id)
            if node_source:
                code_sections.append(
                    f"Node: {node_id}\n```\n{node_source}\n```"
                )
        if code_sections:
            formatted_report += (
                f"\n\n--- AFFECTED NODES SOURCE CODE ---\n"
                f"Source code of nodes affected by the vulnerability (bodies of "
                f"peer nodes are omitted because not relevant).\n"
                f"{'\n\n'.join(code_sections)}"
            )
        # Pre-configured sandbox credentials (from preprocessing); empty if none.
        auth_block = credential_finder.authentication_block()
        human_msg = HumanMessage(content=(
            f"Target Sandbox: {state['sandbox_url']}\n\n"
            + (f"{auth_block}\n\n" if auth_block else "")
            + f"Vulnerability to Prove:\n{formatted_report}"
        ))
        messages = [sys_msg, human_msg]
        response = llm_with_tools.invoke(messages)
        return {"messages": [sys_msg, human_msg, response], "iterations": 1}

    def session_state(self, state) -> dict:
        # Merge cookie jars from tool artifacts into ValidatorState.cookies.
        # Scans the raw (pre-compaction) history so compacted cookie-bearing
        # responses are not missed; browser and HTTP channels share the jar.
        current_cookies = dict(state.get("cookies", {}))
        for msg in reversed(state["messages"]):
            if getattr(msg, "type", "") == "ai":
                break
            if getattr(msg, "type", "") == "tool" and hasattr(msg, "artifact") and msg.artifact:
                name = getattr(msg, "name", "")
                if name == "send_http_request":
                    current_cookies.update(msg.artifact)
                elif name in browser_tools.BROWSER_TOOL_NAMES and isinstance(msg.artifact, dict):
                    jar = msg.artifact.get("cookies") or {}
                    if isinstance(jar, dict):
                        current_cookies.update(jar)
        return {"cookies": current_cookies}

    def tool_batch_done(self, state) -> bool:
        # `ask_for_context` is terminal only on passes where it is bound
        # (mirrors bind_tools). Failed (status='error') tool messages
        # never count: a rejected mark_validation_complete must bounce back.
        first_pass = _is_first_pass(state)
        terminal_names = [
            n for n in self._terminal_names()
            if n != "ask_for_context" or first_pass
        ]
        for msg in reversed(state["messages"]):
            if getattr(msg, "type", "") != "tool":
                break
            if getattr(msg, "status", "") == "error":
                continue
            if getattr(msg, "name", "") in terminal_names:
                return True
        return False

    def fallback(self, state) -> Command:
        """Resolve an iteration-capped validation: keeps the reviewer's
        "confirmed" status (never proven either way) and logs the timeout."""
        timeout_note = (
            f"[validation timeout] No verdict after {state.get('iterations', 0)} "
            f"tool-loop iterations; result on this vulnerability is unproven."
        )

        def with_timeout_note(record: dict) -> dict:
            record = dict(record)
            existing_logs = record.get("execution_logs") or ""
            record["execution_logs"] = (
                f"{existing_logs}\n{timeout_note}" if existing_logs else timeout_note
            )
            return record

        updated_vuln = with_timeout_note(state.get("report_to_test", {}))

        # Save to cache so subsequent runs skip the (doomed) tool-calling loop.
        cache_validator(dict(state.get("report_to_test", {})), state.get("peer_payloads"), updated_vuln)

        # Batched variants stay unproven too: give each its own timeout note
        # (status untouched, same as the seed's).
        updates = [updated_vuln]
        for member in state.get("validation_variants") or []:
            updates.append(with_timeout_note(member))

        # Per-agent cleanup, same as the terminal tool does.
        browser_tools.manager.close_agent_sessions(state.get("agent_id"))
        attacker_tools.manager.close_agent_sessions(state.get("agent_id"))

        return Command(
            update={
                "vulnerabilities": updates,
            }
        )


validator_agent = ValidatorAgent(
    name="validator",
    settings_prefix="validator",
    compaction=CompactionConfig(),
    summary_ledger=VALIDATOR_SUMMARY_LEDGER,
    summary_llm=fast_llm,
)

# Module-level graph node callables (kept so graph.py's imports stay untouched).
validator_agent_node = validator_agent.agent
validator_router = validator_agent.router
validator_fallback_node = validator_agent.fallback
ask_validator_for_tool = validator_agent.ask
