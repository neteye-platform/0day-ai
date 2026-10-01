"""Validator stage: external exploit proofing against the live sandbox."""

import logging
import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command, Send

import attacker_tools
import browser_tools
import credential_finder
import settings
import tools
from llms import fast_llm, smart_llm
from run_stats import _record_stat, as_dicts
from schemas import VALIDATOR_AGENT
from state import MasterState, ReviewerState, ValidatorState
from stage_reviewer import _primary_node, _reviewer_mode_for
from tool_loop import CompactionConfig, ToolLoopAgent
from utils import cache_validator, get_node_code


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

    commands = []
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
        payload = ValidatorState(
            report_to_test=evaluation,
            sandbox_url=state.get("sandbox_url"),
            messages=[],
            iterations=0,
            vulnerabilities=[],
            cookies={},
            agent_id=uuid.uuid4().hex,
        )
        commands.append(Send("validator_agent", payload))

    if not commands:
        # Nothing to validate directly: advance to the integration-audit phase
        # (instead of ENDing) so deferred records still reach the auditor.
        return "integration_audit_dispatch"

    _record_stat("validator_records", len(commands))
    return commands


def route_validator_feedback(state: MasterState):
    """Conditional router from the validator back into the reviewer (or onward).

    'insufficient_context' records within validator_feedback_max_rounds go back
    to the Reviewer for a re-review; the review_round cap drains the loop.
    Otherwise return the `integration_audit_dispatch` marker."""
    all_vulns = as_dicts(state.get("vulnerabilities", []))
    max_rounds = settings.validator_feedback_max_rounds
    flagged = [
        v for v in all_vulns
        if v.get("status") == "insufficient_context"
        and (v.get("review_round") or 0) <= max_rounds
    ]

    if not flagged:
        return "integration_audit_dispatch"

    commands = []
    for record in flagged:
        payload = ReviewerState(
            node_id=_primary_node(record),
            expert_report=record,
            mode=_reviewer_mode_for(record),
            iterations=0,
            vulnerabilities=[],
            messages=[]
        )
        commands.append(Send("reviewer_agent", payload))

    logging.info(
        f"Validator requested more context for {len(commands)} vulnerability(ies); "
        f"dispatching reviewer feedback re-reviews."
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

    def _subject(self, state) -> str:
        report = state.get("report_to_test", {})
        affected = [n for n in (report.get("affected_nodes") or []) if n]
        if affected:
            return ", ".join(affected)
        return report.get("node_id", "Unknown")

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
        if getattr(settings, "attacker_enabled", False):
            validator_tools += [
                attacker_tools.run_command,
                attacker_tools.write_attacker_file,
                attacker_tools.read_attacker_file
            ]
        # ask_for_context is bound ONLY on the first validation pass.
        if (state.get("report_to_test", {}).get("review_round") or 0) < settings.validator_feedback_max_rounds:
            validator_tools.append(tools.ask_for_context)
        return smart_llm.bind_tools(validator_tools)

    def cached_verdict(self, state):
        return cache_validator(
            state.get("report_to_test", {}), state.get("peer_payloads")
        )

    def first_turn(self, state, llm_with_tools) -> dict:
        # System prompt composed from the capabilities this run actually grants
        # (attacker shell only when enabled; insufficient-context hatch only on
        # the first pass, mirroring bind_tools).
        sys_prompt = VALIDATOR_AGENT["prompt"]
        if getattr(settings, "attacker_enabled", False):
            sys_prompt += "\n\n" + VALIDATOR_AGENT.get("attacker_tools", "")
        if (state.get("report_to_test", {}).get("review_round") or 0) < settings.validator_feedback_max_rounds:
            sys_prompt += "\n\n" + VALIDATOR_AGENT.get("insufficient_context", "")
        sys_msg = SystemMessage(content=sys_prompt)
        # Build a structured string for the LLM
        report = state['report_to_test']
        affected = [n for n in (report.get("affected_nodes") or []) if n]
        affected_str = (
            ", ".join(affected) if affected else report.get('node_id', 'Unknown')
        )
        steps = report.get('reproduction_steps') or []
        steps_str = (
            "\n".join(f"  {i}. {s}" for i, s in enumerate(steps, 1))
            if steps else "  None provided by reviewer"
        )
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
        # Source of every affected node so the validator can reason about the
        # exact code under test without extra lookups.
        code_sections = []
        for node_id in affected:
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
        # (mirrors bind_tools gating). Failed (status='error') tool messages
        # never count: a rejected mark_validation_complete must bounce back.
        first_pass = (
            state.get("report_to_test", {}).get("review_round") or 0
        ) < settings.validator_feedback_max_rounds
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
        updated_vuln = dict(state.get("report_to_test", {}))

        timeout_note = (
            f"[validation timeout] No verdict after {state.get('iterations', 0)} "
            f"tool-loop iterations; result on this vulnerability is unproven."
        )
        existing_logs = updated_vuln.get("execution_logs") or ""
        updated_vuln["execution_logs"] = (
            f"{existing_logs}\n{timeout_note}" if existing_logs else timeout_note
        )

        # Save to cache so subsequent runs skip the (doomed) tool-calling loop.
        cache_validator(dict(state.get("report_to_test", {})), state.get("peer_payloads"), updated_vuln)

        # Per-agent cleanup, same as the terminal tool does.
        browser_tools.manager.close_agent_sessions(state.get("agent_id"))
        attacker_tools.manager.close_agent_sessions(state.get("agent_id"))

        return Command(
            update={
                "vulnerabilities": [updated_vuln],
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
