"""Integration-auditor stage: chain requires_integration records with proven peers."""

import logging
import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command, Send

import tools
from llms import fast_llm, smart_llm
from run_stats import _record_stat, affected_nodes_label, as_dicts, steps_block
from schemas import INTEGRATION_AUDITOR_AGENT
from state import IntegrationAuditorState, MasterState, ValidatorState
from tool_loop import CompactionConfig, ToolLoopAgent
from utils import append_note, cache_integration_auditor


def dispatch_integration_audits(state: MasterState):
    """Runs strictly AFTER all direct-to-validator records are proven.

    Fans each pending `requires_integration` record to the auditor with the
    other `exploitable` records as chain candidates; nothing pending →
    `reporter_dispatch`."""
    all_vulns = as_dicts(state.get("vulnerabilities", []))
    pending = [
        v for v in all_vulns
        if v.get("status") == "confirmed"
        and v.get("validation_strategy") == "requires_integration"
    ]
    proven = [v for v in all_vulns if v.get("status") == "exploitable"]
    logging.info(
        f"dispatch_integration_audits sees {len(all_vulns)} records, "
        f"{len(pending)} requires_integration still confirmed, "
        f"{len(proven)} proven exploitable peer(s)."
    )

    if not pending:
        # Validator/auditor phases fully drained: advance to reporter dispatch.
        return "reporter_dispatch"

    commands = []
    for evaluation in pending:
        others = [
            v for v in proven
            if v.get("vuln_id") != evaluation.get("vuln_id")
        ]
        logging.info(
            f"Auditing {evaluation.get('vuln_id')}: {len(others)} proven peer(s) "
            f"to chain with "
            f"({[v.get('vuln_id') for v in others]})."
        )
        payload = IntegrationAuditorState(
            report_to_test=evaluation,
            confirmed_vulns=others,
            iterations=0,
            vulnerabilities=[],
            messages=[]
        )
        commands.append(Send("integration_auditor", payload))

    _record_stat("integration_audits", len(commands))
    return commands


INTEGRATION_AUDITOR_SUMMARY_LEDGER = (
    "You are an expert summarizer. The following is the message history of a "
    "security integration auditor agent deciding whether a `requires_integration` "
    "vulnerability (real but not exploitable in isolation) can be combined with "
    "other confirmed vulnerabilities into a concrete multi-step external exploit "
    "chain. Your summary will REPLACE these messages in the model context, so the "
    "auditor must be able to continue the decision from it WITHOUT re-reading the "
    "original tool outputs.\n\n"
    "Produce an information-dense summary as a security chaining ledger with "
    "exactly these sections:\n"
    "## Objective\n"
    "One or two sentences restating the exact `requires_integration` vulnerability "
    "under audit (vuln_id, CWE, affected nodes) and the chain decision pending.\n"
    "## Candidate Peers Examined\n"
    "Bulleted, deduplicated list of every other confirmed vulnerability whose "
    "details were fetched, with the single most important fact each revealed "
    "about how it could provide a precondition (privilege, session, state, file) "
    "to the chained path. Do NOT include full records — distill them.\n"
    "## Confirmed Chain Facts\n"
    "Bulleted list of verified facts established for the chain (e.g. 'IDOR "
    "leaks any user id so it can obtain the admin cookie', 'XSS fires only after "
    "authentication') stated in final form.\n"
    "## Ruled-Out Candidates\n"
    "Bulleted list of other vulnerabilities or chaining paths already rejected, "
    "with a one-line reason for each.\n"
    "## Active Leads & Next Steps\n"
    "Bulleted list of the most promising remaining chain checks not yet completed.\n\n"
    "RULES:\n"
    "- If the history contains a prior SUMMARY (a system message containing "
    "'CONTEXT COMPACTION SUMMARY'), treat it as the anchoring summary: extend "
    "and refine it with only the new facts gathered since, rather than "
    "regenerating from scratch.\n"
    "- Write in final form ('the IDOR returns any profile', 'auth is required "
    "before the sink'), never 'the model checked X'.\n"
    "- Keep it concise, under 1024 words. Preserve exact vuln_ids, node ids, "
    "and reproduction-step content.\n"
)


def _one_line(text: str, limit: int = 240) -> str:
    """Flatten free text to a single compact line for peer summaries."""
    if not text:
        return ""
    return " ".join(str(text).split())[:limit]


class IntegrationAuditorAgent(ToolLoopAgent):
    """Integration auditor track: chain-building tools (get_vulnerability_details
    for peer records, get_node_connections to verify code-level links) and the
    single terminal submit_integration_audit verdict (chained/unchainable)."""

    terminal_tool = "submit_integration_audit"
    cache_hit_label = "Integration auditor"

    def _subject(self, state) -> str:
        return state.get("report_to_test", {}).get("vuln_id", "Unknown")

    def bind_tools(self, state):
        return smart_llm.bind_tools([
            tools.get_vulnerability_details,
            tools.get_node_connections,
            tools.get_path,
            tools.submit_integration_audit,
        ])

    def cached_verdict(self, state):
        return cache_integration_auditor(
            state.get("report_to_test", {}), state.get("confirmed_vulns")
        )

    def first_turn(self, state, llm_with_tools) -> dict:
        sys_msg = SystemMessage(content=INTEGRATION_AUDITOR_AGENT.get("prompt", ""))

        report = state.get("report_to_test", {})
        affected_str = affected_nodes_label(report, report.get("node_id", "Unknown"))
        formatted_vuln = (
            f"--- CORE VULNERABILITY (requires_integration) ---\n"
            f"Vulnerability ID: {report.get('vuln_id', 'Unknown')}\n"
            f"CWE ID: {report.get('cwe_id', 'Unknown')}\n"
            f"Affected Nodes: {affected_str}\n"
            f"Type: {report.get('vulnerability_type', 'Code Defect')}\n"
            f"Description: {report.get('description', '')}\n"
            f"Reviewer Reasoning: {report.get('reviewer_reasoning', 'None')}\n"
        )
        if report.get("source_cve"):
            formatted_vuln += f"Source CVE: {report.get('source_cve')}\n"

        peers = state.get("confirmed_vulns", [])
        if peers:
            peer_lines = []
            for v in peers:
                peer_nodes = [n for n in (v.get("affected_nodes") or []) if n]
                payload = v.get("poc_payload")
                payload_hint = "proven payload available (see details)" if payload else "no proven payload"
                peer_lines.append(
                    f"- vuln_id={v.get('vuln_id', '?')} | CWE={v.get('cwe_id', '?')} | "
                    f"nodes={', '.join(peer_nodes) or '?'} | {payload_hint} | "
                    f"{_one_line(v.get('description', ''))}"
                )
            formatted_vuln += (
                f"\n--- OTHER PROVEN VULNERABILITIES (candidates to chain with) ---\n"
                f"All of these were validated in the sandbox and are exploitable; their "
                f"working `poc_payload`s are available. Call "
                f"`get_vulnerability_details(<vuln_id>)` to fetch the full record "
                f"(including the proven payload) of any of these before relying on it "
                f"in a chain:\n"
                + "\n".join(peer_lines)
            )
        else:
            # 'chained' is impossible with zero peers: resolve terminal 'unchainable'
            # in place; the empty-messages pre_router guard ends the subgraph here.
            logging.warning(
                f"{report.get('vuln_id', 'Unknown')} is requires_integration but arrived "
                f"at the auditor with an EMPTY confirmed_vulns peer list; resolving "
                f"'unchainable' without invoking the auditor LLM. Possible cause: "
                f"dispatch_integration_audits saw no 'exploitable' records in the "
                f"parent channel (check the dispatch_integration_audits log lines "
                f"in this run)."
            )
            note = (
                "[integration auditor] No other proven vulnerabilities exist to "
                "chain with; resolved 'unchainable' without invoking the LLM (a "
                "'chained' verdict requires at least one other exploitable "
                "vulnerability)."
            )
            record = append_note(report, "integration_audit_reasoning", note)
            record["status"] = "unchainable"
            # Cache so a repeat of the same report short-circuits in the base
            # pre_agent cache hook instead of re-running this branch.
            cache_integration_auditor(report, [], record)
            return Command(update={"vulnerabilities": [record]})

        # strip_numbering: our counter must never double-number the reviewer's steps.
        steps_str = steps_block(report.get("reproduction_steps") or [], strip_numbering=True)
        formatted_vuln += (
            f"\n--- REVIEWER'S ISOLATED REPRODUCTION STEPS (this record alone) ---\n"
            f"{steps_str}"
        )

        human_msg = HumanMessage(content=(
            f"Audit the following `requires_integration` vulnerability for a combinable "
            f"multi-step exploit chain.\n\n{formatted_vuln}"
        ))

        response = llm_with_tools.invoke([sys_msg, human_msg])
        return {"messages": [sys_msg, human_msg, response], "iterations": 1}

    def fallback(self, state) -> Command:
        """Resolve an iteration-capped audit: keeps the record 'confirmed'
        (chain never proven) and records the timeout in the reasoning."""
        timeout_note = (
            f"[integration audit timeout] No verdict after {state.get('iterations', 0)} "
            f"tool-loop iterations; the chain was not resolved. Keeping the record as "
            f"'confirmed' (chain inconclusive) without a chained/unchainable verdict."
        )
        updated_vuln = append_note(
            state.get("report_to_test", {}), "integration_audit_reasoning", timeout_note
        )
        cache_integration_auditor(
            dict(state.get("report_to_test", {})),
            state.get("confirmed_vulns"),
            updated_vuln,
        )
        return Command(update={"vulnerabilities": [updated_vuln]})


integration_auditor_agent = IntegrationAuditorAgent(
    name="integration_auditor",
    settings_prefix="integration_auditor",
    compaction=CompactionConfig(),
    summary_ledger=INTEGRATION_AUDITOR_SUMMARY_LEDGER,
    summary_llm=fast_llm,
)

# Module-level graph node callables (kept so graph.py's imports stay untouched).
integration_auditor_node = integration_auditor_agent.agent
integration_auditor_router = integration_auditor_agent.router
integration_auditor_fallback_node = integration_auditor_agent.fallback
ask_integration_auditor_for_tool = integration_auditor_agent.ask


def route_integration_audit(state: MasterState):
    """Conditional router after the integration auditor completes its tasks.

    `chained` records go to the Validator for final PoC construction, with the
    proven `poc_payload`/`execution_logs` of their `chained_with` peers injected
    as `peer_payloads`. `unchainable` records are terminal (stay in the report).
    """
    all_vulns = as_dicts(state.get("vulnerabilities", []))
    chained = [v for v in all_vulns if v.get("status") == "chained"]

    if not chained:
        # No record needs a final validator pass: advance to reporter dispatch.
        return "reporter_dispatch"

    by_id = {v.get("vuln_id"): v for v in all_vulns}
    commands = []
    for record in chained:
        peer_ids = record.get("chained_with") or []
        peer_payloads = []
        for vid in peer_ids:
            peer = by_id.get(vid)
            if not peer:
                logging.warning(
                    f"Chained record {record.get('vuln_id')} references "
                    f"chained_with vuln '{vid}' not found in state; skipping peer payload."
                )
                continue
            peer_payloads.append({
                "vuln_id": peer.get("vuln_id"),
                "cwe_id": peer.get("cwe_id"),
                "description": peer.get("description"),
                "poc_payload": peer.get("poc_payload"),
                "execution_logs": peer.get("execution_logs"),
            })
        payload = ValidatorState(
            report_to_test=record,
            sandbox_url=state.get("sandbox_url"),
            peer_payloads=peer_payloads,
            messages=[],
            iterations=0,
            vulnerabilities=[],
            cookies={},
            agent_id=uuid.uuid4().hex,
        )
        commands.append(Send("validator_agent", payload))

    logging.info(
        f"Integration auditor chained {len(commands)} vulnerability(ies); "
        f"dispatching to the validator for PoC construction."
    )
    _record_stat("validator_chained_records", len(commands))
    return commands
