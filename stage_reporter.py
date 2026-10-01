"""Reporter stage: single-shot per-vulnerability findings + final report.md assembly."""

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

import settings
from llms import smart_llm
from run_stats import _record_stat, _snapshot_pipeline_stats, as_dict, as_dicts, strip_step_numbering
from schemas import REPORTER_AGENT, ReporterFinding, cwes
from state import MasterState, ReporterState
from utils import cache_reporter, cvss_severity_label, cvss_v3_base_score


# Reportable = proven exploitable, or confirmed with a static-only proof.
# false_positive / unchainable / review_error / unproven are excluded.
def _is_reportable(record: dict) -> bool:
    if record.get("status") == "exploitable":
        return True
    return (
        record.get("status") == "confirmed"
        and record.get("validation_strategy") == "static_finding_only"
    )


def _render_reporter_prompt(record: dict) -> str:
    """Single-shot human prompt for ONE record. poc_payload/execution_logs are
    included as ground truth but quoting them is forbidden by the prompt."""
    cwe = record.get("cwe_id", "OTHER_UNCATEGORIZED")
    cwe_desc = cwes.get(cwe, "")
    lines = [
        f"Target application: {settings.app_path.name} ({settings.app_path})",
        "",
        f"Vulnerability ID: {record.get('vuln_id', 'Unknown')}",
        f"Type: {record.get('vulnerability_type', 'Code Defect')}",
        f"CWE: {f'{cwe} — {cwe_desc}' if cwe_desc else cwe}",
    ]
    affected = [n for n in (record.get("affected_nodes") or []) if n]
    if affected:
        lines.append(f"Affected nodes: {', '.join(affected)}")
    if record.get("source_cve"):
        lines.append(f"Source CVE: {record['source_cve']}")
    if record.get("confidence_score") is not None:
        lines.append(f"Confidence: {record['confidence_score']}/10")
    lines += ["", "Description:", str(record.get("description") or "_none_")]
    if record.get("reviewer_reasoning"):
        lines += ["", "Reviewer reasoning:", str(record["reviewer_reasoning"]).rstrip()]
    steps = record.get("reproduction_steps") or []
    if steps:
        lines += ["", "Original reproduction steps (reviewer):"]
        lines += [f"{i}. {strip_step_numbering(s)}" for i, s in enumerate(steps, 1)]
    if record.get("poc_payload"):
        lines += [
            "",
            "Validator PoC payload (ground truth — use it to correct the steps; "
            "do NOT quote it verbatim in your output):",
            "```",
            str(record["poc_payload"]).rstrip(),
            "```",
        ]
    if record.get("execution_logs"):
        lines += [
            "",
            "Validation evidence / execution logs (ground truth — use it to prove "
            "success; do NOT quote it verbatim in your output):",
            str(record["execution_logs"]).rstrip(),
        ]
    lines += [
        "",
        "Produce ONE ReporterFinding for this vulnerability: a short summary, "
        "rewritten self-sufficient reproduction steps based on the PoC payload and "
        "execution logs, a CVSS v3.1 base vector, a worst-case scenario, and a "
        "remediation.",
    ]
    return "\n".join(lines)


def _assessment_for(finding: dict, record: dict) -> tuple[float | None, str, str]:
    """Deterministic CVSS score + label: the score is computed from the vector
    (never taken from the model) and the label re-derived, so the report can
    never disagree with the vector. Model label only as fallback for a missing
    or malformed vector."""
    vector = str(finding.get("cvss_vector") or "").strip()
    score = cvss_v3_base_score(vector)
    if score is None:
        return None, vector or "N/A", str(finding.get("severity") or "Not assessed")
    return score, vector, cvss_severity_label(score)


def _build_pipeline_statistics(state: MasterState) -> str:
    """Deterministic Pipeline Statistics section: run-local counters
    (_pipeline_stats) mixed with counts derived from the final MasterState —
    never from LLM output."""
    stats = _snapshot_pipeline_stats()

    notes = state.get("notes", [])
    upstream = downstream = note_vulns = 0
    for note in notes:
        n = as_dict(note)
        upstream += len(n.get("upstream") or [])
        downstream += len(n.get("downstream") or [])
        note_vulns += len(n.get("vulns") or [])

    grouped = state.get("grouped_demands", {}) or {}
    routed_demands = sum(len(d) for d in grouped.values())

    status_counts: dict[str, int] = defaultdict(int)
    for v in state.get("vulnerabilities", []):
        rec = as_dict(v)
        status_counts[str(rec.get("status") or "?")] += 1

    confirmed_total = status_counts.get("confirmed", 0)
    static_accepted = sum(
        1
        for v in as_dicts(state.get("vulnerabilities", []))
        if v.get("validation_strategy") == "static_finding_only"
    )

    verified_total = stats.get("verifier_evaluations", 0)
    lines = [
        "## Pipeline Statistics",
        "",
        "_Deterministic counters recorded by the pipeline during this scan._",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| CVEs identified by SCA (OSV records) | {len(state.get('known_vulns', []))} |",
        f"| Explorer notes produced | {len(notes)} |",
        f"| Upstream demands declared | {upstream} |",
        f"| Downstream demands declared | {downstream} |",
        f"| Demands routed to the contract verifier | {routed_demands} |",
        f"| Local vulnerability hypotheses in explorer notes | {note_vulns} |",
        f"| Hypotheses dispatched to the Reviewer | {stats.get('reviewer_hypotheses', 0)} |",
        f"| Reviewer re-reviews (validator feedback) | {stats.get('reviewer_feedback_reviews', 0)} |",
        f"| Records dispatched to the Validator (direct) | {stats.get('validator_records', 0)} |",
        f"| Chained records re-dispatched to the Validator | {stats.get('validator_chained_records', 0)} |",
        f"| Integration audits (requires_integration) | {stats.get('integration_audits', 0)} |",
        f"| Contract-verifier evaluations (total) | {verified_total} |",
        f"| Vulnerability findings reported | {len(state.get('reporter_findings', []))} |",
    ]

    if verified_total:
        lines += [
            "",
            "**Contract-verifier verdicts**",
            "",
            "| Verdict | Count |",
            "| --- | ---: |",
            f"| MET | {stats.get('verifier_MET', 0)} |",
            f"| FAILED | {stats.get('verifier_FAILED', 0)} |",
            f"| DELEGATED | {stats.get('verifier_DELEGATED', 0)} |",
            f"| OUT_OF_SCOPE | {stats.get('verifier_OUT_OF_SCOPE', 0)} |",
        ]

    lines += [
        "",
        "**Final vulnerability record statuses**",
        "",
        "| Status | Count |",
        "| --- | ---: |",
        f"| Exploitable (validated in the sandbox) | {status_counts.get('exploitable', 0)} |",
        f"| Accepted static finding (no network-reachable path) | {static_accepted} |",
        f"| Confirmed, unvalidated | {max(0, confirmed_total - static_accepted)} |",
        f"| False positive | {status_counts.get('false_positive', 0)} |",
        f"| Unchainable | {status_counts.get('unchainable', 0)} |",
        f"| Review error / unresolved | {status_counts.get('review_error', 0)} |",
        f"| Insufficient context | {status_counts.get('insufficient_context', 0)} |",
        f"| Hypothesis (never reviewed) | {status_counts.get('hypothesis', 0)} |",
    ]
    return "\n".join(lines)


def _render_report_markdown(
    records: list[dict],
    findings_by_id: dict[str, dict],
    statistics: str | None = None,
) -> str:
    """Assemble report.md from the reporter findings (summary + rewritten
    steps). Raw description/reviewer_reasoning/poc_payload/execution_logs are
    deliberately NOT written out — the reporter distilled them."""
    rows = []
    for record in records:
        finding = findings_by_id.get(record.get("vuln_id")) or {}
        score, vector, label = _assessment_for(finding, record)
        rows.append((record, finding, vector, score, label))
    # Severity rank: highest CVSS score first, then vuln_id for stability.
    rows.sort(
        key=lambda r: (r[3] is None, -(r[3] or 0.0), r[0].get("vuln_id", "")),
    )

    exploitable = sum(1 for r in records if r.get("status") == "exploitable")
    static = len(records) - exploitable

    lines = [
        f"# Vulnerability Report — {settings.app_path.name}",
        "",
        f"- **Target application:** `{settings.app_path}`",
        f"- **Generated:** {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- **Exploitable findings:** {exploitable}",
        f"- **Static findings (no network-reachable path):** {static}",
        "",
    ]
    if statistics:
        lines += [statistics.rstrip(), ""]
    lines += [
        "## Findings at a Glance",
        "",
        "| # | Vulnerability ID | CWE | CVSS v3.1 | Severity |",
        "|---|------------------|-----|-----------|----------|",
    ]
    for i, (record, finding, vector, score, label) in enumerate(rows, 1):
        cwe = record.get("cwe_id", "OTHER_UNCATEGORIZED")
        score_str = f"{score:.1f}" if score is not None else "N/A"
        lines.append(
            f"| {i} | `{record.get('vuln_id', 'Unknown')}` | {cwe} | {score_str} | {label} |"
        )

    lines += ["", "## Findings", ""]
    for i, (record, finding, vector, score, label) in enumerate(rows, 1):
        cwe = record.get("cwe_id", "OTHER_UNCATEGORIZED")
        cwe_desc = cwes.get(cwe, "")
        score_str = f"{score:.1f}" if score is not None else "N/A"
        lines += [
            f"### {i}. {record.get('vuln_id', 'Unknown')}",
            "",
            f"**Severity:** {label}  \n",
            f"**CVSS v3.1 base score:** {score_str}  ",
            f"**CVSS vector:** `{vector}`  ",
            f"**CWE:** {cwe}{f' — {cwe_desc}' if cwe_desc else ''}",
            f"**Type:** {record.get('vulnerability_type', 'Code Defect')}",
        ]
        affected = [n for n in (record.get("affected_nodes") or []) if n]
        if affected:
            lines.append(f"**Affected nodes:** {', '.join(affected)}")
        if record.get("source_cve"):
            lines.append(f"**Source CVE:** {record['source_cve']}")
        if record.get("confidence_score") is not None:
            lines.append(f"**Audit confidence:** {record['confidence_score']}/10")

        summary = finding.get("summary") or record.get("description") or "_none_"
        steps = finding.get("reproduction_steps") or record.get("reproduction_steps") or []
        lines += ["", "#### Summary", "", str(summary).rstrip()]
        lines += ["", "#### Reproduction steps", ""]
        if steps:
            lines += [f"{j}. {strip_step_numbering(s)}" for j, s in enumerate(steps, 1)]
        else:
            lines.append("_No reproduction steps available._")
        lines += ["", "#### Worst-case impact", ""]
        if finding.get("worst_case_scenario"):
            lines.append(str(finding["worst_case_scenario"]).rstrip())
        else:
            lines.append(
                "_The reporter produced no worst-case assessment for this finding._"
            )
        lines += ["", "#### Remediation", ""]
        if finding.get("remediation"):
            lines.append(str(finding["remediation"]).rstrip())
        else:
            lines.append("_No remediation was provided._")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_empty_report(statistics: str | None = None) -> str:
    """Minimal report written when nothing was proven/accepted."""
    lines = [
        f"# Vulnerability Report — {settings.app_path.name}",
        "",
        f"- **Target application:** `{settings.app_path}`",
        f"- **Generated:** {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "No exploitable vulnerabilities were confirmed and no static findings were "
        "accepted during this scan.",
        "",
    ]
    if statistics:
        lines += [statistics.rstrip(), ""]
    return "\n".join(lines).rstrip() + "\n"


def _write_report(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        logging.info(f"Reporter: wrote {path} ({len(text)} chars).")
    except OSError as exc:
        logging.error(f"Reporter: failed to write {path}: {exc}")


def dispatch_reporters(state: MasterState):
    """Fan out ONE reporter task per reportable record. Routes straight to the
    assembler when nothing is reportable (report.md still gets statistics)."""
    records = [
        record
        for record in as_dicts(state.get("vulnerabilities", []))
        if _is_reportable(record)
    ]

    if not records:
        logging.warning("Reporter: no exploitable/static findings to report.")
        return "report_assembler"

    _record_stat("reporters_dispatched", len(records))
    logging.info(f"Dispatching {len(records)} per-vulnerability reporter task(s).")
    return [Send("reporter", ReporterState(report=record)) for record in records]


def reporter_node(state: ReporterState) -> dict:
    """Single-shot per-vulnerability Reporter: ONE structured LLM call per
    record. Fails open to the record's own evidence when the call errors."""
    report = state.get("report") or {}
    if not report:
        return {}
    vuln_id = report.get("vuln_id", "Unknown")

    finding = cache_reporter(report)
    if finding is not None:
        logging.info(f"Reporter cache hit for {vuln_id}.")
        finding = dict(finding)
    else:
        sys_msg = SystemMessage(content=REPORTER_AGENT.get("prompt", ""))
        human_msg = HumanMessage(content=_render_reporter_prompt(report))
        reporter_llm = smart_llm.with_structured_output(
            ReporterFinding, method="json_schema", strict=True
        )
        try:
            result = reporter_llm.invoke([sys_msg, human_msg])
            finding = result if isinstance(result, dict) else result.model_dump()
        except Exception as exc:
            # Fail open: keep the record's own evidence, losing no finding.
            logging.error(f"Reporter: LLM call failed for {vuln_id} ({exc}); using record evidence.")
            finding = {
                "summary": report.get("description") or "",
                "cvss_vector": "",
                "severity": None,
                "reproduction_steps": list(report.get("reproduction_steps") or []),
                "worst_case_scenario": "",
                "remediation": "",
            }
        cache_reporter(report, finding)

    finding["vuln_id"] = vuln_id
    return {"reporter_findings": [finding]}


def report_assembler_node(state: MasterState) -> dict:
    """Terminal barrier node: writes <target_app>/report.md once, severity-ranked,
    with deterministic CVSS scores and the statistics section. Returns {}."""
    report_path = settings.app_path / "report.md"
    statistics = _build_pipeline_statistics(state)

    findings_by_id = {}
    for finding in state.get("reporter_findings", []) or []:
        if isinstance(finding, dict) and finding.get("vuln_id"):
            findings_by_id[finding["vuln_id"]] = finding

    records = []
    for v in state.get("vulnerabilities", []):
        record = as_dict(v)
        if _is_reportable(record) and record.get("vuln_id") in findings_by_id:
            records.append(record)

    if not records:
        _write_report(report_path, _render_empty_report(statistics))
        return {}

    markdown = _render_report_markdown(records, findings_by_id, statistics)
    _write_report(report_path, markdown)
    return {}
