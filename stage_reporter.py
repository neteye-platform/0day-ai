"""Reporter stage: single-shot per-vulnerability findings + final timestamped
report directory (report.pdf + bundled poc/ scripts) assembly."""

import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send
from markdown_it import MarkdownIt
from weasyprint import HTML

import settings
from llms import smart_llm
from run_stats import _record_stat, _snapshot_pipeline_stats, as_dict, as_dicts, raise_if_stopping, strip_step_numbering
from schemas import REPORTER_AGENT, ReporterFinding, cwes
from state import MasterState, ReporterState
from utils import cache_reporter, cvss_severity_label, cvss_v3_base_score


# Reportable = proven exploitable, or confirmed with a static-only proof.
# false_positive / unchainable / review_error / unproven are excluded — except
# a false positive reached THROUGH the patch lifecycle: the auto-verified fix is
# itself the deliverable, and its section documents the applied patch.
def _is_reportable(record: dict) -> bool:
    if record.get("status") == "exploitable":
        return True
    if (
        record.get("status") == "false_positive"
        and record.get("patch_diff")
        and record.get("patch_state") in ("reviewed", "verified")
    ):
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
        f"| Duplicate records merged by the LLM dedup agent | {stats.get('hypotheses_merged_by_agent', 0)} |",
        f"| Hypotheses dispatched to the Reviewer | {stats.get('reviewer_hypotheses', 0)} |",
        f"| Reviewer re-reviews (validator feedback) | {stats.get('reviewer_feedback_reviews', 0)} |",
        f"| Records dispatched to the Validator (direct) | {stats.get('validator_records', 0)} |",
        f"| Chained records re-dispatched to the Validator | {stats.get('validator_chained_records', 0)} |",
        f"| Integration audits (requires_integration) | {stats.get('integration_audits', 0)} |",
        f"| Contract-verifier evaluations (total) | {verified_total} |",
        f"| Vulnerability findings reported | {len(state.get('reporter_findings', []))} |",
    ]

    if stats.get("patcher_records"):
        patched_verified = sum(
            1
            for v in as_dicts(state.get("vulnerabilities", []))
            if v.get("patch_state") == "verified"
        )
        patched_attempts = sum(
            (v.get("patch_round") or 0)
            for v in as_dicts(state.get("vulnerabilities", []))
        )
        lines += [
            f"| Records dispatched to the Patcher | {stats['patcher_records']} |",
            f"| Patch attempts banked (edits applied to source) | {patched_attempts} |",
            f"| Reviewer patch re-checks | {stats.get('reviewer_patch_rechecks', 0)} |",
            f"| Patches verified in the sandbox (exploit dead, app healthy) | {patched_verified} |",
        ]

    for label, key in (
        ("Demands skipped at output cap (verifier)", "verifier_demands_skipped_output_cap"),
        ("Explorer nodes skipped at output cap", "explorer_nodes_skipped_output_cap"),
        ("Explorer nodes skipped (prompt over explorer_max_prompt_chars)", "explorer_nodes_skipped_oversized"),
        ("CVE analyses skipped at output cap", "cve_analyses_skipped_output_cap"),
        ("Threat-intel enrichments skipped at output cap", "threat_intel_skipped_output_cap"),
        ("Edge-traversal batches skipped at output cap", "edge_traversal_batches_skipped_output_cap"),
        ("LLM dedup-agent group calls", "dedup_agent_groups"),
        ("LLM dedup-agent groups skipped at output cap", "dedup_agent_groups_skipped_output_cap"),
        ("LLM dedup-agent groups skipped on error (failed open)", "dedup_agent_groups_skipped_errors"),
        ("Node resolutions via fallback (global-exact / caller-scoped)", "demand_nodes_resolved_fallback"),
        ("Unresolved node targets (unique, logged once each)", "resolve_targets_unresolved_unique"),
        ("Demands dropped: unresolvable target (unique)", "demands_dropped_unresolved_unique"),
        ("Demands dropped: no qualifying caller (unique)", "demands_dropped_scoped_unique"),
    ):
        if stats.get(key):
            lines.append(f"| {label} | {stats[key]} |")

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


def _step_block(index: int, step: str) -> list[str]:
    """One numbered step as markdown lines. A step embedding a fenced code
    block is indented so the fence nests INSIDE the numbered list item
    (CommonMark needs the 3-space content indent) instead of terminating it."""
    text = strip_step_numbering(step)
    lines = text.splitlines()
    if not lines or "```" not in text:
        return [f"{index}. {text}"]
    out = [f"{index}. {lines[0]}"]
    out += ["   " + line if line else "" for line in lines[1:]]
    return out


def _patch_section_lines(record: dict, patch_file: str | None) -> list[str]:
    """Deterministic `#### Proposed fix` block for the patch lifecycle: the
    banked summary/diff plus a verification line derived ONLY from the record's
    final status (never reinterpreted by any LLM)."""
    diff = record.get("patch_diff")
    if not diff:
        if record.get("patch_state") == "failed":
            return [
                "", "#### Proposed fix", "",
                "_An automatic fix was attempted and abandoned: no safe minimal "
                "first-party patch could be authored — remediate manually._",
            ]
        return []
    status = record.get("status")
    state = record.get("patch_state")
    rounds = record.get("patch_round") or 1
    if state == "verified":
        verdict = (
            "**Verified** — dynamic re-validation against the resynced sandbox: the "
            "exploit no longer reproduces and the legitimate flow still works."
        )
    elif state == "rejected":
        verdict = (
            f"**REJECTED and not fixed** — the exploit still reproduced after "
            f"{rounds} patch attempt(s) on the resynced sandbox: treat the diff as "
            "an unproven proposal and remediate manually."
        )
    elif status == "exploitable":
        verdict = (
            "**NOT verified** — the exploit still reproduced after the patch (or the "
            "sandbox could not adopt it): treat this fix as a proposal only."
        )
    else:
        verdict = (
            "**Statically adjudicated** — the re-review accepted the patch; no "
            "dynamic re-proof was executed."
        )
    lines = [
        "", "#### Proposed fix", "",
        str(record.get("patch_summary") or "_no summary_").rstrip(), "",
        verdict, "",
    ]
    if record.get("patched_files"):
        lines.append(f"**Files changed:** {', '.join(record['patched_files'])}  ")
    if patch_file:
        lines.append(f"**Patch file:** `patches/{patch_file}`  ")
    files = ", ".join(record.get("patched_files") or []) or "n/a"
    lines += ["", f"```diff\n# {files}\n{str(diff).rstrip()}\n```"]
    return lines


def _render_report_markdown(
    records: list[dict],
    findings_by_id: dict[str, dict],
    statistics: str | None = None,
    poc_files: dict[str, str] | None = None,
    patch_files: dict[str, str] | None = None,
) -> str:
    """Assemble the report markdown (summary + rewritten steps) that
    _write_report renders to PDF. poc_files maps vuln_id -> the PoC script
    filename _copy_poc_scripts delivered into <report_dir>/poc/. Raw
    description/reviewer_reasoning/poc_payload/execution_logs are deliberately
    NOT written out — the reporter distilled them."""
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
    patched_verified = sum(
        1 for r in records
        if r.get("status") == "false_positive"
        and r.get("patch_diff") and r.get("patch_state") == "verified"
    )
    patched_static = sum(
        1 for r in records
        if r.get("status") == "false_positive"
        and r.get("patch_diff") and r.get("patch_state") != "verified"
    )
    static = len(records) - exploitable - patched_verified - patched_static

    lines = [
        f"# Vulnerability Report — {settings.app_path.name}",
        "",
        f"- **Target application:** `{settings.app_path}`",
        f"- **Generated:** {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- **Exploitable findings:** {exploitable}",
        *(
            [f"- **Auto-patched & verified (fix applied, exploit dead, feature intact):** {patched_verified}"]
            if patched_verified else []
        ),
        *(
            [f"- **Auto-patched, statically adjudicated (no dynamic re-proof):** {patched_static}"]
            if patched_static else []
        ),
        f"- **Static findings (no network-reachable path):** {static}",
        "",
    ]
    if statistics:
        lines += [statistics.rstrip(), ""]
    lines += [
        "## Findings at a Glance",
        "",
        "| # | Title | Vulnerability ID | CWE | CVSS v3.1 | Severity |",
        "|---|-------|------------------|-----|-----------|----------|",
    ]
    for i, (record, finding, vector, score, label) in enumerate(rows, 1):
        cwe = record.get("cwe_id", "OTHER_UNCATEGORIZED")
        score_str = f"{score:.1f}" if score is not None else "N/A"
        title = str(finding.get("title") or "").strip() or record.get("vuln_id", "Unknown")
        lines.append(
            f"| {i} | {title.replace('|', chr(92) + '|')} | `{record.get('vuln_id', 'Unknown')}` "
            f"| {cwe} | {score_str} | {label} |"
        )

    lines += ["", "## Findings", ""]
    for i, (record, finding, vector, score, label) in enumerate(rows, 1):
        cwe = record.get("cwe_id", "OTHER_UNCATEGORIZED")
        cwe_desc = cwes.get(cwe, "")
        score_str = f"{score:.1f}" if score is not None else "N/A"
        title = str(finding.get("title") or "").strip() or record.get("vuln_id", "Unknown")
        lines += [
            f"### {i}. {title}",
            "",
            f"**ID:** `{record.get('vuln_id', 'Unknown')}`  ",
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
        poc_file = (poc_files or {}).get(record.get("vuln_id"))
        if poc_file:
            lines.append(f"**PoC script:** `poc/{poc_file}`")

        summary = finding.get("summary") or record.get("description") or "_none_"
        steps = finding.get("reproduction_steps") or record.get("reproduction_steps") or []
        lines += ["", "#### Summary", "", str(summary).rstrip()]
        lines += ["", "#### Reproduction steps", ""]
        if steps:
            for j, step in enumerate(steps, 1):
                lines += _step_block(j, step)
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
        lines += _patch_section_lines(record, (patch_files or {}).get(record.get("vuln_id")))
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


_MD = MarkdownIt("commonmark").enable("table")

_PDF_CSS = """
@page { size: A4; margin: 18mm 16mm;
        @bottom-center { content: counter(page) " / " counter(pages);
                         font-size: 8pt; color: #888; } }
body { font-family: "DejaVu Sans", sans-serif; font-size: 9.5pt;
       line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 17pt; margin-bottom: 2mm; }
h2 { font-size: 13pt; color: #0f3b5c; border-bottom: 1px solid #d0d7de;
     padding-bottom: 1mm; margin-top: 6mm; }
h3 { font-size: 11.5pt; margin-top: 5mm; }
h4 { font-size: 10pt; margin-top: 4mm; color: #333; }
p, li { orphans: 2; widows: 2; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.6pt;
       background: #f2f4f6; padding: 0 1px; }
pre { background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 3px;
      padding: 2mm; white-space: pre-wrap; word-wrap: break-word; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0; }
th, td { border: 1px solid #c9d1d9; padding: 1mm 2mm; text-align: left; }
th { background: #eef2f5; }
blockquote { color: #555; border-left: 3px solid #d0d7de; margin-left: 0;
             padding-left: 3mm; }
"""


def _markdown_to_pdf(markdown_text: str) -> bytes:
    html_body = _MD.render(markdown_text)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{_PDF_CSS}</style></head><body>{html_body}</body></html>"
    )
    return HTML(string=html).write_pdf()


def _write_report(path: Path, markdown_text: str) -> None:
    """Render the assembled markdown to PDF and write it; fail-open like every
    other reporter I/O path (a render error logs, never crashes the pipeline)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        pdf_bytes = _markdown_to_pdf(markdown_text)
        path.write_bytes(pdf_bytes)
        logging.info(f"Reporter: wrote {path} ({len(pdf_bytes)} bytes).")
    except Exception as exc:
        logging.error(f"Reporter: failed to write {path}: {exc}")


def dispatch_reporters(state: MasterState):
    """Fan out ONE reporter task per reportable record. Routes straight to the
    assembler when nothing is reportable (the statistics-only report dir is
    still written)."""
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
    raise_if_stopping()
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
                "title": "",
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


def _copy_poc_scripts(records: list[dict], report_dir: Path) -> dict[str, str]:
    """Bundle every reportable record's PoC script into <report_dir>/poc/ so the
    human reader can run it next to the PDF.

    Prefers the script bytes the validator staged under
    .cache/poc_scripts/<vuln_id>/<rel>; a record that declares poc_script whose
    staged file is missing gets its script rebuilt from the record's own
    poc_payload text. Fails open per record. Returns vuln_id -> delivered
    filename (rendered into the report as poc/<filename>)."""
    files: dict[str, str] = {}
    poc_dir = report_dir / "poc"
    for record in records:
        rel = str(record.get("poc_script") or "").strip().lstrip("/")
        vuln_id = str(record.get("vuln_id") or "unknown")
        if not rel or ".." in Path(rel).parts:
            continue
        data: bytes | None = None
        try:
            staged = settings.cache_dir / "poc_scripts" / vuln_id / rel
            if staged.is_file():
                data = staged.read_bytes()
        except OSError:
            data = None
        if not data and record.get("poc_payload"):
            data = (str(record["poc_payload"]).rstrip() + "\n").encode()
        if not data:
            logging.info(f"Reporter: no PoC script to bundle for {vuln_id}.")
            continue
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", vuln_id).strip("_")
        suffix = Path(rel).suffix or ".txt"
        name, counter = f"{slug}{suffix}", 2
        while name in files.values():
            name = f"{slug}_{counter}{suffix}"
            counter += 1
        try:
            poc_dir.mkdir(parents=True, exist_ok=True)
            (poc_dir / name).write_bytes(data)
        except OSError as exc:
            logging.warning(f"Reporter: could not bundle PoC script for {vuln_id}: {exc}")
            continue
        files[vuln_id] = name
    return files


def _copy_patch_files(records: list[dict], report_dir: Path) -> dict[str, str]:
    """Bundle every patched record's diff into <report_dir>/patches/<vuln_id>.patch
    so the reader can `git apply`/`patch -p1` it. Mirrors _copy_poc_scripts'
    fail-open and slug/collision conventions. Returns vuln_id -> filename."""
    files: dict[str, str] = {}
    patches_dir = report_dir / "patches"
    for record in records:
        diff = record.get("patch_diff")
        if not diff:
            continue
        vuln_id = str(record.get("vuln_id") or "unknown")
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", vuln_id).strip("_")
        name, counter = f"{slug}.patch", 2
        while name in files.values():
            name = f"{slug}_{counter}.patch"
            counter += 1
        try:
            patches_dir.mkdir(parents=True, exist_ok=True)
            (patches_dir / name).write_text(str(diff).rstrip() + "\n", encoding="utf-8")
        except OSError as exc:
            logging.warning(f"Reporter: could not bundle patch for {vuln_id}: {exc}")
            continue
        files[vuln_id] = name
    return files


def report_assembler_node(state: MasterState) -> dict:
    """Terminal barrier node: writes a FRESH timestamped report directory
    (<target_app>/report_<YYYY-MM-DD_HHMMSS>/report.pdf + poc/ scripts),
    severity-ranked, with deterministic CVSS scores and the statistics
    section. Returns {}."""
    report_dir = settings.app_path / f"report_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
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
        _write_report(report_dir / "report.pdf", _render_empty_report(statistics))
        return {}

    poc_files = _copy_poc_scripts(records, report_dir)
    patch_files = _copy_patch_files(records, report_dir)
    markdown = _render_report_markdown(records, findings_by_id, statistics, poc_files, patch_files)
    _write_report(report_dir / "report.pdf", markdown)
    return {}
