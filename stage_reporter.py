"""Reporter stage: single-shot per-vulnerability findings + final timestamped
report directory assembly: a main report.pdf (header, statistics, token usage,
glance table) plus one self-contained PDF per finding under findings/ (each
with its own poc/ and patches/ bundles)."""

import logging
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send
from markdown_it import MarkdownIt
from weasyprint import HTML

import settings
from llms import invoke_tracked, smart_llm
from run_stats import _record_stat, _snapshot_pipeline_stats, as_dict, as_dicts, raise_if_stopping, snapshot_token_totals, strip_step_numbering
from schemas import REPORTER_AGENT, ReporterFinding, cwes
from state import MasterState, ReporterState
from utils import cache_reporter, cvss_gate_blocks, cvss_severity_label, cvss_v3_base_score


# Reportable = proven exploitable, or confirmed with a static-only proof.
# false_positive / unchainable / review_error / unproven are excluded — except
# a false positive reached THROUGH the patch lifecycle: the auto-verified fix is
# itself the deliverable, and its section documents the applied patch.
def _below_cvss_gate(record: dict) -> bool:
    """Confirmed finding the CVSS validation gate held back from the
    Validator/Auditor dispatch. Applies the SAME predicate the dispatchers
    applied (incl. the 'missing strategy means direct' resolution), so a record
    reported as unvalidated here is exactly one the gate skipped. Such a finding
    was never dynamically proven — the user asked below-threshold confirmed
    findings to reach the report untested rather than die unseen."""
    strategy = record.get("validation_strategy") or "direct_to_validator"
    return cvss_gate_blocks({**record, "validation_strategy": strategy}, settings.validator_min_cvss)


def _is_patched_closure(record: dict) -> bool:
    """The false_positive that _is_reportable admits because the auto-fix IS the
    deliverable: the section documents the original flaw; the status it conveys
    is PATCHED, never 'false positive'."""
    return (
        record.get("status") == "false_positive"
        and record.get("patch_diff")
        and record.get("patch_state") in ("reviewed", "verified")
    )


def _is_reportable(record: dict) -> bool:
    if record.get("status") == "exploitable":
        return True
    if _is_patched_closure(record):
        return True
    if record.get("status") == "confirmed" and record.get("validation_strategy") == "static_finding_only":
        return True
    return _below_cvss_gate(record)


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
    reviewer_vector = str(record.get("cvss_vector") or "").strip()
    if reviewer_vector:
        reviewer_score = cvss_v3_base_score(reviewer_vector)
        lines.append(
            f"Reviewer CVSS estimate (pre-validation): {reviewer_vector}"
            + (
                f" (score {reviewer_score:.1f}, {cvss_severity_label(reviewer_score)})"
                if reviewer_score is not None else ""
            )
        )
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
    if reviewer_vector:
        lines.append(
            "Your cvss_vector is the report's final assessment: start from the "
            "Reviewer's pre-validation estimate and correct any metric the proven "
            "reproduction contradicts (e.g. a required login ⇒ real PR, a payload "
            "that never fires ⇒ lower impact); when there is no Proof of Concept, "
            "the Reviewer's estimate stands."
        )
    if _below_cvss_gate(record):
        lines.append(
            "This finding was NEVER dynamically validated (its severity estimate "
            "fell below the pipeline's CVSS validation gate). Write the summary and "
            "steps from the Reviewer's evidence and NEVER phrase the finding as "
            "proven, triggered, or confirmed in a live sandbox."
        )
    if record.get("patch_state"):
        lines.append(
            "This finding went through the automatic-fix lifecycle: the report "
            "section describes the ORIGINAL vulnerability as it existed BEFORE the "
            "fix — the pipeline itself stamps the fixed state onto the finding, so "
            "never title or phrase it as a false positive or as 'mitigated'. Your "
            "cvss_vector MUST therefore score the UNPATCHED flaw: never zero out the "
            "C/I/A metrics because the fix works — a C:N/I:N/A:N vector (base score "
            "0.0) is NEVER a valid assessment here."
        )
    return "\n".join(lines)


def _assessment_for(finding: dict, record: dict) -> tuple[float | None, str, str]:
    """Deterministic CVSS score + label: the score is computed from the vector
    (never taken from the model) and the label re-derived, so the report can
    never disagree with the vector. A missing/malformed reporter vector falls
    back to the Reviewer's estimate on the record; the model label is only the
    last resort."""
    vector = str(finding.get("cvss_vector") or "").strip()
    score = cvss_v3_base_score(vector)
    if score is None:
        reviewer_vector = str(record.get("cvss_vector") or "").strip()
        reviewer_score = cvss_v3_base_score(reviewer_vector)
        if reviewer_score is not None:
            return reviewer_score, reviewer_vector, cvss_severity_label(reviewer_score)
        return None, vector or "N/A", str(finding.get("severity") or "Not assessed")
    # Patch-lifecycle guard: a reportable finding (exploitable, FP-with-patch,
    # gate-skipped, static) always carries a real flaw, so a 0.0 recomputed score
    # can only mean the model scored the PATCHED residual risk instead of the
    # flaw. Fall back to the record's (pre-patch) vector in that case.
    if score == 0.0 and record.get("patch_state"):
        reviewer_vector = str(record.get("cvss_vector") or "").strip()
        reviewer_score = cvss_v3_base_score(reviewer_vector)
        if reviewer_score is not None and reviewer_score > 0.0:
            return reviewer_score, reviewer_vector, cvss_severity_label(reviewer_score)
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
    gate_skipped_total = sum(
        1 for v in as_dicts(state.get("vulnerabilities", [])) if _below_cvss_gate(v)
    )
    if gate_skipped_total:
        lines.append(
            f"| Records below CVSS validation gate (reported unvalidated) | {gate_skipped_total} |"
        )

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


# Ledger agent keys -> report labels. Unknown keys render prettified as-is.
_TOKEN_AGENT_LABELS = {
    "explorer": "Expert explorer",
    "cve_analyzer": "CVE analyzer",
    "threat_intel": "Threat intel",
    "credential_finder": "Credential finder",
    "contract_verifier": "Contract verifier",
    "edge_traversal": "Edge traversal",
    "dedup_agent": "LLM dedup agent",
    "reviewer": "Reviewer",
    "validator": "Validator",
    "integration_auditor": "Integration auditor",
    "patcher": "Patcher",
    "reporter": "Reporter",
}


def _build_token_usage() -> str:
    """Deterministic Token Usage section: per-agent LLM input/output token
    totals for the run, from the run_stats token ledger. The ledger merges
    this process's live spend with the usage persisted in cache entries that
    HIT (a cached verdict carries the tokens its original run spent), so the
    section reflects the full work behind this report regardless of how much
    was recomputed today. Values are never derived from LLM output."""
    totals = snapshot_token_totals()
    lines = [
        "## Token Usage",
        "",
        "_LLM tokens attributed to each pipeline agent for this report, "
        "including token counts persisted with reused (cached) verdicts._",
        "",
        "| Agent | LLM calls | Input tokens | Output tokens | Total tokens |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    grand_calls = grand_in = grand_out = 0
    ranked = sorted(
        totals.items(),
        key=lambda kv: -(kv[1].get("input_tokens", 0) + kv[1].get("output_tokens", 0)),
    )
    for agent, usage in ranked:
        calls = usage.get("calls", 0)
        tin = usage.get("input_tokens", 0)
        tout = usage.get("output_tokens", 0)
        grand_calls += calls
        grand_in += tin
        grand_out += tout
        label = _TOKEN_AGENT_LABELS.get(agent, agent.replace("_", " ").capitalize())
        lines.append(
            f"| {label} | {calls:,} | {tin:,} | {tout:,} | {tin + tout:,} |"
        )
    lines.append(
        f"| **Total** | **{grand_calls:,}** | **{grand_in:,}** | "
        f"**{grand_out:,}** | **{grand_in + grand_out:,}** |"
    )
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


def _ranked_rows(records: list[dict], findings_by_id: dict[str, dict]) -> list[tuple]:
    """(record, finding, vector, score, label) per reportable record,
    severity-ranked: highest CVSS first, then vuln_id for stability. The rank
    index also names every finding's report folder, so the main table's
    numbering and the findings/<NN>_ directory names never disagree."""
    rows = []
    for record in records:
        finding = findings_by_id.get(record.get("vuln_id")) or {}
        score, vector, label = _assessment_for(finding, record)
        rows.append((record, finding, vector, score, label))
    rows.sort(key=lambda r: (r[3] is None, -(r[3] or 0.0), r[0].get("vuln_id", "")))
    return rows


def _vuln_slug(vuln_id: str) -> str:
    """Filesystem-safe id slug, capped so paths never overflow a PDF page."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(vuln_id or "")).strip("_")[:60].strip("_")
    return slug or "finding"


def _finding_folder(index: int, vuln_id: str) -> str:
    return f"{index:02d}_{_vuln_slug(vuln_id)}"


# Inline embedding caps for the finding PDFs: a PoC script or patch diff is
# reproduced inside the PDF only when it stays small AND its longest line
# cannot overflow the typeset page; anything bigger stays a separate file.
_INLINE_MAX_LINES = 20
_INLINE_MAX_LINE_CHARS = 90


def _fits_inline(text: str) -> bool:
    lines = text.splitlines()
    return (
        0 < len(lines) <= _INLINE_MAX_LINES
        and all(len(line) <= _INLINE_MAX_LINE_CHARS for line in lines)
        and "```" not in text
    )


def _patch_section_lines(record: dict, patch_file: str | None) -> list[str]:
    """Deterministic `#### Proposed fix` block for the patch lifecycle: the
    banked summary/diff plus a verification line derived ONLY from the record's
    final status (never reinterpreted by any LLM). The diff is reproduced
    inline only when it fits the PDF caps (_fits_inline); otherwise the bundled
    patches/ file is referenced (and embedded as a fail-open when bundling
    failed and no file exists)."""
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
    # The finding PDF's metadata block already lists `patches/<file>`; the
    # too-long fallback below still names it.
    diff_text = str(diff).rstrip()
    if _fits_inline(diff_text) or not patch_file:
        files = ", ".join(record.get("patched_files") or []) or "n/a"
        lines += ["", f"```diff\n# {files}\n{diff_text}\n```"]
    else:
        lines += ["", f"_The diff is too long to reproduce here; it is bundled as `patches/{patch_file}`._"]
    return lines


def _display_title(finding: dict, record: dict) -> str:
    """Section/glance heading: the reporter's title, falling back to the raw
    vuln_id with underscores spaced out — the id's underscore-only runs are
    otherwise unbreakable tokens that can stretch the glance table past the
    page width (weasyprint then clips columns and misfragments the table)."""
    title = str(finding.get("title") or "").strip()
    return title or str(record.get("vuln_id", "Unknown")).replace("_", " ")


_ZWSP = "\u200b"
# Even with punctuation break points, a punctuation-free identifier run longer
# than this could still outstretch its column; one ZWSP every N chars caps the
# minimum content width so the table can never exceed the page margins.
_ZWSP_RUN_CAP_RE = re.compile(rf"(?P<w>[^\s{_ZWSP}]{{24}})(?=[^\s{_ZWSP}])")


def _soft_break(text: str) -> str:
    """Insert zero-width-space wrap opportunities after the characters that
    glue long path segments and identifiers into one unbreakable token. The
    glance table renders such tokens (slug paths in code spans, `Class::method`
    titles) and overflow-wrap is NOT an option (see the _PDF_CSS bug note:
    weasyprint drops trailing rows of a paginated table when it is set on
    cells); ZWSP gives Pango legal break points through the markup alone."""
    for ch in ("_", "/", ".", ":"):
        text = text.replace(ch, ch + _ZWSP)
    return _ZWSP_RUN_CAP_RE.sub(rf"\g<w>{_ZWSP}", text)


def _render_main_report(rows: list[tuple], statistics: str | None = None) -> str:
    """Assemble the MAIN report markdown: header counts, statistics, token
    usage and the glance table — NO per-finding detail sections. Every finding
    is documented in its own PDF under findings/ (each self-contained: its own
    poc/ and patches/ bundles), referenced from the table's Report column.
    Rows come from _ranked_rows so numbering matches the finding folders. Raw
    description/reviewer_reasoning/poc_payload/execution_logs are deliberately
    NOT written out — the reporter distilled them into the finding PDFs."""
    records = [r[0] for r in rows]
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
    gate_skipped = sum(1 for r in records if _below_cvss_gate(r))
    static = len(records) - exploitable - patched_verified - patched_static - gate_skipped

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
        *(
            [f"- **Confirmed, not dynamically validated (below CVSS validation gate):** {gate_skipped}"]
            if gate_skipped else []
        ),
        "",
    ]
    if statistics:
        lines += [statistics.rstrip(), ""]
    lines += [
        "## Findings at a Glance",
        "",
        "_Every finding is documented in detail in its own report — see the Report column._",
        "",
        "| # | Title | CVSS v3.1 | Severity | Report |",
        "|---|-------|-----------|----------|--------|",
    ]
    for i, (record, finding, _vector, score, label) in enumerate(rows, 1):
        score_str = f"{score:.1f}" if score is not None else "N/A"
        title = _display_title(finding, record)
        if _is_patched_closure(record):
            title += " **(PATCHED)**"
        title = _soft_break(title.replace("|", chr(92) + "|"))
        # Display-only abbreviation: a full slug is a ~60-char code token and
        # the real folder keeps the full name. The ZWSP wrap points from
        # _soft_break then guarantee the remaining path segments can break,
        # so the table can never exceed the page margins.
        folder = _finding_folder(i, record.get("vuln_id", ""))
        shown = folder if len(folder) <= 28 else folder[:24] + "…"
        lines.append(
            f"| {i} | {title} | {score_str} | {label} "
            f"| `{_soft_break(f'findings/{shown}/report.pdf')}` |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_finding_report(
    record: dict,
    finding: dict,
    vector: str,
    score: float | None,
    label: str,
    artifacts: dict,
) -> str:
    """Assemble ONE finding's standalone PDF markdown: metadata block, Summary
    / Reproduction steps / Worst-case impact / Remediation, the bundled (and
    inline when small enough) PoC script, and the Proposed fix section.
    Artifacts come from _bundle_finding_artifacts."""
    cwe = record.get("cwe_id", "OTHER_UNCATEGORIZED")
    cwe_desc = cwes.get(cwe, "")
    score_str = f"{score:.1f}" if score is not None else "N/A"
    lines = [
        f"# {_display_title(finding, record)}",
        "",
        f"_Vulnerability report for `{settings.app_path.name}` — scan overview: `../../report.pdf`._",
        "",
        f"- **Target application:** `{settings.app_path}`",
        f"- **Generated:** {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- **ID:** `{record.get('vuln_id', 'Unknown')}`",
        f"- **Severity:** {label}",
        f"- **CVSS v3.1 base score:** {score_str}",
        f"- **CVSS vector:** `{vector}`",
        f"- **CWE:** {cwe}{f' — {cwe_desc}' if cwe_desc else ''}",
        f"- **Type:** {record.get('vulnerability_type', 'Code Defect')}",
    ]
    affected = [n for n in (record.get("affected_nodes") or []) if n]
    if affected:
        lines.append(f"- **Affected nodes:** {', '.join(affected)}")
    if record.get("source_cve"):
        lines.append(f"- **Source CVE:** {record['source_cve']}")
    if record.get("confidence_score") is not None:
        lines.append(f"- **Audit confidence:** {record['confidence_score']}/10")
    poc_file = artifacts.get("poc_file")
    if poc_file:
        lines.append(f"- **PoC script:** `poc/{poc_file}`")
    if artifacts.get("patch_file"):
        lines.append(f"- **Patch file:** `patches/{artifacts['patch_file']}`")
    if _below_cvss_gate(record):
        lines += [
            "",
            "**Validation:** not performed — the Reviewer's CVSS estimate fell below "
            "the pipeline's validation gate, so this finding carries NO dynamic proof.  ",
        ]
    if _is_patched_closure(record):
        verified = record.get("patch_state") == "verified"
        lines += [
            "",
            f"**Status: PATCHED** — an automatic fix was applied; this section "
            f"describes the ORIGINAL vulnerability as proven exploitable BEFORE the "
            f"fix. "
            + (
                "Dynamic re-validation on the patched build confirmed the exploit no "
                "longer reproduces and the legitimate flow still works (see "
                "_Proposed fix_ below)."
                if verified
                else "The fix was accepted on re-review; no dynamic re-proof was "
                "executed (see _Proposed fix_ below)."
            ),
        ]

    summary = finding.get("summary") or record.get("description") or "_none_"
    steps = finding.get("reproduction_steps") or record.get("reproduction_steps") or []
    lines += ["", "## Summary", "", str(summary).rstrip()]
    lines += ["", "## Reproduction steps", ""]
    if steps:
        for j, step in enumerate(steps, 1):
            lines += _step_block(j, step)
    else:
        lines.append("_No reproduction steps available._")
    lines += ["", "## Worst-case impact", ""]
    if finding.get("worst_case_scenario"):
        lines.append(str(finding["worst_case_scenario"]).rstrip())
    else:
        lines.append("_The reporter produced no worst-case assessment for this finding._")
    lines += ["", "## Remediation", ""]
    if finding.get("remediation"):
        lines.append(str(finding["remediation"]).rstrip())
    else:
        lines.append("_No remediation was provided._")
    if poc_file:
        lines += ["", "## PoC script", ""]
        if artifacts.get("poc_inline"):
            lines += [
                f"_Reproduced below and bundled as `poc/{poc_file}`._", "",
                "```", str(artifacts["poc_inline"]).rstrip(), "```",
            ]
        else:
            lines.append(
                f"_Bundled alongside this report as `poc/{poc_file}` (too long to reproduce here)._"
            )
    lines += _patch_section_lines(record, artifacts.get("patch_file"))
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
/* No overflow-wrap on table cells: weasyprint 70 silently drops the
   trailing rows of a paginated table when that property is set on them. */
th, td { border: 1px solid #c9d1d9; padding: 1mm 2mm; text-align: left; }
th { background: #eef2f5; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
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
        usage = None
        try:
            result, usage = invoke_tracked(reporter_llm, [sys_msg, human_msg], "reporter")
            finding = result if isinstance(result, dict) else result.model_dump()
        except Exception as exc:
            # Fail open: keep the record's own evidence, losing no finding.
            logging.error(f"Reporter: LLM call failed for {vuln_id} ({exc}); using record evidence.")
            finding = {
                "title": "",
                "summary": report.get("description") or "",
                "cvss_vector": str(report.get("cvss_vector") or ""),
                "severity": None,
                "reproduction_steps": list(report.get("reproduction_steps") or []),
                "worst_case_scenario": "",
                "remediation": "",
            }
        cache_reporter(report, finding, usage)

    finding["vuln_id"] = vuln_id
    return {"reporter_findings": [finding]}


def _bundle_finding_artifacts(record: dict, finding_dir: Path) -> dict:
    """Bundle ONE finding's PoC script and patch diff into ITS OWN folder
    (<finding_dir>/poc/, <finding_dir>/patches/) so the finding directory and
    its PDF are self-contained.

    Prefers the script bytes the validator staged under
    .cache/poc_scripts/<vuln_id>/<rel>; a record that declares poc_script whose
    staged file is missing gets its script rebuilt from the record's own
    poc_payload text. Fails open per artifact. Returns poc_file / poc_inline /
    patch_file: the delivered filenames plus the PoC text when it fits the
    in-report caps (_fits_inline) — rendered into the finding PDF as
    poc/<filename> and patches/<filename> references."""
    vuln_id = str(record.get("vuln_id") or "unknown")
    slug = _vuln_slug(vuln_id)
    artifacts: dict = {"poc_file": None, "poc_inline": None, "patch_file": None}

    rel = str(record.get("poc_script") or "").strip().lstrip("/")
    if rel and ".." not in Path(rel).parts:
        data: bytes | None = None
        try:
            staged = settings.cache_dir / "poc_scripts" / vuln_id / rel
            if staged.is_file():
                data = staged.read_bytes()
        except OSError:
            data = None
        if not data and record.get("poc_payload"):
            data = (str(record["poc_payload"]).rstrip() + "\n").encode()
        if data:
            try:
                poc_dir = finding_dir / "poc"
                poc_dir.mkdir(parents=True, exist_ok=True)
                name = f"{slug}{Path(rel).suffix or '.txt'}"
                (poc_dir / name).write_bytes(data)
                artifacts["poc_file"] = name
                try:
                    text = data.decode()
                except UnicodeDecodeError:
                    text = ""
                if _fits_inline(text):
                    artifacts["poc_inline"] = text.rstrip("\n")
            except OSError as exc:
                logging.warning(f"Reporter: could not bundle PoC script for {vuln_id}: {exc}")
        else:
            logging.info(f"Reporter: no PoC script to bundle for {vuln_id}.")

    diff = record.get("patch_diff")
    if diff:
        try:
            patches_dir = finding_dir / "patches"
            patches_dir.mkdir(parents=True, exist_ok=True)
            name = f"{slug}.patch"
            (patches_dir / name).write_text(str(diff).rstrip() + "\n", encoding="utf-8")
            artifacts["patch_file"] = name
        except OSError as exc:
            logging.warning(f"Reporter: could not bundle patch for {vuln_id}: {exc}")
    return artifacts


def report_assembler_node(state: MasterState) -> dict:
    """Terminal barrier node: writes THIS SCAN's report directory
    (<app_path>/report_<YYYY-MM-DD_HHMMSS>/): a MAIN report.pdf (header,
    statistics, token usage, glance table) plus one self-contained PDF per
    finding under findings/<NN>_<vuln-id>/ (with its own poc/ and patches/
    bundles). Severity-ranked, deterministic CVSS scores.

    One directory per scan: the name is minted on the first write and banked
    into MasterState.report_dir, and every later arrival rewrites ITS OWN
    directory from scratch. The reporter dispatch is fed by several terminal
    branches, so feedback-loop waves re-trigger this node; the LAST render is
    the one that survives, carrying the fullest record set. Returns
    {"report_dir": <name>}."""
    name = str(state.get("report_dir") or "").strip().strip("/")
    if not (name.startswith("report_") and Path(name).name == name):
        name = f"report_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    report_dir = settings.app_path / name
    if report_dir.exists():
        # Re-triggered arrival: wipe the previous render, because findings
        # folders are severity-ranked (NN_slugs shift between waves) and a
        # stale leftover would masquerade as a current finding.
        shutil.rmtree(report_dir, ignore_errors=True)
    statistics = _build_pipeline_statistics(state) + "\n\n" + _build_token_usage()

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
        return {"report_dir": name}

    rows = _ranked_rows(records, findings_by_id)
    for i, (record, finding, vector, score, label) in enumerate(rows, 1):
        finding_dir = report_dir / "findings" / _finding_folder(i, record.get("vuln_id", ""))
        artifacts = _bundle_finding_artifacts(record, finding_dir)
        _write_report(
            finding_dir / "report.pdf",
            _render_finding_report(record, finding, vector, score, label, artifacts),
        )
    _write_report(report_dir / "report.pdf", _render_main_report(rows, statistics))
    return {"report_dir": name}
