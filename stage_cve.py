import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

import settings
from llms import fast_llm, invoke_structured_capped
from run_stats import _log_agent_completion, _record_stat, _start_agent_progress, raise_if_stopping
from schemas import CVE_ANALYZER_AGENT, CVEAnalysis
from state import CVEAnalyzerState, MasterState
from utils import cache


def dispatch_cve_analyzers(state: MasterState):
    """Reads the deduplicated SCA results and dispatches tasks to the CVE Analyzer."""
    commands: list[Send] = []
    known_vulns = state.get("known_vulns", [])
    progress_id = _start_agent_progress(len(known_vulns))

    for cve_record in known_vulns:
        payload = CVEAnalyzerState(
            cve=cve_record,
            progress_id=progress_id,
        )
        commands.append(Send("cve_analyzer", payload))

    # No-op task keeps the aggregate_demands AND-join barrier satisfiable with zero SCA findings.
    if not commands:
        commands.append(Send("cve_analyzer", CVEAnalyzerState(cve={}, progress_id="")))

    logging.info(
        "Starting CVE analyzer scan: 0/%d complete, %d remaining.",
        len(commands),
        len(commands),
    )
    return commands


def _normalize_cwe_ids(value) -> list[str]:
    """Normalize an OSV `cwe_ids` value (list or bare string): strip, drop
    empties/non-strings, de-duplicate preserving order."""
    entries = value if isinstance(value, list) else ([value] if isinstance(value, str) else [])
    seen = set()
    normalized = []
    for entry in entries:
        cwe = entry.strip() if isinstance(entry, str) else ""
        if cwe and cwe not in seen:
            seen.add(cwe)
            normalized.append(cwe)
    return normalized


def _backfill_osv_cwe_ids(cached: dict, cve: dict) -> dict:
    """Merge current OSV CWEs into a cached analyzer output: the
    CVE/threat-intel caches are keyed by CVE id only and never auto-bust."""
    if not isinstance(cached, dict):
        return cached
    cached = dict(cached)
    osv_cwes = _normalize_cwe_ids(cve.get("cwe_ids"))
    cached["cwe_ids"] = osv_cwes or _normalize_cwe_ids(cached.get("cwe_ids"))
    return cached


def _finalize_cve_analysis(dict_analysis: dict, cve: dict, *, enriched_by: str | None = None) -> dict | None:
    """Apply deterministic CVE output guards and attach routing metadata."""
    cve_id = cve.get("id", "UNKNOWN-CVE")
    fix_category = dict_analysis.get("fix_category")
    if fix_category == "application_mitigation" and not dict_analysis.get("security_assumption"):
        logging.warning(f"{cve_id}: classified as application_mitigation but no security_assumption. Dropping.")
        return None
    if fix_category == "upgrade_only" and not dict_analysis.get("hypothesis"):
        logging.warning(f"{cve_id}: classified as upgrade_only but no hypothesis. Dropping.")
        return None
    if fix_category not in ("application_mitigation", "upgrade_only"):
        logging.warning(f"{cve_id}: invalid fix_category '{fix_category}'. Dropping.")
        return None

    dict_analysis["required_keywords"] = list(dict.fromkeys(
        kw.strip()
        for kw in (dict_analysis.get("required_keywords") or [])
        if kw and kw.strip()
    ))
    dict_analysis["source_cve"] = cve_id
    dict_analysis["package"] = cve.get("package") or "unknown"
    dict_analysis["fixed_version"] = cve.get("fixed_version")
    # Override model-emitted CWEs with the deterministic OSV ones (post-LLM for analyzer and Threat Intel).
    dict_analysis["cwe_ids"] = _normalize_cwe_ids(cve.get("cwe_ids"))
    if enriched_by:
        dict_analysis["enriched_by"] = enriched_by
    return dict_analysis


def cve_descriptions(cve: dict) -> list:
    """Up to 3 distinct descriptions from the preprocessor; `details` for legacy records."""
    descriptions = cve.get("descriptions")
    if not descriptions:
        details = cve.get("details")
        descriptions = [details] if details else []
    return descriptions


def osv_enrichment_lines(cve: dict) -> list[str]:
    """Deterministic OSV enrichment lines shared by the CVE analyzer and Threat Intel prompts."""
    lines = []
    if cve.get("fixed_version"):
        lines.append(f"Fixed version: {cve['fixed_version']}")
    if cve.get("cwe_ids"):
        lines.append(f"OSV CWE classifications: {', '.join(cve['cwe_ids'])}")
    if cve.get("severity_label"):
        lines.append(f"OSV severity: {cve['severity_label']}")
    if cve.get("cvss_vector"):
        lines.append(f"OSV CVSS vector: {cve['cvss_vector']}")
    return lines


def _cve_analyzer_node(state: CVEAnalyzerState) -> dict:
    """Classify one CVE: extract a security demand (application_mitigation) or
    emit a vulnerability hypothesis (upgrade_only)."""
    raise_if_stopping()
    cve = state.get("cve", {})
    package_name = cve.get("package") or "unknown"
    cve_id = cve.get("id", "UNKNOWN-CVE")
    if not cve:
        # No-op task: fires the cve_analyzer -> threat_intel_gate chain for the aggregate_demands barrier.
        return {}
    descriptions = cve_descriptions(cve)
    if not descriptions:
        # Without descriptions the LLM would just hallucinate
        logging.warning(f"{cve_id}: no descriptions provided")
        return {"cve_demands": []}

    # Cached by CVE id only; delete .cache/cve_analyzer/ to re-classify.
    cache_file = settings.cache_dir / "cve_analyzer" / f"{cve_id}.json"
    cached_demand = cache(cache_file, "read")
    if cached_demand:
        return {"cve_demands": [_backfill_osv_cwe_ids(cached_demand, cve)]}

    sys_msg = SystemMessage(content=CVE_ANALYZER_AGENT["prompt"] + "\n\n")

    osv_lines = osv_enrichment_lines(cve)
    enrichment = ""
    if osv_lines:
        # Prompt-byte quirk retained: a present fixed-version line starts with a newline.
        lead = "\n" if cve.get("fixed_version") else ""
        enrichment = f"\n--- OSV ENRICHMENT ---\n{lead}{''.join(f'{line}\n' for line in osv_lines)}"

    desc_block = "\n".join(
        f"Description {i + 1}: {d}\n"
        for i, d in enumerate(descriptions)
    )

    human_msg = HumanMessage(content=(
        f"Analyze this CVE affecting the package '{package_name}':\n\n"
        f"CVE ID: {cve_id}\n"
        f"{desc_block}"
        f"{enrichment}"
    ))

    cve_analyzer_llm = fast_llm.with_structured_output(CVEAnalysis, method="json_schema", strict=True)
    analysis = invoke_structured_capped(
        cve_analyzer_llm, [sys_msg, human_msg], f"CVE analyzer {cve_id}"
    )
    if analysis is None:
        # Fail open: no demands for this CVE; uncached so a re-run retries it.
        _record_stat("cve_analyses_skipped_output_cap")
        return {"cve_demands": []}

    dict_analysis = analysis if isinstance(analysis, dict) else analysis.model_dump()

    dict_analysis = _finalize_cve_analysis(dict_analysis, cve)
    if dict_analysis is None:
        return {"cve_demands": []}

    cache(cache_file, "write", dict_analysis)

    return {
        "cve_demands": [dict_analysis]
    }


def cve_analyzer_node(state: CVEAnalyzerState) -> dict:
    """Graph node wrapper: runs the analyzer and advances its progress ledger."""
    result = _cve_analyzer_node(state)
    cve = state.get("cve", {})
    _log_agent_completion(
        state.get("progress_id", ""),
        "CVE analyzer",
        f"cve={cve.get('id', 'UNKNOWN-CVE')}",
    )
    return result
