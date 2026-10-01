import json
import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send
from tavily import TavilyClient

import settings
from llms import get_llm, invoke_structured_capped
from run_stats import _log_agent_completion, _record_stat, _start_agent_progress, raise_if_stopping, take_cached_usage
from schemas import THREAT_INTEL_AGENT, CVEAnalysis
from stage_cve import _backfill_osv_cwe_ids, _finalize_cve_analysis, cve_descriptions, osv_enrichment_lines
from state import MasterState, ThreatIntelState
from utils import cache, is_high_severity


def threat_intel_gate_node(state: MasterState) -> dict:
    """Barrier after all per-CVE analyzer tasks have completed."""
    return {}


def _analysis_needs_threat_intel(cve: dict, analysis: dict | None) -> bool:
    if is_high_severity(cve):
        return True
    if analysis is None:
        return True
    if "required_keywords" in analysis and not analysis["required_keywords"]:
        return True
    trigger_keys = {"trigger_condition", "attacker_request_primitive"}
    if trigger_keys & analysis.keys() and not any(analysis.get(key) for key in trigger_keys):
        return True
    return False


def _noop_threat_intel_send() -> list[Send]:
    """Stateless Send firing the aggregate_demands join barrier when there is
    nothing to enrich; threat_intel_node returns immediately for empty cve."""
    return [Send("threat_intel", ThreatIntelState(cve={}, prior_analysis=None, progress_id=""))]


def dispatch_threat_intel(state: MasterState):
    """Dispatch only CVEs whose mechanics need external threat intelligence.

    Always returns at least one threat_intel Send: the join barrier requires
    it to run exactly once, so the empty case emits a no-op task."""
    if not getattr(settings, "threat_intel_enabled", True):
        logging.info("Threat Intel disabled via settings.threat_intel_enabled=False.")
        return _noop_threat_intel_send()
    if not os.environ.get("TAVILY_API_KEY"):
        logging.warning("Threat Intel disabled: TAVILY_API_KEY is not configured.")
        return _noop_threat_intel_send()

    analyzed = {}
    for record in state.get("cve_demands", []):
        record = record if isinstance(record, dict) else record.model_dump()
        source_cve = record.get("source_cve")
        if source_cve:
            current = analyzed.get(source_cve)
            if current is None or record.get("enriched_by") == "threat_intel":
                analyzed[source_cve] = record

    candidates = []
    for cve in state.get("known_vulns", []):
        cve_id = cve.get("id", "UNKNOWN-CVE")
        prior = analyzed.get(cve_id)
        if _analysis_needs_threat_intel(cve, prior):
            candidates.append((cve, prior))

    if not candidates:
        logging.info("Threat Intel: no CVEs met the enrichment criteria.")
        return _noop_threat_intel_send()

    progress_id = _start_agent_progress(len(candidates))
    logging.info("Starting Threat Intel scan: 0/%d complete, %d remaining.", len(candidates), len(candidates))
    return [
        Send("threat_intel", ThreatIntelState(
            cve=cve,
            prior_analysis=prior,
            progress_id=progress_id,
        ))
        for cve, prior in candidates
    ]


def _format_threat_intel_results(search_data: dict) -> str:
    sections = []
    answer = search_data.get("answer")
    if answer:
        sections.append(f"Tavily answer:\n{str(answer)[:2500]}")
    for index, result in enumerate(search_data.get("results", [])[:5], 1):
        if not isinstance(result, dict):
            continue
        sections.append(
            f"Result {index}: {result.get('title', '')}\n"
            f"URL: {result.get('url', '')}\n"
            f"Content: {str(result.get('content', ''))[:1500]}"
        )
    return "\n\n".join(sections)[:9000]


def _threat_intel_node(state: ThreatIntelState) -> dict:
    cve = state.get("cve", {})
    prior = state.get("prior_analysis")
    cve_id = cve.get("id", "UNKNOWN-CVE")
    package_name = cve.get("package") or "unknown"
    # Cached by CVE id only; delete .cache/threat_intel/ to re-enrich.
    cache_file = settings.cache_dir / "threat_intel" / f"{cve_id}.json"
    cached = cache(cache_file, "read")
    if cached:
        take_cached_usage("threat_intel", cached)
        return {"cve_demands": [_backfill_osv_cwe_ids(cached, cve)]}

    query = f"{cve_id} {package_name} root cause writeup exploit analysis"
    try:
        search_data = TavilyClient().search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
        )
    except Exception as exc:
        # Fail open: keep the analyzer output on any Tavily error.
        logging.warning(f"{cve_id}: Tavily search failed; keeping analyzer output: {exc}")
        return {"cve_demands": [prior] if prior else []}

    descriptions = cve_descriptions(cve)
    enrichment = osv_enrichment_lines(cve)
    human_msg = HumanMessage(content=(
        f"Analyze and complete this CVE using the external threat intelligence.\n\n"
        f"CVE ID: {cve_id}\nPackage: {package_name}\n"
        f"OSV descriptions:\n{chr(10).join(descriptions)}\n"
        f"{' '.join(enrichment)}\n\n"
        f"Prior CVE analyzer output (may be null or incomplete):\n"
        f"{json.dumps(prior or {}, indent=2)}\n\n"
        f"--- WEB INTEL ---\n{_format_threat_intel_results(search_data)}"
    ))
    sys_msg = SystemMessage(content=THREAT_INTEL_AGENT.get("prompt", ""))
    structured_llm = get_llm("threat_intel").with_structured_output(CVEAnalysis, method="json_schema", strict=True)
    response, usage = invoke_structured_capped(
        structured_llm, [sys_msg, human_msg], f"Threat Intel {cve_id}", "threat_intel"
    )
    if response is None:
        # Fail open to the prior analyzer output.
        _record_stat("threat_intel_skipped_output_cap")
        logging.warning(f"{cve_id}: Threat Intel hit the output cap; keeping prior output.")
        return {"cve_demands": [prior] if prior else []}
    response = response if isinstance(response, dict) else response.model_dump()
    enriched = _finalize_cve_analysis(response, cve, enriched_by="threat_intel")
    if enriched is None:
        logging.warning(f"{cve_id}: Threat Intel returned an invalid analysis; keeping prior output.")
        return {"cve_demands": [prior] if prior else []}

    # token_usage rides ONLY the cache payload (see stage_cve note).
    cache(cache_file, "write", {**enriched, "token_usage": usage})
    return {"cve_demands": [enriched]}


def threat_intel_node(state: ThreatIntelState) -> dict:
    """Graph node wrapper: the no-op task fires the barrier with no external
    calls; otherwise enrich and advance the progress ledger."""
    raise_if_stopping()
    if not state.get("cve"):
        return {}
    result = _threat_intel_node(state)
    cve = state.get("cve", {})
    _log_agent_completion(
        state.get("progress_id", ""),
        "Threat Intel",
        f"cve={cve.get('id', 'UNKNOWN-CVE')}",
    )
    return result
