"""Backward-compat facade for the former monolith.

The pipeline code now lives in llms.py, run_stats.py, tool_loop.py and the
stage_* modules; every name previously defined at this module's top level is
re-exported here so `from nodes import X` (and `import nodes`) keeps working
unchanged. Unused re-exports below are deliberate.

Mutable module-level bindings are authoritative in their defining module; the
alias here captures the initial object (in-place mutations are visible,
rebindings such as stage_verifier._DISPLAY_NODES are not).
"""

import hashlib
import json
import logging
import os
import re
import subprocess
import threading
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command, Send
from tavily import TavilyClient

import attacker_tools
import browser_tools
import credential_finder
import settings
import tools
from boundary_edges import (
    boundary_batch_fingerprint,
    build_boundary_edges,
    cluster_boundary_edges,
    render_batch_prompt,
    summarize_boundary_edges,
)
from dedup import Embeddings, cluster_vulnerabilities, deduplicate_demands
from languages import SYMBOL_QUERIES
from llms import fast_llm, reviewer_llm, smart_llm
from run_stats import (
    _agent_progress,
    _agent_progress_lock,
    _log_agent_completion,
    _pipeline_stats,
    _pipeline_stats_lock,
    _record_stat,
    _reset_pipeline_stats,
    _snapshot_pipeline_stats,
    _start_agent_progress,
)
from schemas import (
    CVE_ANALYZER_AGENT,
    EDGE_TRAVERSAL_AGENT,
    EXPERT_AGENTS,
    INTEGRATION_AUDITOR_AGENT,
    MANAGER_AGENT,
    REPORTER_AGENT,
    REVIEWER_AGENT,
    THREAT_INTEL_AGENT,
    VALIDATOR_AGENT,
    VERIFIER_AGENT,
    AnalysisNote,
    BatchedAnalysisResult,
    CVEAnalysis,
    EdgeTraversalOutput,
    ExpertTask,
    ReporterFinding,
    VerifierOutput,
    cwes,
)
from state import (
    CVEAnalyzerState,
    ExplorerState,
    IntegrationAuditorState,
    MasterState,
    ReporterState,
    ReviewerState,
    ThreatIntelState,
    ValidatorState,
    VerifierState,
)
from tool_loop import CompactionConfig, ToolLoopAgent
from utils import (
    build_images,
    build_networkx_graph,
    cache,
    cache_integration_auditor,
    cache_reporter,
    cache_reviewer,
    cache_validator,
    clear_aggregate_caches,
    cvss_severity_label,
    cvss_v3_base_score,
    deduplicate_cves,
    extract_container_artifacts,
    extract_imports,
    find_container_builds,
    find_unsupported_code_files,
    format_node_context,
    get_cached_graph_data,
    get_node_code,
    index_file,
    is_feedback_review,
    is_high_severity,
    is_node_worth_scanning,
    is_path_excluded,
    read_file_text,
    resolve_node_id,
    reviewer_cache_key,
    run_osv_scanner,
    run_osv_scanner_image,
    safe_cache_filename,
    scan_codebase_for_keywords,
    start_sandbox,
    uses_namespace_in_ast,
)
from stage_aggregate import (
    _CLASS_PROP_DECL_RE,
    _CLASS_PROP_WINDOW_LINES,
    _MEMBER_PARAM_RE,
    _TARGET_ARGS_RE,
    _build_container_members,
    _build_cve_hypothesis,
    _caller_invokes_symbol,
    _class_property_names,
    _dedupe_enriched_cve_demands,
    _member_param_names,
    _note_demands,
    _process_cve_demands,
    _route_downstream,
    _route_explorer_notes,
    _route_upstream,
    _scope_bare_container_demand,
    aggregate_demands_node,
    build_caller_map,
    build_import_map,
    build_sub_nodes_index,
    filter_cve_demands_by_keywords,
)
from stage_cve import (
    _backfill_osv_cwe_ids,
    _cve_analyzer_node,
    _finalize_cve_analysis,
    _normalize_cwe_ids,
    cve_analyzer_node,
    dispatch_cve_analyzers,
)
from stage_edge_traversal import (
    CROSS_BOUNDARY_VULN_TYPES,
    _edge_traversal_finding_to_record,
    _run_edge_traversal_batch,
    edge_traversal_node,
)
from stage_explorer import (
    _explore_batch,
    _explore_single,
    _log_explorer_completion,
    _pack_node_batches,
    dispatch_explorers,
    expert_explorer_node,
)
from stage_integration_auditor import (
    INTEGRATION_AUDITOR_SUMMARY_LEDGER,
    IntegrationAuditorAgent,
    _one_line,
    dispatch_integration_audits,
    integration_auditor_agent,
    integration_auditor_fallback_node,
    integration_auditor_node,
    integration_auditor_router,
    ask_integration_auditor_for_tool,
    route_integration_audit,
)
from stage_manager import manager_agent_node
from stage_preprocess import bootstrap_node, preprocessor_node
from stage_reporter import (
    _assessment_for,
    _build_pipeline_statistics,
    _is_reportable,
    _render_empty_report,
    _render_report_markdown,
    _render_reporter_prompt,
    _write_report,
    dispatch_reporters,
    report_assembler_node,
    reporter_node,
)
from stage_reviewer import (
    CODE_LEVEL_REVIEWER_TOOLS,
    CROSS_BOUNDARY_REVIEWER_TOOLS,
    FRAMEWORK_DEPENDENCY_REVIEWER_TOOLS,
    REVIEWER_SUMMARY_LEDGER,
    ReviewerAgent,
    _primary_node,
    _reviewer_mode_for,
    ask_reviewer_for_tool,
    dispatch_reviewers,
    reviewer_agent,
    reviewer_agent_node,
    reviewer_fallback_node,
    reviewer_router,
)
from stage_threat_intel import (
    _analysis_needs_threat_intel,
    _format_threat_intel_results,
    _noop_threat_intel_send,
    _threat_intel_node,
    dispatch_threat_intel,
    threat_intel_gate_node,
    threat_intel_node,
)
from stage_validator import (
    VALIDATOR_SUMMARY_LEDGER,
    ValidatorAgent,
    ask_validator_for_tool,
    dispatch_validators,
    route_validator_feedback,
    validator_agent,
    validator_agent_node,
    validator_fallback_node,
    validator_router,
)
from stage_verifier import (
    _DISPLAY_NODES,
    _NODE_DISPLAY_MEMO,
    _contract_verifier_node,
    _node_display_name,
    contract_verifier_node,
    dispatch_verifiers,
    synchronization_node,
)
