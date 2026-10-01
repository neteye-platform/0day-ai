import operator
from pathlib import Path
from typing import TypedDict, Any, Annotated, Optional
from langgraph.graph.message import add_messages
from schemas import AnalysisNote, ExpertTask, VulnerabilityRecord
from utils import merge_vulnerabilities


class MasterState(TypedDict):
    # Minted by bootstrap (resume-safe), copied into every subagent dispatch:
    # stamped as LangSmith metadata on the main and the project-split subagent
    # traces so all traces of one pipeline run correlate.
    pipeline_run_id: str

    known_vulns: list[dict]
    expert_tasks: list[ExpertTask]

    # Set only by the preprocessor; validator output is constrained to
    # `vulnerabilities` by compile_validator's output_schema.
    sandbox_url: Optional[str]

    notes: Annotated[list[AnalysisNote], operator.add]
    cve_demands: Annotated[list[dict], operator.add]
    grouped_demands: dict

    vulnerabilities: Annotated[list[VulnerabilityRecord], merge_vulnerabilities]

    # Per-vulnerability reporter outputs (one dict per reportable record,
    # assembled into the report dir by report_assembler_node). Append-reduced since
    # the reporter fan-out writes them concurrently.
    reporter_findings: Annotated[list[dict], operator.add]

class ExplorerState(TypedDict):
    node_ids: list[str]
    role: str
    task_description: str
    progress_id: str

class ReporterState(TypedDict):
    # The single reportable vulnerability record this reporter task summarizes.
    report: dict

class CVEAnalyzerState(TypedDict):
    cve: dict
    progress_id: str


class ThreatIntelState(TypedDict):
    cve: dict
    prior_analysis: Optional[dict]
    progress_id: str


class VerifierState(TypedDict):
    target_node_id: str
    target_code: str
    incoming_demands: list[dict] # List of assumptions about one node
    progress_id: str

class ReviewerState(TypedDict):
    # LangSmith correlation id inherited from MasterState at dispatch.
    pipeline_run_id: Optional[str]
    node_id: str
    expert_report: dict
    mode: str  # "code_level" | "framework_dependency" | "dependency_mitigation" | "systemic"
    # Ledger id from dispatch_reviewers; the base router advances it on every
    # terminal route so the run log tracks reviewer fan-out.
    progress_id: str
    # Number of LLM invocations in the tool loop. Bounds the loop so a model
    # that never submits a verdict ends gracefully via the fallback node.
    iterations: Annotated[int, operator.add]
    vulnerabilities: Annotated[list[VulnerabilityRecord], merge_vulnerabilities]
    messages: Annotated[list, add_messages]

class ValidatorState(TypedDict):
    # LangSmith correlation id inherited from MasterState at dispatch.
    pipeline_run_id: Optional[str]
    report_to_test: dict # The specific vulnerability to validate
    sandbox_url: Optional[str]     # The endpoint/IP of the sandbox
    cookies: dict
    # Proven results (vuln_id / cwe_id / description / poc_payload / execution_logs)
    # of the OTHER validated vulnerabilities a `chained` record depends on, sent
    # only by route_integration_audit. Rendered into the validator's first turn so
    # the final PoC can reuse the peers' proven payloads.
    peer_payloads: Optional[list[dict]]
    # Unique per-validator id (uuid) used to namespace browser session ids, so
    # concurrent validators sharing the single web browser never collide even
    # if their LLM picks identical session labels.
    agent_id: Optional[str]
    # Confirmed records that share (cwe, vulnerable_component) exactly with
    # report_to_test: one validator exercises every variant's reproduction
    # steps and its terminal verdict is written to the seed AND every variant
    # (each variant still flows through the vulnerabilities channel as its own
    # record — sharing only coalesces the validation runs, never the findings).
    validation_variants: Optional[list[dict]]
    # Ledger id from dispatch_validators; the base router advances it on every
    # terminal route so the run log tracks validator fan-out (turns included).
    progress_id: str
    # Number of LLM invocations in the tool loop (same role as ReviewerState.iterations).
    iterations: Annotated[int, operator.add]
    vulnerabilities: Annotated[list[VulnerabilityRecord], merge_vulnerabilities]
    messages: Annotated[list, add_messages]

class IntegrationAuditorState(TypedDict):
    # LangSmith correlation id inherited from MasterState at dispatch.
    pipeline_run_id: Optional[str]
    # A single `requires_integration` record to be combined into a multi-step exploit chain.
    report_to_test: dict
    # Full records of all OTHER confirmed vulnerabilities (excludes report_to_test
    # itself). Rendered as a summary for the agent and served to
    # get_vulnerability_details for deep dives into a specific peer.
    confirmed_vulns: list[dict]
    # Number of LLM invocations in the tool loop (same role as ReviewerState.iterations).
    iterations: Annotated[int, operator.add]
    vulnerabilities: Annotated[list[VulnerabilityRecord], merge_vulnerabilities]
    messages: Annotated[list, add_messages]
