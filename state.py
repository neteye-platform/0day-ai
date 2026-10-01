import operator
from pathlib import Path
from typing import TypedDict, Any, Annotated, Optional
from langgraph.graph.message import add_messages
from schemas import AnalysisNote, ExpertTask, ValidationResult, VulnerabilityEvaluation, VulnerabilityRecord
from utils import merge_vulnerabilities


class MasterState(TypedDict):
    known_vulns: list[dict]
    expert_tasks: list[ExpertTask]

    # Container runtime data, populated by the preprocessor after it starts the
    # built image in the background. None when no sandbox could be started.
    sandbox_url: Optional[str]
    container_name: Optional[str]

    notes: Annotated[list[AnalysisNote], operator.add]
    cve_demands: Annotated[list[dict], operator.add]
    grouped_demands: dict

    vulnerabilities: Annotated[list[VulnerabilityRecord], merge_vulnerabilities]

class ExplorerState(TypedDict):
    node_ids: list[str]
    role: str
    task_description: str
    progress_id: str

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
    node_id: str
    expert_report: dict
    mode: str  # "code_level" | "framework_dependency" | "dependency_mitigation" | "systemic"
    # Number of LLM invocations in the tool loop. Bounds the loop so a model
    # that never submits a verdict ends gracefully via the fallback node.
    iterations: Annotated[int, operator.add]
    vulnerabilities: Annotated[list[VulnerabilityRecord], merge_vulnerabilities]
    messages: Annotated[list, add_messages]

class ValidatorState(TypedDict):
    report_to_test: dict # The specific vulnerability to validate
    sandbox_url: Optional[str]     # The endpoint/IP of the sandbox
    container_name: Optional[str]  # The running sandbox container
    cookies: dict
    # Unique per-validator id (uuid) used to namespace browser session ids, so
    # concurrent validators sharing the single web browser never collide even
    # if their LLM picks identical session labels.
    agent_id: Optional[str]
    # Number of LLM invocations in the tool loop (same role as ReviewerState.iterations).
    iterations: Annotated[int, operator.add]
    vulnerabilities: Annotated[list[VulnerabilityRecord], merge_vulnerabilities]
    messages: Annotated[list, add_messages]

class IntegrationAuditorState(TypedDict):
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
