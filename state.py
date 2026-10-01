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

class VerifierState(TypedDict):
    target_node_id: str
    target_code: str
    incoming_demands: list[dict] # List of assumptions about one node
    progress_id: str

class ReviewerState(TypedDict):
    node_id: str
    expert_report: dict
    mode: str  # "code_level" | "framework_dependency"
    vulnerabilities: Annotated[list[VulnerabilityRecord], merge_vulnerabilities]
    messages: Annotated[list, add_messages]

class ValidatorState(TypedDict):
    report_to_test: dict # The specific vulnerability to validate
    sandbox_url: Optional[str]     # The endpoint/IP of the sandbox
    container_name: Optional[str]  # The running sandbox container
    cookies: dict
    vulnerabilities: Annotated[list[VulnerabilityRecord], merge_vulnerabilities]
    messages: Annotated[list, add_messages]
