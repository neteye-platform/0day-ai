import operator
from pathlib import Path
from typing import TypedDict, Any, Annotated
from langgraph.graph.message import add_messages
from schemas import AnalysisNote, ExpertTask, ValidationResult, VulnerabilityEvaluation, VulnerabilityRecord
from utils import merge_vulnerabilities


class MasterState(TypedDict):
    known_vulns: list[dict]
    expert_tasks: list[ExpertTask]

    notes: Annotated[list[AnalysisNote], operator.add]
    cve_demands: Annotated[list[dict], operator.add]
    grouped_demands: dict

    vulnerabilities: Annotated[list[VulnerabilityRecord], merge_vulnerabilities]

class ExplorerState(TypedDict):
    node_ids: list[str]
    role: str
    task_description: str

class CVEAnalyzerState(TypedDict):
    cve: dict

class VerifierState(TypedDict):
    target_node_id: str
    target_code: str
    incoming_demands: list[dict] # List of assumptions about one node

class ReviewerState(TypedDict):
    node_id: str
    expert_report: dict
    vulnerabilities: Annotated[list[VulnerabilityRecord], merge_vulnerabilities]
    messages: Annotated[list, add_messages]

class ValidatorState(TypedDict):
    report_to_test: dict # The specific vulnerability to validate
    sandbox_url: str     # The endpoint/IP of the sandbox
    cookies: dict
    vulnerabilities: Annotated[list[VulnerabilityRecord], merge_vulnerabilities]
    messages: Annotated[list, add_messages]
