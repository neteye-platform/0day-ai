import operator
from pathlib import Path
from typing import TypedDict, Any, Annotated
from langgraph.graph.message import add_messages
from schemas import AnalysisNote, ExpertTask, ValidationResult, VulnerabilityEvaluation


class MasterState(TypedDict):
    graph: dict[str, Any]
    app_summary: str
    known_vulns: list[dict]
    expert_tasks: list[ExpertTask]
    notes: Annotated[list[AnalysisNote], operator.add]
    cve_demands: Annotated[list[dict], operator.add]
    grouped_demands: dict
    vulnerability_hypothesis: Annotated[list[dict], operator.add]
    confirmed_vulnerabilities: Annotated[list[ValidationResult], operator.add]

class ExplorerState(TypedDict):
    node_id: str
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
    expert_report: list[dict]
    messages: Annotated[list, add_messages]
    filtered_reports: Annotated[list[VulnerabilityEvaluation], operator.add]

class ValidatorState(TypedDict):
    report_to_test: dict # The specific vulnerability to validate
    sandbox_url: str     # The endpoint/IP of the sandbox
    messages: Annotated[list, add_messages]
    confirmed_vulnerabilities: Annotated[list[ValidationResult], operator.add]
    cookies: dict
    notes: Annotated[list[str], operator.add]

