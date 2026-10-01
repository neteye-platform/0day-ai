import operator
from typing import TypedDict, Any, Annotated
from langgraph.graph.message import add_messages
from schemas import ExpertTask, ValidationResult, VulnerabilityEvaluation


class MasterState(TypedDict):
    graph_path: str
    app_summary: str
    communities_map: dict[str, list[str]] # Maps community ID to list of node IDs
    expert_tasks: list[ExpertTask]
    vulnerability_reports: Annotated[list[dict[str, Any]], operator.add] # Aggregated findings
    filtered_reports: Annotated[list[VulnerabilityEvaluation], operator.add]
    confirmed_vulnerabilities: Annotated[list[ValidationResult], operator.add]
    manager_message: Any

class ExpertState(TypedDict):
    task: ExpertTask
    subgraph_nodes: list[str]
    unprocessed_nodes: list[str]
    messages: Annotated[list, add_messages] # Tracks the conversation and tool calls
    vulnerability_reports: Annotated[list[dict[str, Any]], operator.add]
    notes: Annotated[list[dict], operator.add]

class ReviewerState(TypedDict):
    report_id: str
    expert_report: list[dict]
    messages: Annotated[list, add_messages]
    filtered_reports: Annotated[list[VulnerabilityEvaluation], operator.add]
    notes: Annotated[list[str], operator.add]

class ValidatorState(TypedDict):
    report_to_test: dict # The specific vulnerability to validate
    sandbox_url: str     # The endpoint/IP of the sandbox
    messages: Annotated[list, add_messages]
    confirmed_vulnerabilities: Annotated[list[ValidationResult], operator.add]
    cookies: dict
    notes: Annotated[list[str], operator.add]

