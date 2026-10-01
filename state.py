import operator
from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph.message import add_messages
from schemas import ExpertTask, ValidationResult


class MasterState(TypedDict):
    graph_path: str
    app_summary: str
    communities_map: Dict[str, List[str]] # Maps community ID to list of node IDs
    expert_tasks: List[ExpertTask]
    vulnerability_reports: Annotated[List[Dict[str, Any]], operator.add] # Aggregated findings
    filtered_reports: Annotated[list[dict], operator.add]
    confirmed_vulnerabilities: Annotated[List[ValidationResult], operator.add]
    manager_message: Any

class ExpertState(TypedDict):
    task: ExpertTask
    subgraph_nodes: List[str]
    messages: Annotated[list, add_messages] # Tracks the conversation and tool calls
    vulnerability_reports: Annotated[List[Dict[str, Any]], operator.add]

class ReviewerState(TypedDict):
    report_id: str
    messages: Annotated[list, add_messages]

class ValidatorState(TypedDict):
    report_to_test: dict # The specific vulnerability to validate
    sandbox_url: str     # The endpoint/IP of the sandbox
    messages: Annotated[list, add_messages]

