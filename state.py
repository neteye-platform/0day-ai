import operator
from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph.message import add_messages
from schemas import ExpertTask # Import your Pydantic models here


class MasterState(TypedDict):
    graph_path: str
    app_summary: str
    communities_map: Dict[str, List[str]] # Maps community ID to list of node IDs
    expert_tasks: List[ExpertTask]
    vulnerability_reports: Annotated[List[Dict[str, Any]], operator.add] # Aggregated findings
    filtered_reports: list[dict]
    messages: Annotated[list, add_messages]

class ExpertState(TypedDict):
    task: ExpertTask
    subgraph_nodes: List[str]
    messages: Annotated[list, add_messages] # Tracks the conversation and tool calls
    vulnerability_reports: Annotated[List[Dict[str, Any]], operator.add]

