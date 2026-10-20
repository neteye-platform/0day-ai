"""Manager stage: deterministic heuristic task assignment per community."""

from collections import defaultdict
from typing import Any

import settings
from schemas import MANAGER_AGENT, ExpertTask
from state import MasterState
from utils import build_networkx_graph


def _make_task(community_id: int, agent: str) -> dict[str, Any]:
    return ExpertTask(
        target_community=f"Community {community_id}",
        agent_role=agent,
        task_description=MANAGER_AGENT["expert_descriptions"][agent],
    ).model_dump()


def manager_agent_node(state: MasterState) -> dict[str, Any]:
    """The Manager agent assigns tasks completely deterministically using heuristics."""
    G = build_networkx_graph(settings.graph, settings.communities_to_analyze)

    community_groups = defaultdict(list)
    for node_id, data in G.nodes(data=True):
        comm_id = data.get("community")
        if comm_id is not None:
            community_groups[comm_id].append((node_id, data))

    expert_keywords = MANAGER_AGENT["expert_keywords"]
    heuristic_tasks = []

    for comm_id, nodes in community_groups.items():
        scores = {agent: 0 for agent in expert_keywords}

        for node_id, node in nodes:
            direct_string = f"{node.get('label', '')} {node.get('source_file', '')} {node_id}".lower()
            # Neighbor matches weigh less to prevent inheritance skew.
            neighbor_string = ""
            for neighbor in G.successors(node_id):
                neighbor_data = G.nodes.get(neighbor, {})
                neighbor_string += f" {neighbor_data.get('label', '')}".lower()

            for agent, keywords in expert_keywords.items():
                for keyword in keywords:
                    if keyword in direct_string:
                        scores[agent] += 2
                    if keyword in neighbor_string:
                        scores[agent] += 1

        max_score = max(scores.values())
        ASSIGNMENT_THRESHOLD = max(2, int(max_score * 0.5))

        assigned = 0
        # Assign only the top-K best-scoring roles: distinct expert roles must
        # not re-scan the same nodes.
        for agent, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            if (
                assigned >= settings.max_experts_per_community
                or score < ASSIGNMENT_THRESHOLD
            ):
                break
            heuristic_tasks.append(_make_task(comm_id, agent))
            assigned += 1

        # Fallback: no role reached the assignment threshold.
        if not assigned:
            heuristic_tasks.append(_make_task(comm_id, "LogicFlowAuditor"))

    return {"expert_tasks": heuristic_tasks}
