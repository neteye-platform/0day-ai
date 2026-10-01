"""Reviewer stage: multi-track hypothesis adjudication (tool-loop subgraph)."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command, Send

import settings
import tools
from dedup import cluster_vulnerabilities
from llms import fast_llm, reviewer_llm
from run_stats import _record_stat, _start_agent_progress, as_dicts, get_embedder
from schemas import REVIEWER_AGENT
from stage_edge_traversal import CROSS_BOUNDARY_VULN_TYPES
from state import MasterState, ReviewerState
from tool_loop import CompactionConfig, ToolLoopAgent
from utils import (
    cache_reviewer,
    get_node_code,
    is_feedback_review,
    reviewer_cache_key,
)

# Each track binds only its own tool subset; the ToolNode in graph.py registers
# the union so all tracks run through the same compiled subgraph.
CODE_LEVEL_REVIEWER_TOOLS = [
    tools.read_source_code,
    tools.read_file,
    tools.get_node_connections,
    tools.search_codebase,
    tools.get_definition,
    tools.submit_evaluation,
]
FRAMEWORK_DEPENDENCY_REVIEWER_TOOLS = [
    tools.read_file,
    tools.search_codebase,
    tools.list_container_artifacts,
    tools.find_in_container,
    tools.read_container_artifact,
    tools.submit_evaluation,
]
# cross_boundary: the flaw lives in the composition of two endpoints AND the
# transport between them, so both source-reading and container-artifact tools.
CROSS_BOUNDARY_REVIEWER_TOOLS = [
    tools.read_source_code,
    tools.read_file,
    tools.get_node_connections,
    tools.search_codebase,
    tools.get_definition,
    tools.list_container_artifacts,
    tools.find_in_container,
    tools.read_container_artifact,
    tools.submit_evaluation,
]


def _reviewer_mode_for(hypothesis: dict) -> str:
    """Route a hypothesis to its reviewer track: 'framework_dependency',
    'dependency_mitigation', 'systemic', 'cross_boundary' or 'code_level'."""
    vuln_type = hypothesis.get("vulnerability_type")
    if vuln_type == "Known Dependency Vulnerability":
        return "framework_dependency"
    if vuln_type == "Dependency Mitigation Vulnerability":
        return "dependency_mitigation"
    if vuln_type == "Systemic Vulnerability":
        return "systemic"
    if vuln_type in CROSS_BOUNDARY_VULN_TYPES:
        return "cross_boundary"
    return "code_level"


def _primary_node(record: dict, default: str = "Unknown") -> str:
    """First affected node of a record — the primary anchor for reviewer
    state/cache keys. The full list rides inside `expert_report`."""
    affected = record.get("affected_nodes") or []
    return affected[0] if affected else default


def dispatch_reviewers(state: MasterState):
    """Groups reports and dispatches parallel reviewer threads using the Send API."""
    all_vulns = as_dicts(state.get("vulnerabilities", []))
    hypotheses = [v for v in all_vulns if v.get("status") == "hypothesis"]

    if not hypotheses:
        logging.warning(f"No vulnerabilities hypotheses to dispatch.")
        # Advance straight to the reporter-dispatch barrier: it writes an empty
        # report rather than silently ENDing (reviewer/validator phases skipped).
        return "reporter_dispatch"

    # Semantic dedup before fan-out so one reviewer adjudicates a pattern once.
    # Fails open: exact-key dedup always runs, embedding clustering only when
    # the embeddings server serves the configured model.
    embedder = get_embedder(
        settings.semantic_dedup_enabled,
        "Semantic dedup",
        "exact-key dedup only",
    )
    hypotheses = cluster_vulnerabilities(
        hypotheses,
        settings.semantic_dedup_threshold,
        embedder,
        cross_threshold=settings.dedup_cross_node_similarity,
        anchor_confirmed_threshold=settings.dedup_anchor_confirmed_similarity,
        anchor_min_jaccard=settings.dedup_anchor_min_jaccard,
        max_merged_cluster=settings.dedup_max_merged_cluster,
        disk_cache_dir=settings.cache_dir / "hypothesis_embeddings",
    )

    progress_id = _start_agent_progress(len(hypotheses))
    logging.info(
        "Starting reviewer pass: 0/%d complete, %d remaining.",
        len(hypotheses),
        len(hypotheses),
    )

    commands = []
    for hypothesis in hypotheses:
        payload = ReviewerState(
            node_id=_primary_node(hypothesis),
            expert_report=hypothesis,
            mode=_reviewer_mode_for(hypothesis),
            progress_id=progress_id,
            iterations=0,
            vulnerabilities=[],
            messages=[]
        )
        commands.append(Send("reviewer_agent", payload))

    logging.info(f"Dispatching {len(commands)} reviewers.")
    _record_stat("reviewer_hypotheses", len(commands))
    return commands


# Compaction ledger prompts: the generic machinery lives in tool_loop.py; these
# constants carry the exact per-agent prompt text.
REVIEWER_SUMMARY_LEDGER = (
    "You are an expert summarizer. The following is the message history of a "
    "security reviewer agent investigating whether a reported vulnerability "
    "hypothesis in a target application is a true positive or a false positive. "
    "Your summary will REPLACE these messages in the model context, so the "
    "reviewer must be able to continue the investigation from it WITHOUT "
    "re-reading the original tool outputs.\n\n"
    "Produce an information-dense summary as a security investigation ledger "
    "with exactly these sections:\n"
    "## Objective\n"
    "One or two sentences restating the exact hypothesis under review and the "
    "target component/node.\n"
    "## Checks & Artifacts Examined\n"
    "Bulleted, deduplicated list of every node, file, container artifact, "
    "search, and request already examined, with the single most important fact "
    "each one revealed. Do NOT include full code or full tool outputs — distill "
    "them into their conclusions.\n"
    "## Confirmed Facts\n"
    "Bulleted list of verified facts established so far, stated in final form.\n"
    "## Ruled-Out Dead Ends\n"
    "Bulleted list of hypotheses or investigation paths already disproven, with "
    "a one-line reason for each.\n"
    "## Active Leads & Next Steps\n"
    "Bulleted list of the most promising remaining checks not yet completed.\n\n"
    "RULES:\n"
    "- If the history contains a prior SUMMARY (a system message containing "
    "'CONTEXT COMPACTION SUMMARY'), treat it as the anchoring summary: extend "
    "and refine it with only the new facts gathered since, rather than "
    "regenerating from scratch.\n"
    "- Write in final form ('the code does X', 'the sink is reachable'), never "
    "'the model checked X'.\n"
    "- Keep it concise, under 1024 words. Preserve exact file paths, node ids, "
    "CVE ids, and tool argument names.\n"
)


class ReviewerAgent(ToolLoopAgent):
    """Reviewer track: mode-dependent toolsets, cache-hit Command, and the
    trailing-batch submit_evaluation end condition."""

    terminal_tool = "submit_evaluation"
    cache_hit_label = "Reviewer"
    progress_label = "Reviewer"

    def bind_tools(self, state):
        mode = state.get("mode", "code_level")
        if mode == "framework_dependency":
            reviewer_tools = FRAMEWORK_DEPENDENCY_REVIEWER_TOOLS
        elif mode == "cross_boundary":
            reviewer_tools = CROSS_BOUNDARY_REVIEWER_TOOLS
        else:
            # code_level, dependency_mitigation, systemic all trace first-party
            # code and share the source-reading toolset.
            reviewer_tools = CODE_LEVEL_REVIEWER_TOOLS
        return reviewer_llm.bind_tools(
            reviewer_tools,
            parallel_tool_calls=True
        )

    def cached_verdict(self, state):
        # Feedback re-reviews are never served from cache: the same round-N
        # flagged record is re-dispatched verbatim on checkpoint replay, so a
        # content-hash cache would collapse each genuine re-answer into the
        # earlier verdict without an LLM turn.
        report = state.get("expert_report", {})
        if is_feedback_review(report):
            return None
        return cache_reviewer(reviewer_cache_key(report, state.get("node_id", "Unknown")), report)

    def first_turn(self, state, llm_with_tools) -> dict:
        # System prompt = shared directives + the mode-specific reachability
        # standard (mode names match agents.yaml keys).
        mode = state.get("mode", "code_level")
        mode_prompt = REVIEWER_AGENT.get(mode, "")
        sys_prompt = REVIEWER_AGENT.get('prompt', '')
        if mode_prompt:
            sys_prompt = f"{sys_prompt}\n\n{mode_prompt}"
        sys_msg = SystemMessage(content=sys_prompt)

        report = state.get("expert_report", {})
        node_id = state.get("node_id")

        affected = [n for n in (report.get("affected_nodes") or []) if n]
        affected_str = ", ".join(affected) if affected else node_id
        formatted_vuln = (
            f"Target: {node_id}\n"
            f"Affected Nodes: {affected_str}\n\n"
            f"Potential Issue to Investigate:\n"
            f"- Type: {report.get('vulnerability_type', 'Code Defect')}\n"
            f"- CWE ID: {report.get('cwe_id', 'Unknown')}\n"
            f"- Component: {report.get('vulnerable_component', 'Unknown')}\n"
            f"- Description: {report.get('description', '')}\n"
        )
        if report.get("source_cve"):
            formatted_vuln += f"- Source CVE: {report.get('source_cve')}\n"

        feedback_qs = report.get("open_questions") or []
        if feedback_qs:
            qs_str = "\n".join(f"  {i}. {q}" for i, q in enumerate(feedback_qs, 1))
            formatted_vuln += (
                f"\n--- VALIDATOR FEEDBACK (INSUFFICIENT CONTEXT) ---\n"
                f"The downstream Validator could not confirm this vulnerability because it "
                f"lacked the information below. Resolve EACH question using your tools, then "
                f"re-emit fully self-sufficient reproduction steps (exact HTTP method, path, "
                f"parameters/headers/body, and any session state) with `submit_evaluation`. "
                f"This is feedback round {report.get('review_round', 0)}.\n"
                f"{qs_str}\n"
            )

        # Synthetic nodes (dependency:/infra:) don't exist in the app graph:
        # there is no source code to attach, so skip the lookup.
        if node_id and not node_id.startswith(("dependency:", "infra:")):
            target_node_source = get_node_code(node_id, reviewer_mode=True)
            if target_node_source:
                formatted_vuln += (
                    f"--- TARGET NODE SOURCE CODE ---\n"
                    f"```\n"
                    f"{target_node_source}\n"
                    f"```\n"
                )

        human_msg = HumanMessage(content=(
            f"Review the following potential issues found in the target node.\n\n"
            f"{formatted_vuln}"
        ))

        response = llm_with_tools.invoke([sys_msg, human_msg])
        return {"messages": [sys_msg, human_msg, response], "iterations": 1}

    def fallback(self, state) -> Command:
        """Resolve a review that hit the iteration cap without a verdict:
        review_error (never silently confirmed or discarded)."""
        report = dict(state.get("expert_report", {}))

        updated_vuln = dict(report)
        updated_vuln["status"] = "review_error"
        updated_vuln["reviewer_reasoning"] = (
            f"Review terminated after {state.get('iterations', 0)} tool-loop iterations "
            f"without a submit_evaluation verdict (loop budget exceeded)."
        )

        # Feedback re-reviews are never cached (see cached_verdict).
        if not is_feedback_review(report):
            cache_reviewer(reviewer_cache_key(report, state.get("node_id", "Unknown")), report, updated_vuln)

        return Command(
            update={
                "vulnerabilities": [updated_vuln],
            }
        )


reviewer_agent = ReviewerAgent(
    name="reviewer",
    settings_prefix="reviewer",
    compaction=CompactionConfig(),
    summary_ledger=REVIEWER_SUMMARY_LEDGER,
    summary_llm=fast_llm,
)

# Module-level graph node callables (kept so graph.py's imports stay untouched).
reviewer_agent_node = reviewer_agent.agent
reviewer_router = reviewer_agent.router
reviewer_fallback_node = reviewer_agent.fallback
ask_reviewer_for_tool = reviewer_agent.ask
