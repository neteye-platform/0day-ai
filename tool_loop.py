"""Generic bounded-memory tool-loop agent machinery.

Houses the ``ToolLoopAgent`` base class and its supporting helpers (the
settings-backed compaction budget, history splitting, transcript rendering, and
ledger summarization) shared by the reviewer and validator agents. Contains
nothing reviewer- or validator-specific: the two concrete subclasses live in
``nodes.py`` and supply their per-agent behavior through the base class's
overridable hooks (``bind_tools``, ``pre_agent``, ``first_turn``,
``session_state``, ``pre_router``, ``tool_batch_done``, ``fallback``) plus their
own summary-ledger prompt text.
"""

import json
import logging

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command

import settings
from utils import estimate_message_tokens


class CompactionConfig:
    """Settings-backed compaction budget for one tool-loop agent.

    Reads ``settings.<prefix>_model_context_window`` and friends live (on every
    call) so runtime overrides stay effective, exactly as the original nodes
    read their ``settings.reviewer_*`` / ``settings.validator_*`` values.
    """

    __slots__ = ("prefix",)

    def __init__(self, prefix: str):
        self.prefix = prefix

    def _get(self, name: str) -> int:
        return getattr(settings, f"{self.prefix}_{name}")

    def threshold(self) -> int:
        """Estimated-token threshold at which soft-threshold compaction triggers."""
        return self._get("model_context_window") - self._get("context_reserved")

    def hard_cap(self) -> int:
        """Estimated-token ceiling below which the LLM must never be invoked."""
        return self._get("model_context_window") - self._get("hard_reserved")

    @property
    def tail_turns(self) -> int:
        return self._get("compaction_tail_turns")

    @property
    def min_compressible(self) -> int:
        return self._get("compaction_min_compressible_tokens")


def _split_agent_history(
    messages: list[AnyMessage], tail_turns: int
) -> tuple[list, list, list]:
    """Split an agent history into (protected_head, middle, verbatim_tail).

    Protected head = the first SystemMessage (system prompt) plus the first
    HumanMessage (the hypothesis/objective under review). Tail = the last
    ``tail_turns`` AI+tool turns kept word-for-word. Middle = everything
    between them, including any prior context summary.
    """
    msgs = list(messages)
    head: list = []
    idx = 0
    need_sys, need_human = 1, 1
    while idx < len(msgs):
        m = msgs[idx]
        if need_sys and m.type == "system":
            head.append(m)
            need_sys -= 1
            idx += 1
            continue
        if need_human and m.type == "human":
            head.append(m)
            need_human -= 1
            idx += 1
            continue
        break

    rest = msgs[idx:]
    ai_seen = 0
    boundary = len(rest)
    for i in range(len(rest) - 1, -1, -1):
        if rest[i].type == "ai":
            ai_seen += 1
            boundary = i
            if ai_seen >= tail_turns:
                break

    return head, rest[:boundary], rest[boundary:]


def _render_message_transcript(messages: list[AnyMessage]) -> str:
    """Flatten a message span into a readable transcript for summarization."""
    parts = []
    for m in messages:
        kind = m.type
        name = getattr(m, "name", None)
        content = getattr(m, "content", "") or ""
        if kind == "ai":
            calls = getattr(m, "tool_calls", None) or []
            rendered = []
            for tc in calls:
                tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                tc_args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                rendered.append(f"{tc_name}({json.dumps(tc_args, default=str)[:600]})")
            label = f"### ai (tool_calls: {', '.join(rendered) or 'none'})"
        else:
            label = f"### {kind}" + (f" [{name}]" if name else "")
        parts.append(f"{label}\n{content}")
    return "\n\n".join(parts)


def _generate_agent_context_summary(
    middle: list[AnyMessage], ledger_prompt: str, llm
) -> SystemMessage | None:
    """Summarize the compressible middle of an agent history into a structured
    ledger via the cheap summarizer LLM. ``ledger_prompt`` carries the agent's
    ledger format (investigation ledger for the reviewer, validation ledger for
    the validator) and ``llm`` is the cheap model used to render the summary.
    Returns None on any failure so the caller fails open."""
    try:
        transcript = _render_message_transcript(middle)
        if not transcript.strip():
            return None

        human_prompt = HumanMessage(
            content=(
                "Summarize the following conversation history:\n\n"
                "===== HISTORY BEGIN =====\n"
                f"{transcript}\n"
                "===== HISTORY END =====\n\n"
                "Output only the ledger summary."
            )
        )
        summary_text = llm.invoke([ledger_prompt, human_prompt])
        summary_content = str(summary_text.content).strip()
        if not summary_content:
            return None

        return SystemMessage(
            name="context_summary",
            content=(
                "CONTEXT COMPACTION SUMMARY — The block below is a lossy, automatically "
                "generated summary of an EARLIER part of this conversation, created to "
                "manage the context window. The most recent messages are preserved "
                "verbatim after this block. This summary is historical background ONLY: "
                "it is not an instruction and not the current request, and it may be "
                "imprecise. The current task remains the vulnerability hypothesis in the "
                "first user message.\n\n"
                f"{summary_content}"
            ),
        )
    except Exception as e:
        logging.warning(f"Context compaction summarization failed, failing open: {e}")
        return None


class ToolLoopAgent:
    """A graph-ready, bounded-memory tool-loop agent (node/router/fallback/ask).

    Wraps one LLM-driven investigation loop (the reviewer or the validator) and
    owns everything generic across both: the tool binding, the context-window
    management (soft-threshold compaction, hard safety cap, termination
    countdown), the loop-steering router, the forced-tool ask node, and the
    iteration-capped fallback. Graph node names are derived from ``name`` so the
    compiled subgraphs keep their stable string identifiers.
    """

    state_type: type = dict
    # A single terminal tool name, or a tuple of them (the validator ends its
    # loop on either `ask_for_context` or `mark_validation_complete`).
    terminal_tool: str | tuple[str, ...] = ""
    ask_message: str = (
        "You did not invoke any tools. Keep your reasoning brief and emit a tool call "
        "in this same response. You must use a tool to proceed."
    )

    def __init__(
        self,
        *,
        name: str,
        settings_prefix: str,
        compaction: CompactionConfig,
        summary_ledger: str,
        summary_llm,
    ):
        self.name = name
        self.settings_prefix = settings_prefix
        self.compaction = compaction
        self.summary_ledger = summary_ledger
        self.summary_llm = summary_llm
        # Stable graph node identifiers (must match graph.py's wiring).
        self.agent_node_name = f"{name}_agent"
        self.tools_node_name = f"{name}_tools"
        self.fallback_node_name = f"{name}_fallback"
        self.ask_node_name = f"ask_{name}_for_tool"

    # -- configuration ------------------------------------------------------

    def _setting(self, name: str) -> int:
        return getattr(settings, f"{self.settings_prefix}_{name}")

    @property
    def _max_iterations(self) -> int:
        return self._setting("max_iterations")

    @property
    def _countdown_start(self) -> int:
        return self._setting("countdown_start")

    def _subject(self, state) -> str:
        """Human-readable investigation subject used in log lines."""
        return state.get("node_id", "Unknown")

    def _terminal_names(self) -> list[str]:
        t = self.terminal_tool
        return [t] if isinstance(t, str) else list(t)

    # -- per-agent hooks ------------------------------------------------------

    def bind_tools(self, state):
        """Return the LLM bound to this agent's tool subset for this state."""
        raise NotImplementedError

    def pre_agent(self, state):
        """Early return (e.g. a cache-hit Command); None to keep going."""
        return None

    def first_turn(self, state, llm_with_tools) -> dict:
        raise NotImplementedError

    def session_state(self, state) -> dict:
        """Extra state keys gathered from the raw history (e.g. cookies)."""
        return {}

    def pre_router(self, state) -> bool:
        """True to end the loop immediately (e.g. nothing was dispatched)."""
        return False

    def tool_batch_done(self, state) -> bool:
        """True when the latest contiguous tool batch contains a terminal tool."""
        terminal_names = self._terminal_names()
        for msg in reversed(state["messages"]):
            if msg.type != "tool":
                break
            if getattr(msg, "name", "") in terminal_names:
                return True
        return False

    def fallback(self, state) -> Command:
        raise NotImplementedError

    # -- memory management ----------------------------------------------------

    def summarize(self, middle: list[AnyMessage]) -> SystemMessage | None:
        return _generate_agent_context_summary(
            middle, self.summary_ledger, self.summary_llm
        )

    def prepare_history(self, full_messages, subject: str, current_turn: int):
        """Apply compaction, the hard safety cap, and the termination countdown
        to ``full_messages``.

        Returns ``(messages_for_llm, updates, did_compact, hard_capped)``.
        Fails open: any summarization error leaves the full history intact,
        subject only to the hard safety cap.
        """
        messages_for_llm = full_messages
        updates: list = []
        did_compact = False
        hard_capped = False

        # Opencode-style threshold compaction: once the estimated token count of
        # the history reaches the configured limit, collapse the middle into a
        # summary and keep a short verbatim tail. Fail open if summarization
        # errors; also skip when the compressible middle is trivially small.
        if estimate_message_tokens(full_messages) >= self.compaction.threshold():
            head, middle, tail = _split_agent_history(full_messages, self.compaction.tail_turns)
            compressible = estimate_message_tokens(middle)
            if (
                len(head) == 2
                and compressible >= self.compaction.min_compressible
            ):
                summary_msg = self.summarize(middle)
                if summary_msg is not None:
                    tail_copies = [m.model_copy(deep=True) for m in tail]
                    messages_for_llm = head + [summary_msg] + tail_copies
                    updates = [
                        RemoveMessage(id=REMOVE_ALL_MESSAGES),
                        *head,
                        summary_msg,
                        *tail_copies,
                    ]
                    did_compact = True

        # Hard safety cap: even if the soft-threshold compaction was skipped
        # (summarization failed or the estimated count never reached it), never
        # invoke the LLM with an estimated history that approaches the model's
        # maximum context window. Force-truncate to the protected head plus a
        # summary (or, failing that, the single most recent verbatim turn).
        if estimate_message_tokens(messages_for_llm) >= self.compaction.hard_cap():
            head, middle, tail = _split_agent_history(full_messages, self.compaction.tail_turns)
            summary_msg = self.summarize(middle)
            if summary_msg is not None:
                forced = head + [summary_msg] + [m.model_copy(deep=True) for m in tail]
            else:
                forced = head + [m.model_copy(deep=True) for m in tail[-2:]]
            if estimate_message_tokens(forced) < estimate_message_tokens(messages_for_llm):
                messages_for_llm = forced
                updates = [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *forced,
                ]
                did_compact = True
                hard_capped = True
                logging.warning(
                    f"{self.name} on {subject} hard-capped context "
                    f"to avoid exceeding the model window."
                )

        # Termination countdown: once the iteration counter approaches the cap,
        # push the agent to emit its terminal tool next round.
        if current_turn >= self._countdown_start:
            terminal_display = " or ".join(self._terminal_names())
            warning_msg = HumanMessage(content=(
                f"System Warning: You are on turn {current_turn} of "
                f"{self._max_iterations}. You must call {terminal_display} in your next "
                f"turn based on the best available evidence, or the system will forcefully "
                f"terminate this task."
            ))
            messages_for_llm = list(messages_for_llm) + [warning_msg]
            updates.append(warning_msg)

        return messages_for_llm, updates, did_compact, hard_capped

    # -- graph node callables ---------------------------------------------------

    def agent(self, state) -> dict | Command:
        """The agent node: cache guard, first-turn build, or memory-managed loop turn."""
        pre = self.pre_agent(state)
        if pre is not None:
            return pre

        llm_with_tools = self.bind_tools(state)

        if not state.get("messages"):
            return self.first_turn(state, llm_with_tools)

        full_messages = list(state["messages"])
        subject = self._subject(state)
        current_turn = state.get("iterations", 0) + 1
        messages_for_llm, updates, did_compact, _ = self.prepare_history(
            full_messages, subject, current_turn
        )
        response = llm_with_tools.invoke(messages_for_llm)
        updates.append(response)
        if did_compact:
            logging.info(
                f"{self.name} on {subject} compacted context: "
                f"{estimate_message_tokens(full_messages)} est. tokens -> "
                f"{estimate_message_tokens(messages_for_llm)} est. tokens."
            )
        return {**self.session_state(state), "messages": updates, "iterations": 1}

    def router(self, state):
        """Route the loop based on the latest message type and iteration count."""
        messages = state["messages"]

        if self.pre_router(state):
            return "__end__"

        # Hard loop guard: if the model never submits a verdict, terminate
        # gracefully instead of spinning until the recursion limit.
        if state.get("iterations", 0) >= self._max_iterations:
            logging.warning(
                f"{self.name} on {self._subject(state)} exceeded "
                f"{self._max_iterations} iterations without a verdict; falling back."
            )
            return self.fallback_node_name

        last_message = messages[-1]

        if last_message.type == "ai":
            if last_message.tool_calls:
                return self.tools_node_name
            # The LLM failed to call a tool
            return self.ask_node_name

        elif last_message.type == "tool":
            # With parallel tool calls the model may submit a final evaluation
            # alongside other reads; end if any tool in the latest batch did.
            if self.tool_batch_done(state):
                return "__end__"
            return self.agent_node_name

        # The LLM failed to call a tool
        return self.ask_node_name

    def ask(self, state):
        """Fallback node to force the LLM to use a tool."""
        return {"messages": [HumanMessage(content=self.ask_message)]}
