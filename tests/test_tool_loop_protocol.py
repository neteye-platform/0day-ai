from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tool_loop import ToolLoopAgent, _split_agent_history


def _ai(calls, content="x"):
    return AIMessage(
        content=content,
        tool_calls=[{"name": "t", "args": {}, "id": i} for i in calls],
    )


def _tool(i, content="ok", name="t"):
    return ToolMessage(content=content, tool_call_id=i, name=name)


def _assert_protocol(msgs):
    pending: set[str] = set()
    for m in msgs:
        if m.type == "tool":
            assert m.tool_call_id in pending, "tool result without its AI call"
            pending.discard(m.tool_call_id)
        else:
            assert not pending, "AI tool_call missing results before next message"
            if m.type == "ai":
                pending = {tc["id"] for tc in m.tool_calls}
    assert not pending


def _history(big_tool_content):
    return [
        SystemMessage(content="s"),
        HumanMessage(content="h"),
        _ai(["a"]),
        _tool("a"),
        _ai(["b", "c"]),
        _tool("b", big_tool_content),
        _tool("c"),
        _ai(["d"]),
        _tool("d"),
    ]


def test_oversized_tool_result_demotes_whole_unit_in_order():
    msgs = _history("Z" * 500)
    head, middle, tail = _split_agent_history(msgs, tail_turns=3, max_chars=100)
    assert head + middle + tail == msgs
    _assert_protocol(head + middle + tail)
    _assert_protocol(tail)
    assert [m.type for m in tail] == ["ai", "tool"]


def test_no_oversized_keeps_tail():
    msgs = _history("ok")
    head, middle, tail = _split_agent_history(msgs, tail_turns=2, max_chars=100)
    assert head + middle + tail == msgs
    _assert_protocol(tail)
    assert len(tail) == 5


def _agent(max_iter):
    a = object.__new__(ToolLoopAgent)
    a.name = "n"
    a.settings_prefix = "n"
    a.terminal_tool = "done"
    a.progress_label = ""
    a.fallback_node_name = "n_fallback"
    a.tools_node_name = "n_tools"
    a.agent_node_name = "n_agent"
    a.ask_node_name = "ask"
    a._setting = lambda name: max_iter  # noqa: ARG005
    return a


def _call(name):
    return AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": "1"}])


def test_final_turn_terminal_call_executes():
    a = _agent(1)
    state = {"messages": [_call("done")], "iterations": 1}
    assert a.router(state) == "n_tools"


def test_final_turn_non_terminal_call_falls_back():
    a = _agent(1)
    state = {"messages": [_call("search")], "iterations": 1}
    assert a.router(state) == "n_fallback"


def test_final_turn_terminal_result_ends_but_error_falls_back():
    a = _agent(1)
    ok = {"messages": [_call("done"), _tool("1", name="done")], "iterations": 1}
    assert a.router(ok) == "__end__"
    bad = _tool("1", name="done")
    bad.status = "error"
    assert a.router({"messages": [_call("done"), bad], "iterations": 1}) == "n_fallback"


def test_prepare_history_hard_cap_keeps_full_last_unit():
    a = _agent(10)
    a.compaction = SimpleNamespace(
        max_response_chars=0,
        tail_turns=3,
        min_compressible=10**9,
        threshold=lambda: 10**9,
        hard_cap=lambda: 0,
    )
    a.summarize = lambda middle: (None, None)  # noqa: ARG005
    msgs = _history("ok")
    msgs = msgs[:7] + [_ai(["d", "e"]), _tool("d"), _tool("e")]
    out, *_ = a.prepare_history(msgs, "s", 1)
    _assert_protocol(out)
    assert out[-3:] == msgs[-3:]
