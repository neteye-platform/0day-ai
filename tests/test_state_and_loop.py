import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from state import merge_token_usage
from tool_loop import _repeats_previous_tool_batch, _tool_batch_fingerprint


class TestMergeTokenUsage:
    def test_sums_every_field(self):
        prev = {"calls": 1, "input_tokens": 10, "output_tokens": 5}
        new = {"calls": 2, "input_tokens": 7, "output_tokens": 1}
        assert merge_token_usage(prev, new) == {
            "calls": 3,
            "input_tokens": 17,
            "output_tokens": 6,
        }

    @pytest.mark.parametrize("empty", [None, {}])
    def test_empty_sides_pass_the_other_through(self, empty):
        usage = {"calls": 1, "input_tokens": 1, "output_tokens": 1}
        assert merge_token_usage(empty, usage) == usage
        assert merge_token_usage(usage, empty) == usage

    def test_missing_fields_count_as_zero(self):
        assert merge_token_usage({"calls": 1}, {"input_tokens": 4}) == {
            "calls": 1,
            "input_tokens": 4,
            "output_tokens": 0,
        }


def ai(*calls):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"id{i}"}
            for i, (name, args) in enumerate(calls)
        ],
    )


class TestToolBatchFingerprint:
    def test_is_insensitive_to_call_and_key_order(self):
        a = ai(("read", {"x": 1, "y": 2}), ("search", {"q": "a"}))
        b = ai(("search", {"q": "a"}), ("read", {"y": 2, "x": 1}))
        assert _tool_batch_fingerprint(a) == _tool_batch_fingerprint(b)

    def test_different_args_differ(self):
        assert _tool_batch_fingerprint(ai(("read", {"x": 1}))) != (
            _tool_batch_fingerprint(ai(("read", {"x": 2})))
        )

    def test_message_without_calls_is_empty(self):
        assert _tool_batch_fingerprint(AIMessage(content="hi")) == ()


class TestRepeatDetection:
    def history(self, first, second):
        return [
            HumanMessage(content="go"),
            first,
            ToolMessage(content="out", tool_call_id="id0"),
            second,
        ]

    def test_identical_consecutive_batches_are_flagged(self):
        call = ("read", {"path": "a.py"})
        assert _repeats_previous_tool_batch(self.history(ai(call), ai(call)))

    def test_different_batches_are_not_flagged(self):
        history = self.history(ai(("read", {"p": 1})), ai(("read", {"p": 2})))
        assert not _repeats_previous_tool_batch(history)

    def test_a_single_batch_is_not_a_repeat(self):
        assert not _repeats_previous_tool_batch([ai(("read", {"p": 1}))])
