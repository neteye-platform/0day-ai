import pytest

from utils.dedup import component_tokens, is_degenerate_anchor


class TestDegenerateAnchors:
    @pytest.mark.parametrize(
        "component",
        ["", None, "   ", "$str", "uid", "$itemtype", "some_node_id"],
    )
    def test_bare_labels_are_degenerate(self, component):
        assert is_degenerate_anchor(component)

    @pytest.mark.parametrize(
        "component",
        [
            "$db->prepare()",
            "request_params['filter']",
            "cursor.execute",
            "user supplied sort column",  # 3+ word prose is identifying
        ],
    )
    def test_structural_or_descriptive_anchors_identify(self, component):
        assert not is_degenerate_anchor(component)

    def test_two_words_are_still_degenerate(self):
        assert is_degenerate_anchor("sort column")


class TestComponentTokens:
    def test_is_lowercased_and_deduplicated(self):
        assert component_tokens("Foo foo BAR") == frozenset({"foo", "bar"})

    def test_empty_inputs(self):
        assert component_tokens("") == frozenset()
        assert component_tokens(None) == frozenset()
