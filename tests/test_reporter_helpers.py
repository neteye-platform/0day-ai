import pytest

import settings
from stages.reporter import _ZWSP, _below_cvss_gate, _fits_inline, _soft_break

XSS = "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N"


class TestFitsInline:
    def test_short_snippet_fits(self):
        assert _fits_inline("print('hi')\nprint('there')")

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "\n".join(f"line {i}" for i in range(21)),
            "x" * 91,
            "before\n```\ncode\n```",
        ],
    )
    def test_oversized_or_fenced_text_is_referenced_instead(self, text):
        assert not _fits_inline(text)

    def test_limits_are_inclusive(self):
        assert _fits_inline("\n".join(["x" * 90] * 20))


class TestSoftBreak:
    def test_adds_break_points_after_separators(self):
        out = _soft_break("src/a_b.c:d")
        assert out.replace(_ZWSP, "") == "src/a_b.c:d"
        assert out.count(_ZWSP) >= 4

    def test_plain_words_are_unchanged(self):
        assert _soft_break("plain") == "plain"


class TestBelowCvssGate:
    @pytest.fixture(autouse=True)
    def _threshold(self, monkeypatch):
        monkeypatch.setattr(settings, "validator_min_cvss", 7.0)

    def test_missing_strategy_is_treated_as_direct(self):
        assert _below_cvss_gate({"status": "confirmed", "cvss_vector": XSS})

    def test_validated_records_are_not_below_the_gate(self):
        assert not _below_cvss_gate({"status": "exploitable", "cvss_vector": XSS})

    def test_gate_can_be_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "validator_min_cvss", 0)
        assert not _below_cvss_gate({"status": "confirmed", "cvss_vector": XSS})
