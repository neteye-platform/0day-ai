import pytest

from utils.general import (
    boundary_deferred,
    cvss_gate_blocks,
    cvss_severity_label,
    cvss_v3_base_score,
    is_high_severity,
)

CRITICAL = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
XSS = "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N"


@pytest.mark.parametrize(
    ("vector", "score"),
    [
        (CRITICAL, 9.8),
        ("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H", 10.0),
        (XSS, 6.1),
        ("CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:N/A:N", 6.5),
        ("CVSS:3.1/AV:L/AC:L/PR:N/UI:R/S:U/C:N/I:N/A:N", 0.0),
        ("CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", 9.8),
    ],
)
def test_base_score_matches_spec_examples(vector, score):
    assert cvss_v3_base_score(vector) == score


@pytest.mark.parametrize(
    "vector",
    [
        None,
        "",
        42,
        "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",  # no CVSS:3.x prefix
        "CVSS:2.0/AV:N/AC:L/Au:N/C:P/I:P/A:P",  # other generation
        "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H",  # missing metric
        "CVSS:3.1/AV:X/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",  # invalid value
        "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:Z/C:H/I:H/A:H",  # scope is strict U/C
    ],
)
def test_base_score_rejects_unusable_vectors(vector):
    assert cvss_v3_base_score(vector) is None


@pytest.mark.parametrize(
    ("score", "label"),
    [
        (None, "Not assessed"),
        (0.0, "None"),
        (0.1, "Low"),
        (3.9, "Low"),
        (4.0, "Medium"),
        (6.9, "Medium"),
        (7.0, "High"),
        (8.9, "High"),
        (9.0, "Critical"),
        (10.0, "Critical"),
    ],
)
def test_severity_label_boundaries(score, label):
    assert cvss_severity_label(score) == label


def test_is_high_severity_prefers_label_over_vector():
    assert is_high_severity({"severity_label": "critical"})
    assert not is_high_severity({"severity_label": "LOW", "cvss_vector": CRITICAL})
    assert is_high_severity({"cvss_vector": CRITICAL})
    assert not is_high_severity({"cvss_vector": XSS})
    assert not is_high_severity({})


class TestCvssGate:
    def record(self, **kw):
        return {
            "status": "confirmed",
            "validation_strategy": "direct_to_validator",
            "cvss_vector": XSS,
            **kw,
        }

    def test_blocks_below_threshold(self):
        assert cvss_gate_blocks(self.record(), 7.0)

    def test_passes_at_or_above_threshold(self):
        assert not cvss_gate_blocks(self.record(cvss_vector=CRITICAL), 7.0)
        assert not cvss_gate_blocks(self.record(), 6.1)

    @pytest.mark.parametrize("threshold", [None, 0, -1])
    def test_disabled_thresholds_never_block(self, threshold):
        assert not cvss_gate_blocks(self.record(), threshold)

    def test_fails_open_without_a_usable_vector(self):
        assert not cvss_gate_blocks(self.record(cvss_vector=None), 7.0)
        assert not cvss_gate_blocks(self.record(cvss_vector="garbage"), 7.0)

    @pytest.mark.parametrize("status", ["hypothesis", "exploitable", "false_positive"])
    def test_only_confirmed_records_are_gated(self, status):
        assert not cvss_gate_blocks(self.record(status=status), 7.0)

    def test_static_findings_are_not_gated(self):
        assert not cvss_gate_blocks(
            self.record(validation_strategy="static_finding_only"), 7.0
        )

    def test_requires_integration_is_gated(self):
        assert cvss_gate_blocks(
            self.record(validation_strategy="requires_integration"), 7.0
        )

    @pytest.mark.parametrize("field", ["patch_state", "patch_diff"])
    def test_patch_lifecycle_records_always_validate(self, field):
        assert not cvss_gate_blocks(self.record(**{field: "applied"}), 7.0)


class TestBoundaryDeferred:
    def test_requires_flag_and_gate_block(self):
        flagged = {"cvss_vector": XSS, "changes_security_boundary": True}
        assert boundary_deferred(flagged, 7.0)
        assert not boundary_deferred({"cvss_vector": XSS}, 7.0)

    def test_gate_passing_record_is_never_deferred(self):
        record = {"cvss_vector": CRITICAL, "changes_security_boundary": True}
        assert not boundary_deferred(record, 7.0)

    def test_evaluates_as_confirmed_regardless_of_current_status(self):
        record = {
            "status": "unchainable",
            "cvss_vector": XSS,
            "changes_security_boundary": True,
        }
        assert boundary_deferred(record, 7.0)
