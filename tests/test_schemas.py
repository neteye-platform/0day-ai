import pytest
from pydantic import ValidationError

from schemas import EvaluationToolInput, VulnerabilityRecord

VECTOR = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"


def exploitable(**kw):
    base = {
        "reasoning": "traced",
        "is_exploitable": True,
        "cvss_vector": VECTOR,
        "validation_strategy": "direct_to_validator",
    }
    return EvaluationToolInput(**{**base, **kw})


class TestEvaluationToolInput:
    def test_valid_exploitable_verdict(self):
        verdict = exploitable(reservations=["unaudited field"])
        assert verdict.changes_security_boundary is False
        assert verdict.reservations == ["unaudited field"]

    def test_valid_false_positive(self):
        verdict = EvaluationToolInput(
            reasoning="r", is_exploitable=False, mitigation="auth.py:check halts"
        )
        assert verdict.cvss_vector is None

    def test_exploitable_requires_a_strategy(self):
        with pytest.raises(ValidationError, match="validation_strategy"):
            exploitable(validation_strategy=None)

    @pytest.mark.parametrize("vector", [None, "", "   "])
    def test_exploitable_requires_a_vector(self, vector):
        with pytest.raises(ValidationError, match="cvss_vector is required"):
            exploitable(cvss_vector=vector)

    def test_false_positive_must_not_carry_a_vector(self):
        with pytest.raises(ValidationError, match="must stay unset"):
            EvaluationToolInput(
                reasoning="r",
                is_exploitable=False,
                mitigation="m",
                cvss_vector=VECTOR,
            )

    @pytest.mark.parametrize("mitigation", [None, "", "  "])
    def test_false_positive_requires_a_mitigation(self, mitigation):
        with pytest.raises(ValidationError, match="mitigation is required"):
            EvaluationToolInput(
                reasoning="r", is_exploitable=False, mitigation=mitigation
            )

    @pytest.mark.parametrize(
        "vector",
        [
            "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            "CVSS:3.1/AV:N/AC:L",
            "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:Q/C:H/I:H/A:H",
            "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N",
        ],
    )
    def test_unparsable_vectors_bounce(self, vector):
        with pytest.raises(ValidationError, match="CVSS"):
            exploitable(cvss_vector=vector)

    def test_vector_whitespace_is_trimmed(self):
        assert exploitable(cvss_vector=f"  {VECTOR}  ").cvss_vector == VECTOR

    def test_unknown_strategy_is_rejected(self):
        with pytest.raises(ValidationError):
            exploitable(validation_strategy="whatever")


class TestVulnerabilityRecord:
    def make(self, **kw):
        base = {
            "affected_nodes": ["node-a"],
            "cwe_id": "CWE-89",
            "description": "d",
            "vulnerable_component": "cursor.execute",
        }
        return VulnerabilityRecord(**{**base, **kw})

    def test_vuln_id_is_derived_and_stable(self):
        first, second = self.make(), self.make()
        assert first.vuln_id
        assert first.vuln_id == second.vuln_id
        assert "node-a" in first.vuln_id
        assert "CWE-89" in first.vuln_id

    def test_distinct_cves_get_distinct_ids_on_one_package(self):
        common = {
            "affected_nodes": ["dependency:lib"],
            "vulnerability_type": "Known Dependency Vulnerability",
            "vulnerable_component": None,
        }
        one = self.make(source_cve="CVE-2024-1", **common)
        two = self.make(source_cve="CVE-2024-2", **common)
        assert one.vuln_id != two.vuln_id
        assert "CVE-2024-1" in one.vuln_id

    def test_default_status_is_hypothesis(self):
        assert self.make().status == "hypothesis"

    def test_unknown_status_is_rejected(self):
        with pytest.raises(ValidationError):
            self.make(status="pwned")
