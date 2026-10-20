import pytest

from utils.general import merge_vulnerabilities


def merged(existing, *updates):
    return {r["vuln_id"]: r for r in merge_vulnerabilities(existing, list(updates))}


def test_new_records_are_added(make_record):
    result = merge_vulnerabilities([make_record("a")], [make_record("b")])
    assert {r["vuln_id"] for r in result} == {"a", "b"}


@pytest.mark.parametrize(
    ("old", "new", "winner"),
    [
        ("hypothesis", "confirmed", "confirmed"),
        ("confirmed", "exploitable", "exploitable"),
        ("confirmed", "hypothesis", "confirmed"),
        ("exploitable", "confirmed", "exploitable"),
        ("review_error", "confirmed", "confirmed"),
        # the validator's ask must displace the auditor's intermediate verdict
        ("chained", "insufficient_context", "insufficient_context"),
        ("confirmed", "insufficient_context", "insufficient_context"),
    ],
)
def test_status_ladder(make_record, old, new, winner):
    result = merged([make_record(status=old)], make_record(status=new))
    assert result["n1:CWE-89:a"]["status"] == winner


@pytest.mark.parametrize(
    "new_status", ["confirmed", "false_positive", "exploitable", "review_error"]
)
def test_reviewer_reanswer_replaces_insufficient_context(make_record, new_status):
    result = merged(
        [make_record(status="insufficient_context", open_questions=["q"])],
        make_record(status=new_status),
    )
    record = result["n1:CWE-89:a"]
    assert record["status"] == new_status
    assert "open_questions" not in record


def test_same_status_merges_description_and_nodes(make_record):
    old = make_record(description="first", affected_nodes=["n1"])
    new = make_record(description="second", affected_nodes=["n2"])
    record = merged([old], new)["n1:CWE-89:a"]
    assert "first" in record["description"]
    assert "Additional context: second" in record["description"]
    assert set(record["affected_nodes"]) == {"n1", "n2"}


def test_same_description_is_not_duplicated(make_record):
    record = merged([make_record()], make_record())["n1:CWE-89:a"]
    assert record["description"] == "desc"


def test_distinct_cves_never_text_merge(make_record):
    old = make_record(source_cve="CVE-1", description="alpha")
    new = make_record(source_cve="CVE-2", description="beta")
    record = merged([old], new)["n1:CWE-89:a"]
    assert record["description"] == "alpha"


def test_patch_applied_replaces_exploitable_record(make_record):
    old = make_record(status="exploitable")
    new = make_record(status="confirmed", patch_state="applied", patch_diff="--- x")
    record = merged([old], new)["n1:CWE-89:a"]
    assert record["patch_state"] == "applied"
    assert record["status"] == "confirmed"


@pytest.mark.parametrize("state", ["verified", "rejected"])
def test_validator_patch_verdict_replaces_unconditionally(make_record, state):
    old = make_record(status="false_positive", patch_state="reviewed")
    new = make_record(status="exploitable", patch_state=state)
    assert merged([old], new)["n1:CWE-89:a"]["patch_state"] == state


def test_force_reset_replaces_verbatim_and_pops_marker(make_record):
    old = make_record(status="exploitable", poc_payload="stale")
    reset = make_record(status="confirmed", _force_reset=True)
    record = merged([old], reset)["n1:CWE-89:a"]
    assert record["status"] == "confirmed"
    assert "_force_reset" not in record
    assert "poc_payload" not in record


def test_force_reset_requires_a_vuln_id():
    with pytest.raises(ValueError, match="vuln_id"):
        merge_vulnerabilities([], [{"_force_reset": True, "status": "confirmed"}])


def test_missing_vuln_id_is_derived_from_the_record():
    result = merge_vulnerabilities(
        [], [{"affected_nodes": ["node-1"], "cwe_id": "CWE-79", "description": "d"}]
    )
    assert result[0]["vuln_id"]
    assert "node-1" in result[0]["vuln_id"]
