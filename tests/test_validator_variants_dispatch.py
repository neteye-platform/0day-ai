"""Standalone re-queue of unresolved grouped-run members."""

from stages import validator as vs
from utils import dispatch_claim_taken, mark_solo_validation, peek_solo_pending


def _rec(vuln_id):
    return {
        "vuln_id": vuln_id,
        "status": "confirmed",
        "cwe_id": "CWE-89",
        "vulnerable_component": "login form query builder",
        "validation_strategy": "direct_to_validator",
    }


def _state(*recs):
    return {"vulnerabilities": list(recs), "sandbox_url": "http://x"}


def _patch(monkeypatch):
    monkeypatch.setattr(vs, "ensure_sandbox_reachable", lambda *a, **k: None)
    monkeypatch.setattr(vs, "_start_agent_progress", lambda n: "pid")
    monkeypatch.setattr(vs, "_record_stat", lambda *a, **k: None)
    monkeypatch.setattr(vs.settings, "validator_min_cvss", 0)


def test_group_then_standalone_not_suppressed_by_group_claim(monkeypatch):
    _patch(monkeypatch)
    recs = [_rec("a"), _rec("b")]
    sends = vs.dispatch_validators(_state(*recs))
    assert len(sends) == 1
    payload = sends[0].arg
    assert len(payload["validation_variants"]) == 1
    assert dispatch_claim_taken("b", "direct", 0, 0)

    # The shared run left "b" unresolved: it is queued and re-sent alone.
    mark_solo_validation("b")
    solo = vs._solo_validator_sends(_state(*recs), recs)
    assert len(solo) == 1
    assert solo[0].arg["report_to_test"]["vuln_id"] == "b"
    assert solo[0].arg["validation_variants"] is None
    assert not peek_solo_pending()

    # Termination: a repeat firing neither re-sends nor re-queues it.
    assert vs._solo_validator_sends(_state(*recs), recs) == []


def test_solo_member_is_never_rebatched(monkeypatch):
    _patch(monkeypatch)
    recs = [_rec("a"), _rec("b")]
    mark_solo_validation("b")
    sends = vs.dispatch_validators(_state(*recs))
    assert len(sends) == 2
    assert all(s.arg["validation_variants"] is None for s in sends)


def test_resolved_member_leaves_queue_without_run(monkeypatch):
    _patch(monkeypatch)
    done = {**_rec("b"), "status": "false_positive"}
    mark_solo_validation("b")
    assert vs._solo_validator_sends(_state(done), [done]) == []
    assert not peek_solo_pending()
