"""Per-variant verdicts in grouped validator runs: no blanket propagation."""

import tools
from schemas import VariantResult
from utils import is_solo_validation, peek_solo_pending

SEED = {"vuln_id": "s", "status": "confirmed"}
A = {"vuln_id": "a", "status": "confirmed"}
B = {"vuln_id": "b", "status": "confirmed"}


def _state(*variants):
    return {"validation_variants": list(variants)}


def _verdict(status="exploitable"):
    return {
        **SEED,
        "status": status,
        "poc_payload": "seedpoc",
        "execution_logs": "seedlog",
    }


def _by_id(updates):
    return {u["vuln_id"]: u for u in updates}


def _res(vuln_id, ok, payload="p", logs="l"):
    return {
        "vuln_id": vuln_id,
        "is_confirmed": ok,
        "poc_payload": payload,
        "execution_logs": logs,
    }


def test_each_variant_gets_its_own_verdict_and_evidence():
    results = [
        _res("a", True, "pa", "la"),
        VariantResult(
            vuln_id="b", is_confirmed=False, poc_payload=None, execution_logs="lb"
        ),
    ]
    out = _by_id(tools.propagate_validation_update(_state(A, B), _verdict(), results))
    assert out["s"]["status"] == "exploitable"
    assert out["a"]["status"] == "exploitable"
    assert out["a"]["poc_payload"] == "pa" and out["a"]["execution_logs"] == "la"
    assert out["b"]["status"] == "false_positive"
    assert out["b"]["execution_logs"] == "lb"
    assert out["b"]["poc_payload"] is None  # never the seed's payload
    assert not peek_solo_pending()


def test_omitted_variant_stays_unresolved_and_is_requeued():
    out = _by_id(
        tools.propagate_validation_update(_state(A, B), _verdict(), [_res("a", True)])
    )
    assert "b" not in out  # no update => channel record stays 'confirmed'
    assert peek_solo_pending() == {"b"}
    assert is_solo_validation("b") and not is_solo_validation("a")


def test_unknown_and_duplicate_ids_are_ignored():
    results = [
        _res("zzz", True),  # not in the group
        _res("s", False),  # the seed is not a variant
        _res("a", True, "x"),
        _res("a", False, "y"),  # contradictory duplicate => untested
    ]
    out = _by_id(tools.propagate_validation_update(_state(A, B), _verdict(), results))
    assert set(out) == {"s"}
    assert peek_solo_pending() == {"a", "b"}


def test_group_of_one_is_unchanged():
    updates = tools.propagate_validation_update({}, _verdict(), [_res("a", True)])
    assert [u["vuln_id"] for u in updates] == ["s"]
    assert not peek_solo_pending()


def test_ask_for_context_never_resolves_variants():
    out = _by_id(
        tools.propagate_validation_update(
            _state(A), _verdict("insufficient_context"), [_res("a", True)]
        )
    )
    assert set(out) == {"s"}
    assert peek_solo_pending() == {"a"}


def test_legacy_replay_without_variant_results_does_not_blanket_copy():
    # A legacy cache entry is a bare seed record with no private results key.
    out = _by_id(tools.propagate_validation_update(_state(A, B), _verdict()))
    assert set(out) == {"s"}
    assert peek_solo_pending() == {"a", "b"}


def test_replay_reads_results_from_cached_record_key():
    cached = {
        **_verdict(),
        tools.VARIANT_RESULTS_CACHE_KEY: [_res("a", False, None, "la")],
    }
    out = _by_id(tools.propagate_validation_update(_state(A, B), cached))
    assert out["a"]["status"] == "false_positive"
    assert peek_solo_pending() == {"b"}


def test_validator_cache_identity_includes_variant_reports(tmp_path, monkeypatch):
    from utils import general

    monkeypatch.setattr(general.settings, "app_path", str(tmp_path), raising=False)
    seed = {"vuln_id": "s", "status": "confirmed"}
    member = {"vuln_id": "m", "status": "confirmed", "reproduction_steps": ["a"]}
    general.cache_validator(
        seed, None, {**seed, "status": "exploitable"}, None, [member]
    )

    hit = general.cache_validator(seed, None, variants=[member])
    changed = general.cache_validator(
        seed, None, variants=[{**member, "reproduction_steps": ["b"]}]
    )
    assert hit is not None and hit["status"] == "exploitable"
    assert changed is None
