import json
from unittest.mock import MagicMock

import pytest

import stages.verifier as v


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Run the verifier on `demands`, with the LLM replaced by `evals(prompt)`."""
    monkeypatch.setattr(v.settings, "cache_dir", tmp_path)
    monkeypatch.setattr(v, "get_llm", lambda _a: MagicMock())
    monkeypatch.setattr(v, "_record_stat", lambda *a, **k: None)

    def go(demands, evals):
        prompts = []

        def fake(_llm, msgs, *_a):
            prompts.append(msgs[-1].content)
            return {"evaluations": evals}, {}

        monkeypatch.setattr(v, "invoke_structured_capped", fake)
        out, tag = v._contract_verifier_node(
            {"target_node_id": "n1", "target_code": "x", "incoming_demands": demands}
        )
        return out["vulnerabilities"], prompts, tag

    return go


def _ev(i, **kw):
    return {"demand_id": f"D{i}", "status": "FAILED", "evidence": "e", "cwe": "CWE-89", **kw}


def test_colliding_labels_get_unique_ids_and_distinct_records(run):
    demands = [
        {"type": "explorer_upstream_assumption", "parameter_name": "$id", "source": "a", "description": "first"},
        {"type": "explorer_upstream_assumption", "parameter_name": "$id", "source": "b", "description": "second"},
    ]
    vulns, prompts, _ = run(demands, [_ev(0), _ev(1)])
    assert "[ID: D0]" in prompts[0] and "[ID: D1]" in prompts[0]
    assert "first" in vulns[0]["description"] and "second" in vulns[1]["description"]
    assert vulns[0]["vulnerable_component"] != vulns[1]["vulnerable_component"]


def test_unique_label_stays_readable_and_cve_type_kept(run):
    demands = [
        {"type": "cve_assumption", "source": "CVE-1", "description": "d", "cwe_ids": ["CWE-89"]},
        {"type": "explorer_upstream_assumption", "parameter_name": "$q", "source": "a", "description": "e"},
    ]
    vulns, _, _ = run(demands, [_ev(0), _ev(1)])
    assert vulns[0]["vulnerable_component"] == "CVE-1"
    assert vulns[0]["source_cve"] == "CVE-1"
    assert vulns[1]["vulnerable_component"] == "$q"
    assert "source_cve" not in vulns[1]


def test_legacy_label_keyed_batch_cache_is_ignored(run, tmp_path):
    import hashlib

    demands = [{"type": "x", "source": "s", "description": "d"}]
    h = hashlib.md5(json.dumps(demands, sort_keys=True).encode()).hexdigest()
    d = tmp_path / "contract_verifier"
    d.mkdir()
    (d / f"n1_batch0_{h}.json").write_text(
        json.dumps({"evaluations": [{"demand_id": "s", "status": "FAILED"}]})
    )
    vulns, prompts, _ = run(demands, [_ev(0)])
    assert prompts  # legacy entry was a miss: the LLM was called
    assert "d" in vulns[0]["description"]
