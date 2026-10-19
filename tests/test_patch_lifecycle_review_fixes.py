import pytest

import settings
import tools.patch as patch_tools
from stages import integration_auditor as ia
from stages import patcher as pt
from utils import clear_feedback_dispatch_state


def _hunk(rel, start, old, new):
    return {
        "file": rel,
        "diff": "d",
        "undo": {"start": start, "new_len": len(new), "old_lines": old},
    }


def test_rollback_restores_multi_hunk_file(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "app_path", tmp_path)
    f = tmp_path / "a.py"
    f.write_text("a\nb\nc\nd\n")
    # hunk1: b -> X,Y (shifts); hunk2 (against shifted file): d (now line 5) -> Z
    f.write_text("a\nX\nY\nc\nZ\n")
    log = [
        _hunk("a.py", 2, ["b"], ["X", "Y"]),
        _hunk("a.py", 5, ["d"], ["Z"]),
        {"file": "a.py", "diff": "", "note": "noop"},
    ]
    assert patch_tools.rollback_patch_log(log) == ["a.py"]
    assert f.read_text() == "a\nb\nc\nd\n"


def test_rollback_skips_changed_or_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "app_path", tmp_path)
    assert patch_tools.rollback_patch_log([_hunk("nope.py", 1, ["a"], ["b"])]) == []
    (tmp_path / "s.py").write_text("x\n")
    assert patch_tools.rollback_patch_log([_hunk("s.py", 9, ["a"], ["b"])]) == []


def test_dispatch_patchers_claims_once(monkeypatch):
    clear_feedback_dispatch_state()
    monkeypatch.setattr(settings, "patcher_enabled", True)
    rec = {
        "vuln_id": "n:CWE-1:x",
        "status": "exploitable",
        "vulnerability_type": "SQLi",
        "patch_round": 0,
    }
    state = {"vulnerabilities": [rec]}
    first = pt.dispatch_patchers(state)
    assert isinstance(first, list) and len(first) == 1
    assert pt.dispatch_patchers(state) == "integration_audit_dispatch"
    clear_feedback_dispatch_state()


def _audit_state(**extra):
    rec = {
        "vuln_id": "n:CWE-2:y",
        "status": "confirmed",
        "validation_strategy": "requires_integration",
        "integration_audit_reasoning": "old audit",
        **extra,
    }
    return {"vulnerabilities": [rec], "sandbox_url": "http://x"}


def test_patched_audited_record_is_reaudited(monkeypatch):
    clear_feedback_dispatch_state()
    monkeypatch.setattr(ia, "ensure_sandbox_reachable", lambda *a, **k: None)
    monkeypatch.setattr(settings, "validator_min_cvss", 0)
    out = ia.dispatch_integration_audits(_audit_state(patch_state="reviewed", patch_round=1))
    assert isinstance(out, list) and len(out) == 1
    # claim bounds it: same patch round does not re-audit
    assert (
        ia.dispatch_integration_audits(_audit_state(patch_state="reviewed", patch_round=1))
        == []
    )
    clear_feedback_dispatch_state()


def test_unpatched_audited_record_still_skipped(monkeypatch):
    clear_feedback_dispatch_state()
    monkeypatch.setattr(ia, "ensure_sandbox_reachable", lambda *a, **k: None)
    monkeypatch.setattr(settings, "validator_min_cvss", 0)
    assert ia.dispatch_integration_audits(_audit_state()) == "reporter_dispatch"
