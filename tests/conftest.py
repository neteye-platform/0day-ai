"""Shared fixtures.

The analyzer resolves ``agents.yaml`` and ``settings.app_path`` relative to the
current directory, so every test runs from the repository root. The target app
is never needed: ``settings`` only warns when it is missing.
"""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def _reset_dispatch_claims():
    from utils.general import clear_feedback_dispatch_state

    clear_feedback_dispatch_state()
    yield
    clear_feedback_dispatch_state()


@pytest.fixture
def make_record():
    """Build a minimal vulnerability record dict with a fixed ``vuln_id``."""

    def _make(vuln_id="n1:CWE-89:a", **overrides):
        record = {
            "vuln_id": vuln_id,
            "status": "hypothesis",
            "cwe_id": "CWE-89",
            "description": "desc",
            "affected_nodes": ["n1"],
        }
        record.update(overrides)
        return record

    return _make
