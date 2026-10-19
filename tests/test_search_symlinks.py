"""search_codebase must not read through symlinks escaping the app tree."""

import pytest

import settings
from tools import base


@pytest.fixture
def app(tmp_path, monkeypatch):
    root = tmp_path / "app"
    root.mkdir()
    monkeypatch.setattr(settings, "app_path", root)
    monkeypatch.setattr(base, "get_cached_graph_data", lambda _g: {"nodes": []})
    return root


def _search(query):
    return base.search_codebase.func(
        query=query, state={"messages": []}, match_whole_word=False
    )


def test_external_symlink_is_not_read(app, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET_TOKEN=1\n")
    (app / "leak.py").symlink_to(secret)
    (app / "real.py").write_text("TOPSECRET_TOKEN=local\n")
    out = _search("TOPSECRET_TOKEN")
    assert "real.py" in out
    assert "leak.py" not in out


def test_symlinked_dir_outside_app_is_not_read(app, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.py").write_text("NEEDLE_VALUE\n")
    (app / "link").symlink_to(outside, target_is_directory=True)
    assert "File:" not in _search("NEEDLE_VALUE")


def test_internal_symlink_still_searched(app):
    (app / "a.py").write_text("INSIDE_NEEDLE\n")
    (app / "b.py").symlink_to(app / "a.py")
    assert "File: a.py" in _search("INSIDE_NEEDLE")
