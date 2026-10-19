import settings
from tools.credential import _collect_sql


def _run(tmp_path, monkeypatch, sql):
    monkeypatch.setattr(settings, "app_path", tmp_path)
    f = tmp_path / "seed.sql"
    f.write_text(sql)
    return _collect_sql(f, allow_excluded=True)


def test_leading_comment_line_does_not_swallow_statement(tmp_path, monkeypatch):
    hits = _run(
        tmp_path,
        monkeypatch,
        "-- create app user\nCREATE USER 'app'@'%' IDENTIFIED BY 'S3cretPw';\n",
    )
    assert [(h["username"], h["secret"]) for h in hits] == [("app", "S3cretPw")]


def test_statement_split_across_lines(tmp_path, monkeypatch):
    hits = _run(
        tmp_path,
        monkeypatch,
        "CREATE USER 'app'@'%'\nIDENTIFIED BY 'S3cretPw';\n",
    )
    assert [(h["username"], h["secret"]) for h in hits] == [("app", "S3cretPw")]


def test_comment_between_statements(tmp_path, monkeypatch):
    hits = _run(
        tmp_path,
        monkeypatch,
        "SELECT 1;\n# note\n-- another\nCREATE USER 'b'@'%' IDENTIFIED BY 'Pw12345';\n",
    )
    assert [h["secret"] for h in hits] == ["Pw12345"]
