import pytest

from stages.reporter import _MD, _PDF_URL_FETCHER, _markdown_to_pdf

EVIL = '# T\n\n<img src="file:///etc/passwd">\n\n<script>alert(1)</script>\n\n| a | b |\n|---|---|\n| 1 | `x` |\n'


def test_raw_html_is_escaped():
    out = _MD.render(EVIL)
    assert "<img" not in out and "<script" not in out
    assert "&lt;img" in out and "&lt;script&gt;" in out
    assert "<table>" in out  # tables still render


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "http://127.0.0.1:1/x", "https://example.com/a.png"]
)
def test_fetcher_refuses_non_data_urls(url):
    with pytest.raises(ValueError):
        _PDF_URL_FETCHER.fetch(url)


def test_fetcher_allows_data_uri():
    resp = _PDF_URL_FETCHER.fetch("data:text/plain;base64,aGk=")
    assert resp is not None


def test_pdf_renders_and_attempts_no_fetch(monkeypatch):
    calls = []
    orig = type(_PDF_URL_FETCHER).fetch

    def spy(self, url, headers=None):
        calls.append(url)
        return orig(self, url, headers)

    monkeypatch.setattr(type(_PDF_URL_FETCHER), "fetch", spy)
    pdf = _markdown_to_pdf(EVIL)
    assert pdf.startswith(b"%PDF")
    assert calls == []
