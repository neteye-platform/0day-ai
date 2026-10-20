"""Sandbox origin confinement of send_http_request and the shared helper."""

import pytest
import requests
from requests.adapters import HTTPAdapter

from tools import base
from tools.base import is_sandbox_origin, send_http_request

SANDBOX = "http://127.0.0.1:8080"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8080/a?b=1",
        "HTTP://127.0.0.1:8080/x",
    ],
)
def test_same_origin_accepted(url):
    assert is_sandbox_origin(url, SANDBOX)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080@other-host/",
        "http://127.0.0.1:8080:x@other-host/",
        "http://user@127.0.0.1:8080/",
        "http://127.0.0.1:8081/",
        "https://127.0.0.1:8080/",
        "http://127.0.0.1/",
        "http://127.0.0.1:8080.evil.com/",
        "http://127.0.0.1:8080\\@evil/",
        "ftp://127.0.0.1:8080/",
        "file:///etc/passwd",
        "//other-host/",
    ],
)
def test_foreign_origin_rejected(url):
    assert not is_sandbox_origin(url, SANDBOX)


def test_default_ports_and_host_case():
    assert is_sandbox_origin("http://Example.com:80/x", "http://example.com")
    assert is_sandbox_origin("https://example.com/", "https://example.com:443")
    assert not is_sandbox_origin("https://example.com/", "http://example.com")


@pytest.fixture
def fake_net(monkeypatch):
    """Route every adapter send to a scripted table; record requested URLs."""
    sent = []
    table = {}

    def fake_send(self, request, **kwargs):
        sent.append((request.method, request.url, request.body))
        status, headers = table.get(request.url, (200, {}))
        resp = requests.Response()
        resp.status_code = status
        resp.headers.update({"Content-Type": "text/plain", **headers})
        resp._content = b"ok"
        resp._content_consumed = True
        resp.url = request.url
        resp.request = request
        return resp

    monkeypatch.setattr(HTTPAdapter, "send", fake_send)
    return sent, table


def _call(endpoint, **kw):
    state = {"sandbox_url": SANDBOX, "agent_id": "a"}
    return send_http_request.func(
        method=kw.pop("method", "GET"), endpoint=endpoint, state=state, **kw
    )


def test_userinfo_endpoint_blocked_before_sending(fake_net):
    sent, _ = fake_net
    out, _ = _call("http://127.0.0.1:8080@other-host/")
    assert "only make requests to the sandbox" in out
    assert sent == []


def test_relative_endpoint_still_works(fake_net):
    sent, _ = fake_net
    out, _ = _call("page")
    assert out.endswith("ok")
    assert sent[0][1] == f"{SANDBOX}/page"


def test_in_origin_redirect_followed(fake_net):
    sent, table = fake_net
    table[f"{SANDBOX}/a"] = (302, {"Location": "/b"})
    out, _ = _call("/a")
    assert [u for _, u, _ in sent] == [f"{SANDBOX}/a", f"{SANDBOX}/b"]
    assert out.startswith("HTTP/1.1 200")


def test_off_origin_redirect_blocked_before_sending(fake_net):
    sent, table = fake_net
    table[f"{SANDBOX}/a"] = (302, {"Location": "http://169.254.169.254/latest"})
    out, _ = _call("/a")
    assert "Request blocked" in out and "leaves the sandbox origin" in out
    assert [u for _, u, _ in sent] == [f"{SANDBOX}/a"]


def test_no_follow_returns_redirect_even_off_origin(fake_net):
    sent, table = fake_net
    table[f"{SANDBOX}/a"] = (302, {"Location": "http://elsewhere/"})
    out, _ = _call("/a", follow_redirects=False)
    assert out.startswith("HTTP/1.1 302")
    assert len(sent) == 1


def test_max_redirects_cap(fake_net):
    sent, table = fake_net
    table[f"{SANDBOX}/loop"] = (302, {"Location": "/loop"})
    out, _ = _call("/loop", max_redirects=2)
    assert "Request failed" in out and "redirects" in out.lower()
    assert len(sent) == 3


def test_csrf_prefetch_skips_foreign_origin(fake_net):
    sent, _ = fake_net
    session = base._OriginPinnedSession(SANDBOX)
    # Foreign target is skipped; only the in-origin sandbox root is fetched.
    base._fetch_csrf_tokens(session, "http://evil/x", SANDBOX, True)
    assert [u for _, u, _ in sent] == [f"{SANDBOX}/"]
