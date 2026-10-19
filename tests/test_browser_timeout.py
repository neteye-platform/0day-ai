"""Browser origin pinning and wedge recovery, with no real browser."""

import concurrent.futures
import threading

import pytest

import settings
from tools import browser


class _Req:
    def __init__(self, url):
        self.url = url


class _Route:
    def __init__(self, url):
        self.request = _Req(url)
        self.action = None

    def continue_(self):
        self.action = "continue"

    def abort(self):
        self.action = "abort"


class _Ctx:
    def route(self, pattern, handler):
        self.pattern, self.handler = pattern, handler


def _decide(url):
    ctx = _Ctx()
    browser.BrowserSessionManager._pin_origin(ctx, "http://127.0.0.1:8080")
    route = _Route(url)
    ctx.handler(route)
    return route.action


@pytest.mark.parametrize(
    ("url", "action"),
    [
        ("http://127.0.0.1:8080/x", "continue"),
        ("data:text/html,hi", "continue"),
        ("about:blank", "continue"),
        ("blob:http://127.0.0.1:8080/uuid", "continue"),
        ("http://other-host/", "abort"),
        ("http://127.0.0.1:8080@other-host/", "abort"),
        ("file:///etc/passwd", "abort"),
    ],
)
def test_route_guard(url, action):
    assert _decide(url) == action


def test_url_from_rejects_userinfo_prefix_bypass():
    state = {"sandbox_url": "http://127.0.0.1:8080"}
    url, err = browser._url_from(state, "http://127.0.0.1:8080@other-host/")
    assert url is None and err
    url, err = browser._url_from(state, "/p")
    assert url == "http://127.0.0.1:8080/p" and err is None


def test_evaluate_wrapper_passes_deadline():
    calls = []

    class Page:
        def evaluate(self, *a):
            calls.append(a)
            return 1

    assert browser._evaluate_bounded(Page(), "1+1") == 1
    assert calls[0][1] == ["1+1", int(settings.browser_timeout_ms)]


def test_evaluate_falls_back_when_csp_blocks_eval():
    class Page:
        n = 0

        def evaluate(self, *a):
            Page.n += 1
            if len(a) == 2:
                raise RuntimeError("blocked by Content Security Policy")
            return 7

    assert browser._evaluate_bounded(Page(), "x") == 7


def test_wedged_worker_is_replaced_and_sessions_dropped(monkeypatch):
    monkeypatch.setattr(settings, "browser_timeout_ms", -29000)  # 1s total
    mgr = browser.BrowserSessionManager()
    mgr._sessions["a:s"] = object()
    release = threading.Event()
    old_executor = mgr._executor

    with pytest.raises(concurrent.futures.TimeoutError):
        mgr._browser_thread_run(lambda: release.wait(30))

    try:
        assert mgr._executor is not old_executor
        assert mgr._sessions == {}
        # A fresh worker serves the next call without queueing behind the stuck one.
        assert mgr._browser_thread_run(lambda: "ok") == "ok"
    finally:
        release.set()
        mgr._executor.shutdown(wait=False)
