"""Headless-browser toolset for the Validator agent.

Renders pages, executes JavaScript, and captures console/pageerror/dialog
evidence so DOM-based vulnerabilities (XSS, DOM clobbering, client-side
injection) can be proven end-to-end - a path the HTTP-only
``send_http_request`` (which strips ``<script>``) can never reach.

Isolation / concurrency contract
--------------------------------
- ONE shared Firefox process (Playwright's own patched build via
  ``channel="firefox"``, or ``settings.browser_executable``) serves all
  concurrent validator agents; each session gets a dedicated Playwright
  BrowserContext (isolated cookies, storage, pages) - exactly like tabs.
- ``agent_id`` (from ValidatorState) namespaces session ids, so concurrent
  validators can never collide even if they pick the same session label.
- ``session_id`` is a required, explicit argument on every browser call;
  there is no shared "last-used" session to fall back on.
- ALL Playwright object traffic (browser/context/page ops, cookie seeding and
  reading, session teardown) is marshalled onto ONE dedicated browser thread
  (``ThreadPoolExecutor(max_workers=1)``). The sync Playwright API binds every
  object to the thread+greenlet that created it, and LangGraph's ``ToolNode``
  offloads sync tool calls to arbitrary workers of the shared default thread
  pool - a later call landing on a different worker would make Playwright
  raise ``greenlet.error: Cannot switch to a different thread``. Pinning every
  Playwright call to the single worker guarantees that affinity invariant, so
  concurrent validators are serialized at the browser and safe.
- Everything fails open: if Playwright/Firefox is unavailable the tools
  report a clear "browser unavailable" error and the validator continues
  HTTP-only.
- Console/pageerror/dialog events are captured per session into a bounded
  ring buffer; every call reports only the messages that have arrived since
  the previous call (a per-session watermark), and ``browser_console`` dumps
  them.

Lifecycle
---------
Sessions are closed per-agent at the terminal tool (``mark_validation_complete``)
or the validator fallback via ``close_agent_sessions``. An idle-TTL reaper acts
as a last-resort safety net and never reaps in-use sessions. All teardown is
routed through the same dedicated browser thread.
"""

import atexit
import concurrent.futures
import json
import logging
import threading
import time
from typing import Annotated
from urllib.parse import urlparse

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

import settings

logger = logging.getLogger(__name__)

BROWSER_TOOL_NAMES = {
    "browser_navigate",
    "browser_click",
    "browser_fill",
    "browser_evaluate",
    "browser_console",
}


class _Session:
    __slots__ = (
        "console_msgs",
        "context",
        "delivered",
        "in_use",
        "last_used",
        "page",
        "session_id",
    )

    def __init__(self, session_id, context, page):
        self.session_id = session_id
        self.context = context
        self.page = page
        self.console_msgs = []  # bounded ring of already-truncated event lines
        self.delivered = 0  # watermark: index already surfaced to the LLM
        self.last_used = time.time()
        self.in_use = False


def _host(url) -> str:
    try:
        return (urlparse(url).netloc or "").split(":")[0]
    except ValueError:
        return ""


def _agent_key(state) -> str:
    return (state.get("agent_id") if isinstance(state, dict) else None) or "no-agent"


class BrowserSessionManager:
    """Thread-safe owner of the shared browser and the per-session contexts."""

    def __init__(self):
        self._browser = None
        self._playwright = None
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()
        self._browser_lock = threading.Lock()
        self._disabled = False
        self._disabled_reason = None
        self._reaper_started = False
        # Single dedicated worker for EVERY Playwright object call. The sync
        # Playwright API binds objects to the thread+greenlet that created
        # them; ToolNode runs sync tools on arbitrary shared-pool workers, so
        # without this pin a later browser_* call would land on a different
        # thread and Playwright would raise `greenlet.error`.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pw-browser"
        )
        self._browser_tid = None

    def _browser_thread_run(self, fn):
        """Run zero-arg callable ``fn`` on the dedicated browser thread.

        Inline-runs when already on that thread (e.g. page event callbacks);
        otherwise submits the job and waits, bounded by a timeout so a wedged
        page op cannot block the browser thread forever. Any exception raised
        by ``fn`` (including a timeout) propagates to the caller and is handled
        by the tool's fail-open path.
        """
        if self._browser_tid == threading.get_ident():
            return fn()
        future = self._executor.submit(fn)
        attempt_s = (
            int(getattr(settings, "browser_timeout_ms", 30000)) / 1000.0
        ) + 30.0
        try:
            return future.result(timeout=attempt_s)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise

    # -- availability ------------------------------------------------------

    def _browser_boot(self):
        """Return a usable shared browser or None (fail open, one attempt).

        Runs on the dedicated browser thread (via ``_browser_thread_run``), so
        the launched browser is always owned by that thread and the thread id
        is recorded there for the affinity check.
        """
        if self._disabled or not getattr(settings, "browser_enabled", True):
            return None
        with self._browser_lock:
            if self._browser is not None:
                try:
                    if self._browser.is_connected():
                        return self._browser
                except Exception:  # noqa: BLE001, S110
                    pass
                # Dead browser (crash): drop it so the next call relaunches.
                self._stop_locked()
            try:
                from playwright.sync_api import sync_playwright

                self._playwright = sync_playwright().start()
                executable = getattr(settings, "browser_executable", None)
                if executable:
                    self._browser = self._playwright.firefox.launch(
                        executable_path=executable, headless=True
                    )
                else:
                    self._browser = self._playwright.firefox.launch(
                        channel="firefox", headless=True
                    )
                self._browser_tid = threading.get_ident()
                return self._browser
            except (
                Exception  # noqa: BLE001
            ) as e:  # fail open: no browser, keep HTTP path
                self._disabled = True
                self._disabled_reason = str(e)[:300]
                logger.warning(
                    "Headless browser unavailable (%s); validator continues HTTP-only.",
                    self._disabled_reason,
                )
                return None

    def _stop_locked(self):
        try:
            if self._browser is not None:
                self._browser_thread_run(self._browser.close)
        except Exception:  # noqa: BLE001, S110
            pass
        self._browser = None

    def shutdown(self):
        with self._lock:
            for s in list(self._sessions.values()):
                self._close_session(s)
            self._sessions.clear()
        with self._browser_lock:
            self._stop_locked()
        if self._playwright is not None:
            try:
                self._browser_thread_run(self._playwright.stop)
            except Exception:  # noqa: BLE001, S110
                pass
            self._playwright = None
        self._executor.shutdown(wait=True)

    # -- console/dialog capture ----------------------------------------------

    def _on_console(self, session, msg_obj):
        try:
            text = f"[console:{msg_obj.type}] {msg_obj.text}"
        except Exception:  # noqa: BLE001
            return
        self._append_event(session, text)

    def _on_pageerror(self, session, exc):
        self._append_event(
            session, f"[pageerror] {exc}".strip()[: settings.browser_console_msg_chars]
        )

    def _on_dialog(self, session, dialog):
        self._append_event(
            session, f"[dialog] {dialog.message}"[: settings.browser_console_msg_chars]
        )
        try:
            self._browser_thread_run(dialog.dismiss)
        except Exception:  # noqa: BLE001, S110
            pass

    def _append_event(self, session, text):
        if not text:
            return
        if len(text) > settings.browser_console_msg_chars:
            text = text[: settings.browser_console_msg_chars]
        session.console_msgs.append(text)
        overflow = len(session.console_msgs) - settings.browser_console_max_messages
        if overflow > 0:
            session.console_msgs = session.console_msgs[overflow:]
            if session.delivered > overflow:
                session.delivered -= overflow
            else:
                session.delivered = 0

    def _drain_new(self, session) -> str:
        new = session.console_msgs[session.delivered :]
        session.delivered = len(session.console_msgs)
        return "\n".join(new)

    # -- session lifecycle ---------------------------------------------------

    def _bind_handlers(self, session):
        session.page.on("console", lambda m: self._on_console(session, m))
        session.page.on("pageerror", lambda e: self._on_pageerror(session, e))
        session.page.on("dialog", lambda d: self._on_dialog(session, d))

    def session_for(self, state, session_id, create: bool):
        """Resolve ``(agent_id, session_id)`` to a session.

        ``create``=True boots the browser and builds a fresh (isolated,
        cookie-seeded) context when the session does not yet exist; returns
        (session, error_msg). Fails open on any browser failure.
        """
        key = f"{_agent_key(state)}:{session_id}"
        with self._lock:
            s = self._sessions.get(key)
        if s is not None:
            return s, None
        if not create:
            return None, (
                f"Error: unknown browser session_id {session_id!r}. Call "
                "browser_navigate first to open a session, then reuse the same "
                "session_id for browser_click/browser_fill/browser_evaluate/"
                "browser_console."
            )

        browser = self._browser_thread_run(self._browser_boot)
        if browser is None:
            return None, self._unavailable_msg()

        with self._lock:
            s = self._sessions.get(key)
            if s is not None:
                return s, None  # another thread created it while we launched

            def _build():
                context = browser.new_context()
                self._seed_cookies(context, state)
                page = context.new_page()
                s = _Session(session_id, context, page)
                self._bind_handlers(s)
                return s

            try:
                s = self._browser_thread_run(_build)
            except Exception as e:  # noqa: BLE001
                return None, (
                    f"Error: failed to create browser session: {str(e)[:300]}"
                )
            self._sessions[key] = s
            self._start_reaper()
            return s, None

    def _seed_cookies(self, context, state):
        cookies = (state or {}).get("cookies") if isinstance(state, dict) else None
        if not cookies:
            return
        host = _host((state or {}).get("sandbox_url"))
        if not host:
            return
        try:
            defs = [
                {
                    "name": str(name),
                    "value": str(value),
                    "domain": host,
                    "path": "/",
                    "secure": False,
                    "http_only": False,
                    "same_site": "Lax",
                }
                for name, value in cookies.items()
                if str(value)
            ]
            if defs:
                context.add_cookies(defs)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to seed browser cookies: %s", e)

    def cookie_artifact(self, session) -> dict:
        """Flatten the context's cookie jar into a ``{name: value}`` dict."""
        flat = {}
        try:

            def _cookies():
                return {
                    c["name"]: c["value"]
                    for c in session.context.cookies()
                    if c.get("name") and c.get("value") is not None
                }

            flat = self._browser_thread_run(_cookies)
        except Exception:  # noqa: BLE001, S110
            pass
        return flat or {}

    def browser_artifact(self, session) -> dict:
        return {
            "session_id": session.session_id,
            "cookies": self.cookie_artifact(session),
        }

    def unavailable_art(self) -> dict:
        return {"session_id": "", "cookies": {}}

    def _unavailable_msg(self) -> str:
        reason = self._disabled_reason or "disabled by settings"
        return (
            "Error: Browser is unavailable in this environment "
            f"({reason}). The HTTP-only send_http_request tool remains "
            "available."
        )

    def close_agent_sessions(self, agent_id: str):
        if not agent_id:
            return
        prefix = f"{agent_id}:"
        with self._lock:
            doomed = [s for key, s in self._sessions.items() if key.startswith(prefix)]
            for s in doomed:
                self._close_session(s)
            for key in [k for k in self._sessions if k.startswith(prefix)]:
                self._sessions.pop(key, None)

    def _close_session(self, session: _Session):
        try:
            self._browser_thread_run(session.context.close)
        except Exception:  # noqa: BLE001, S110
            pass

    def use(self, session: _Session, body):
        """Run zero-arg callable ``body`` (touching only Playwright objects) on
        the dedicated browser thread, marking the session in-use with a fresh
        timestamp so the idle reaper never reaps it mid-flight. Returns the
        body's return value; exceptions propagate to the caller's fail-open
        handlers."""
        session.in_use = True
        session.last_used = time.time()
        try:
            return self._browser_thread_run(body)
        finally:
            session.in_use = False
            session.last_used = time.time()

    # -- idle reaper (safety net) ----------------------------------------------

    def _start_reaper(self):
        if self._reaper_started:
            return
        self._reaper_started = True
        interval = max(30, int(getattr(settings, "browser_idle_timeout_sec", 600) / 4))
        ttl = int(getattr(settings, "browser_idle_timeout_sec", 600))
        grace = max(2.0, ttl / 10.0)

        def reap():
            while True:
                time.sleep(interval)
                now = time.time()
                with self._lock:
                    doomed = [
                        s
                        for s in self._sessions.values()
                        if not s.in_use and (now - s.last_used) > ttl + grace
                    ]
                    for s in doomed:
                        self._close_session(s)
                    for key in list(self._sessions):
                        if self._sessions[key] in doomed:
                            self._sessions.pop(key, None)

        threading.Thread(target=reap, daemon=True, name="browser-reaper").start()


manager = BrowserSessionManager()
atexit.register(manager.shutdown)


def _describe(session) -> str:
    parts = []
    try:
        parts.append(f"URL: {session.page.url}")
    except Exception:  # noqa: BLE001, S110
        pass
    try:
        parts.append(f"Title: {session.page.title()}")
    except Exception:  # noqa: BLE001, S110
        pass
    body = ""
    try:
        body = session.page.locator("body").inner_text(
            timeout=min(settings.browser_timeout_ms, 5000)
        )
    except Exception:  # noqa: BLE001
        try:
            body = session.page.content()
        except Exception:  # noqa: BLE001
            body = ""
    body = body.strip()
    if len(body) > settings.browser_describe_max_chars:
        body = (
            body[: settings.browser_describe_max_chars] + "\n... [body truncated] ..."
        )
    parts.append("\n--- VISIBLE TEXT ---\n" + (body or "(no visible text)"))
    events = manager._drain_new(session)
    parts.append(
        "\n--- NEW BROWSER EVENTS (console/pageerror/dialog since last call) ---\n"
        + (events or "(none)")
    )
    return "\n".join(parts)


def _url_from(state, url) -> tuple[str | None, str | None]:
    sandbox_url = (state or {}).get("sandbox_url") if isinstance(state, dict) else None
    if not sandbox_url:
        return None, (
            "Error: No sandbox is configured. The preprocessor could not start "
            "a sandbox container."
        )
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        if not url.startswith("/"):
            url = f"/{url}"
        url = f"{sandbox_url}{url}"
    if not url.startswith(sandbox_url):
        return (
            None,
            f"Error: You can only navigate the sandbox application at {sandbox_url}",
        )
    return url, None


@tool(response_format="content_and_artifact")
def browser_navigate(
    session_id: str,
    url: str,
    state: Annotated[dict, InjectedState],
    wait_for: str = "load",
) -> tuple[str, dict]:
    """
    Opens a headless-browser session on a fresh, isolated context and loads a
    page in it, executing the page's JavaScript (unlike send_http_request,
    which strips <script> tags). Use this to prove DOM/client-side vulnerabilities
    (XSS, DOM clobbering, client-side injection) and to interact with login
    flows that require a real browser engine.

    Args:
        session_id (str): Pick a unique label for THIS validation (e.g.
            's1'). Reuse the exact same session_id in every subsequent
            browser_* call so the browser keeps your page, cookies, and
            console history. Do not reuse another vulnerability's session_id.
        url (str): Absolute sandbox URL or an absolute path under the sandbox
            (e.g. '/search?q=xss'). Only the sandbox origin is reachable.
        wait_for (str): 'load' (default, waits for full load), 'domcontentloaded',
            or 'networkidle'.
    """
    session, err = manager.session_for(state, session_id, create=True)
    if err:
        return err, manager.unavailable_art()
    target, err = _url_from(state, url)
    if err:
        return err, manager.browser_artifact(session)
    try:

        def _navigate():
            try:
                session.page.goto(
                    target,
                    wait_until=wait_for,
                    timeout=settings.browser_timeout_ms,
                )
            except Exception as e:  # noqa: BLE001
                return (
                    f"Error: navigation failed: {str(e)[:500]}",
                    manager.browser_artifact(session),
                )
            return _describe(session), manager.browser_artifact(session)

        snap, art = manager.use(session, _navigate)
        hint = (
            f"\n\n[Keep using session_id {session_id!r} for all browser calls "
            "on this vulnerability.]"
        )
        return snap + hint, art
    except Exception as e:  # noqa: BLE001
        return (
            f"Error: browser_navigate failed: {str(e)[:500]}",
            manager.unavailable_art(),
        )


@tool(response_format="content_and_artifact")
def browser_click(
    session_id: str,
    selector: str,
    state: Annotated[dict, InjectedState],
) -> tuple[str, dict]:
    """
    Clicks the element matching a CSS selector in the active browser session
    opened by browser_navigate. Use after browser_navigate to trigger
    event handlers, submit buttons, or navigate client-side routes.

    Args:
        session_id (str): The session_id returned/used by browser_navigate.
        selector (str): CSS selector of the element to click.
    """
    session, err = manager.session_for(state, session_id, create=False)
    if err:
        return err, manager.unavailable_art()
    try:

        def _click():
            try:
                session.page.click(selector, timeout=settings.browser_timeout_ms)
            except Exception as e:  # noqa: BLE001
                return (
                    f"Error: could not click {selector!r}: {str(e)[:500]}",
                    manager.browser_artifact(session),
                )
            try:
                session.page.wait_for_load_state(
                    "domcontentloaded", timeout=min(settings.browser_timeout_ms, 5000)
                )
            except Exception:  # noqa: BLE001, S110
                pass
            return _describe(session), manager.browser_artifact(session)

        snap, art = manager.use(session, _click)
        return snap, art
    except Exception as e:  # noqa: BLE001
        return f"Error: browser_click failed: {str(e)[:500]}", manager.unavailable_art()


@tool(response_format="content_and_artifact")
def browser_fill(
    session_id: str,
    selector: str,
    value: str,
    state: Annotated[dict, InjectedState],
) -> tuple[str, dict]:
    """
    Fills an input/textarea with a value in the active browser session. Use to
    type into search boxes, comment fields, or login forms before a
    browser_click.

    Args:
        session_id (str): The session_id returned/used by browser_navigate.
        selector (str): CSS selector of the element to fill.
        value (str): Text to type into the element.
    """
    session, err = manager.session_for(state, session_id, create=False)
    if err:
        return err, manager.unavailable_art()
    try:

        def _fill():
            try:
                session.page.fill(selector, value, timeout=settings.browser_timeout_ms)
            except Exception as e:  # noqa: BLE001
                return (
                    f"Error: could not fill {selector!r}: {str(e)[:500]}",
                    manager.browser_artifact(session),
                )
            return _describe(session), manager.browser_artifact(session)

        snap, art = manager.use(session, _fill)
        return snap, art
    except Exception as e:  # noqa: BLE001
        return f"Error: browser_fill failed: {str(e)[:500]}", manager.unavailable_art()


@tool(response_format="content_and_artifact")
def browser_evaluate(
    session_id: str,
    expression: str,
    state: Annotated[dict, InjectedState],
) -> tuple[str, dict]:
    """
    Executes a JavaScript expression in the current page of the active
    browser session and returns its JSON-serialized result. Use to read DOM
    state after an injection (e.g. document.getElementById('x').innerHTML),
    or to trigger/manipulate client-side behavior.

    Args:
        session_id (str): The session_id returned/used by browser_navigate.
        expression (str): JavaScript expression to evaluate (e.g.
            "window.localStorage.getItem('token')").
    """
    session, err = manager.session_for(state, session_id, create=False)
    if err:
        return err, manager.unavailable_art()
    try:

        def _evaluate():
            try:
                result = session.page.evaluate(expression)
            except Exception as e:  # noqa: BLE001
                return (
                    f"Error: evaluate failed: {str(e)[:500]}",
                    manager.browser_artifact(session),
                )
            rendered = json.dumps(result, default=str, ensure_ascii=False)
            if len(rendered) > 4000:
                rendered = rendered[:4000] + "\n... [result truncated] ..."
            events = manager._drain_new(session)
            out = (
                f"--- EVALUATE RESULT ---\n{rendered}\n\n"
                f"--- NEW BROWSER EVENTS (console/pageerror/dialog since last call) ---\n"
                + (events or "(none)")
            )
            return out, manager.browser_artifact(session)

        out, art = manager.use(session, _evaluate)
        return out, art
    except Exception as e:  # noqa: BLE001
        return (
            f"Error: browser_evaluate failed: {str(e)[:500]}",
            manager.unavailable_art(),
        )


@tool(response_format="content_and_artifact")
def browser_console(
    session_id: str,
    state: Annotated[dict, InjectedState],
) -> tuple[str, dict]:
    """
    Dumps console.log/error/warn output, uncaught page errors, and any JS
    alert/confirm/prompt (dialog) messages that the active browser session has
    produced since the last browser call. Console evidence (e.g. an executed
    XSS payload calling document.cookie) is the proof that client-side code
    actually ran. Each message is returned at most once.

    Args:
        session_id (str): The session_id returned/used by browser_navigate.
    """
    session, err = manager.session_for(state, session_id, create=False)
    if err:
        return err, manager.unavailable_art()
    try:

        def _console():
            events = manager._drain_new(session)
            head = ""
            try:
                head = f"URL: {session.page.url}"
            except Exception:  # noqa: BLE001, S110
                pass
            return (
                f"{head}\n--- BROWSER EVENTS (console/pageerror/dialog since last call) ---\n"
                + (events or "(none)")
            ), manager.browser_artifact(session)

        out, art = manager.use(session, _console)
        return out, art
    except Exception as e:  # noqa: BLE001
        return (
            f"Error: browser_console failed: {str(e)[:500]}",
            manager.unavailable_art(),
        )
