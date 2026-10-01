"""Opt-in capture of serialized LLM requests for provider debugging."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


_write_lock = threading.Lock()
_SENSITIVE_HEADERS = {"authorization", "proxy-authorization", "x-api-key"}
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
}


def _enabled() -> bool:
    return os.getenv("LLM_CAPTURE_RAW_REQUESTS", "").lower() in {"1", "true", "yes", "on"}


def _log_path() -> Path:
    configured = os.getenv("LLM_RAW_REQUEST_LOG")
    path = Path(configured) if configured else Path(".cache") / "llm_raw_requests.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key.lower() in _SENSITIVE_KEYS else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _request_body(request: httpx.Request) -> tuple[Any, str]:
    body = request.content
    digest = hashlib.sha256(body).hexdigest()
    max_bytes = int(os.getenv("LLM_RAW_REQUEST_MAX_BYTES", "2000000"))
    if len(body) > max_bytes:
        return f"[BODY OMITTED: {len(body)} bytes exceeds {max_bytes} byte limit]", digest

    try:
        return _redact(json.loads(body.decode("utf-8"))), digest
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body[:max_bytes].decode("utf-8", errors="replace"), digest


def _capture_request(request: httpx.Request) -> None:
    if not _enabled():
        return

    body, digest = _request_body(request)
    headers = {
        key: "[REDACTED]" if key.lower() in _SENSITIVE_HEADERS else value
        for key, value in request.headers.items()
    }
    record = {
        "kind": "llm_request",
        "request_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": request.method,
        "url": str(request.url),
        "headers": headers,
        "body_sha256": digest,
        "body": body,
    }
    path = _log_path()
    with _write_lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        try:
            path.chmod(0o600)
        except OSError:
            pass


def _capture_response(response: httpx.Response) -> None:
    if not _enabled() or response.is_success:
        return

    path = _log_path()
    record = {
        "kind": "llm_response_error",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status_code": response.status_code,
        "request_url": str(response.request.url),
        "request_body_sha256": hashlib.sha256(response.request.content).hexdigest(),
    }
    with _write_lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_debug_http_client() -> httpx.Client | None:
    """Return an instrumented client only when raw request capture is enabled."""
    if not _enabled():
        return None
    return httpx.Client(
        event_hooks={"request": [_capture_request], "response": [_capture_response]},
    )
