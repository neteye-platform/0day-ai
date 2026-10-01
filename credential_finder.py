"""Credential-finder preprocessing agent (runs after container build/sandbox).

Finds pre-configured credentials in repo env/compose/Dockerfile definitions,
application source/seed/SQL, and extracted container-image artifacts.
Discovery is deterministic; with ``settings.credential_finder_use_llm`` one
structured ``llms.get_llm("credential_finder")`` call normalizes the
candidates, failing open to
the raw list. Result goes to ``<target_app>/.cache/credentials.json`` for
validator agents. Importing this module never touches docker or the network.
"""

from pathlib import Path
import hashlib
import json
import logging
import re
from typing import Any, Iterator, Optional

import yaml

import settings
import utils
from llms import get_llm, invoke_tracked
from utils import COMPOSE_FILENAMES, is_path_excluded, get_container_artifacts_root, safe_cache_filename

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

# Token test so `tokenizer` / `password_expiration_delay` / `csrf_token` do not match.
_SECRET_WORDS = {
    "password", "passwd", "pwd", "passphrase", "pass", "secret",
    "apikey", "apisecret", "auth",
}
_PREFIX_KEY_WORDS = {
    "secret", "api", "apikey", "access", "private", "client",
    "auth", "app", "session", "signing", "root",
}


def _is_secret_key(key: str) -> bool:
    tokens = [t for t in re.split(r"[^A-Za-z0-9]", key) if t]
    if not tokens:
        return False
    low = [t.lower() for t in tokens]
    last = low[-1]
    if last in _SECRET_WORDS:
        return True
    if last in ("key", "secret", "token") and len(low) >= 2 and low[-2] in _PREFIX_KEY_WORDS:
        return True
    return False

# Keys that identify a principal (paired with a secret for a full credential).

# Trailing components stripped to derive a pairing "namespace".
_USER_SUFFIXES = ("USERNAME", "USER", "LOGIN", "ACCOUNT")
_SECRET_SUFFIXES = (
    "PASSWORD", "PASSWD", "PWD", "PASSPHRASE", "PASS", "TOKEN", "SECRET",
    "APIKEY", "API_KEY", "APISECRET", "CLIENT_SECRET", "ACCESS_KEY",
    "SECRET_KEY", "PRIVATE_KEY",
)

# Generic `KEY = 'value'` / `KEY: "value"` / `'key' => "value"` assignment
# where KEY carries a secret hint. Value must be a quoted literal.
_ASSIGN = re.compile(
    r"""(?ix)
    (?:['"]?(?P<key>[A-Za-z_][A-Za-z0-9_]*(?:[.-][A-Za-z0-9_]+)*)['"]?\s*
        (?:=>|[:=])\s*)
    (?P<q>['"])(?P<val>(?:(?!\2)[^\\]|\\.){1,200})\2
    """
)
# PHP-style `define('DB_PASSWORD', 'x');`
_DEFINE = re.compile(
    r"""(?ix)
    define\s*\(\s*['"](?P<key>[A-Za-z_][A-Za-z0-9_]*)['"]\s*,\s*
    (?P<q>['"])(?P<val>(?:(?!\2).){1,200})\2
    """
)
# Seed code `password_hash('admin', PASSWORD_DEFAULT)` -> the literal is the
# plaintext default password.
_PASSWORD_HASH = re.compile(r"""password_hash\s*\(\s*['"](?P<val>[^'"]{1,100})['"]""", re.I)
# `'name' => 'admin'` (or login/username) literals, used to recover the account
# name sitting near a password_hash call.
_NAME_ASSIGN = re.compile(r"""['"](?:name|login|username)['"]\s*=>\s*['"]([^'"]+)['"]""", re.I)

# SQL statements carrying an explicit DB user + plaintext password.
_SQL_CREATE_USER = re.compile(
    r"""CREATE\s+USER(?:\s+IF\s+NOT\s+EXISTS)?\s+['"](?P<user>[^'"]+)['"]@[^\s]+
        \s+IDENTIFIED\s+BY\s+['"](?P<pass>[^'"]+)['"]""",
    re.I | re.X,
)
_SQL_SET_PASSWORD = re.compile(
    r"""SET\s+PASSWORD\s+FOR\s+['"](?P<user>[^'"]+)['"]@[^\s]+
        \s*=\s*(?:PASSWORD\()?['"](?P<pass>[^'"]+)['"]\)?""",
    re.I | re.X,
)
_SQL_GRANT = re.compile(
    r"""GRANT[^;]*?TO\s+['"](?P<user>[^'"]+)['"]@[^\s]+
        \s+IDENTIFIED\s+BY\s+['"](?P<pass>[^'"]+)['"]""",
    re.I | re.X,
)
_SQL_STMTS = (_SQL_CREATE_USER, _SQL_SET_PASSWORD, _SQL_GRANT)

# Rootfs paths likely to carry baked-in credentials.
_ARTIFACT_HINT = re.compile(
    r"(?i)(\.env$|/env$|docker-entrypoint|entrypoint|Dockerfile|compose|"
    r"\.sql$|supervisord|\.conf$|\.ini$|\.cnf$|\.properties$|initdb|"
    r"\.htpasswd|\.pem$|\.key$|secrets?|credentials?)"
)

_PLACEHOLDER = re.compile(
    r"^(?:\s*<.*>\s*|changeme|your[-_ ]?(?:password|secret|key|pass)|"
    r"example|xxxx+|\.\.\.+|todo|fixme|insert\s+your|none|null|"
    r"\$\{.*\}|%[^%]+%|\*+|unknown)$",
    re.I,
)
_HASHED_VALUE = re.compile(r"^(?:\$2[aby]\$\d{2}\$|sha1:|md5:|sha256:|{SHA}|\$1\$|\$5\$|\$6\$|pbkdf2:)")
_BOOLISH = {"true", "false", "yes", "no", "1", "0", "on", "off", "null", "none", "undefined", "nan", "inf"}

_MAX_CANDIDATES_TO_LLM = 150


# ---------------------------------------------------------------------------
# Candidate helpers
# ---------------------------------------------------------------------------

def _is_placeholder(value: str) -> bool:
    v = value.strip()
    if not v:
        return True
    if v.lower() in _BOOLISH:
        return True
    if len(v) > 200 or len(v) < 2:
        return True
    if _HASHED_VALUE.match(v):
        return True
    if _PLACEHOLDER.match(v):
        return True
    return False


def _namespace(key: str) -> str:
    """Strip a trailing user/secret suffix so ``APP_DB_USER`` and
    ``APP_DB_PASSWORD`` collapse to the same namespace ``APP_DB``."""
    upper = key.upper()
    for suffix in _SECRET_SUFFIXES + _USER_SUFFIXES:
        if upper.endswith(suffix) and len(upper) > len(suffix):
            return upper[: -len(suffix)].rstrip("_")
    return upper


def _candidates_from_pairs(
    source: str, key_values: list[tuple[str, str]], scope: Optional[str] = None
) -> list[dict]:
    """Turn ``(key, value)`` env-style pairs into raw candidates, pairing each
    secret with a sibling ``*_USER`` value in the same namespace."""
    users: dict[str, str] = {}
    secrets: dict[str, list[dict]] = {}

    def _entry(key: str, value: str) -> None:
        ns = _namespace(key)
        upper = key.upper()
        if any(upper.endswith(s) for s in _USER_SUFFIXES):
            users.setdefault(ns, value)
        elif any(upper.endswith(s) for s in _SECRET_SUFFIXES):
            secrets.setdefault(ns, []).append({"key": key, "value": value})

    for key, value in key_values:
        _entry(key, value)

    candidates = []
    for ns, items in secrets.items():
        username = users.get(ns)
        for item in items:
            if _is_placeholder(item["value"]):
                continue
            candidates.append({
                "source": source,
                "scope": scope,
                "key": item["key"],
                "username": username,
                "secret": item["value"],
            })
    return candidates


def _looks_literal(value: str) -> bool:
    """Only concrete literals count: reject shell/function syntax (``$(...)``,
    quotes, concat) that marks a value as code or substitution."""
    v = str(value)
    if len(v) < 2 or len(v) > 200:
        return False
    return not any(ch in v for ch in "'\"$()<>[]{}\\`")


def _add(candidates: list[dict], **fields: Any) -> None:
    secret = str(fields.get("secret") or "")
    if not _looks_literal(secret) or _is_placeholder(secret):
        return
    fields.setdefault("key", None)
    fields.setdefault("username", None)
    fields.setdefault("scope", None)
    candidates.append(fields)


# ---------------------------------------------------------------------------
# Text readers
# ---------------------------------------------------------------------------

def _iter_lines(path: Path) -> Optional[Iterator[str]]:
    try:
        return iter(path.open("r", encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as e:
        logger.debug("Skipping unreadable file %s: %s", path, e)
        return None


def _file_size_ok(path: Path) -> bool:
    try:
        max_bytes = getattr(settings, "credential_finder_max_file_bytes", 2 * 1024 * 1024)
        return path.stat().st_size <= max_bytes
    except OSError:
        return False


def _windows(path: Path, lineno: int) -> str:
    """A single display line: relative path, optional line number, trimmed."""
    rel = str(path.relative_to(settings.app_path)) if path.is_relative_to(settings.app_path) else str(path)
    return f"{rel}:{lineno}" if lineno else rel


# ---------------------------------------------------------------------------
# Env / compose / dockerfile collectors
# ---------------------------------------------------------------------------

def _env_file_pairs(path: Path) -> list[tuple[str, str]]:
    """Parse a dotenv-style file into ``(KEY, VALUE)`` pairs."""
    pairs = []
    for line in _iter_lines(path) or []:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = re.sub(r"^export\s+", "", key.strip())
        if not key:
            continue
        value = value.strip()
        # Inline comments only apply to unquoted values; a quoted value may
        # legitimately contain `#`.
        if value and value[0] not in ("'", '"'):
            value = re.sub(r"\s+#.*$", "", value).strip()
        elif len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        pairs.append((key, value))
    return pairs


def _collect_env_files() -> list[dict]:
    """Scan dotenv files under the repo for secret pairs."""
    candidates = []
    for path in settings.app_path.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if not name.endswith(".env") and ".env." not in name:
            continue
        if is_path_excluded(str(path)):
            continue
        if not _file_size_ok(path):
            continue
        pairs = _env_file_pairs(path)
        if not pairs:
            continue
        candidates.extend(_candidates_from_pairs(f"{path.relative_to(settings.app_path)}", pairs))
    return candidates


def _resolve_compose_value(value: Any, env: dict[str, str]) -> Any:
    if isinstance(value, str):
        return re.sub(
            r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}",
            lambda m: env.get(m.group("name").upper(), m.group("default") or ""),
            value,
        )
    return value


def _collect_compose() -> list[dict]:
    """Parse compose files: ``services.*.environment`` + ``env_file``. Every
    compose file present is scanned so overrides are not missed."""
    candidates = []
    env = {}
    for env_path in sorted(settings.app_path.rglob(".env")):
        if env_path.is_file() and not is_path_excluded(str(env_path)):
            env.update({k.upper(): v for k, v in _env_file_pairs(env_path)})

    for path in sorted(settings.app_path.rglob("*")):
        if not path.is_file() or path.name not in COMPOSE_FILENAMES:
            continue
        if is_path_excluded(str(path)):
            continue
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        for service, conf in (doc.get("services") or {}).items():
            if not isinstance(conf, dict):
                continue
            env_pairs: list[tuple[str, str]] = []
            local_env = dict(env)
            for ef in conf.get("env_file") or []:
                ef_path = (path.parent / str(ef)).resolve() if isinstance(ef, str) else None
                if ef_path and ef_path.is_file() and ef_path.is_relative_to(settings.app_path):
                    local_env.update({k.upper(): v for k, v in _env_file_pairs(ef_path)})
            raw_env = conf.get("environment") or {}
            if isinstance(raw_env, list):
                for item in raw_env:
                    if isinstance(item, str) and "=" in item:
                        k, _, v = item.partition("=")
                        env_pairs.append((k.strip(), v.strip()))
            elif isinstance(raw_env, dict):
                for k, v in raw_env.items():
                    env_pairs.append((str(k), str(v)))
            resolved = [(k, str(_resolve_compose_value(v, local_env))) for k, v in env_pairs]
            if not resolved:
                continue
            source = f"{path.name} service '{service}'"
            candidates.extend(_candidates_from_pairs(source, resolved, scope=service))
    return candidates


# Dockerfile `ENV KEY=value` / `ENV KEY value` / `ARG KEY=value`, where the
# value may be a quoted string containing spaces.
_DOCKER_ENV = re.compile(
    r"""(?ix)
    ^\s*(ENV|ARG)\s+(?P<key>[A-Za-z_][A-Za-z0-9_]*)
    \s*(?:=\s*|\s+)
    (?P<val>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s#]+)
    """
)


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        quote = value[0]
        inner = value[1:-1]
        return inner.replace("\\" + quote, quote)
    return value


def _collect_dockerfiles() -> list[dict]:
    """Scan Dockerfiles for `ENV KEY=value` / `ARG KEY=value` secrets."""
    candidates = []
    for path in settings.app_path.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        is_dockerfile = (
            name == "Dockerfile"
            or name.startswith("Dockerfile.")
            or name.endswith(".dockerfile")
            or name == "Containerfile"
        )
        if not is_dockerfile or is_path_excluded(str(path)):
            continue
        if not _file_size_ok(path):
            continue
        for lineno, raw in enumerate(_iter_lines(path) or [], 1):
            line = raw.rstrip("\n")
            for match in _DOCKER_ENV.finditer(line):
                key, val = match.group("key"), match.group("val")
                if not _is_secret_key(key):
                    continue
                _add(
                    candidates,
                    source=_windows(path, lineno),
                    key=key,
                    username=None,
                    secret=_unquote(val),
                    scope="dockerfile",
                )
    return candidates


# ---------------------------------------------------------------------------
# SQL collector
# ---------------------------------------------------------------------------

def _sql_candidates_from_statement(stmt: str, source: str) -> list[dict]:
    found = []
    for pat in _SQL_STMTS:
        for m in pat.finditer(stmt):
            found.append({"user": m.group("user"), "pass": m.group("pass"), "source": source})
    return found


def _collect_sql(path: Path, allow_excluded: bool = False) -> list[dict]:
    """Buffer SQL statements and extract `CREATE USER ... IDENTIFIED BY` etc.
    ``allow_excluded`` bypasses the path-exclusion check for already-allowlisted
    files (e.g. ``initdb.sql`` extracted inside the container-artifact snapshot,
    which lives under the excluded ``.cache`` tree)."""
    candidates = []
    if not path.is_file() or not _file_size_ok(path):
        return candidates
    if not allow_excluded and is_path_excluded(str(path)):
        return candidates
    buffer = ""
    source = str(path.relative_to(settings.app_path))
    for lineno, raw in enumerate(_iter_lines(path) or [], 1):
        line = raw.rstrip("\n")
        buffer += line
        while ";" in buffer:
            head, _, rest = buffer.partition(";")
            stmt = re.sub(r"(?m)^\s*(?:--|#).*$", "", head)
            for hit in _sql_candidates_from_statement(stmt, source):
                if not _is_placeholder(hit["pass"]):
                    _add(
                        candidates,
                        source=f"{source}:{lineno}",
                        key=None,
                        username=hit["user"],
                        secret=hit["pass"],
                        scope="sql",
                    )
            buffer = rest
    return candidates


# ---------------------------------------------------------------------------
# Source collector
# ---------------------------------------------------------------------------

def _norm_name(value: str) -> str:
    """Normalize an account identifier for pairing: lowercase, drop separators
    so `post-only` and `postonly` compare equal."""
    return re.sub(r"[^a-zA-Z0-9]", "", value).lower()


def _collect_source() -> list[dict]:
    """Line-oriented scan of application source/seed files for literal
    credential assignments, password_hash seeds, and SQL user statements."""
    candidates = []
    scanned = 0
    budget = getattr(settings, "credential_finder_max_scan_bytes", 64 * 1024 * 1024)

    for path in settings.app_path.rglob("*"):
        if not path.is_file():
            continue
        if scanned >= budget:
            break
        if path.name in utils.MANIFEST_NAMES:
            continue
        if is_path_excluded(str(path)):
            continue
        if not _file_size_ok(path):
            continue
        suffix = path.suffix.lower()
        if suffix == ".sql":
            candidates.extend(_collect_sql(path))
            try:
                scanned += path.stat().st_size
            except OSError:
                pass
            continue
        lines_iter = _iter_lines(path)
        if lines_iter is None:
            continue
        # (raw name, normalized name) literals seen in the file, used to pair a
        # `password_hash('<lit>')` seed with the account name near it.
        recent_names: list[tuple[str, str]] = []
        window: list[str] = []
        try:
            for lineno, raw in enumerate(lines_iter, 1):
                scanned += len(raw)
                line = raw.rstrip("\n")
                window.append(line)
                if len(window) > 24:
                    window.pop(0)
                for m in _NAME_ASSIGN.finditer(line):
                    recent_names.append((m.group(1), _norm_name(m.group(1))))
                    if len(recent_names) > 64:
                        recent_names.pop(0)
                for m in _PASSWORD_HASH.finditer(line):
                    literal = m.group("val")
                    username = literal if literal in {r for r, _ in recent_names} else None
                    if username is None:
                        norm = _norm_name(literal)
                        for raw_name, raw_norm in recent_names:
                            if norm and raw_norm == norm:
                                username = raw_name
                                break
                    _add(
                        candidates,
                        source=_windows(path, lineno),
                        key=None,
                        username=username,
                        secret=literal,
                        scope="seed",
                    )
                joined = " ".join(window[-3:])  # small lookbehind for define(...) spans
                for m in list(_DEFINE.finditer(joined)):
                    if not _is_secret_key(m.group("key")):
                        continue
                    _add(
                        candidates,
                        source=_windows(path, lineno),
                        key=m.group("key"),
                        username=None,
                        secret=m.group("val"),
                    )
                for m in _ASSIGN.finditer(line):
                    key = m.group("key")
                    if not _is_secret_key(key):
                        continue
                    val = m.group("val")
                    if not key.isupper() and "." in key:
                        continue
                    _add(
                        candidates,
                        source=_windows(path, lineno),
                        key=key,
                        username=None,
                        secret=val,
                    )
        except (UnicodeDecodeError, OSError):
            continue
    return candidates


# ---------------------------------------------------------------------------
# Container artifact collector
# ---------------------------------------------------------------------------

def _collect_container_artifacts() -> list[dict]:
    """Read baked-in ENV vars and credential-bearing files from the built-image
    snapshots the preprocessor wrote under `.cache/container_artifacts/`."""
    candidates = []
    root = get_container_artifacts_root()
    if not root.is_dir():
        return candidates
    for image_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        slug = image_dir.name
        meta = image_dir / "image_metadata.json"
        if meta.is_file():
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            env_pairs = []
            for pair in data.get("Env") or []:
                if isinstance(pair, str) and "=" in pair:
                    k, _, v = pair.partition("=")
                    env_pairs.append((k, v))
            if env_pairs:
                candidates.extend(
                    _candidates_from_pairs(
                        f"image:{slug}/Env", env_pairs, scope=f"image {slug}"
                    )
                )
        rootfs = image_dir / "rootfs"
        if not rootfs.is_dir():
            continue
        for file_path in rootfs.rglob("*"):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(rootfs).as_posix()
            if not _ARTIFACT_HINT.search(rel):
                continue
            if not _file_size_ok(file_path):
                continue
            if rel.endswith(".sql"):
                candidates.extend(_collect_sql(file_path, allow_excluded=True))
                continue
            source = f"image:{slug}/rootfs/{rel}"
            for lineno, raw in enumerate(_iter_lines(file_path) or [], 1):
                line = raw.rstrip("\n")
                for m in _ASSIGN.finditer(line):
                    if not _is_secret_key(m.group("key")):
                        continue
                    _add(
                        candidates,
                        source=f"{source}:{lineno}",
                        key=m.group("key"),
                        username=None,
                        secret=m.group("val"),
                    )
    return candidates


# ---------------------------------------------------------------------------
# Aggregation / dedup / output
# ---------------------------------------------------------------------------

def _dedupe(candidates: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for c in sorted(candidates, key=lambda c: (c.get("source") or "", c.get("key") or "", c.get("username") or "", c.get("secret") or "")):
        key = (c.get("username"), c.get("secret"))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _raw_to_record(c: dict) -> dict:
    username = c.get("username")
    key = c.get("key") or ""
    if key and re.search(r"(?i)api[_-]?key|token", key):
        kind = "api_key"
    elif username:
        kind = "login"
    else:
        kind = "secret"
    service = key or username or "credential"
    notes = f"found in {c.get('source')}" + (f" ({c.get('scope')})" if c.get("scope") else "")
    return {
        "service": service,
        "kind": kind,
        "username": username,
        "secret": c.get("secret"),
        "source": c.get("source"),
        "notes": notes,
    }


def _fallback_records(candidates: list[dict]) -> list[dict]:
    return [_raw_to_record(c) for c in candidates]


def _render_candidates(candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates[: _MAX_CANDIDATES_TO_LLM], 1):
        parts = [
            f"key={c.get('key') or '-'}",
            f"username={c.get('username') or '-'}",
            f"secret={c.get('secret')}",
            f"source={c.get('source') or '-'}",
            f"scope={c.get('scope') or '-'}",
        ]
        lines.append(f"{i}. " + " | ".join(parts))
    return "\n".join(lines)


def _llm_normalize(candidates: list[dict]) -> tuple[Optional[list[dict]], Optional[dict]]:
    """One structured LLM call to label/dedupe raw candidates into records.
    Returns ``(records_or_None, token_usage)``; None records (and a log) on any
    failure so the caller fails open. The usage is booked to the run ledger by
    invoke_tracked; it is returned so the caller can persist it in the cache."""
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        from schemas import CREDENTIAL_FINDER_AGENT, CredentialList

        sys_msg = SystemMessage(content=CREDENTIAL_FINDER_AGENT.get("prompt", ""))
        human_msg = HumanMessage(content=(
            f"Target application: {settings.app_path.name}\n\n"
            f"RAW CREDENTIAL CANDIDATES ({len(candidates)} total, up to "
            f"{_MAX_CANDIDATES_TO_LLM} shown):\n{_render_candidates(candidates)}"
        ))
        structured = get_llm("credential_finder").with_structured_output(CredentialList, method="json_schema", strict=True)
        result, usage = invoke_tracked(structured, [sys_msg, human_msg], "credential_finder")
        result = result if isinstance(result, dict) else result.model_dump()
        records = []
        for item in result.get("credentials") or []:
            item = item if isinstance(item, dict) else item.model_dump()
            if not str(item.get("secret") or "").strip():
                continue
            records.append({
                "service": str(item.get("service") or "credential").strip(),
                "kind": item.get("kind", "secret"),
                "username": item.get("username"),
                "secret": str(item["secret"]).strip(),
                "source": item.get("source"),
                "notes": item.get("notes"),
            })
        return records, usage
    except Exception as e:
        logger.warning("Credential finder LLM pass failed; using raw candidates: %s", e)
        return None, None


def _write_credentials(records: list[dict]) -> None:
    target = settings.cache_dir / "credentials.json"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(records, indent=2), encoding="utf-8")
        logger.info("Credential finder: wrote %d credential(s) to %s.", len(records), target)
    except OSError as e:
        logger.error("Credential finder: failed to write %s: %s", target, e)


def load_credentials() -> list[dict]:
    """Read persisted credential records (``service``, ``kind``, ``username``,
    ``secret``, ``source``, ``notes``) with a non-empty secret; missing or
    corrupt file yields ``[]`` (fail open)."""
    target = settings.cache_dir / "credentials.json"
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [
        r for r in data
        if isinstance(r, dict) and str(r.get("secret") or "").strip()
    ]


def authentication_block() -> str:
    """Render the validator prompt's ``TARGET AUTHENTICATION`` block.

    Returns an empty string when no credentials are available (finder disabled,
    found nothing, or the file is missing/corrupt). Otherwise renders one bullet
    per credential: ``username``/``password`` when an account name is known,
    else the bare secret (api key / session cookie / db password), each with the
    record's ``notes`` appended when present.
    """
    records = load_credentials()
    if not records:
        return ""

    lines = []
    for record in records:
        service = str(
            record.get("service") or record.get("username") or "credential"
        ).strip()
        username = str(record.get("username") or "").strip()
        secret = str(record.get("secret") or "").strip()
        notes = str(record.get("notes") or "").strip()

        if username:
            line = f'- {service}: username="{username}", password="{secret}"'
        else:
            line = f'- {service}: "{secret}"'
        if notes:
            line += f"  — {notes}"
        lines.append(line)

    return (
        "TARGET AUTHENTICATION:\n"
        "Use the following sandbox credentials/sessions when authenticated "
        "access is required:\n"
        + "\n".join(lines)
    )


def _llm_cache(candidates: list[dict]) -> tuple[Optional[list[dict]], bool]:
    """Read (or write) the LLM normalization cache keyed on the raw
    candidates. Returns ``(records, was_cached)``."""
    digest = hashlib.md5(
        json.dumps(candidates, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache_file = settings.cache_dir / "credential_finder" / safe_cache_filename(f"{digest}.json")
    cached = utils.cache(cache_file, "read")
    if cached and isinstance(cached.get("credentials"), list):
        from run_stats import take_cached_usage
        take_cached_usage("credential_finder", cached)
        return cached["credentials"], True
    records, usage = _llm_normalize(candidates)
    if records is None:
        return None, False
    utils.cache(cache_file, "write", {"credentials": records, "token_usage": usage})
    return records, False


def credential_finder_node(state) -> dict:
    """Graph node: discover pre-configured credentials and persist them to
    ``<target_app>/.cache/credentials.json``. Runs after container setup.
    Returns {} (no state change)."""
    from run_stats import raise_if_stopping
    raise_if_stopping()
    if not getattr(settings, "credential_finder_enabled", True):
        logger.info("Credential finder disabled via settings.credential_finder_enabled=False.")
        return {}

    collected: dict[str, list[dict]] = {
        "env": _collect_env_files(),
        "compose": _collect_compose(),
        "dockerfile": _collect_dockerfiles(),
        "source": _collect_source(),
        "artifacts": _collect_container_artifacts(),
    }
    candidates = _dedupe([c for group in collected.values() for c in group])
    logger.info(
        "Credential finder: collected %d raw candidate(s) after dedup "
        "(env=%d, compose=%d, dockerfile=%d, source/sql=%d, artifacts=%d).",
        len(candidates),
        len(collected["env"]),
        len(collected["compose"]),
        len(collected["dockerfile"]),
        len(collected["source"]),
        len(collected["artifacts"]),
    )

    records = None
    use_llm = getattr(settings, "credential_finder_use_llm", True)
    if use_llm and candidates:
        records, cached = _llm_cache(candidates)
        if cached:
            logger.info("Credential finder: reused cached LLM normalization (%d records).", len(records or []))
    if records is None:
        records = _fallback_records(candidates)

    _write_credentials(records)
    return {}
