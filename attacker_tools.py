"""Attacker-container toolset for the Validator agent.

Gives the validator real command-line capability by provisioning a dedicated
Kali attacker container (``kalilinux/kali-rolling`` + ``kali-linux-headless``)
exposed via bounded shell/file tools that execute INSIDE it - so the agent can
run ``curl``/``nmap``/``sqlmap``/custom PoCs, draft a PoC script with a file
tool, run it, and read the evidence back for ``poc_payload``/``execution_logs``.

Design / lifecycle
------------------
- LAZY provisioning: importing this module never touches docker. Nothing is
  pulled, built, or started unless a validator actually calls a shell tool for
  the first time. The shared image is built once (the Dockerfile content is
  md5-stamped in ``attacker/.build.hash``, mirroring utils.build_images) and
  the container is started on the default bridge network under a pinned name.
  A stale container of the same name is replaced; after that the running box is
  reused without any build cost.
- BRIDGE + TARGET: the sandbox is published on all host interfaces and is
  reachable from both the host and the attacker container at the docker bridge
  gateway (e.g. 172.17.0.1). The preprocessor sets ``sandbox_url`` to that gateway
  URL for every validator path, so ``run_command`` shares one target address with
  the HTTP/browser tools, and every ``run_command`` result echoes it as a
  ``SHELL TARGET`` header.
- FAIL OPEN: if docker is absent, the daemon is down, the image cannot be
  built, or the container cannot start, the tools report "attacker container
  unavailable" and the validator continues with its HTTP/browser tools.
- The docker CLI is shelled out via ``subprocess`` (bounded by explicit
  timeouts), matching the sandbox tooling in utils.py. Thread-safe: the lazy
  boot is guarded by a lock so concurrent validators never double-build.
"""

import hashlib
import logging
import os
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

import settings

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent
ATTACKER_DIR = _REPO_ROOT / "attacker"
ATTACKER_DOCKERFILE = ATTACKER_DIR / "Dockerfile"
# Caps on reading attacker files back into the LLM.
ATTACKER_FILEREAD_MAX_LINES = 150
ATTACKER_FILEREAD_MAX_BYTES = 1_000_000


class AttackerManager:
    """Thread-safe owner of the shared attacker container (lazy boot)."""

    def __init__(self):
        self._container_name = None
        self._disabled = False
        self._disabled_reason = None
        self._boot_lock = threading.Lock()

    # -- availability ------------------------------------------------------

    def _blocked(self) -> bool:
        return self._disabled or not getattr(settings, "attacker_enabled", True)

    @staticmethod
    def _docker_available() -> bool:
        try:
            subprocess.run(
                ["docker", "version"], capture_output=True, text=True, timeout=10
            )
            return True
        except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
            return False

    def unavailable_msg(self) -> str:
        reason = self._disabled_reason or (
            "docker is unavailable or the attacker image could not be built/started"
        )
        return (
            "Error: The attacker container is unavailable in this environment "
            f"({reason}). The HTTP/browser validation tools remain available."
        )

    # -- lazy provisioning ---------------------------------------------------

    def ensure(self) -> str | None:
        """Ensure the attacker container exists and is running; return its name.

        Idempotent and thread-safe. Returns the pinned container name or None
        (fail open) with ``_disabled``/``_disabled_reason`` set. Only the first
        caller performs the build/start work.
        """
        if self._blocked():
            return None
        # Fast path: already provisioned and running.
        if self._container_name is not None and self._running(self._container_name):
            return self._container_name

        with self._boot_lock:
            if self._disabled:
                return None
            if self._container_name is not None and self._running(self._container_name):
                return self._container_name

            if not self._docker_available():
                self._disable("docker is not installed or the daemon is not running")
                return None

            image = getattr(settings, "attacker_image_tag", "vulnscan-kali-attacker:latest")
            name = getattr(settings, "attacker_container_name", "vulnscan-kali-attacker")

            if not self._build_image(image):
                self._disable(f"image '{image}' could not be built")
                return None
            if not self._start_container(image, name):
                self._disable(f"container '{name}' could not be started")
                return None

            self._container_name = name
            return name

    def _disable(self, reason: str):
        self._disabled = True
        self._disabled_reason = reason
        logger.warning("Attacker container unavailable: %s", reason)

    @staticmethod
    def _running(name: str) -> bool:
        try:
            r = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return r.returncode == 0 and r.stdout.strip() == "true"
        except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
            return False

    # -- image build (hash-cached, like utils.build_images) -------------------

    @staticmethod
    def _image_exists(image: str) -> bool:
        try:
            r = subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return r.returncode == 0
        except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
            return False

    def _dockerfile_hash(self) -> str:
        digest = hashlib.md5()
        try:
            if ATTACKER_DOCKERFILE.is_file():
                digest.update(ATTACKER_DOCKERFILE.read_bytes())
        except OSError:
            pass
        return digest.hexdigest()

    def _build_image(self, image: str) -> bool:
        digest = self._dockerfile_hash()
        stamp = ATTACKER_DIR / ".build.hash"
        if (
            not getattr(settings, "force_rebuild", False)
            and self._image_exists(image)
            and stamp.is_file()
            and stamp.read_text().strip() == digest
        ):
            logger.info("Reusing attacker image %s (Dockerfile unchanged).", image)
            return True

        try:
            build = subprocess.run(
                ["docker", "build", "-t", image, "-f", str(ATTACKER_DOCKERFILE), str(ATTACKER_DIR)],
                capture_output=True,
                text=True,
                timeout=int(getattr(settings, "attacker_build_timeout", 3600)),
            )
        except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
            logger.warning("Attacker image build failed: %s", e)
            return False
        if build.returncode != 0:
            logger.warning("Attacker image build failed: %s", build.stderr[-2000:])
            return False
        try:
            stamp.write_text(digest)
        except OSError:
            pass
        return True

    # -- container start + gateway discovery -----------------------------------

    @staticmethod
    def _start_container(image: str, name: str) -> bool:
        # Replace any stale container holding the pinned name.
        try:
            subprocess.run(
                ["docker", "rm", "-f", name], capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
            pass
        try:
            run = subprocess.run(
                [
                    "docker", "run", "-d", "--name", name,
                    # Full capability set: on SELinux-enforcing hosts Docker's
                    # default caps make nmap's kernel-level scans fail to even
                    # exec ("Operation not permitted"), and many pentest tools
                    # (raw sockets, pcap, ptrace) need more than the defaults.
                    # The container stays namespaced and unprivileged - ALL
                    # applies inside its own namespaces only.
                    "--cap-add", "ALL",
                    image, "sleep", "infinity",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
            logger.warning("Attacker container start failed: %s", e)
            return False
        if run.returncode != 0:
            logger.warning("Attacker container start failed: %s", run.stderr.strip())
            return False
        logger.info("Started attacker container '%s' from image %s.", name, image)
        return True


manager = AttackerManager()


# -- executor -------------------------------------------------------------------

def _contains_workdir(path: str) -> Path | None:
    """Resolve an attacker-side path and confine it to the workdir tree."""
    base = Path(getattr(settings, "attacker_workdir", "/work"))
    raw = Path(path)
    candidate = raw if raw.is_absolute() else base / raw
    normalized = Path(os.path.normpath(str(candidate)))
    if os.path.abspath(str(normalized)) != str(normalized) or normalized == base:
        # Reject escapes (..) and the workdir root itself.
        return None
    if str(base) == os.path.commonpath([str(base), str(normalized)]):
        return normalized
    return None


def _exec_input(name: str, argv: list[str], *, input_bytes: bytes | None = None,
                timeout: int, max_chars: int) -> tuple[int, str, bool]:
    """Run ``docker exec`` in the attacker container; return (exit_code, out, truncated)."""
    try:
        result = subprocess.run(
            argv,
            input=input_bytes,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return (
            124,
            f"[COMMAND TIMED OUT after {timeout}s; the process was killed.]",
            False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return 255, f"[docker exec failed: {e}]", False

    def _decode(data) -> str:
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return ""

    stdout = _decode(result.stdout)
    stderr = _decode(result.stderr)
    out = stdout
    if stderr:
        out += f"\n--- STDERR ---\n{stderr}"
    truncated = False
    if len(out) > max_chars:
        out = out[:max_chars]
        out += "\n\n... [OUTPUT TRUNCATED at {} chars] ...".format(max_chars)
        truncated = True
    return result.returncode, out, truncated


@tool
def run_command(
    command: str,
    state: Annotated[dict, InjectedState],
    workdir: str = "/work",
    timeout: int | None = None,
) -> str:
    """
    Runs a shell command INSIDE the dedicated Kali attacker container
    (kalilinux/kali-rolling with the kali-linux-headless tool suite installed:
    nmap, sqlmap, curl, netcat, Burp-light tooling, password/exploit helpers,
    etc.). Use this to actively probe the sandbox with real tools, run or debug
    a PoC, or retrieve evidence that HTTP/browser tools cannot (e.g. raw TCP,
    TLS fingerprinting, payload fuzzing).

    The sandbox application is published on all host interfaces and is reachable
    from inside the attacker container through the IP address of the ``sandbox_url``

    Args:
        command (str): The shell command to run, e.g.
            "curl -sS -v http://<SHELL TARGET>/api/export?title=OR+1=1" or
            "nmap -sV -p- <SHELL TARGET host>".
        workdir (str): Directory to run the command in inside the container.
            Defaults to /work
        timeout (int): Max seconds for this command (default and hard cap come
            from settings.attacker_command_timeout).
    """
    name = manager.ensure()
    if not name:
        return manager.unavailable_msg()

    eff_timeout = int(timeout or getattr(settings, "attacker_command_timeout", 60))
    eff_timeout = min(eff_timeout, int(getattr(settings, "attacker_command_timeout", 60)))
    max_chars = int(getattr(settings, "attacker_output_max_chars", 8000))

    cwd = (workdir or str(Path(getattr(settings, "attacker_workdir", "/work")))).strip()
    if not cwd.startswith("/"):
        cwd = f"/{cwd}"

    argv = ["docker", "exec", "-w", cwd, name, "/bin/bash", "-lc", command]
    exit_code, out, truncated = _exec_input(
        name, argv, timeout=eff_timeout, max_chars=max_chars
    )
    return (
        f"ATTACKER SHELL (workdir {cwd}, exit code: {exit_code}{', output truncated' if truncated else ''})\n"
        f"{out}"
    )


@tool
def write_attacker_file(file_path: str, content: str, mode: str = "0600") -> str:
    """
    Writes text into a file INSIDE the Kali attacker container, confined to the
    attacker workdir. Use this to draft a PoC script (e.g. main.py) before
    running it with run_command, or to save payload lists for fuzzing. The file
    can be read back with read_attacker_file.

    Args:
        file_path (str): Path relative to the attacker workdir (e.g.
            'pocs/exploit.py'; subdirectories are created automatically).
        content (str): The full text to write (UTF-8).
        mode (str): Octal permission bits, e.g. '0600' or '0755' for scripts.
    """
    name = manager.ensure()
    if not name:
        return manager.unavailable_msg()
    target = _contains_workdir(file_path)
    if target is None:
        return (
            f"Error: '{file_path}' must resolve to a path inside the attacker "
            "workdir (/work). Absolute and '..' escape paths are not allowed."
        )
    target_s = str(target)
    mkdir_cmd = f"mkdir -p {shlex.quote(str(Path(target_s).parent))}"
    _exec_input(
        name,
        ["docker", "exec", name, "/bin/bash", "-lc", mkdir_cmd],
        timeout=30,
        max_chars=2000,
    )
    write_cmd = (
        f"cat > {shlex.quote(target_s)} && "
        f"chmod {mode} {shlex.quote(target_s)}"
    )
    exit_code, out, truncated = _exec_input(
        name,
        ["docker", "exec", "-i", name, "/bin/bash", "-lc", write_cmd],
        input_bytes=content.encode("utf-8", errors="replace"),
        timeout=30,
        max_chars=2000,
    )
    if exit_code != 0:
        return f"Error writing '{file_path}':\n{out}"
    return f"Wrote {len(content.encode('utf-8'))} bytes to attacker file '{target_s}' (mode {mode})."


@tool
def read_attacker_file(
    file_path: str, start_line: int = 1, end_line: int | None = None
) -> str:
    """
    Reads a line range of a file inside the Kali attacker container, confined
    to the attacker workdir and capped like the source reader at 150 lines per
    call. Use this to inspect a saved PoC script or read back captured evidence
    that run_command produced (e.g. a written request/response file) to include
    in poc_payload / execution_logs.

    Args:
        file_path (str): Path relative to the attacker workdir (e.g.
            'pocs/exploit.py').
        start_line (int): First line to read, 1-indexed and inclusive.
        end_line (int): Last line to read, 1-indexed and inclusive. Defaults to
            the end of the file (or the 150-line cap).
    """
    name = manager.ensure()
    if not name:
        return manager.unavailable_msg()
    target = _contains_workdir(file_path)
    if target is None:
        return (
            f"Error: '{file_path}' must resolve to a path inside the attacker "
            "workdir (/work). Absolute and '..' escape paths are not allowed."
        )
    target_s = str(target)
    try:
        result = subprocess.run(
            ["docker", "exec", name, "cat", target_s],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
        return f"Error reading attacker file '{file_path}': {e}"
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        return f"Error: cannot read '{file_path}' inside the attacker container: {err}"
    text = result.stdout.decode("utf-8", errors="replace")
    if len(text) > ATTACKER_FILEREAD_MAX_BYTES:
        return (
            f"Error: '{file_path}' is too large to page "
            f"(> {ATTACKER_FILEREAD_MAX_BYTES} bytes); use run_command "
            "with head/tail/grep instead."
        )
    lines = text.splitlines()
    total = len(lines)
    if total == 0:
        return f"Attacker file '{file_path}' is empty (0 lines)."

    requested_start = start_line
    if start_line < 1:
        start_line = 1
    if start_line > total:
        return (
            f"Error: start_line {requested_start} is beyond the end of "
            f"'{file_path}' (file has {total} lines)."
        )
    requested_end = end_line if end_line is not None else total
    if requested_end < start_line:
        return (
            f"Error: end_line ({requested_end}) is smaller than start_line "
            f"({start_line})."
        )
    end = min(requested_end, total)
    truncated = False
    if end - start_line + 1 > ATTACKER_FILEREAD_MAX_LINES:
        end = start_line + ATTACKER_FILEREAD_MAX_LINES - 1
        truncated = True

    body = "".join(
        f"{i:>6}: {line}" for i, line in enumerate(lines[start_line - 1:end], start_line)
    )
    header = f"Attacker file: {target_s} (lines {start_line}-{end} of {total})\n"
    if truncated:
        body += (
            f"\n... [TRUNCATED: requested lines {requested_start}-{requested_end} "
            f"exceeds the {ATTACKER_FILEREAD_MAX_LINES}-line limit. Showed lines "
            f"{start_line}-{end}. Call read_attacker_file again with "
            f"start_line={end + 1} to continue.] ..."
        )
    return f"{header}\n{body}"


# -- host-facing helper for tools.send_http_request --------------------------------
# send_http_request runs on the host, but files the attacker references are written
# INSIDE the attacker container under the workdir. This lets the HTTP tool read
# those bytes out of the container so a PoC payload can be uploaded via
# files=... even though the request itself is issued from the host.

def read_attacker_file_bytes(file_path: str) -> bytes | None:
    """Read a whole file from the attacker container as raw bytes, or None.

    Only paths confined to the attacker workdir are considered (non-workdir paths
    never touch docker and keep the lazy-provisioning contract). Returns None when
    the attacker container is unavailable or the file cannot be read.
    """
    target = _contains_workdir(file_path)
    if target is None:
        return None
    name = manager.ensure()
    if not name:
        return None
    try:
        result = subprocess.run(
            ["docker", "exec", name, "cat", str(target)],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout
