from pathlib import Path
import logging
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

_REQUIRED_KEYS = ("OPENAI_API_KEY", "TAVILY_API_KEY", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")
_missing = [k for k in _REQUIRED_KEYS if not os.environ.get(k)]
if _missing:
    logging.getLogger(__name__).warning(
        ".env loaded but missing required keys: %s", ", ".join(_missing)
    )


app_path = Path("../apps/glpi-11.0.7-clean")

# LLM provider.
# "openai" = ChatOpenAI against the internal gateway (needs OPENAI_API_KEY).
# "ollama" = ChatOllama against a local Ollama server.
llm_provider = "openai"
openai_base_url = "http://localhost:11434/v1"
openai_model = "deepseek-v4-flash"
ollama_model = "gemma4:cloud"
ollama_base_url = "http://localhost:11434"

# Concurrency
simple_agents_concurrency = 4

# Tool-loop guards for the compiled reviewer/validator subgraphs. If the model
# never calls submit_evaluation / mark_validation_complete within this many LLM
# rounds, the loop terminates gracefully via the fallback node instead of
# crashing on the LangGraph recursion limit.
reviewer_max_iterations = 12
# First reviewer LLM turn at which the countdown note ("submit_evaluation in
# your next turn or be terminated") is injected; turns >= this get a fresh note
# each round. Derived so there are always COUNTDOWN_LEAD_TURNS of cushion
# before the iterations cap.
COUNTDOWN_LEAD_TURNS = 4
reviewer_countdown_start = max(1, reviewer_max_iterations - COUNTDOWN_LEAD_TURNS)
validator_max_iterations = 150
# Same derivation for the validator track ("mark_validation_complete in your
# next turn or be terminated").
validator_countdown_start = max(1, validator_max_iterations - COUNTDOWN_LEAD_TURNS)
# How many times the Validator may request more context (insufficient_context)
# from the Reviewer for a single record. Past this cap the Validator must
# conclude with the evidence it has.
validator_feedback_max_rounds = 1

# Reviewer context compaction (opencode-style). When the estimated token count
# of the reviewer's message history reaches model_context_window minus
# context_reserved, the middle of the conversation is collapsed into a prior
# LLM-generated summary and the most recent verbatim tail is preserved. Token
# estimates use the conservative ~2 chars/token heuristic
# (utils.estimate_message_tokens), so the estimate intentionally exceeds the
# model's real token count for code-heavy tool histories.
reviewer_model_context_window = 131072
reviewer_context_reserved = 24000
# Number of most-recent AI+tool interaction turns kept verbatim after compaction.
reviewer_compaction_tail_turns = 1
# Do not compact unless the compressible middle is worth at least this many
# estimated tokens (avoids thrashing on tiny histories).
reviewer_compaction_min_compressible_tokens = 4000
# Hard safety margin: if the estimated token count still approaches the model
# window even after the soft-threshold compaction was skipped, the reviewer
# node force-truncates before invoking the LLM so it can never overflow the
# model's maximum context length.
reviewer_hard_reserved = 8192

# Headless-browser toolset (browser_tools.py) for the validator. One shared
# Firefox process serves all concurrent validators; each session_id gets an
# isolated BrowserContext. browser_executable = None uses Playwright's own
# patched Firefox channel build; set it to a path to point at a custom build.
browser_enabled = True
browser_timeout_ms = 30000
# Bounded per-session console/pageerror/dialog ring buffer: max messages kept,
# and per-message char cap, and the visible-text cap for browser_navigate output.
browser_console_max_messages = 60
browser_console_msg_chars = 500
browser_describe_max_chars = 6000
browser_executable = None
# Idle-TTL safety-net reaper for browsers sessions not closed by the terminal
# tool (grace-perioded so in-use sessions are never reaped).
browser_idle_timeout_sec = 600

# Kali attacker container (attacker_tools.py) for the validator: run_command /
# write_attacker_file / read_attacker_file execute inside a
# kalilinux/kali-rolling + kali-linux-headless box started lazily on the default
# bridge network. The sandbox is then reachable from it via the bridge gateway
# (run_command prints the rewritten SHELL TARGET). Falls open (HTTP-only
# validation) if docker/build/container startup fails.
attacker_enabled = True
attacker_image_tag = "vulnscan-kali-attacker:latest"
attacker_container_name = "vulnscan-kali-attacker"
# Working directory for validator shell activity; file tools confine reads and
# writes to this tree so PoC scripts and evidence stay predictable.
attacker_workdir = "/work"
# Hard cap (and default) for any single run_command; commands exceeding it are
# killed. Building kali-linux-headless on first run can take a long time, so
# the image build has its own generous timeout.
attacker_command_timeout = 60
attacker_build_timeout = 3600
attacker_output_max_chars = 8000

# Validator context compaction: same mechanism as the reviewer, applied to the
# validator's HTTP-proving loop. HTTP responses from send_http_request can grow
# without bound over long validation sessions, so the same soft-threshold
# compaction plus hard safety cap keep the history under the model window.
validator_model_context_window = 131072
validator_context_reserved = 24000
# Number of most-recent AI+tool interaction turns kept verbatim after compaction.
validator_compaction_tail_turns = 1
# Do not compact unless the compressible middle is worth at least this many
# estimated tokens (avoids thrashing on tiny histories).
validator_compaction_min_compressible_tokens = 4000
# Hard safety margin: force-truncate before invoking the LLM if the estimate
# approaches the model window even after soft-threshold compaction was skipped.
validator_hard_reserved = 8192


graph = app_path / "graphify-out" / "graph.json"
cache_dir = app_path / ".cache"

# Image tag used when the preprocessor builds the target container. If None,
# it is derived from the app directory name (e.g. vulnscan-web_reactoops:latest).
docker_image_tag = None


# None = all
communities_to_analyze = None # [32, 33, 54, 61]

# When True, the explorer dispatches multiple small nodes sharing a file in a
# single batch. Set to False to force one dispatch per node (no batching).
explorer_batching_enabled = True

# Max expert roles assigned per community (top-K by heuristic score). Reduces
# duplicate explorer scans of the same nodes by multiple expert roles.
max_experts_per_community = 2

# When True, the preprocessor always rebuilds the container image even if the
# build definition (Dockerfile/compose) is unchanged since the last run.
force_rebuild = False

