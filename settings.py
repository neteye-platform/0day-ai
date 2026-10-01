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


app_path = Path("../apps/htb")

# LLM provider.
# "openai" = ChatOpenAI against the internal gateway (needs OPENAI_API_KEY).
# "ollama" = ChatOllama against a local Ollama server.
llm_provider = "openai"
openai_base_url = "http://localhost:11434/v1"
openai_model = "deepseek-v4-flash"
ollama_model = "gemma4:cloud"
ollama_base_url = "http://localhost:11434"

# Concurrency
agents_concurrency = 1

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

# Tool-loop guards for the compiled integration-auditor subgraph. The auditor
# decides whether a `requires_integration` vulnerability combines with other
# confirmed findings; if it never calls submit_integration_audit within this
# many LLM rounds, the loop terminates via the fallback node.
integration_auditor_max_iterations = 20
# Same countdown derivation as the reviewer/validator tracks.
integration_auditor_countdown_start = max(1, integration_auditor_max_iterations - COUNTDOWN_LEAD_TURNS)
# Integration-auditor context compaction settings (mirror the reviewer's).
integration_auditor_model_context_window = 131072
integration_auditor_context_reserved = 24000
integration_auditor_compaction_tail_turns = 1
integration_auditor_compaction_min_compressible_tokens = 4000
integration_auditor_hard_reserved = 8192

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
# bridge network. The sandbox is published on all host interfaces and is
# reachable from the attacker box at the bridge gateway (default 172.17.0.1);
# the preprocessor sets sandbox_url to that gateway URL so every validator
# tool (HTTP, browser, attacker shell) shares one target address. Falls open
# (HTTP-only validation) if docker/build/container startup fails.
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

# File/path-level scan exclusion. Nodes/code whose source_file matches any
# pattern are skipped before the explorer/contract-verifier work, dropped from
# the CVE keyword-corpus, and blocked from reviewer file reads (search_codebase,
# read_file). This lets a full repo (including third-party trees, tests, docs)
# be scanned without manually pruning non-relevant paths first. Bare directory
# names ("vendor") and globs ("tests/*", "*.md", "**/migrations/*") are both
# supported, matched relative to the app root.
scan_exclude_paths = []          # e.g. ["tests/", "docs/api/*", "**/migrations/*"]
# Auto-exclude well-known dependency/test/doc paths even when the list above is
# empty. Set to False to rely solely on scan_exclude_paths.
scan_exclude_defaults = True

# When True, the explorer dispatches multiple small nodes sharing a file in a
# single batch. Set to False to force one dispatch per node (no batching).
explorer_batching_enabled = True
# Maximum combined code size (in chars) for a batched explorer dispatch.
explorer_batch_char_threshold = 15000

# Max expert roles assigned per community (top-K by heuristic score). Reduces
# duplicate explorer scans of the same nodes by multiple expert roles.
max_experts_per_community = 2

# Semantic dedup: before reviewers are dispatched, hypotheses that
# are the same real vulnerability described differently by different agents
# (e.g. explorer roles calling it "plaintext password logging" vs "plaintext
# credential logging") are merged via local Ollama embeddings so one reviewer
# subgraph adjudicates the pattern once. Clustering groups by
# (vulnerability_type, cwe_id) at semantic_dedup_threshold (default 0.80,
# validated against real cached hypotheses). Dependency-origin records
# (carry source_cve) are never merged — each is a distinct known CVE. Fails
# open: if Ollama/embeddings is unreachable or semantic_dedup_enabled is False,
# every hypothesis is dispatched unchanged.
semantic_dedup_enabled = True
semantic_dedup_threshold = 0.80
embeddings_model = "embeddinggemma"
embeddings_base_url = "http://localhost:11434"
embeddings_timeout = 60

# When True, the preprocessor always rebuilds the container image even if the
# build definition (Dockerfile/compose) is unchanged since the last run.
force_rebuild = False

