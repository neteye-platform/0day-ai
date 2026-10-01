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


# ============================ Target settings ============================

app_path = Path("../apps/glpi-11.0.7-clean")

graph = app_path / "graphify-out" / "graph.json"
cache_dir = app_path / ".cache"

# None = all
communities_to_analyze = None # [32, 33, 54, 61]

# Path patterns (relative to app root) skipped before analysis and blocked from
# reviewer file reads; globs and bare dir names supported.
scan_exclude_paths = []          # e.g. ["tests/", "docs/api/*", "**/migrations/*"]
# Auto-exclude well-known dependency/test/doc paths even when the list is empty.
scan_exclude_defaults = True


# ========================== Model / LLM settings ==========================

llm_base_url = "http://localhost:11434/v1"
llm_model = "deepseek-v4-flash"
llm_api_key = os.environ.get("OPENAI_API_KEY")
# llm_base_url = "http://localhost:11434/v1"
# llm_model = "nemotron-3-ultra:cloud"
# llm_api_key = "ollama"

# Single context-window size shared by the reviewer, validator, and integration
# auditor (their compaction hard caps derive from it).
model_context_window = 131072

# Single fixed output budget for EVERY LLM call in the pipeline — fast
# structured JSON (explorer, CVE analyzer, threat-intel, contract verifier),
# compaction summaries, the reviewer, and the validator/integration-auditor
# loops. Window-independent: stays put no matter model_context_window.
llm_max_completion_tokens = 16384


# =============================== Agents ==================================

agents_concurrency = 4

# Reviewer/validator loop caps: if the terminal tool isn't called within this many
# LLM rounds, the loop ends via the fallback node instead of hitting the recursion
# limit. Countdown notes are injected from COUNTDOWN_LEAD_TURNS before the cap.
reviewer_max_iterations = 25
validator_max_iterations = 150
integration_auditor_max_iterations = 20
COUNTDOWN_LEAD_TURNS = 4
reviewer_countdown_start = max(1, reviewer_max_iterations - COUNTDOWN_LEAD_TURNS)
validator_countdown_start = max(1, validator_max_iterations - COUNTDOWN_LEAD_TURNS)
integration_auditor_countdown_start = max(1, integration_auditor_max_iterations - COUNTDOWN_LEAD_TURNS)
# Max times the Validator may request more context from the Reviewer per record;
# past this it must conclude on the evidence it has.
validator_feedback_max_rounds = 1

## ---- Tool-loop context compaction ----

# Shared context-compaction budget for ALL tool-loop agents (reviewer,
# validator, integration auditor): when estimated history tokens reach the
# window minus context_reserved, the middle is collapsed into an LLM summary,
# keeping the most recent verbatim tail (estimated ~2 chars/token, deliberately
# conservative).
context_reserved = 24000
# Messages longer than this move out of the verbatim tail into the compressible
# middle (a degenerate long model dump is never pinned verbatim).
max_response_chars = 12000
# Most-recent AI+tool turns kept verbatim after compaction.
compaction_tail_turns = 1
# Minimum compressible tokens before compacting (avoids thrashing).
compaction_min_compressible_tokens = 4000
# Force-truncate before invoking the LLM if the estimate still nears the window.
hard_reserved = 8192

## ---- Contract verifier ----

# Max demands per structured contract-verifier call: output scales with demand
# count, so batches stay well under llm_max_completion_tokens.
verifier_max_demands_per_call = 40

## ---- Deduplication ----

# Semantic dedup: merge hypotheses describing the same real vulnerability across
# agents via local Ollama embeddings before review. Never merges source_cve
# records; fails open to no dedup if Ollama is unreachable.
semantic_dedup_enabled = True
semantic_dedup_threshold = 0.80
embeddings_model = "embeddinggemma"
embeddings_base_url = "http://localhost:11434"

# Demand dedup (contract-verifier input): collapse paraphrases of one requirement
# per target node (exact identity, then embedding similarity). cve_assumption
# demands never merge; fails open to exact-only merging.
demand_dedup_enabled = True
embeddings_timeout = 60

## ---- Explorer ----

# Batch small nodes sharing a file into one explorer dispatch.
explorer_batching_enabled = True
explorer_batch_char_threshold = 15000

# Max expert roles assigned per community (top-K by heuristic score).
max_experts_per_community = 1

## ---- Validator tools ----

# Headless-browser toolset: one shared Firefox serves all validators; each
# session_id gets an isolated context. None uses Playwright's patched Firefox.
browser_enabled = True
browser_timeout_ms = 30000
# Bounded per-session console/pageerror/dialog ring buffer + navigate output cap.
browser_console_max_messages = 60
browser_console_msg_chars = 500
browser_describe_max_chars = 6000
browser_executable = None
# Idle-TTL reaper for sessions not closed by the terminal tool.
browser_idle_timeout_sec = 600

# Kali attacker container (run_command/write_attacker_file/read_attacker_file),
# started lazily on the default bridge network; the sandbox is reachable at the
# bridge gateway so all validator tools share one target URL. Fails open.
attacker_enabled = True
attacker_image_tag = "vulnscan-kali-attacker:latest"
attacker_container_name = "vulnscan-kali-attacker"
# Confined working tree for validator shell activity.
attacker_workdir = "/work"
# Per-command cap; image build has its own generous timeout.
attacker_command_timeout = 60
attacker_build_timeout = 3600
attacker_output_max_chars = 8000

## ---- Edge Traversal agent ----

# Detects composite vulnerabilities arising from interactions between
# individually-benign components (trust-boundary crossings). Runs as a single
# synchronous node after the contract-verifier superstep; findings are emitted
# as standard hypotheses into the `vulnerabilities` channel and adjudicated by
# the reviewer's `cross_boundary` track.
edge_traversal_enabled = True
# Edges are grouped by boundary category into homogeneous batches of this many
# related crossings; one structured LLM call is spent per batch.
edge_traversal_batch_size = 8
# Cap on total batched dispatches per run (a mid/large repo exposes only a
# handful of proxy ingress points, queue topologies, and network listeners).
edge_traversal_max_batches = 12
# Deterministic extraction toggles (all on by default).
edge_traversal_direct_enabled = True     # cross-community AST calls/references
edge_traversal_queue_enabled = True      # task/event dispatch -> worker entry
edge_traversal_http_enabled = True       # HTTP client call -> route entry
edge_traversal_infra_enabled = True      # reverse-proxy config -> app route
# Only traverse edges that touch at least one node carrying an explorer
# interface note (exit/ingress profile). Keeps the candidate set scoped to the
# analyzed surface instead of flooding on unexamined cross-community edges.
edge_traversal_require_note = True
# Cap on candidate edges per boundary category before batching.
edge_traversal_max_edges_per_category = 150

## ---- Threat intel / build ----

# Enrich HIGH/CRITICAL CVEs with external web evidence (Tavily); False runs the
# analyzer results as-is (barrier still fires via a no-op task).
threat_intel_enabled = False

# Always rebuild the container image even if Dockerfile/compose is unchanged.
force_rebuild = False
