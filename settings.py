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

# Repairs graphify's `Class::method()` -> CLASS-node binding: at graph load
# (in memory only) each such `calls` edge is re-anchored to the member node
# parsed from the edge's own call line; unresolvable parses keep it verbatim.
repair_call_edges = True

# Scopes bare-target demands (`$input`, `input`) emitted by class CONTAINER
# nodes to the callers of the member(s) whose signature declares that
# parameter; property names stay class-wide, undeclared names drop. Needs
# repair_call_edges (falls back to the plain broadcast without it).
container_demands_scope_to_members = True

# None = all
communities_to_analyze = None
# communities_to_analyze = [402]

# Path patterns (relative to app root) skipped before analysis and blocked from
# reviewer file reads; globs and bare dir names supported.
scan_exclude_paths = []          # e.g. ["tests/", "docs/api/*", "**/migrations/*"]
# Auto-exclude well-known dependency/test/doc paths even when the list is empty.
scan_exclude_defaults = True


# ========================== Model / LLM settings ==========================

llm_base_url = "http://localhost:11434/v1"
llm_model = "qwen3-8-flash-next"
llm_api_key = os.environ.get("OPENAI_API_KEY")
# llm_base_url = "http://localhost:11434/v1"
# llm_model = "nemotron-3-super:cloud"
# llm_api_key = "ollama"

# Single context-window size shared by the reviewer, validator, and integration
# auditor (their compaction hard caps derive from it).
model_context_window = 250112

# Single fixed output budget for EVERY LLM call in the pipeline — fast
# structured JSON (explorer, CVE analyzer, threat-intel, contract verifier),
# compaction summaries, the reviewer, and the validator/integration-auditor
# loops. Window-independent: stays put no matter model_context_window.
llm_max_completion_tokens = 16384


# =============================== Agents ==================================

agents_concurrency = 6

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

# Cross-node merging of code-level hypotheses (same defect reported from
# different caller nodes). Thresholds measured on the GLPI run's embeddings:
# hub-utility hypotheses anchor on bare parameter names ($str, $itemtype) whose
# token identity is meaningless (cosine ~0.6 between different defects), while
# genuine paraphrased duplicates sit at cosine >= 0.93 even with low component
# overlap. At 0.85 + jaccard alone, degenerate anchors coalesce hundreds of
# distinct per-target claims into one oversized review.
dedup_cross_node_similarity = 0.93      # high-confidence tier (embedding decides)
dedup_anchor_confirmed_similarity = 0.85  # mid tier: needs a strong descriptive component
dedup_anchor_min_jaccard = 0.6          # component token overlap for the mid tier
dedup_max_merged_cluster = 25           # cap on cross-node cluster growth

# Demand dedup (contract-verifier input): collapse paraphrases of one requirement
# per target node (exact identity, then embedding similarity). cve_assumption
# demands never merge; fails open to exact-only merging.
demand_dedup_enabled = True
# 0.86, not the hypothesis 0.80: the global pairwise-cosine histogram of the
# cached demand vectors shows 99.95% of DISTINCT demand pairs below 0.815 with
# a monotone tail — the [0.80, 0.86) band is dominated by related-but-distinct
# contract checks, while true paraphrases cluster above 0.90.
demand_dedup_threshold = 0.86
embeddings_timeout = 60

# Confirmed records sharing (cwe, vulnerable_component) exactly are validated by
# ONE validator that tests every finding's reproduction steps and records its
# verdict on all of them (records are never dropped; each keeps its own report
# section). Groups larger than this are validated individually.
validator_variant_max_group = 6

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
# bridge gateway so all validator tools share one target URL. Each validator
# gets its OWN container (named "<attacker_container_name>-<agent_id>") whose
# "/work" is a bind mount of that validator's host directory under the cache,
# so PoC files never leak between validators nor accumulate across runs. Fails
# open.
attacker_enabled = True
# Shared, hash-cached image (built once, used by every validator container).
attacker_image_tag = "vulnscan-kali-attacker:latest"
# Container-name PREFIX; the per-validator container is "<name>-<agent_id>".
attacker_container_name = "vulnscan-kali-attacker"
# Container mount point for the per-validator host dir
# (<cache_dir>/validator/<agent_id>).
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

## ---- Credential finder ----

# Preprocessing agent that discovers pre-configured credentials (default
# accounts, DB passwords, baked-in secrets) from the container artifacts,
# docker-compose/.env files, and application source, writing them to
# `<target_app>/.cache/credentials.json` for downstream consumers (validators).
credential_finder_enabled = True
# Optional single structured-LLM pass that labels/dedupes the raw deterministic
# candidates into {service, kind, username, secret, notes}. Off -> the raw
# (paired) candidates are written as-is. Fails open to raw candidates on error.
credential_finder_use_llm = True
# Per-file size cap (bytes) and total scan budget for the source/artifact walk.
credential_finder_max_file_bytes = 2 * 1024 * 1024
credential_finder_max_scan_bytes = 64 * 1024 * 1024

## ---- Threat intel / build ----

# Enrich HIGH/CRITICAL CVEs with external web evidence (Tavily); False runs the
# analyzer results as-is (barrier still fires via a no-op task).
threat_intel_enabled = False

# Always rebuild the container image even if Dockerfile/compose is unchanged.
force_rebuild = False
