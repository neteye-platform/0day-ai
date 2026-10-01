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

# Target repository to scan (lives OUTSIDE this repo). Edit this to switch targets.
app_path = Path("../apps/target-app")

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

# None = all; a list of ints restricts the scan to those graphify communities.
communities_to_analyze = None

# Path patterns (relative to app root) skipped before analysis and blocked from
# reviewer file reads; globs and bare dir names supported.
scan_exclude_paths = ["*.sql"]   # e.g. ["install/mysql/", "tests/", "docs/api/*", "**/migrations/*"]
# SQL dumps/seeds (e.g. an app's *-empty.sql installer files can be hundreds of KB)
# are unanalyzable (no tree-sitter grammar) and were blowing the explorer's
# unmanaged prompt past the model window; excluded by default.
# Auto-exclude well-known dependency/test/doc paths even when the list is empty.
scan_exclude_defaults = True


# ========================== Model / LLM settings ==========================

llm_base_url = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
llm_model = os.environ.get("LLM_MODEL", "qwen3.8:35b")
llm_api_key = os.environ.get("LLM_API_KEY", "ollama")

# Default model context window (registry key `context_window`); each agent's
# tool-loop compaction caps derive from ITS effective window.
model_context_window = 250112

# Single fixed output budget for EVERY LLM call in the pipeline — fast
# structured JSON (explorer, CVE analyzer, threat-intel, contract verifier),
# compaction summaries, the reviewer, and the validator/integration-auditor
# loops. Window-independent: stays put no matter model_context_window.
llm_max_completion_tokens = 16384

# Per-agent LLM assignment (consumed by llms.get_llm / llms.get_config).
# `llm_defaults` is the base for every agent; each `llm_overrides` entry may
# override any of: model, base_url, api_key, temperature, reasoning_effort,
# max_completion_tokens, context_window — e.g. to route one agent to a
# different endpoint, give it more thinking, or switch to a model with a
# smaller window (the tool-loop compaction caps and the explorer prompt cap
# then follow the agent's own window). Unset keys inherit the default.
# Agent keys:
# explorer, cve_analyzer, threat_intel, contract_verifier, edge_traversal,
# credential_finder, compaction (tool-loop context summaries), dedup_agent,
# reviewer, validator, integration_auditor, patcher, reporter.
llm_defaults = {
    "model": llm_model,
    "base_url": llm_base_url,
    "api_key": llm_api_key,
    "temperature": 0.2,
    "reasoning_effort": "none",
    "max_completion_tokens": llm_max_completion_tokens,
    "context_window": model_context_window,
}
llm_overrides = {
    # Full-override example (every key is optional; unset ones inherit
    # llm_defaults):
    # "reviewer": {
    #     "model": "openai/gpt-6.1-sol",
    #     "base_url": "https://openrouter.ai/api/v1",
    #     "api_key": os.environ.get("OPENROUTER_API_KEY"),
    #     "temperature": 0.6,
    #     "reasoning_effort": "high",
    #     "max_completion_tokens": 32768,
    #     "context_window": 1073741824,
    # },
    # cheap fast structured-JSON agents + compaction summaries: inherit defaults
    "explorer": {},
    "cve_analyzer": {},
    "threat_intel": {},
    "contract_verifier": {},
    "edge_traversal": {},
    "credential_finder": {},
    "compaction": {},
    # deliberate-reasoning agents
    "dedup_agent": {"temperature": 0.8, "reasoning_effort": "medium"},
    "integration_auditor": {"temperature": 0.8, "reasoning_effort": "medium"},
    "patcher": {"temperature": 0.8, "reasoning_effort": "medium"},
    "reporter": {"temperature": 0.8, "reasoning_effort": "medium"},
    # tool-loop agents
    "reviewer": {"temperature": 0.8, "reasoning_effort": "low"},
    "validator": {"temperature": 0.8, "reasoning_effort": "low"},
}


# =============================== Agents ==================================

agents_concurrency = 3

# Reviewer/validator loop caps: if the terminal tool isn't called within this many
# LLM rounds, the loop ends via the fallback node instead of hitting the recursion
# limit. Countdown notes are injected from COUNTDOWN_LEAD_TURNS before the cap.
reviewer_max_iterations = 25
validator_max_iterations = 100
integration_auditor_max_iterations = 20
COUNTDOWN_LEAD_TURNS = 8
reviewer_countdown_start = max(1, reviewer_max_iterations - COUNTDOWN_LEAD_TURNS)
validator_countdown_start = max(1, validator_max_iterations - COUNTDOWN_LEAD_TURNS)
integration_auditor_countdown_start = max(1, integration_auditor_max_iterations - COUNTDOWN_LEAD_TURNS)
# Max times the Validator may request more context from the Reviewer per record;
# past this it must conclude on the evidence it has.
validator_feedback_max_rounds = 1

# CVSS validation gate: a Reviewer-confirmed record whose own CVSS vector estimate
# (submitted via submit_evaluation) computes below this base score is never sent to
# the Validator/Integration Auditor — it stays 'confirmed' and is reported without
# dynamic proof. Findings with a missing/unparseable estimate ALWAYS validate
# (fail-open). 0 (or negative) disables the gate entirely.
validator_min_cvss = 7.0

## ---- Patcher agent ----

# Post-validator fixer: patches the SOURCE CODE of the target application to
# block the exact flow of every validator-proven exploitable record (minimal,
# surgical edits only — no refactoring, no best-practice extras). After the
# edits, the sandbox is resynced to the patched code and each patched record is
# re-reviewed against the patched source; a re-confirmed record is validated
# again in the resynced sandbox. When False the whole stage is inert and the
# pipeline behaves exactly as before the feature existed.
patcher_enabled = True
# Tool-loop cap for one patcher run (read + edit + submit_patch turns).
patcher_max_iterations = 30
patcher_countdown_start = max(1, patcher_max_iterations - COUNTDOWN_LEAD_TURNS)
# Patch attempts allowed per record. The patcher -> reviewer -> validator fix
# loop re-runs while a re-validated patch was REJECTED and this budget is left;
# a record still rejected at budget exhaustion ships flagged as not fixed.
patcher_max_attempts = 2

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
# count, so batches stay well under llm_max_completion_tokens. 15 keeps even a
# degenerate rambling batch far from the 16k cap; the call sites also salvage a
# capped batch via llms.invoke_structured_capped.
verifier_max_demands_per_call = 15

## ---- Deduplication ----

# Semantic dedup: merge hypotheses describing the same real vulnerability across
# agents via local Ollama embeddings before review. Never merges source_cve
# records; fails open to no dedup if Ollama is unreachable.
semantic_dedup_enabled = True
semantic_dedup_threshold = 0.80
# Plain official model (ollama pull). Keep-alive/prewarm/timeout tuning below
# matters more than model size here. If throughput ever collapses (single text
# takes seconds while load is zero) the resident llama-server is wedged —
# usually orphaned request backlogs left by force-killed runs — run
# `ollama stop <model>` to respawn it.
embeddings_model = "embeddinggemma"
embeddings_base_url = "http://localhost:11434"

# Cross-node merging of code-level hypotheses (same defect reported from
# different caller nodes). Thresholds calibrated on representative scans:
# hub-utility hypotheses anchor on bare parameter names ($str, $id) whose
# token identity is meaningless (cosine ~0.6 between different defects), while
# genuine paraphrased duplicates sit at cosine >= 0.93 even with low component
# overlap. At 0.85 + jaccard alone, degenerate anchors coalesce hundreds of
# distinct per-target claims into one oversized review.
dedup_cross_node_similarity = 0.93      # high-confidence tier (embedding decides)
dedup_anchor_confirmed_similarity = 0.85  # mid tier: needs a strong descriptive component
dedup_anchor_min_jaccard = 0.6          # component token overlap for the mid tier
dedup_max_merged_cluster = 25           # cap on cross-node cluster growth

# LLM dedup agent (stage_dedup node, between edge_traversal and the reviewer
# fan-out): groups hypotheses by cwe_id and spends ONE structured
# get_llm("dedup_agent")
# call per group to decide true-duplicate equivalence classes (the embedding
# pass above only catches surface paraphrases — reworded duplicates and the
# same sink re-anchored under different node ids survive it). Groups larger
# than group_max subdivide by the source root directory of the affected
# component to keep each call's context focused. Fails open: a group whose
# call errors/caps passes through un-deduplicated.
dedup_agent_enabled = True
dedup_agent_group_max = 50     # CWE groups larger than this get dir-ordered packing
dedup_agent_max_group = 50     # hard cap of records per LLM call (chunk size);
                               # 50 was picked for prompt-size focus (the
                               # embedding pass has already collapsed verbatim
                               # twins before this)
dedup_agent_parallel = 4       # concurrent group calls
dedup_agent_desc_chars = 900   # per-record description budget in the prompt
                               # (verifier "Fails to satisfy demand… Evidence:"
                               # texts often run ~1 KB; cutting mid-Evidence
                               # would hide the discriminating sink call)

# Demand dedup (contract-verifier input): collapse paraphrases of one requirement
# per target node (exact identity, then embedding similarity). cve_assumption
# demands never merge; fails open to exact-only merging.
demand_dedup_enabled = True
# 0.86, not the hypothesis 0.80: offline cosine histograms over demand vectors
# show distinct pairs dominated by the [0.80, 0.86) band (related-but-distinct
# contract checks), while true paraphrases cluster above 0.90.
demand_dedup_threshold = 0.86
embeddings_timeout = 180
# Texts per /api/embed request. Failed chunks subdivide (halved down to 1)
# instead of aborting the pass, so an oversized batch is a slowdown, not a
# fail-open.
embeddings_batch_size = 200
# Serial by design: one llama.cpp server queues concurrent requests anyway,
# and each concurrent request can materialize its own model copy, which caused
# pile-up timeouts with >1 worker.
embeddings_parallel_chunks = 1
# One-shot cold-load allowance: the model load normally happens INSIDE the
# first request, so a healthy-but-cold server can burn the whole
# embeddings_timeout before serving a byte. Prewarm absorbs that with a
# dedicated long timeout, only when uncached texts actually exist.
embeddings_prewarm_timeout = 600
# Pinned across the dedup passes (demands then hypotheses run far apart within
# one scan); the model is small, and without this every pass re-pays the cold
# load.
embeddings_keep_alive = "6h"
# Progress watchdog: the pass aborts only after this many seconds with ZERO
# newly embedded texts (a healthy pass can never stall longer than one request
# timeout), instead of a fixed total budget that long lists would outlive.
# 0 disables the watchdog entirely.
embeddings_stall_budget_sec = 540

# Confirmed records sharing (cwe, vulnerable_component) exactly are validated by
# ONE validator that tests every finding's reproduction steps and records its
# verdict on all of them (records are never dropped; each keeps its own report
# section). Groups larger than this are validated individually.
validator_variant_max_group = 6

## ---- Explorer ----

# Batch small nodes sharing a file into one explorer dispatch.
explorer_batching_enabled = True
explorer_batch_char_threshold = 15000
# Hard cap on ONE explorer prompt (code + context + overhead). Explorer prompts
# have NO compaction and _pack_node_batches lets an oversized node solo: a
# DB-dump-sized file tokenizes past the model window -> deterministic HTTP 400.
# 0.75 chars/token is a deliberately pessimistic worst case that keeps even
# pathological tokenizers inside window - output budget - context_reserved.
_explorer_llm_cfg = {**llm_defaults, **llm_overrides.get("explorer", {})}
explorer_max_prompt_chars = int(
    (_explorer_llm_cfg["context_window"] - _explorer_llm_cfg["max_completion_tokens"] - context_reserved) * 0.75
)

# Max expert roles assigned per community (top-K by heuristic score).
max_experts_per_community = 2

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

## ---- LangSmith trace splitting ----

# Tracing itself stays purely env-driven (LANGSMITH_TRACING / LANGSMITH_API_KEY
# in .env); these settings only control HOW traces are split. When enabled, the
# four tool-loop subagents (reviewer, validator, integration auditor, patcher)
# run with their own LangSmith project, so their runs navigate as separate
# traces instead
# of bloating the single giant pipeline trace (25k-run-per-trace cap). Every
# run is tagged `agent:<name>` and carries `pipeline_run_id` metadata for
# cross-project correlation with the main trace. No effect unless tracing is
# actually active; fails open (plain env-level tracing) on any langsmith issue.
langsmith_tracing = os.environ.get("LANGSMITH_TRACING", "").lower() in ("true", "1", "yes")
# "default" is LangSmith's fallback project when LANGSMITH_PROJECT is unset —
# keeping them aligned preserves the root-project == "-reviewer" prefix invariant.
langsmith_project = os.environ.get("LANGSMITH_PROJECT") or "default"
langsmith_split_subagents = True
langsmith_split_projects = {
    "reviewer": f"{langsmith_project}-reviewer",
    "validator": f"{langsmith_project}-validator",
    "integration_auditor": f"{langsmith_project}-integration-auditor",
    "patcher": f"{langsmith_project}-patcher",
}

# osv-scanner results are cached under <cache_dir>/osv keyed by target identity
# (container image content id / repo path) and reused while younger than this
# many hours; the upstream OSV database grows continuously, so freshness is
# TTL-based. 0 disables the cache (scan every run).
osv_cache_max_age_hours = 168
