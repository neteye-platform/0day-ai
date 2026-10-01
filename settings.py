from pathlib import Path

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

