from pathlib import Path

app_path = Path("../apps/glpi-11.0.7-clean")
# app_path = Path("../apps/web_reactoops")
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

# Concurrency
simple_agents_concurrency = 4
tool_agents_concurrency = 1
