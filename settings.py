from pathlib import Path

app_path = Path("../apps/web_reactoops")
graph = app_path / "graphify-out" / "graph.json"
sandbox_url = "http://127.0.0.1:8000"
# sandbox_url = "http://127.0.0.1:9000"
cache_dir = app_path / ".cache"
container_name = "notebookconverter"

# Image tag used when the preprocessor builds the target container. If None,
# it is derived from the app directory name (e.g. vulnscan-web_reactoops:latest).
docker_image_tag = None


# None = all
communities_to_analyze = None # [32, 33, 54, 61]

# When True, the explorer dispatches multiple small nodes sharing a file in a
# single batch. Set to False to force one dispatch per node (no batching).
explorer_batching_enabled = True

# Concurrency
simple_agents_concurrency = 4
tool_agents_concurrency = 1
