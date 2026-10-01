import os
from pathlib import Path

app_path = Path("../apps/htb")
graph = app_path / "graphify-out" / "graph.json"
sandbox_url = "http://127.0.0.1:8000"
cache_dir = app_path / ".cache"
container_name = "notebookconverter"
