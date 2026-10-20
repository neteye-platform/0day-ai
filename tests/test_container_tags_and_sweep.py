import os
import socket
from pathlib import Path

import settings
from tools import attacker
from utils.general import container_image_tag


def test_single_build_keeps_default_tag(monkeypatch):
    monkeypatch.setattr(settings, "app_path", Path("/x/MyApp"))
    assert container_image_tag(Path("/x/MyApp/Dockerfile")) == "vulnscan-myapp:latest"


def test_multiple_builds_get_distinct_tags(monkeypatch):
    monkeypatch.setattr(settings, "app_path", Path("/x/MyApp"))
    a = container_image_tag(Path("/x/MyApp/Dockerfile"), True)
    b = container_image_tag(Path("/x/MyApp/Dockerfile.dev"), True)
    c = container_image_tag(Path("/x/MyApp/sub/Dockerfile"), True)
    assert len({a, b, c}) == 3
    assert all(t.startswith("vulnscan-myapp:") for t in (a, b, c))


def test_abandoned_containers_filtering():
    host = socket.gethostname()
    out = "\n".join(
        [
            f"k-live\t{os.getpid()}@{host}",
            f"k-dead\t999999999@{host}",
            "k-legacy\t",
            "k-foreign\t999999999@other-host",
        ]
    )
    assert attacker.abandoned_containers(out) == ["k-dead", "k-legacy"]
