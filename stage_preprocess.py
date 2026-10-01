"""Bootstrap + preprocessor stage: graph existence, SCA scan, sandbox, artifacts, symbol index."""

import json
import logging
import subprocess
from typing import Any

import settings
from languages import SYMBOL_QUERIES
from run_stats import _reset_pipeline_stats
from state import MasterState
from utils import (
    run_osv_scanner,
    run_osv_scanner_image,
    deduplicate_cves,
    find_container_builds,
    build_images,
    start_sandbox,
    extract_container_artifacts,
    find_unsupported_code_files,
    get_cached_graph_data,
    index_file,
    is_path_excluded,
)


def bootstrap_node(state: MasterState) -> dict[str, Any]:
    """Ensure the knowledge graph exists before the parallel branches start."""
    _reset_pipeline_stats()  # fresh ledger per pipeline invocation
    if not settings.graph.exists():
        logging.info(f"Graph {settings.graph} not found. Running graphify extract...")
        subprocess.run(
            ["graphify", "extract", str(settings.app_path), "--code-only"],
            check=True,
        )
    return {}


def preprocessor_node(state: MasterState) -> dict[str, Any]:
    """Build/scan the container image(s), start the sandbox, snapshot artifacts,
    run SCA, and write the AST symbol index to disk."""
    raw_vulns = []

    builds = find_container_builds(settings.app_path)
    sandbox_data = None
    built_images = []
    if builds:
        for kind, build_file in builds:
            tag = f"vulnscan-{settings.app_path.name.lower()}:latest"
            images = build_images(kind, build_file, tag)
            if images:
                built_images.extend(images)
                for image in images:
                    logging.info(f"Scanning container image {image} with osv-scanner.")
                    raw_vulns.extend(run_osv_scanner_image(image))
                sandbox_data = start_sandbox(kind, build_file, images[0], settings.app_path.name)
            else:
                logging.warning(f"Failed to build image from {build_file}. Falling back to repo scan.")
                raw_vulns.extend(run_osv_scanner(settings.app_path))
    else:
        logging.warning("No Dockerfile or compose file found. Falling back to repo scan.")
        raw_vulns = run_osv_scanner(settings.app_path)

    # Container artifact snapshot: deterministic, image-based, independent of
    # sandbox success; lets the reviewer inspect effective runtime config.
    if built_images:
        artifact_summary = extract_container_artifacts(built_images)
        logging.info(f"Extracted container artifacts for {len(artifact_summary)} image(s).")
    else:
        logging.info("No container images built; skipping container artifact extraction.")

    logging.info(f"Found {len(raw_vulns)} raw vulns")
    clean_vulns = deduplicate_cves(raw_vulns)
    logging.info(f"{len(clean_vulns)} remaining CVEs after deduplication")

    for ext, files in find_unsupported_code_files(get_cached_graph_data(settings.graph)).items():
        logging.error(
            f"Unsupported language '{ext}': {len(files)} code file(s) "
            f"cannot be analyzed ({', '.join(files)})."
        )

    # Build the AST symbol index for application code only (excluding dep trees).
    global_symbol_index = []
    for filepath in settings.app_path.rglob("*"):
        if filepath.is_file() and not is_path_excluded(str(filepath)) \
                and filepath.suffix.lower() in SYMBOL_QUERIES:
            try:
                file_symbols = index_file(filepath)
                if file_symbols:
                    global_symbol_index.extend(file_symbols)
            except Exception as e:
                logging.warning(f"Failed to index {filepath}: {e}")

    index_file_path = settings.app_path / ".ast_symbol_index.json"

    with open(index_file_path, "w", encoding="utf-8") as f:
        json.dump(global_symbol_index, f)

    logging.info(f"Saved {len(global_symbol_index)} symbols to {index_file_path}.")

    return {
        "known_vulns": clean_vulns,
        "sandbox_url": sandbox_data["sandbox_url"] if sandbox_data else None,
    }
