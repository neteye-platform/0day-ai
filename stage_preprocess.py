"""Bootstrap + preprocessor stage: graph existence, SCA scan, sandbox, artifacts, symbol index."""

import json
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import settings
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
    build_symbol_index,
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


def _write_symbol_index() -> None:
    """Build the AST symbol index for application code (excluding dep trees)
    and write it to disk. Reads only app files, so it runs in the background of
    the docker phase."""
    global_symbol_index = build_symbol_index(settings.app_path)
    index_file_path = settings.app_path / ".ast_symbol_index.json"
    with open(index_file_path, "w", encoding="utf-8") as f:
        json.dump(global_symbol_index, f)
    logging.info(f"Saved {len(global_symbol_index)} symbols to {index_file_path}.")


def preprocessor_node(state: MasterState) -> dict[str, Any]:
    """Build/scan the container image(s), start the sandbox, snapshot artifacts,
    run SCA, and write the AST symbol index to disk; independent steps overlap
    on worker threads."""
    started = time.monotonic()
    raw_vulns = []
    builds = find_container_builds(settings.app_path)

    with ThreadPoolExecutor(thread_name_prefix="preprocess") as pool:
        index_future = pool.submit(_write_symbol_index)

        sandbox_data = None
        built_images = []
        sandbox_target = None  # (kind, build_file, first_image) of the last successful build
        scan_futures = []
        for kind, build_file in builds:
            tag = f"vulnscan-{settings.app_path.name.lower()}:latest"
            images = build_images(kind, build_file, tag)
            if images:
                built_images.extend(images)
                scan_futures.extend(pool.submit(run_osv_scanner_image, img) for img in images)
                sandbox_target = (kind, build_file, images[0])
            else:
                logging.warning(f"Failed to build image from {build_file}. Falling back to repo scan.")
                raw_vulns.extend(run_osv_scanner(settings.app_path))
        if not builds:
            logging.warning("No Dockerfile or compose file found. Falling back to repo scan.")
            raw_vulns = run_osv_scanner(settings.app_path)

        sandbox_future = (
            pool.submit(start_sandbox, *sandbox_target, settings.app_path.name)
            if sandbox_target else None
        )
        # Artifact snapshot depends on the images only, not on sandbox success.
        artifacts_future = pool.submit(extract_container_artifacts, built_images) if built_images else None

        for scan in scan_futures:
            raw_vulns.extend(scan.result())
        if sandbox_future:
            sandbox_data = sandbox_future.result()

        if artifacts_future:
            artifact_summary = artifacts_future.result()
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

        index_future.result()  # surface indexing/writing errors; pool exit also joins

    logging.info(f"Preprocessing completed in {time.monotonic() - started:.1f}s.")

    return {
        "known_vulns": clean_vulns,
        "sandbox_url": sandbox_data["sandbox_url"] if sandbox_data else None,
    }
