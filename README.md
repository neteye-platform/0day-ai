# 0day-ai

A LangGraph-based multi-agent pipeline that audits a target repository
end-to-end. It performs static analysis, dependency (SCA) triage, adversarial
review, dynamic exploitation against a live sandbox, proof-of-concept (PoC)
generation, auto-patching, and generates a per-finding PDF report bundled with
PoC scripts.

**Note:** This repository scans a separate target repository defined in
`settings.app_path` by consuming the knowledge graph produced by
[Graphify](https://github.com/Graphify-Labs/graphify) (automatically run in the
initial phase).

## Pipeline Overview

The pipeline operates in sequential, agent-driven phases. For an in-depth
breakdown of the internal mechanics, see `architecture.md`.

```text
0. Bootstrap ─── Graphify (Create the application graph divided in communities)
                 ↓
1. Discovery ─┬─ Code Explorers (Static logic analysis per community)
              └─ Preprocessor (Dependency SCA & CVE evaluation)
                 ↓
2. Triage    ─── Reviewer Agent (Adjudicates all hypotheses statically)
                 ↓
3. Proof     ─── Validator (Live sandbox exploitation & exploit chaining) ⇄ Auto-Patcher
                 ↓
4. Output    ─── Reporter (PDF generation & bundled PoC scripts)
```

### Core Capabilities

- **Static + Dynamic Adjudication:** Vulnerability hypotheses are deduplicated
  locally, adversarially reviewed by specialized LLM agents, and strictly proven
  or disproven in a live Docker sandbox.
- **Live Sandbox Exploitation:** Validators are armed with HTTP tools (including
  automatic CSRF fetching), a shared headless Firefox instance via Playwright,
  and a dedicated Kali Linux attacker container for true network-level exploits.
- **Optional Auto-Patching:** Exploitable findings can receive a minimal
  auto-fix, trigger a rebuilt/resynced sandbox, and undergo a dynamic
  re-verification run.
- **Per-Agent Model Tiering & Heterogeneous Routing:** Each agent profile in
  `settings.py` can bind to different models and gateways within the same run.
  Run fast, local models (e.g., lightweight Qwen instances via Ollama/vLLM) for
  high-throughput exploration, and reserve heavier reasoning models for complex
  exploit validation and auto-patching.
- **Resumable & Cached:** Everything is cached under `<target>/.cache/` (SCA
  results, artifacts, agent verdicts, token ledgers). Scans interrupted via
  `Ctrl+C` stop cooperatively and can resume cheaply.

## Prerequisites

- Python >= 3.11
- Docker (required for the target sandbox and Kali attacker containers). The
  target application is automatically built if the repository contains a
  `Dockerfile` or `docker-compose.yaml`.
- [osv-scanner](https://google.github.io/osv-scanner/installation/) (required
  for Software Composition Analysis)
- An OpenAI-compatible LLM gateway (configured in `settings.py`)

## Installation & Setup

1. **Clone and initialize the environment:**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

1. **Install browser dependencies:** The validator requires a patched Playwright
   Firefox build to drive DOM-based exploits.

```bash
playwright install firefox
```

1. **Configure the pipeline (`settings.py`):** Point `app_path` to your target
   repository. Define provider endpoints and map specific models per agent role
   (e.g., assigning low-latency local endpoints to explorers and high-reasoning
   models to reviewers/validators). Update the configuration for the pipeline
   feature flags (e.g., auto-patching, browser tools, severity thresholds). All
   available options are thoroughly explained via inline comments.

1. **Environment Variables:** Create a `.env` file at the repository root to
   store your API keys for whatever providers you configured in `settings.py`.

```dotenv
# LLM Endpoint Configuration (OpenAI-compatible)
# Local example (vLLM / Ollama):
LLM_BASE_URL="http://localhost:11434/v1"
LLM_MODEL="qwen2.5-coder:32b"
LLM_API_KEY="ollama"

# Or cloud gateway (e.g. OpenRouter / OpenAI):
# LLM_BASE_URL="https://openrouter.ai/api/v1"
# LLM_MODEL="deepseek/deepseek-chat"
# LLM_API_KEY="your-api-key-here"

TAVILY_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=...
```

## Running a Scan

Run the main pipeline directly. The pipeline saves state automatically after
every step. Interrupting a scan (via `Ctrl+C`) triggers a cooperative shutdown
that lets in-flight agents finish their current tasks, ensuring the run can be
resumed later without losing progress.

```bash
python graph.py    # Full scan
langgraph dev      # Serves graphs for LangGraph Studio debugging
```

### Outputs

The scan generates a timestamped report directory (e.g.,
`<target>/report_<timestamp>/`) containing:

- `report.pdf`: The main executive document featuring pipeline statistics, token
  usage ledgers, and a severity-ranked findings table.
- `findings/`: A subdirectory containing one self-contained PDF for every proven
  vulnerability.
- `poc/` & `patches/`: Bundled directories containing the functional
  proof-of-concept scripts and applied `.patch` diffs.

## Repository Layout

| Module                                                                | Role                                                                                                              |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `graph.py`                                                            | Main entrypoint, graph wiring, fan-out barriers, and conditional routers.                                         |
| `settings.py`, `agents.yaml`, `schemas.py`                            | Configuration, agent system prompts, and Pydantic schemas.                                                        |
| `languages.py`                                                        | Supported target language definitions, file mappings, and AST parser configurations.                              |
| `state.py`, `utils.py`, `run_stats.py`, `llms.py`                     | State channels, caching, progress ledgers, token accounting, and LLM instances.                                   |
| `stage_*.py`                                                          | Modular implementations for each pipeline stage (e.g., manager, cve, threat intel, reviewer, validator, patcher). |
| `tool_loop.py`                                                        | Shared bounded-memory tool-loop machinery for subgraphs.                                                          |
| `tools.py`, `browser_tools.py`, `attacker_tools.py`, `patch_tools.py` | Agent toolsets spanning HTTP, headless browser manipulation, Kali shells, and source code patching.               |
| `boundary_edges.py`, `dedup.py`, `credential_finder.py`               | Deterministic cross-boundary edge synthesis, embedding deduplication, and credential harvesting.                  |

## Documentation

- **`architecture.md`**: High-level, stage-by-stage description of the
  pipeline's logic and data flow.
- **`AGENTS.md`**: Exhaustive operational reference detailing runtime
  constraints, cache semantics, prompt routing, and strict subsystem contracts.

## Acknowledgments

This project was developed by Simone Avancini, as part of a Master's thesis at
the University of Trento, in collaboration with Würth IT Italy during an
internship program.

---

**Disclaimer:** This tool includes automated dynamic exploitation capabilities.
It is intended strictly for authorized security auditing and defensive
evaluation in isolated sandbox environments. Do not use this tool against
external systems or targets without explicit authorization.
