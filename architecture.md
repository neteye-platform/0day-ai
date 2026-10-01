# System Architecture

## Pre-processing

Creates the foundational context by parsing every file in the repository.

- **Bootstrap:** run graphify to divide the application into smaller communities (group of nodes) and creates a global graph.
- **Software Composition Analysis:** Discovers the dependency manifest (requirements.txt, package.json) and container definitions (Dockerfile / compose). If a Dockerfile/compose exists, it builds the image(s) and scans each with `osv-scanner scan image --format json`; otherwise it falls back to `osv-scanner -r` on the repo. Deduplicates the OSV results into `known_vulns` for the CVE analyzer.
    - **Sandbox bootstrap:** After building, the preprocessor starts the container(s) in the background for the Validator, publishing them at one bridge-gateway URL shared by all tools.
    - **Container artifacts:** each built image's filesystem is snapshotted into `.cache/container_artifacts/`. The framework_dependency reviewer reads these to inspect the *effective* runtime configuration decoupled from the live sandbox.
    - **Symbol index:** Builds a global AST symbol index (`.ast_symbol_index.json`) with tree-sitter queries; read by the `get_definition` tool.
- **Credential Finder (single LLM call):** harvest pre-configured credentials (dotenv, compose env, Dockerfile ENV/ARG, SQL statements, source assignments, container artifacts) and normalizes them into `.cache/credentials.json`. Rendered into a TARGET AUTHENTICATION block on the validator's first turn.

## Orchestration Phase

Organize the work of explorer agents.

- **Manager:** Assigns one or more expert agents to each community and creates a specialized task for each expert.
- **Queue Filter:** Deterministically drops dependency manifests and nodes not worth an LLM scan.
    - Path exclusion: `is_path_excluded` + `scan_exclude_paths`, with defaults for dependency trees, tests, docs, build dirs, `*.md`
    - Other exclusion rules (with a safeguards that keeps security-relevant exceptions):
        - **Pure types/interfaces only**: keep types with default values (e.g. $role="admin"$) or runtime validation wrappers (Pydantic).
        - **Empty bodies** (e.g. {} or pass): keep security-named nodes (verify, auth_), magic methods (`__wakeup`), substantive docstrings.
        - **Flat primitive-literal assignments only**: keep security-sensitive names (secret, tls, role), regex literals, nested objects.
        - It could miss hardcoded secrets under non-standard names (covered by standard SAST tools).

## Generation Phase

Analyze every node separately to find localized vulnerabilities and define trust assumptions.

- **Expert Explorer (single LLM call).** Task executed by specialized expert agents (WebSurfaceAuditor, LogicFlowAuditor, DataPersistenceAuditor, InfraConfigAuditor). Reads one node (or a batch of small nodes sharing a file), relying on its knowledge (much faster than fetching modules documentation) to identify dangerous functions (e.g., `yaml.load()`, `pickle.loads()`, `eval()`) or other vulnerabilities, and creates a *demand* when the code uses ambiguous functions.
    - Output: Structured notes per node: business interfaces (architectural entry/exit points), demands (security assumptions to be verified), vulnerability_hypotheses. For architectural CWE classes (CWE-327/319/306/200/352/384/840) each hypothesis also carries a short canonical `pattern_label` (≤6 lowercase words); the pipeline maps it to `vulnerable_component` so identical patterns in different nodes share one identity.
    - Optimizations: output schema minimizes output tokens; server-side prompt caching saves ~1500 input tokens per explorer; **Batching** packs small same-file nodes into one dispatch under `explorer_batch_char_threshold` so one explorer reads the file once (attribution rules in a dedicated `batch_prompt`).
- **CVE Analyzer (single LLM call):** classify each SCA-found CVE and turn it into actionable app-level demand. For each CVE picks one track via `fix_category`:
    - `application_mitigation`: app code can prevent the exploit (safe API, config flag, sanitization, avoiding an optional feature). Extracts a security assumption as a demand. Example:
        - CVE description (input): In the nbconvert tool, when HTMLExporter.embed_images=True, the markdown renderer allows arbitrary file read
        - Security assumption (output): embed_images property of HTMLExporter must not be set to True
        - Target: all nodes that import and use nbconvert
    - `upgrade_only`: the flaw lives inside the dependency itself (e.g. RCE in the HTTP server's parsing code); only a package upgrade fixes it. Emits a direct vulnerability hypothesis, skipping the contract-verifier path.
    - Optimization: Every analysis also emits `required_keywords` (strict machine-readable code triggers, e.g. `yaml.load`, `parseNested`). A deterministic keyword pre-filter drops any `application_mitigation` record whose keywords are all absent from the codebase before it reaches the LLM stages; `upgrade_only` records are exempt (anchored to synthetic `dependency:<package>` nodes).
- **Threat Intel Agent (single LLM call).** Gated behind `threat_intel_enabled` (default off); a Tavily search enriches HIGH/CRITICAL CVEs with web evidence (root-cause writeups, exploit analysis) the analyzer missed.
- **Aggregation:** groups all assumptions about a given node. Routes explorer upstream/downstream demands to their resolved targets and dispatches upgrade-only CVE hypotheses straight into the `vulnerabilities` channel. Hypotheses in the deterministic `SYSTEMIC_CWES` allowlist collapse into node-independent `systemic:CWE:signature` records whose `affected_nodes` list unions across every occurrence.

## Verification Phase

Deduplicate the hypotheses and adjudicate each one against the code.

- **Contract Verifier (single LLM call):** check whether a node honors the security assumptions made about it. Takes a target node ($n$) and all the demands about it ($a_{n_i}$) and evaluates if the code meets them; FAILED evaluations become new Hypotheses. FAILED evaluations for `application_mitigation` CVE demands are tagged `vulnerability_type="Dependency Mitigation Vulnerability"` + `source_cve` (routed to the dependency_mitigation reviewer track).
    - Optimizations:
        - De-duplication based on the vulnerability ID (`affected_nodes[0]:cwe_id:anchor`, where anchor is the demand_id for contract-verifier reports, the vulnerable_component/pattern label for explorer reports, and the source CVE id for Known Dependency Vulnerability records). If two agents find the same vulnerability the ID collides and they are merged via a status-priority ladder.
- **Semantic Dedup (local embedding model):** before reviewers are dispatched, hypotheses that describe the same real vulnerability differently (e.g. "plaintext password logging" vs "plaintext credential logging") are merged via local Ollama embeddings. Cross-node merges need a stricter two-tier gate (cosine ≥0.93, or ≥0.85 with strong text-similarity agreement; degenerate bare-identifier anchors never merge cross-node alone).
- **LLM Dedup Agent (single LLM call): catch duplicates the embeddings cannot see** (reworded duplicates, the same shared-callee defect re-anchored to different callers). Groups surviving hypotheses by `cwe_id` (large groups cut into directory-locality-ordered chunks), spends one structured LLM call per group returning true-duplicate equivalence classes, and keeps one canonical per cluster (systemic preferred, `affected_nodes` unioned, provenance in `agent_merged_from`). Cached per group fingerprint; fails open.
- **Reviewer Agent (multiple LLM calls):** judge each hypothesis (confirm, dismiss, or defer to dynamic proof). Takes all hypotheses (from Explorers, the Verifier, and the CVE analyzer) and routes each into one of up to five tracks by `vulnerability_type`: **code_level** (localized app defects), **framework_dependency** (CVEs in dependencies that can be fixed only by upgrading), **dependency_mitigation** (CVEs in dependencies that can be mitigated by the application code), **systemic** (pattern confirmation across every `affected_nodes` member), and **cross_boundary** (Edge Traversal findings, gap between both endpoints of a trust boundary). All run through the single compiled reviewer subgraph bound to a dedicated `get_llm("reviewer")` instance. Different modes defer only by system prompt and available tools.
        - Every exploitable submission must carry a complete, parseable CVSS v3 `cvss_vector`.

## Cross-Community and Integration Analysis

Analyze interactions between communities to find cross-component and configuration bugs.

- **Edge Traversal Agent (single LLM call):** finds bugs that exist only where two components hand data to each other (e.g. a queue producer and its worker, an HTTP call and the route it hits, a reverse proxy and the app behind it). The candidate connections are built deterministically from the code graph and the container configs (cached). Each small batch is then given to one LLM call that checks whether what one side emits actually satisfies what the other side expects. Findings (contract mismatch, differential parsing, confused deputy) go to the reviewer's cross_boundary track.
- **Integration Auditor (multiple LLM calls):** decide whether a finding can be combined into a multi-step exploit chain. A compiled subgraph (`IntegrationAuditorAgent`) receiving every confirmed `requires_integration` finding; emits a single `chained` / `unchainable` verdict via `submit_integration_audit`. `chained` is routed back to the Validator for PoC construction (with the peers' already-proven `poc_payload`s injected, so the final exploit reuses existing primitives); `unchainable` is terminal and stays in the report.

## Validation Phase

- **Triage:** the reviewer's `validation_strategy` routes confirmed findings:
    - `direct_to_validator` -> Validator
    - `static_finding_only` (real in source, no network-reachable path) -> accepted into the report without validation
    - `requires_integration` (real but locked behind auth/state/another exploit) -> Integration Auditor
- **CVSS severity gate:** a confirmed record whose reviewer-estimated CVSS base score recomputes below `validator_min_cvss` (default 7.0) never spends the sandbox: it stays `confirmed` but is reported with an explicit "Validation: not performed" label and no proven-exploit phrasing. **Boundary exception:** a gate-blocked record that shifts the server-side security boundary (wire-autonomous loopback reach, tenant switch, elevated backend rights) still gets a free chain-audit attempt; a no-chain outcome revokes back to plain `confirmed` rather than being lost.
- **Variant batching (no LLM):** records with identical `(cwe_id, component)` are proven in a single validator run that receives an EQUIVALENT VARIANTS block; the terminal verdict is cloned onto every member. Each record still gets its own report section.
- **Validator:** weaponize each confirmed finding into a working PoC against the live sandbox. A tool-loop subgraph that authenticates (using the Credential Finder's TARGET AUTHENTICATION block), reproduces the issue, and files `exploitable` / `false_positive` with the PoC script and execution logs as evidence.
    - **Reviewer ⇄ Validator feedback loop:** if the validator cannot reach a verdict because the report left it unable to test (no reachable sandbox/route, ambiguous reproduction steps, missing auth/session details), it flags the record `insufficient_context` via the dedicated `ask_for_context` terminal tool (bounded by `validator_feedback_max_rounds`, so it is bound only on the first pass). The record is Sent back into the reviewer, which renders the `open_questions` as a feedback block and re-emits self-sufficient `reproduction_steps`; the loop drains because terminal verdicts (`exploitable`/`false_positive`) never loop. An exploit that merely fails is treated as a false positive, not a context problem.
    - **Tooling:** Beyond `send_http_request` (per-session cookie jars, redirect cap, and automatic CSRF-token fetch via a `__CSRF__` placeholder), the validator drives client-side DOM/PoC proof with a shared headless Firefox (one process, one isolated BrowserContext per `session_id`, cookies shared with the HTTP channel) and can execute real network-level attacks inside a lazily-provisioned Kali attacker container (`run_command` / `write_attacker_file` / `read_attacker_file`) with a private `/work` per validator; its final PoC script is staged for the report. All channels target a single sandbox URL that is the docker-bridge gateway, so HTTP, browser, and shell activity share one address.
- **Patcher Agent (multiple LLM calls):** write and prove a minimal fix for each exploitable finding. An agent reads first-party code and applies minimal edits via patch_source_file, terminated by submit_patch. The sandbox is then deterministically re-synced (image rebuild for build-from-source targets, otherwise docker cp + restart of the live container), the reviewer re-adjudicates the patched code, and the validator re-tests:
    - exploit dead AND legitimate flow healthy -> `verified`
    - exploit still fires -> `rejected` (retry while budget lasts, prior attempts rendered to the next patcher)
    - give-up -> `failed` (terminal)
    - Verdicts and diffs land in the report as a Proposed fix section + .patch file.

## Reporting Phase

- **Reporter (single LLM call):** turn each proven finding into a human-readable section. A terminal barrier collects every reportable record — `exploitable`, `static_finding_only` confirmed, CVSS-gate-skipped confirmed, and patch-verified false positives — then spends exactly one structured LLM call per finding (title, distilled summary, reproduction steps rewritten from the proven PoC, corrected CVSS vector, worst-case impact, remediation). A failed call falls back to the record's own text, so nothing is dropped.
- **Assembler:** severity-ranks the sections and writes a fresh `report_<timestamp>/report.pdf` bundling each PoC script (`poc/`) and patch (`patches/`), a deterministic Pipeline Statistics section, and a per-agent Token Usage section.
- **Token accounting:** every LLM turn in the pipeline is booked to a per-agent ledger (run_stats.py) that feeds the report.
