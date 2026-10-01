# System Architecture

## Pre-processing
Creates the foundational context by parsing every file in the repository.
- **Bootstrap:** run graphify to divide the application into smaller communities (group of nodes) and creates a global graph.
- **Threat-Modeler (not yet implemented):** Analyzes the application graph and documentation to generate a global Threat Model.
- **Software Composition Analysis:** Discovers the dependency manifest (requirements.txt, package.json) and container definitions (Dockerfile / compose). If a Dockerfile/compose exists, it builds the image(s) and scans each with `osv-scanner scan image --format json`; otherwise it falls back to `osv-scanner -r` on the repo. Deduplicates the OSV results into `known_vulns` for the CVE analyzer.
    - **Sandbox bootstrap:** After building, the preprocessor starts the container(s) in the background for the Validator.
    - **Container artifacts:** each built image's filesystem is snapshotted into `.cache/container_artifacts/`. The framework_dependency reviewer reads these to inspect the *effective* runtime configuration decoupled from the live sandbox.
    - **Symbol index:** Builds a global AST symbol index (`.ast_symbol_index.json`) with tree-sitter queries; read by the `get_definition` tool.

## Orchestration Phase
Organize the work.
- **Manager:** assign one or more expert agents to each community. It create a specialized task for each expert.
- **Queue Filter:** Deterministically drops all dependency manifests and nodes that are not worth scanning with an LLM. It parses the nodes using tree-sitter and drops nodes according to these rules:
    - A path-level gate (`is_path_excluded`) first drops whole paths matching `scan_exclude_paths` plus built-in defaults (dependency trees like `vendor`/`node_modules`, `tests`, `docs`, build/cache dirs, `*.md`), so a full repo can be scanned without manual pruning.
    - **Pure Types & Interfaces**
        - Drop: Nodes consisting only of type definitions, aliases, or interfaces.
        - Keep (Safeguards): Any type that sets default values (e.g., $role="admin"$) or uses runtime validation wrappers (like Pydantic). This prevents missing dangerous default states.
    - **Empty Skeletons & No-Ops**
        - Drop: Functions, methods, or classes with entirely empty bodies (e.g., {} or pass).
        - Keep (Safeguards): Nodes with security-related names (e.g., verify, auth_), language magic methods (e.g., __ wakeup), or Python nodes with substantive docstrings. This prevents missing fail-open overrides, deserialization gadgets, or runtime docstring leaks.
    - **Primitive Constants & Configs**
        - Drop: Files or classes containing only flat, primitive literal assignments (strings, numbers, booleans) with zero expressions or function calls.
        - Keep (Safeguards): Variables with security-sensitive names (e.g., secret, tls, role), regex literals, or complex nested objects. This prevents missing hardcoded credentials, severe misconfigurations, or ReDoS bombs.
    - Note: it will miss hardcoded secrets and similar vulnerabilities, if the variable does not use a standard name (e.g. secret|key|token|password), but those are easily detected by standard SAST tools.
    - Example for glpi: Dispatching 21856 explorers, skipped 4567 inert nodes

## Generation Phase
Analyze every node separately to find localized vulnerabilities and define trust assumptions. 
Task executed by specialized expert agents (WebSurfaceAuditor, LogicFlowAuditor, DataPersistenceAuditor, InfraConfigAuditor)
- **Expert Explorer:** Reads one node at a time (or a batch of small nodes sharing a file). Relies on its knowledge (much faster than fetching modules documentation) to identify the use of dangerous functions (e.g., yaml.load(), pickle.loads(), eval()) or other vulnerabilities in the target node and creates a demands when the code uses other ambiguous functions.
    - Output: Structured notes for each node containing: business interfaces (architectural entry and exit points), demands (security assumptions that needs to be verified), vulnerability_hypotheses. For architectural CWE classes (CWE-327/319/306/200/352/384/840) each hypothesis also carries a short canonical `pattern_label` (≤6 lowercase words); the pipeline maps it to `vulnerable_component` so identical patterns found in different nodes share one identity.
    - Optimizations:
        - The output schema is designed to minimize the output tokens required to generate the response.
        - Prompt caching (server side) saves many input tokens (~1500 for every explorer agent).
        - **Batching:** multiple small nodes sharing a source file are packed into a single dispatch as long as their combined size stays under the configurable `explorer_batch_char_threshold`, so one explorer reads the file once. Batch-specific vulnerability-attribution rules live in a dedicated `batch_prompt`.
- **CVE Analyzer:** An LLM reads the CVEs found during the SCA phase and classifies each into one of two tracks via `fix_category`:
    - `application_mitigation` — app code can prevent the exploit (safe API, config flag, sanitization, avoiding an optional feature). Extracts a security assumption as a demand. Example:
        - CVE description (input): In the nbconvert tool, when HTMLExporter.embed_images=True, the markdown renderer allows arbitrary file read
        - Security assumption (output): embed_images property of HTMLExporter must not be set to True
        - Target: all nodes that import and use nbconvert
    - `upgrade_only` — the flaw lives inside the dependency itself (e.g. RCE in the HTTP server's parsing code); only a package upgrade fixes it. Emits a direct vulnerability hypothesis, skipping the contract-verifier path.
    - A deterministic keyword pre-filter drops any record whose keywords are all absent from the codebase before it reaches the LLM stages.
    - The OSV-suggested `cwe_ids` are attached to every analysis after the LLM call and forwarded to the contract verifier as a `[SUGGESTED CWE: ...]` hint (deterministic fallback when the verifier omits one).
    - **Threat Intel (optional):** gated behind `threat_intel_enabled` (default off); a backend (Tavily search) enriches HIGH/CRITICAL CVEs with web evidence (root-cause writeups, exploit analysis) that the analyzer missed. Fails open without `TAVILY_API_KEY`; results cached per CVE id.
- **Aggregation (no LLM):** Groups all assumptions about a given node. Routes explorer upstream/downstream demands to their resolved targets and dispatches upgrade-only CVE hypotheses straight into the `vulnerabilities` channel. Hypotheses in the deterministic `SYSTEMIC_CWES` allowlist collapse into node-independent `systemic:CWE:signature` records whose `affected_nodes` list unions across every occurrence (signature built from the canonical `pattern_label`, capped so verbose phrasing degrades to CWE-level grouping).

## Verification Phase
- **Contract Verifier:** Takes a target node ($n$) and all the security assumptions about that node ($a_{n_i}$). Evaluates if the code meets the demands. Flags failures as new Hypotheses. FAILED evaluations for `application_mitigation` CVE demands are tagged `vulnerability_type="Dependency Mitigation Vulnerability"` + `source_cve` (routed to the dependency_mitigation reviewer track); CWE selection prefers the OSV-suggested `[SUGGESTED CWE: ...]` hint.
    - Optimizations:
        - De-duplication based on the vulnerability ID (`affected_nodes[0]:cwe_id:anchor`, where anchor is the demand_id for contract-verifier reports, the vulnerable_component/pattern label for explorer reports, and the source CVE id for Known Dependency Vulnerability records whose identity is `node:CWE:<source_cve>` so distinct CVEs on one package never collide). If two agents find the same vulnerability the ID collides and they are merged via a status-priority ladder.
- **Semantic Dedup (no LLM):** before reviewers are dispatched, hypotheses that describe the same real vulnerability differently (e.g. "plaintext password logging" vs "plaintext credential logging") are merged via local Ollama embeddings — clustering groups by `(vulnerability_type, cwe_id)` at `semantic_dedup_threshold` (0.80) so one reviewer subgraph adjudicates the pattern once. Dependency-origin records (carry `source_cve`) are never merged. Fails open to exact-identity dedup if embeddings are unreachable or `semantic_dedup_enabled` is off.
- **Reviewer Agent (multi-track):** Takes all surviving Hypotheses (from Explorers, the Verifier, and the CVE analyzer) and routes each into one of up to four tracks by `vulnerability_type`: **code_level** (localized app defects), **framework_dependency** (Known Dependency Vulnerability, incl. upgrade-only CVEs — exposure/active-handling plus container-artifact config checks), **dependency_mitigation** (Dependency Mitigation Vulnerability — first-party taint flow up to the vulnerable library call plus the specific missing-mitigation check), and **systemic** (Systemic Vulnerability — pattern confirmation across every `affected_nodes` member). All run through the single compiled reviewer subgraph bound to a dedicated low-reasoning-effort `reviewer_llm`; only the per-mode system prompt section (`agents.yaml`) and the LLM-bound tool subset differ, and the subgraph's ToolNode registers the union of all toolsets.

## Cross-Community and Integration Analysis
Analyze interactions between communities to find cross-component and configuration bugs.
- **Integration Auditor:** A compiled subgraph (`IntegrationAuditorAgent`) that receives every confirmed `requires_integration` finding and decides whether it combines with other confirmed vulnerabilities into a multi-step exploit chain. Emits a single `chained` / `unchainable` verdict via `submit_integration_audit`; `chained` is routed back to the Validator for PoC construction, while `unchainable` is terminal and stays in the report. A record with **zero other confirmed findings** to chain with is resolved `unchainable` in place by `first_turn` — no auditor LLM call is spent, and `pre_router` ends the subgraph before the router can read the still-empty `messages`. Toolset: `get_vulnerability_details`, `get_node_connections`, `get_path`, `submit_integration_audit`.
- **Roadmap — Edge Traversal:** Analyzes the edges between communities in the global graph. It compares Community A's "Exposed Sinks" against Community B's "Trust Assumptions" using their Interface Notes.
    - Component Interaction Matrix: Scans infrastructure edges for known parsing discrepancies (e.g., Nginx → Node.js) to flag HTTP Request Smuggling.

## Validation Phase
- **Triage (no LLM):** The reviewer's `validation_strategy` routes confirmed findings: `direct_to_validator` → Validator; `static_finding_only` (real in source, no network-reachable path) → accepted into the report without validation; `requires_integration` (real but locked behind auth/state/another exploit) → Integration Auditor. Validation is sequential-by-phase: every `direct_to_validator` record is proven first, and only afterwards are the deferred `requires_integration` records audited — their chain candidates are the just-proven `exploitable` peers carrying real `poc_payload`s.
- **Validator:** Tries to exploit the vulnerability confirmed by the reviewer to create a PoC against the live sandbox.
- **Reviewer ⇄ Validator feedback loop:** If the validator cannot reach a verdict because the report left it unable to test (no reachable sandbox/route, ambiguous reproduction steps, missing auth/session details), it flags the record `insufficient_context` via the dedicated `ask_for_context` terminal tool (bounded by `validator_feedback_max_rounds`, so it is bound only on the first pass). The record is Sent back into the reviewer, which renders the `open_questions` as a feedback block and re-emits self-sufficient `reproduction_steps`; the loop drains because terminal verdicts (`exploitable`/`false_positive`) never loop. An exploit that merely fails is treated as a false positive, not a context problem.
- **Validator tooling:** Beyond `send_http_request`, the validator drives client-side DOM/PoC proof with a shared headless Firefox (one process, one isolated BrowserContext per `session_id`, cookies shared with the HTTP channel) and can execute real network-level attacks inside a lazily-provisioned Kali attacker container (`run_command` / `write_attacker_file` / `read_attacker_file`), all targeting a single sandbox URL that is the docker-bridge gateway, so HTTP, browser, and shell activity share one address.
