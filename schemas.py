from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional
import yaml
import re


with open("agents.yaml", "r") as f:
    data = yaml.safe_load(f)
    MANAGER_AGENT = data.get("manager_agent")
    EXPERT_AGENTS = data.get("expert_agents")
    CVE_ANALYZER_AGENT = data.get("cve_analyzer")
    THREAT_INTEL_AGENT = data.get("threat_intel")
    VERIFIER_AGENT = data.get("contract_verifier_agent")
    REVIEWER_AGENT = data.get("reviewer_agent")
    VALIDATOR_AGENT = data.get("validator_agent")
    INTEGRATION_AUDITOR_AGENT = data.get("integration_auditor")

cwes = {
    # --- MEMORY SAFETY (C / C++ / Rust-unsafe) ---
    "CWE-119": "Improper Restriction of Operations within the Bounds of a Memory Buffer",
    "CWE-416": "Use After Free",
    "CWE-476": "NULL Pointer Dereference",
    "CWE-190": "Integer Overflow or Wraparound",
    # --- CONCURRENCY & EXECUTION (Go / Java / C# / Python) ---
    "CWE-362": "Concurrent Execution using Shared Resource with Improper Synchronization (Race Condition)",
    # --- INJECTION (Web / Cloud / DB) ---
    "CWE-89": "SQL Injection",
    "CWE-78": "OS Command Injection",
    "CWE-79": "Cross-Site Scripting (XSS)",
    "CWE-94": "Code Injection",
    "CWE-918": "Server-Side Request Forgery (SSRF)",
    # --- SPECIFIC ACCESS CONTROL (Flat Tier) ---
    "CWE-862": "Missing Authorization",
    "CWE-863": "Incorrect Authorization",
    "CWE-639": "Authorization Bypass Through User-Controlled Key (IDOR)",
    "CWE-306": "Missing Authentication for Critical Function",
    # --- STATE & SESSION (Web / API) ---
    "CWE-352": "Cross-Site Request Forgery (CSRF)",
    "CWE-384": "Session Fixation",
    # --- DATA & CRYPTOGRAPHY ---
    "CWE-200": "Exposure of Sensitive Information to an Unauthorized Actor",
    "CWE-319": "Cleartext Transmission of Sensitive Information",
    "CWE-327": "Use of a Broken or Risky Cryptographic Algorithm",
    "CWE-502": "Deserialization of Untrusted Data",
    # --- CONFIGURATION & FILE SYSTEM ---
    "CWE-22": "Path Traversal",
    "CWE-434": "Unrestricted Upload of File with Dangerous Type",
    "CWE-770": "Allocation of Resources Without Limits or Throttling",
    # --- ESCAPE HATCHES (Broad Parent Categories) ---
    "CWE-284": "Improper Access Control (Use ONLY if no specific access control CWE fits)",
    "CWE-20": "Improper Input Validation (Use ONLY if no specific injection CWE fits)",
    "CWE-840": "Business Logic Errors",
    "OTHER_UNCATEGORIZED": "Use ONLY if no other CWE fits"
}

# CWE classes whose flaws are systemic/architectural (the same insecure pattern
# repeated across many nodes, e.g. "plaintext password storage") rather than a
# defect localized to one function. Hypotheses carrying one of these CWEs are
# grouped into a single node-independent record during aggregation so the
# Reviewer adjudicates the pattern once, against all affected nodes.
SYSTEMIC_CWES = {
    "CWE-327",  # Use of a Broken or Risky Cryptographic Algorithm
    "CWE-319",  # Cleartext Transmission of Sensitive Information
    "CWE-306",  # Missing Authentication for Critical Function
    "CWE-200",  # Exposure of Sensitive Information to an Unauthorized Actor
    "CWE-352",  # Cross-Site Request Forgery (CSRF)
    "CWE-384",  # Session Fixation
    "CWE-840",  # Business Logic Errors
}

CWE_KEYS = Literal[
    "CWE-119", "CWE-416", "CWE-476", "CWE-190", "CWE-362", "CWE-89",
    "CWE-78", "CWE-79", "CWE-94", "CWE-918", "CWE-862", "CWE-863",
    "CWE-639", "CWE-306", "CWE-352", "CWE-384", "CWE-200", "CWE-319",
    "CWE-327", "CWE-502", "CWE-22", "CWE-434", "CWE-770", "CWE-284",
    "CWE-20", "CWE-840", "OTHER_UNCATEGORIZED"
]


# A canonical systemic signature must be a short pattern label. Verbose
# signatures are node-specific phrasing (an explorer ignoring the pattern_label
# rule), never a stable cross-node identity — so anything longer falls back to
# CWE-level grouping instead of producing per-node-unique keys.
SIGNATURE_MAX_WORDS = 6
SIGNATURE_MAX_CHARS = 60


def _normalize_signature(value) -> str:
    """Canonicalize free text into a stable grouping token (lowercase, quotes
    stripped, non-alphanumerics collapsed to '_')."""
    if not value or not str(value).strip():
        return ""
    s = re.sub(r"[`'\"]", "", str(value))
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s


def canonical_signature(
    vulnerable_component: Optional[str] = None,
    demand_id: Optional[str] = None,
    source_cve: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    """Deterministic identity for grouping systemic findings across nodes.

    Falls through the record's most-identity-bearing fields in order of
    stability: the explorer's enforced `pattern_label` (carried as
    `vulnerable_component`), the verifier's demand id, the CVE's usage pattern,
    and finally the free-text description. Returns "" when no candidate yields
    a short enough canonical label (the caller then groups at CWE level)."""
    for candidate in (
        vulnerable_component,
        demand_id if demand_id and demand_id != "unknown_anchor" else None,
        source_cve,
        description,
    ):
        sig = _normalize_signature(candidate)
        if not sig:
            continue
        # Cap: verbose free text is node-specific phrasing, not a shared label.
        if sig.count("_") + 1 > SIGNATURE_MAX_WORDS or len(sig) > SIGNATURE_MAX_CHARS:
            return ""
        return sig
    return ""


class VulnerabilityRecord(BaseModel):
    vuln_id: Optional[str] = None

    # Lifecycle tracking
    status: Literal["hypothesis", "confirmed", "exploitable", "false_positive", "review_error", "insufficient_context", "chained", "unchainable"] = "hypothesis"

    # Core details (from Explorer/Verifier). Systemic records accumulate every
    # affected graph node here; localized records carry exactly one.
    affected_nodes: list[str] = Field(
        default_factory=list,
        description=(
            "Graph node IDs affected by this vulnerability. Localized defects "
            "list a single node; systemic patterns list every node where the "
            "insecure pattern appears (e.g. login AND registration endpoints)."
        ),
    )
    cwe_id: str = Field(
        description="CWE ID of the vulnerability."
    )
    vulnerability_type: str = "Code Defect"
    description: str
    demand_id: Optional[str] = None
    source_cve: Optional[str] = Field(
        default=None,
        description="The CVE ID a dependency-related hypothesis was derived from. Set on hypotheses emitted directly by the CVE analyzer (upgrade_only CVEs) and on contract-verifier findings for application_mitigation CVEs."
    )
    vulnerable_component: Optional[str] = Field(
        default=None,
        description="The structural anchor from the Explorer hypothesis."
    )

    # Reviewer additions
    reviewer_reasoning: Optional[str] = None
    reproduction_steps: Optional[list[str]] = Field(
        default=None,
        description=(
            "Chronological, numbered steps the downstream Validator must execute to "
            "trigger and prove the vulnerability from the outside. Each step must be "
            "self-sufficient (HTTP method, path, parameters/headers/body, and any session "
            "state carried from earlier steps) because the Validator cannot read source code."
        ),
    )
    validation_strategy: Optional[Literal["direct_to_validator", "requires_integration", "static_finding_only"]] = Field(
        default=None,
        description=(
            "Determines graph routing. 'direct_to_validator': Use this if the vulnerability can "
            "be triggered directly OR if its only prerequisites are freely attainable via public "
            "endpoints (e.g., open self-registration, standard login). The Validator agent can "
            "handle basic account creation. 'requires_integration': Use this ONLY if the "
            "vulnerability requires privileges that cannot be freely registered (e.g., requires "
            "an Admin account), or if it strictly requires the output of another exploit to "
            "function. 'static_finding_only': real in source but with no network-reachable path, "
            "so it is accepted as static evidence into the final report without Validator testing."
        ),
    )

    # Validator additions
    poc_payload: Optional[str] = None
    execution_logs: Optional[str] = None

    # Integration Auditor additions
    integration_audit_reasoning: Optional[str] = Field(
        default=None,
        description=(
            "The Integration Auditor's reasoning for the `chained`/`unchainable` "
            "verdict on a `requires_integration` record."
        ),
    )
    chained_with: Optional[list[str]] = Field(
        default=None,
        description=(
            "vuln_ids of the other confirmed vulnerabilities this record chains "
            "with into a single multi-step exploit (set when status is 'chained')."
        ),
    )

    # Validator -> Reviewer feedback loop
    open_questions: Optional[list[str]] = Field(
        default=None,
        description=(
            "Specific questions the Validator raised about the evidence it needs "
            "(set when status is 'insufficient_context'). The Reviewer must resolve "
            "each one and re-emit self-sufficient reproduction steps."
        ),
    )
    review_round: int = Field(
        default=0,
        description=(
            "How many times the Validator has requested more context (insufficient_context) "
            "for this record. The Validator may request context only once; round is "
            "incremented on each request."
        ),
    )

    @model_validator(mode='before')
    @classmethod
    def migrate_legacy_node_id(cls, values):
        """Back-compat: old cached records carry the removed `node_id` field.
        Fold it into `affected_nodes` so stale cache entries keep loading."""
        if isinstance(values, dict) and "node_id" in values:
            legacy = values.pop("node_id")
            if not values.get("affected_nodes"):
                if isinstance(legacy, list):
                    nodes = [n for n in legacy if n]
                elif isinstance(legacy, str) and legacy.strip():
                    nodes = [legacy.strip()]
                else:
                    nodes = []
                if nodes:
                    values["affected_nodes"] = nodes
        return values

    @field_validator('cwe_id', mode='before')
    @classmethod
    def validate_cwe(cls, value: str) -> str:
        # Clean up LLM formatting quirks (whitespace, lowercase)
        if not value or not str(value).strip():
            return "OTHER_UNCATEGORIZED"
        value = value.strip().upper()
        # Safely fallback if the LLM hallucinates an invalid CWE
        if value not in cwes:
            return "OTHER_UNCATEGORIZED"
        return value

    @model_validator(mode='after')
    def set_vuln_id(self) -> 'VulnerabilityRecord':
        # Systemic classification: deterministic CWE allowlist. Dependency-CVE
        # records are excluded — they already carry a stable per-CVE identity
        # and route to the framework/dependency reviewer track. application_mitigation
        # CVE findings are likewise excluded: they route to the dependency_mitigation
        # reviewer track as node:CWE:<CVE-id>, never folded into systemic grouping.
        systemic = (
            self.cwe_id in SYSTEMIC_CWES
            and not self.source_cve
            and self.vulnerability_type
            not in ("Known Dependency Vulnerability", "Dependency Mitigation Vulnerability")
        )
        if systemic:
            self.vulnerability_type = "Systemic Vulnerability"
            if not self.vuln_id:
                sig = canonical_signature(
                    vulnerable_component=self.vulnerable_component,
                    demand_id=self.demand_id,
                    source_cve=self.source_cve,
                    description=self.description,
                )
                # Node-independent: identical patterns from different nodes
                # share one vuln_id, so merge_vulnerabilities groups them. An
                # empty signature (no short canonical label available) degrades
                # to CWE-level grouping instead of a per-node-unique key.
                self.vuln_id = f"systemic:{self.cwe_id}:{sig}" if sig else f"systemic:{self.cwe_id}"
            return self

        if not self.vuln_id:
            # Known Dependency Vulnerability records (from CVE analyzer,
            # upgrade_only path) route to the framework/dependency reviewer
            # track. Identity is node:CWE:<source_cve>: several distinct CVEs
            # can share one package (same synthetic dependency:<package> node)
            # and the analyzer may guess the same CWE for them, so the CVE id
            # itself must anchor the identity or the channel reducer would
            # collapse them into one record.
            if self.vulnerability_type == "Known Dependency Vulnerability":
                primary = self.affected_nodes[0] if self.affected_nodes else "general"
                cve_anchor = self.source_cve or self.demand_id or "unknown-cve"
                self.vuln_id = f"{primary}:{self.cwe_id}:{cve_anchor}"
                return self

            # If it came from the Contract Verifier, use the demand_id anchor
            if self.demand_id and self.demand_id != "unknown_anchor":
                anchor = self.demand_id
            # If it came from the Explorer, use the vulnerable_component anchor
            elif self.vulnerable_component:
                # Normalize the string (lowercase, strip spaces) to prevent trivial mismatches
                anchor = self.vulnerable_component.strip().lower()
            # Fallback if neither exists
            else:
                anchor = "general"

            primary = self.affected_nodes[0] if self.affected_nodes else "general"
            self.vuln_id = f"{primary}:{self.cwe_id}:{anchor}"
        return self


class ExpertTask(BaseModel):
    agent_role: str = Field(
        description="The PREDEFINED role best suited for this specific architectural area.",
        json_schema_extra={"enum": list(EXPERT_AGENTS.keys())}
    )
    target_community: str = Field(
        description="The EXACT single community ID this agent must focus on (e.g., '3'). Extract this exact ID from the summary."
    )
    task_description: str = Field(
        description="Detailed instructions on what specific vulnerability classes, architectural risks, or cross-component interactions to investigate within this subgraph."
    )

# Matches patterns like 15.0.5, v1.2, 19.0.0-rc.1. The bounds avoid corrupting
# IPv4-like dotted numerics (e.g. 192.168.1.1 must survive untouched).
_VERSION_PATTERN = r'(?<![\d.])\b(?:v|version\s*)?\d+\.\d+(?:\.\d+)?(?!\.\d)(?:-[a-zA-Z0-9.]+)?\b'

# Matches full phrases the LLM likes to generate (e.g. "fixed in 15.0.5"),
# optionally swallowing a trailing , or . so sentences stay clean.
_PHRASE_PATTERN = r'(?i)(?:fixed\s+in|prior\s+to|before|upgrading\s+to)\s+(?:' + _VERSION_PATTERN + r'[.,]?)'


def strip_version_numbers(text: str) -> str:
    """Deterministically strip version numbers and fix-version phrases from
    agent-facing text so downstream agents (e.g. the Reviewer) don't go down
    the rabbit hole of checking package versions.

    - Removes whole phrases like "fixed in 2.3.0" / "prior to 3.0.0" entirely.
    - Redacts standalone versions (15.0.5, v1.2, 19.0.0-rc.1) with a token.
    - Never corrupts IPv4-like dotted numerics (e.g. 192.168.1.1 stays intact).
    - Preserves newlines (only horizontal whitespace is collapsed).
    """
    if not isinstance(text, str):
        return text
    cleaned = re.sub(_PHRASE_PATTERN, '', text)
    cleaned = re.sub(_VERSION_PATTERN, '[VERSION_REDACTED]', cleaned)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = '\n'.join(line.strip() for line in cleaned.split('\n'))
    return cleaned.strip()


class CVEHypothesis(BaseModel):
    cwe: CWE_KEYS = Field(description="The matching CWE ID from the provided list.")
    description: str = Field(
        description="Reviewer-facing description: the CVE ID, the exact flaw inside the library, which library feature/usage exposes it, the attacker-controlled trigger, and the fixed version when known."
    )
    affected_component: str = Field(
        description="The concrete, greppable usage pattern the Reviewer should hunt for in application code (e.g., 'app.run(debug=True)', 'serve(app)', 'express-fileupload middleware'). For transitive dependencies, name the likely parent-framework usage (e.g., 'render_template implies Jinja2')."
    )
    framework_exposure_mechanism: str = Field(
        description="How the framework/library inherently exposes the flaw to the attacker. Frame it as framework behavior (e.g., 'The framework intercepts payloads on all routes', 'The middleware parses all multipart requests')."
    )

    @field_validator('description', 'framework_exposure_mechanism')
    @classmethod
    def _strip_versions(cls, v: str) -> str:
        return strip_version_numbers(v)

class CVEAnalysis(BaseModel):
    reasoning: str = Field(
        description="Briefly explain your logic for the classification and the resulting demand or hypothesis. Do your thinking here."
    )
    fix_category: Literal["application_mitigation", "upgrade_only"] = Field(
        description=(
            "'application_mitigation': the application developer can prevent the exploit through visible source code "
            "(safe alternative function, configuration flag, sanitization before the call, avoiding an optional feature). "
            "'upgrade_only': the flaw lives entirely inside the dependency's own code and NO application-level workaround "
            "exists; the only fix is upgrading the package."
        )
    )
    is_enabled_by_default: bool = Field(
        description="True if the CVE description implies it's standard behavior, or if the workaround requires explicitly DISABLING it."
    )
    import_namespace: str = Field(
        description="The SINGLE top-level root module name used to import this package (e.g., 'bs4' for beautifulsoup4, 'flask' for Flask). You MUST output exactly one word. Required for BOTH fix categories to locate usage sites in application code.",
        # No "\-" escape: ollama's regex-to-grammar converter fails on it,
        # which 400s every json_schema structured-output call. Hyphen-last is
        # the identical character class.
        pattern=r"^[a-zA-Z0-9_-]+$"
    )
    required_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Strict, machine-readable list of code-level triggers that MUST appear verbatim in the "
            "application's source code. CRITICAL: You MUST include the high-level public API wrappers "
            "(e.g., 'requests.get', 'app.use') that developers actually write, especially if the vulnerability "
            "resides in a hidden internal class or sub-dependency. "
            "REQUIRED for BOTH fix categories. NEVER use generic terms. Keep to 1-5 highly specific keywords."
        )
    )
    security_assumption: Optional[str] = Field(
        default=None,
        description=(
            "REQUIRED iff fix_category is 'application_mitigation'. The specific demand or configuration requirement. "
            "This MUST be framed around the public API the developer interacts with. If the vulnerable internal "
            "component is enabled by default by a higher-level class, state that explicitly in the assumption."
        )
    )
    trigger_condition: Optional[str] = Field(
        default=None,
        description="The explicit data flow, function call, OR network request required for the exploit. For code-level library flaws, specify the function call (e.g., 'calling yaml.load()'). For framework/middleware flaws, specify the exact HTTP request primitive. If vague, leave empty."
    )
    attacker_request_primitive: Optional[str] = Field(
        default=None,
        description="The exact theoretical request or input primitive an attacker uses (e.g., 'POST request with Next-Action header', 'crafted Transfer-Encoding header', 'multipart/form-data payload')."
    )
    hypothesis: Optional[CVEHypothesis] = Field(
        default=None,
        description="REQUIRED iff fix_category is 'upgrade_only'. The vulnerability hypothesis describing the dependency-internal flaw."
    )

    @field_validator('security_assumption', 'trigger_condition')
    @classmethod
    def _strip_versions(cls, v: Optional[str]) -> Optional[str]:
        return strip_version_numbers(v) if v else None

class VulnerabilityEvaluation(BaseModel):
    # report_id: str = Field(description="The unique identifier or title of the vulnerability report.")
    is_exploitable: bool = Field(
        description="True if the vulnerability has a realistic path to exploitation. False if it is a false positive, purely theoretical, or blocked by standard mitigations."
    )
    confidence_score: int = Field(description="Confidence in this assessment from 1 to 10.")
    reasoning: str = Field(description="Brief technical explanation for the decision.")
    original_report: list[dict]
    reproduction_steps: Optional[list[str]] = Field(
        default=None,
        description="Chronological, numbered external actions required to trigger and prove the vulnerability."
    )

class ReviewerOutput(BaseModel):
    vulnerabilities: list[VulnerabilityEvaluation]

class ValidatorOutput(BaseModel):
    # Constrained to vulnerabilities only: validator subgraphs must not echo their
    # input fields (sandbox_url etc.) back to MasterState — concurrent writes to
    # those scalars raised "Can receive only one value per step" at checkpoint time.
    vulnerabilities: list[VulnerabilityRecord]

class UpstreamDemand(BaseModel):
    target: str = Field(description="Exact parameter name (e.g., 'query', 'user_id').")
    description: str = Field(description="The security contract.")

class DownstreamDemand(BaseModel):
    target: str = Field(description="Format as 'module::symbol' (e.g., 'app.auth::get_jobs').")
    description: str = Field(description="The security contract.")

class Hypothesis(BaseModel):
    cwe: CWE_KEYS = Field(description="The matching CWE ID from the provided list.")
    component: str = Field(description="The exact parameter, state transition, or function call that is flawed.")
    pattern_label: Optional[str] = Field(
        default=None,
        description=(
            "REQUIRED when cwe is a systemic/architectural class (CWE-327, CWE-319, "
            "CWE-306, CWE-200, CWE-352, CWE-384, CWE-840): a SHORT canonical name of "
            "at most six lowercase words identifying the insecure pattern, e.g. "
            "'plaintext password storage', 'no csrf token validation', 'weak tls ciphers'. "
            "Use the EXACT SAME label for every occurrence of the same pattern, so findings "
            "from different nodes can be grouped into a single review. Leave null for "
            "localized defects."
        ),
    )

class AnalysisNote(BaseModel):
    sources: list[str] | None = Field(default=None, description="External data entering this snippet.")
    sinks: list[str] | None = Field(default=None, description="Dangerous operations performed with data.")
    upstream: list[UpstreamDemand] | None = Field(default=None)
    downstream: list[DownstreamDemand] | None = Field(default=None)
    vulns: list[Hypothesis] | None = Field(default=None)

    @model_validator(mode='after')
    def set_empty_lists(self) -> "AnalysisNote":
        if self.sources is None: self.sources = []
        if self.sinks is None: self.sinks = []
        if self.upstream is None: self.upstream = []
        if self.downstream is None: self.downstream = []
        if self.vulns is None: self.vulns = []
        return self

class BatchedAnalysisNote(AnalysisNote):
    node_id: str = Field(description="The exact graph node ID this analysis note refers to.")

class BatchedAnalysisResult(BaseModel):
    notes: list[BatchedAnalysisNote] = Field(
        description="List of analysis notes. MUST contain one note per input node."
    )

class DemandEvaluation(BaseModel):
    demand_id: str = Field(description="The exact ID extracted from the [ID: ...] tag provided in the demand description.")
    status: Literal["MET", "FAILED", "DELEGATED", "OUT_OF_SCOPE"] = Field(
        description=(
            "MET: If the visible code explicitly implements standard, robust security controls (e.g., parameterized queries) that neutralize the threat.\n"
            "FAILED: the code explicitly manipulates data insecurely IN PLAIN SIGHT, or implements a visibly weak/fragile mitigation (e.g., custom regex for path traversal).\n"
            "DELEGATED: the code passes the untrusted data to a helper function, validator, or sanitizer whose implementation is NOT visible in the snippet.\n"
            "OUT_OF_SCOPE: the demand targets a different layer (e.g., expecting a database helper to handle HTTP cookies) or a different context (e.g., HTML configuration on a Markdown exporter)."
        )
    )
    reasoning: str = Field(description="Brief explanation referencing specific lines of code.")
    cwe: Optional[CWE_KEYS] = Field(description="The matching CWE ID from the provided list.")

class VerifierOutput(BaseModel):
    evaluations: list[DemandEvaluation]

# ==========================================
# Tools
# ==========================================

class EvaluationToolInput(BaseModel):
    reasoning: str = Field(description="Brief technical explanation for the decision.")
    is_vulnerable: bool = Field(
        description="True if the specific node contains a defect, unsafe configuration, or lacks mitigation, regardless of whether it can currently be reached from the outside."
    )
    is_exploitable: bool = Field(
        description="True if the vulnerability has a realistic path to exploitation. False if it is a false positive, purely theoretical, or blocked by mitigations, implemented by the application."
    )
    validation_strategy: Optional[Literal["direct_to_validator", "requires_integration", "static_finding_only"]] = Field(
        default=None,
        description=(
            "How this finding must be handled downstream, chosen from the triage rules.\n"
            "REQUIRED when is_exploitable is true (pick exactly one); leave unset for "
            "false positives (is_exploitable false).\n"
            "'static_finding_only' — 100% real in the source but has NO network-reachable "
            "exploit path (e.g. plaintext passwords stored in the DB).\n"
            "'direct_to_validator' — use this if the vulnerability can be triggered directly "
            "OR if its only prerequisites are freely attainable via public endpoints (e.g., "
            "open self-registration, standard login). The Validator agent can handle basic "
            "account creation.\n"
            "'requires_integration' — use this ONLY if the vulnerability requires privileges "
            "that cannot be freely registered (e.g., requires an Admin account), or if it "
            "strictly requires the output of another exploit to function."
        )
    )
    reproduction_steps: list[str] = Field(
        default_factory=list,
        description=(
            "Chronological, numbered sequence of exact external actions required to "
            "trigger and prove the vulnerability, e.g. '1. Authenticate by POSTing valid "
            "credentials to /api/login (fields: username, password) and capture the session "
            "cookie.', '2. Send POST /api/export with JSON body {\"title\":\"<payload>\"} "
            "while carrying the session cookie.', '3. Confirm the reflected payload in the "
            "response body.'. The downstream Validator CANNOT read source code, so every "
            "step must be self-sufficient and executable over HTTP alone: state the HTTP "
            "method, path, required parameters/headers/body, and any session state carried "
            "from earlier steps. Fill this when is_exploitable is true; leave empty for "
            "false positives."
        )
    )

    @model_validator(mode="after")
    def strategy_required_when_exploitable(self):
        if self.is_exploitable and not self.validation_strategy:
            raise ValueError(
                "validation_strategy is required when is_exploitable is true. "
                "Pick one of 'direct_to_validator', 'requires_integration', 'static_finding_only'. "
                "Leave it unset only for false positives."
            )
        return self

class ValidationToolInput(BaseModel):
    is_confirmed: bool = Field(description="True if the exploit successfully triggered in the sandbox.")
    poc_payload: Optional[str] = Field(description="The exact payload, script, or HTTP request that triggered the vulnerability.")
    execution_logs: str = Field(description="Relevant logs or output from the sandbox confirming the exploit.")


class AskForContextInput(BaseModel):
    reasoning: str = Field(
        description=(
            "Why you cannot reach a verdict: the specific blocker (e.g. no sandbox is "
            "reachable, the reproduction steps lack an exact HTTP method/path/body, the "
            "endpoint or feature is unreachable or undocumented, or blocking "
            "authentication/session details are missing)."
        )
    )
    open_questions: list[str] = Field(
        description=(
            "The exact, specific questions the Reviewer must answer to make the record "
            "testable (e.g. 'What is the exact HTTP method and path to reach the export "
            "endpoint?', 'What credentials/session state are needed to reach it?', 'Is the "
            "route exposed publicly or behind an unauthenticated login?')."
        )
    )


class IntegrationAuditInput(BaseModel):
    is_chained: bool = Field(
        description=(
            "True if this vulnerability can be combined with the other confirmed "
            "vulnerabilities into a concrete multi-step external exploit chain. False "
            "if no usable chain exists in isolation."
        )
    )
    confidence_score: int = Field(description="Confidence in this assessment from 1 to 10.")
    reasoning: str = Field(description="Brief technical explanation for the decision.")
    chained_with: Optional[list[str]] = Field(
        default=None,
        description=(
            "The vuln_ids of the other confirmed vulnerabilities this record chains "
            "with, in step order. REQUIRED when is_chained is true."
        ),
    )
    reproduction_steps: Optional[list[str]] = Field(
        default=None,
        description=(
            "REQUIRED when is_chained is true. The complete chronological, numbered "
            "external actions of the combined multi-step exploit, self-sufficient over "
            "the wire (HTTP method, path, required parameters/headers/body, and any "
            "session state carried from earlier steps). The downstream Validator CANNOT "
            "read source code, so every step must be executable over HTTP alone."
        ),
    )

    @model_validator(mode="after")
    def chained_requires_steps(self):
        if self.is_chained and not self.chained_with:
            raise ValueError("chained_with is required when is_chained is true.")
        if self.is_chained and not self.reproduction_steps:
            raise ValueError("reproduction_steps is required when is_chained is true.")
        return self


class VulnerabilityDetailsInput(BaseModel):
    vuln_id: str = Field(
        description=(
            "The exact vuln_id of another confirmed vulnerability to fetch full details for."
        )
    )
