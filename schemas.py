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
    EDGE_TRAVERSAL_AGENT = data.get("edge_traversal")
    DEDUP_AGENT = data.get("dedup_agent")
    REPORTER_AGENT = data.get("reporter_agent")
    CREDENTIAL_FINDER_AGENT = data.get("credential_finder")
    PATCHER_AGENT = data.get("patcher_agent")

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
    "CWE-915": "Improperly Controlled Modification of Dynamically-Determined Object Attributes (Mass Assignment)",
    "CWE-639": "Authorization Bypass Through User-Controlled Key (IDOR)",
    "CWE-306": "Missing Authentication for Critical Function",
    "CWE-287": "Improper Authentication",
    "CWE-307": "Improper Restriction of Excessive Authentication Attempts",
    # --- STATE & SESSION (Web / API) ---
    "CWE-352": "Cross-Site Request Forgery (CSRF)",
    "CWE-384": "Session Fixation",
    "CWE-444": "Inconsistent Interpretation of HTTP Requests (HTTP Request/Response Smuggling)",
    # --- DATA & CRYPTOGRAPHY ---
    "CWE-807": "Reliance on Untrusted Inputs in a Security Decision",
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
    "CWE-915",  # Improperly Controlled Modification of Dynamically-Determined Object Attributes (Mass Assignment)
}

# Deterministic (numerically sorted) rendering used in LLM-facing field
# descriptions, so the wording never drifts from the allowlist above.
_SYSTEMIC_CWE_LABEL = ", ".join(sorted(SYSTEMIC_CWES, key=lambda c: int(c.split("-")[1])))

CWE_KEYS = Literal[tuple(cwes.keys())]


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
    mitigation: Optional[str] = Field(
        default=None,
        description=(
            "Reviewer's cited blocking defense for a `false_positive` record "
            "(file/function; per-field for enumerated filters)."
        ),
    )
    reservations: Optional[list[str]] = Field(
        default=None,
        description=(
            "'Confirmed with reservations': the reviewer's unresolved points, which the "
            "Validator must prove or refute in the sandbox."
        ),
    )
    out_of_scope_concern: Optional[str] = Field(
        default=None,
        description=(
            "A source-to-sink flow the Reviewer observed while tracing that this record's "
            "hypothesis does not cover; rendered for the Validator alongside reservations."
        ),
    )
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
            "Determines graph routing. 'direct_to_validator': Triggerable directly, or the "
            "only barrier is authentication the Validator may already be provisioned for and "
            "will attempt. 'requires_integration': Requires privileges the Validator cannot "
            "obtain, or the output of another exploit. 'static_finding_only': real in source "
            "but no network-reachable path."
        ),
    )

    cvss_vector: Optional[str] = Field(
        default=None,
        description=(
            "Reviewer's CVSS v3.1 base-vector estimate for the adjudicated finding "
            "(set on exploitable verdicts). The pipeline recomputes the numeric score "
            "from it; a confirmed record estimated below settings.validator_min_cvss "
            "is not dispatched to the Validator/Auditor and is reported unvalidated."
        ),
    )

    # Validator additions
    poc_payload: Optional[str] = None
    poc_script: Optional[str] = None
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

    # Patcher additions (post-validator source-code fix; inert when
    # settings.patcher_enabled is False — these fields stay None on every record).
    patch_summary: Optional[str] = Field(
        default=None,
        description=(
            "Patcher's one-paragraph statement of the applied fix: what hunk "
            "changes and exactly which step of the proven exploit it blocks."
        ),
    )
    patch_diff: Optional[str] = Field(
        default=None,
        description="Unified diff of every edit the Patcher applied for this record.",
    )
    patched_files: Optional[list[str]] = Field(
        default=None,
        description="App-relative paths of the files the Patcher modified.",
    )
    patch_round: int = Field(
        default=0,
        description=(
            "How many patch attempts the Patcher has spent on this record; caps "
            "the fix loop at settings.patcher_max_attempts."
        ),
    )
    patch_history: Optional[list[dict]] = Field(
        default=None,
        description=(
            "Prior patch attempts for this record ({round, summary, diff, files, "
            "outcome}), each appended by submit_patch before it overwrites the "
            "current patch_* fields. Rendered into the Patcher's first turn on "
            "attempt > 1 so a retry neither repeats a proven-failing edit nor "
            "discards the audit trail; may be empty/None on the first attempt."
        ),
    )
    patch_state: Optional[Literal["applied", "reviewed", "verified", "rejected", "failed"]] = Field(
        default=None,
        description=(
            "Patch lifecycle marker: 'applied' = edits landed, re-review pending; "
            "'reviewed' = the Reviewer re-adjudicated the patched code; 'verified' "
            "= the Validator confirmed the fix (exploit dead, legitimate flow "
            "intact); 'rejected' = the exploit still fired on the patched build "
            "(retry-eligible until patch_round reaches settings."
            "patcher_max_attempts, then terminal); 'failed' = the Patcher produced "
            "no edit (terminal)."
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

class PatcherOutput(BaseModel):
    # Same constraint as ValidatorOutput: the patcher subgraph writes ONLY the
    # patched record back to MasterState (patch_log stays subgraph-internal).
    vulnerabilities: list[VulnerabilityRecord]

class UpstreamDemand(BaseModel):
    target: str = Field(description="Parameter or context variable requiring upstream restriction.")
    description: str = Field(description="The security invariant required of the caller.")

class DownstreamDemand(BaseModel):
    target: str = Field(description="External symbol or route called (format: 'module::symbol').")
    description: str = Field(description="The security requirement the callee must enforce.")

class Hypothesis(BaseModel):
    cwe: CWE_KEYS = Field(description="The matching CWE ID.")
    component: str = Field(description="The vulnerable parameter, logic check, or missing control.")
    description: str = Field(
        description=(
            "One or two sentences: which untrusted input reaches which exact "
            "sink call or omitted control, and the missing guard. A bare "
            "parameter, function, or CWE name is invalid."
        )
    )
    pattern_label: Optional[str] = Field(
        default=None,
        description=(
            f"Short canonical name (≤6 lowercase words) REQUIRED for architectural flaws: "
            f"{_SYSTEMIC_CWE_LABEL}. "
            "Leave null for localized injection defects."
        ),
    )

class AnalysisNote(BaseModel):
    sources: list[str] | None = Field(default=None, description="External untrusted data entering this snippet.")
    sinks: list[str] | None = Field(default=None, description="Dangerous execution sinks, state changes, or auth checks.")
    upstream: list[UpstreamDemand] | None = Field(
        default=None,
        description=(
            "Populate ONLY if an attacker with full control over an input can achieve "
            "an exploitable security impact (e.g., unauthorized access, injection, data tampering) "
            "within this snippet unless the caller enforces an invariant. "
            "Leave empty if malicious input cannot cause a security compromise here."
        )
    )
    downstream: list[DownstreamDemand] | None = Field(
        default=None,
        description=(
            "Populate ONLY when delegating sensitive operations (e.g. authentication, "
            "token verification, access checks, or database persistence) where the callee "
            "must enforce specific security guarantees to prevent a security bypass."
        )
    )
    vulns: list[Hypothesis] | None = Field(
        default=None,
        description=(
            "Populate ONLY if an exploitable flaw is self-contained within this snippet: "
            "either untrusted data reaches an unmitigated execution sink, or a security-critical "
            "operation (e.g. authentication, authorization, or sensitive state changes) "
            "omits necessary constraints or rate limits. "
            "Leave empty if safety depends on caller data validation."
        )
    )

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
            "FAILED: The code manipulates data insecurely, implements a weak/fragile mitigation, or completely omits a required check/counter (e.g. unmetered credential verification).\n"
            "DELEGATED: the code passes the untrusted data to a helper function, validator, or sanitizer whose implementation is NOT visible in the snippet.\n"
            "OUT_OF_SCOPE: the demand targets a different layer (e.g., expecting a database helper to handle HTTP cookies) or a different context (e.g., HTML configuration on a Markdown exporter)."
        )
    )
    evidence: str = Field(
        description="A single factual statement (max 30 words) specifying the exact function call, sanitizer, or missing check that justifies the status."
    )
    cwe: Optional[CWE_KEYS] = Field(
        default=None,
        description="The matching CWE ID if status is FAILED; null otherwise."
    )

class VerifierOutput(BaseModel):
    evaluations: list[DemandEvaluation]

# ==========================================
# Tools
# ==========================================

# Shared CVSS v3.1 base-metric primer, reused by every schema that makes the
# model emit a vector (the reviewer's pre-validation estimate and the
# reporter's final assessment) so the definitions can never diverge.
CVSS_V31_BASE_HELP = (
    "A complete CVSS v3.1 BASE vector string, e.g. 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H'. "
    "Pick the metrics honestly from the evidence, using these definitions:\n"
    "AV  Attack Vector: N(network) / A(adjacent) / L(local) / P(physical)\n"
    "AC  Attack Complexity: L(low) / H(high)\n"
    "PR  Privileges Required: N(none) / L(low) / H(high)\n"
    "UI  User Interaction: N(none) / R(required)\n"
    "S   Scope: U(unchanged) / C(changed)\n"
    "C,I,A  Confidentiality / Integrity / Availability impact: H(high) / L(low) / N(none)\n"
)
CVSS_V31_BASE_EXAMPLES = (
    "Choose metrics that match the proven reproduction: a finding triggered over HTTP by an "
    "unauthenticated attacker is AV:N/AC:L/PR:N/UI:N/S:U; a client-side XSS requires UI:R; an "
    "admin-only route is PR:H; a compromise that moves past the vulnerable component into adjacent "
    "assets (e.g. sandbox escape, RCE that reaches the host) is S:C. "
    "The pipeline recomputes the numeric base score from this vector."
)

class EvaluationToolInput(BaseModel):
    reasoning: str = Field(
        description="Brief technical explanation for the decision."
    )
    mitigation_bypass: Optional[str] = Field(
        default=None,
        description="Optionally describe how an attacker bypasses the defenses or can abuse the application logic. Optional — do NOT set it to justify a false positive; the blocking defense goes in `mitigation`."
    )
    untrusted_uses: Optional[list[str]] = Field(
        default=None,
        description=(
            "REQUIRED for code-level false positives: every use of the untrusted value in the "
            "traced flow, one per line — file:line, operation, why it cannot reach a sink. "
            "Include derived variables and warn-and-continue branches; follow the request "
            "value under every name it takes (not just the flagged parameter or stored "
            "field). If the value is never read at all, one entry citing the searches "
            "proving that is enough. For systemic false positives: one entry per affected "
            "node citing its own blocking defense. Any use you cannot exclude: "
            "is_exploitable=true + `reservations`."
        ),
    )
    out_of_scope_concern: Optional[str] = Field(
        default=None,
        description=(
            "A source-to-sink flow you OBSERVED while tracing that this hypothesis does not "
            "cover: name the request parameter, file:line, sink. A false positive carrying "
            "one is rejected — rule it out via `untrusted_uses` or resubmit as exploitable "
            "with the full chain in `reproduction_steps`."
        ),
    )
    reservations: Optional[list[str]] = Field(
        default=None,
        description=(
            "Points you could not settle statically (unaudited fields, assumptions only a "
            "live exploit can confirm). Submit with is_exploitable=true + "
            "'direct_to_validator' to mark the finding 'confirmed with reservations'; the "
            "Validator must prove or refute each one in the sandbox."
        ),
    )
    is_exploitable: bool = Field(
        description="True if there is a realistic path to exploitation. False if it is a false positive, purely theoretical, or blocked by application mitigations."
    )
    cvss_vector: Optional[str] = Field(
        default=None,
        description=(
            "REQUIRED when is_exploitable is true; never set for false positives. "
            "Your honest CVSS severity estimate of the adjudicated flaw, judged from the "
            "flow you traced (not a number you wish it were).\n"
            + CVSS_V31_BASE_HELP +
            "The pipeline recomputes the numeric base score from this vector and it decides "
            "whether a live Validator is spent on the finding, so an inflated vector wastes "
            "sandbox time and a deflated one hides the finding's true risk."
        )
    )
    mitigation: Optional[str] = Field(
        default=None,
        description=(
            "REQUIRED when is_exploitable is false: the defense blocking exploitation, cited "
            "to file + function. A defense must HALT the flow: log/warn-only checks are not "
            "defenses. Name every user-controllable field reaching the sink and its guard; "
            "if the list cannot be complete, use is_exploitable=true + `reservations`."
        ),
    )
    validation_strategy: Optional[Literal["direct_to_validator", "requires_integration", "static_finding_only"]] = Field(
        default=None,
        description=(
            "REQUIRED when is_exploitable is true (pick exactly one). Leave unset for false positives.\n"
            "- 'direct_to_validator': Triggerable directly, or the only barrier is authentication the Validator may already be provisioned for and will attempt. Do NOT search for credentials — judge reachability only.\n"
            "- 'requires_integration': Demands privileges the Validator cannot obtain OR the output of another confirmed exploit.\n"
            "- 'static_finding_only': 100% real in source code but NO network-reachable exploit path (e.g., plaintext DB passwords)."
        )
    )
    reproduction_steps: Optional[list[str]] = Field(
        default_factory=list,
        description=(
            "REQUIRED when is_exploitable is true. Leave unset for false positives.\n"
            "Chronological, self-sufficient external actions required to trigger the vulnerability over HTTP. "
            "The downstream Validator cannot read source code. You MUST state exact HTTP methods, paths, parameters, headers, and carried session state. "
            "Example: '1. POST credentials to /api/login and capture cookie. 2. POST /api/export with JSON {\"title\":\"<payload>\"} using cookie.' "
            "Leave empty for false positives."
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
        if self.is_exploitable and not (self.cvss_vector or "").strip():
            raise ValueError(
                "cvss_vector is required when is_exploitable is true: a complete CVSS v3.x "
                "base vector (e.g. 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H') honestly "
                "estimating the adjudicated flaw. False positives carry no severity estimate."
            )
        if not self.is_exploitable and (self.cvss_vector or "").strip():
            raise ValueError(
                "cvss_vector must stay unset for false positives: a blocked flow has no "
                "severity estimate. If the flow is not actually blocked, submit "
                "is_exploitable=true with your vector instead."
            )
        if not self.is_exploitable and not (self.mitigation or "").strip():
            raise ValueError(
                "mitigation is required for false positives: cite the concrete defense "
                "blocking the exploit in code. If you cannot, submit is_exploitable=true "
                "with 'direct_to_validator' and the open points in `reservations`."
            )
        return self

    @field_validator("cvss_vector", mode="before")
    @classmethod
    def _validate_cvss_vector(cls, v):
        """Strict for the reviewer: the numeric score gates Validator dispatch, so the
        vector must be fully parseable, not just CVSS-prefixed."""
        if v is None or not isinstance(v, str):
            return v
        v = v.strip()
        if not v:
            return None
        # Lazy import: utils imports schemas at module level (merge_vulnerabilities).
        from utils import cvss_v3_base_score
        if not v.startswith("CVSS:3.") or cvss_v3_base_score(v) is None:
            raise ValueError(
                "cvss_vector must be a complete CVSS v3.x base vector: start with "
                "'CVSS:3.' and carry all eight base metrics AV/AC/PR/UI/S/C/I/A, e.g. "
                "'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H'. The pipeline scores it "
                "deterministically; a partial or malformed vector is rejected."
            )
        return v

    @field_validator("reproduction_steps", mode="before")
    @classmethod
    def coerce_steps_to_list(cls, v):
        if isinstance(v, str):
            # Split the string by newlines, strip whitespace, and drop empty lines
            return [step.strip() for step in v.split("\n") if step.strip()]
        return v

class ValidationToolInput(BaseModel):
    is_confirmed: bool = Field(description="True if the exploit successfully triggered in the sandbox.")
    poc_payload: Optional[str] = Field(description="The exact payload, script, or HTTP request that triggered the vulnerability.")
    poc_script: Optional[str] = Field(
        default=None,
        description=(
            "When the proof is a script you wrote with write_attacker_file: path of "
            "that complete, runnable PoC script RELATIVE TO YOUR WORKDIR (e.g. "
            "'main.py' or 'pocs/exploit.py'). The final report ships this script to "
            "the human reader. Leave unset for pure single-request proofs."
        )
    )
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


class SubmitPatchInput(BaseModel):
    summary: str = Field(
        description=(
            "One paragraph describing the applied fix: which file/hunk changed, the "
            "mechanism of the defense (e.g. prepared statement, allowlist, escape at "
            "the sink), and exactly which step of the proven exploit this blocks. "
            "Reference the edits actually applied via patch_source_file — a summary "
            "with no applied edit is rejected."
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


# ==========================================
# Edge Traversal agent
# ==========================================

EDGE_BOUNDARY_TYPES = Literal["in_process", "async_messaging", "network_ipc", "infra"]


class EdgeTraversalFinding(BaseModel):
    vulnerability_type: Literal["cross_boundary_contract_mismatch", "differential_parsing", "confused_deputy"] = Field(
        description=(
            "The composite vulnerability class: "
            "'cross_boundary_contract_mismatch' (an assumption at one side of the "
            "boundary is not upheld by the other, e.g. the source strips/omits auth "
            "context before dispatch while the target assumes an authenticated/trusted "
            "caller); "
            "'differential_parsing' (the two sides parse/encode the same payload "
            "differently — header or normalization/serialization discrepancies such as "
            "HTTP request smuggling, path normalization mismatch, unsafe deserialization "
            "handoff); "
            "'confused_deputy' (a privileged component acts on attacker-influenced "
            "instructions from a less-trusted component without re-checking authority)."
        )
    )
    cwe_id: CWE_KEYS = Field(
        description="The matching CWE ID from the provided list (e.g., CWE-862 for missing authorization across queues, CWE-444 for HTTP smuggling, CWE-502 for uncoordinated serialization, CWE-22 for normalization bypasses)."
    )
    affected_nodes: list[str] = Field(
        description="EXACTLY two graph node IDs: [source_node, target_node] (or [infra_config_endpoint, app_route_node]). Use the exact u/v node IDs shown in the boundary edge header."
    )
    gap_details: str = Field(
        description="Explicit description of the semantic mismatch across the boundary (e.g., 'Node A strips the user auth token before enqueuing the task; Node B assumes every incoming queue task is pre-authorized')."
    )
    validation_strategy: Literal["direct_to_validator", "requires_integration", "static_finding_only"] = Field(
        description="'requires_integration' for multi-step logic gaps that depend on another exploit output or privileges the Validator cannot obtain; 'direct_to_validator' for infrastructure/parsing discrepancies reproducible over HTTP; 'static_finding_only' for real-in-source mismatches with no network-reachable path."
    )

    @field_validator('affected_nodes')
    @classmethod
    def _exactly_two_nodes(cls, v: list[str]) -> list[str]:
        if not v or len(v) > 2:
            raise ValueError("affected_nodes must contain exactly the [source, target] node IDs.")
        return v


class EdgeInvariantAssertion(BaseModel):
    source_node: str = Field(description="The upstream/source node of the boundary edge.")
    target_node: str = Field(description="The downstream/target node of the boundary edge.")
    satisfied: bool = Field(
        description="True when the upstream node's sanitization/validation demonstrably satisfies the downstream node's entry demands (the edge is SAFE and must NOT produce a finding). False otherwise."
    )
    reasoning: str = Field(description="Brief technical explanation referencing the visible exit/ingress code.")


class EdgeTraversalOutput(BaseModel):
    findings: list[EdgeTraversalFinding] = Field(
        default_factory=list,
        description="One entry per genuinely composition-only vulnerability found across the boundary edges in this batch. Empty when none exist."
    )
    assertions: list[EdgeInvariantAssertion] = Field(
        default_factory=list,
        description="Edge invariant assertions: flags confirming when an upstream boundary's validation satisfies a downstream node's entry demands, pruning unnecessary false-positive evaluations. Emit one per edge you explicitly verified as safe."
    )


class DedupCluster(BaseModel):
    reason: str = Field(
        description=(
            "One short sentence naming the shared sink or root cause that "
            "justifies the merge (file/function or the shared vulnerable call)."
        ),
    )
    member_vuln_ids: list[str] = Field(
        min_length=2,
        description=(
            "Two or more vuln_ids from THIS group that describe the SAME "
            "underlying defect (same sink at the same code location reached by "
            "the same kind of untrusted input, or one shared root cause fixed "
            "by one change at one place). Copy the vuln_id strings VERBATIM."
        ),
    )


class DedupAgentOutput(BaseModel):
    clusters: list[DedupCluster] = Field(
        default_factory=list,
        description=(
            "Equivalence classes of true duplicate hypotheses in this group. "
            "Clusters must be DISJOINT (one vuln_id in at most one cluster). "
            "Empty when every record is a distinct defect."
        ),
    )


# ==========================================
# Reporter agent
# ==========================================

class ReporterFinding(BaseModel):
    title: str = Field(
        description="Short, clean, human-readable title for this finding (a few words). You may reuse the meaningful parts of the raw vulnerability ID or CWE description (e.g. 'Unauthenticated SQL Injection in Export Endpoint'). Keep it brief and readable: no file paths, no full node IDs, no boilerplate."
    )
    summary: str = Field(
        description="As short as possible: the finding in one or two direct sentences, distilled from the record's description and reviewer reasoning. Nothing beyond what a reader needs to grasp it."
    )
    cvss_vector: str = Field(
        description=CVSS_V31_BASE_HELP + CVSS_V31_BASE_EXAMPLES
    )
    severity: Literal["Critical", "High", "Medium", "Low", "None"] = Field(
        description="Qualitative severity matching the CVSS v3 score ranges (Critical >= 9.0, High >= 7.0, Medium >= 4.0, Low >= 0.1, None = 0.0). Overridden by the pipeline if it disagrees with the vector's computed score."
    )
    reproduction_steps: list[str] = Field(
        description="Chronological, self-sufficient reproduction steps for this vulnerability, rewritten/updated from the validator's PoC payload and execution logs so a reader can reproduce it from scratch. State the exact HTTP method, path, parameters/headers/body, carried session state, and the observable evidence of success. Do NOT paste the raw PoC payload or raw execution logs."
    )
    worst_case_scenario: str = Field(
        description="Decisive and terse answer to 'what is the worst thing that could happen if a malicious actor exploits this vulnerability?', grounded in this vulnerability's real mechanics and the application's actual function."
    )
    remediation: Optional[str] = Field(
        default=None,
        description="A one-line statement of the concrete fix (code change, configuration, or library upgrade) followed by a MINIMAL, correctly fenced and language-tagged code snippet showing the change (e.g. ```php / ```yaml / ```sql). Prefer the snippet over prose; put the fenced block at the start of its own line. Leave null when no concrete remediation is known."
    )

    @field_validator('cvss_vector')
    @classmethod
    def _validate_cvss_vector(cls, v: str) -> str:
        if not isinstance(v, str):
            return v
        v = v.strip()
        if not v.startswith("CVSS:3."):
            raise ValueError("cvss_vector must be a CVSS v3.x base vector (start with 'CVSS:3.').")
        return v


# ==========================================
# Credential finder agent (preprocessing)
# ==========================================

class CredentialRecord(BaseModel):
    service: str = Field(
        description=(
            "Short label identifying what this credential grants access to "
            "(e.g. 'GLPI administrator login', 'MySQL root', 'GLPI application "
            "database user'). Prefer the application/service name over the env "
            "key alone."
        )
    )
    kind: Literal["login", "database", "api_key", "secret", "other"] = Field(
        description="'login' for an interactive user account, 'database' for a DB user, 'api_key' for a token/API key, 'secret' for a raw secret (signing key, root password), 'other' otherwise."
    )
    username: Optional[str] = Field(
        default=None,
        description="The username/account identifier when one is known (e.g. 'glpi', 'root'). Leave unset for bare secrets without a principal."
    )
    secret: str = Field(
        description="The credential value: the plaintext password, token, API key, or secret."
    )
    source: Optional[str] = Field(
        default=None,
        description="Where this was found, e.g. '.env:3', 'docker-compose.yml' service 'db', 'install/empty_data.php:9420', 'image_metadata Env'."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional context: which account/profile this belongs to (e.g. 'administrator'), whether it is a well-known default, and anything that helps a consumer use it."
    )


class CredentialList(BaseModel):
    credentials: list[CredentialRecord] = Field(
        description="The normalized, deduplicated set of pre-configured credentials. Empty when none are real."
    )
