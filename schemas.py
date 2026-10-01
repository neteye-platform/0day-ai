from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional
import json
import yaml
import re


with open("agents.yaml", "r") as f:
    data = yaml.safe_load(f)
    TOOLS = data.get("tools")
    MANAGER_AGENT = data.get("manager_agent")
    EXPERT_AGENTS = data.get("expert_agents")
    CVE_ANALYZER_AGENT = data.get("cve_analyzer")
    THREAT_INTEL_AGENT = data.get("threat_intel")
    VERIFIER_AGENT = data.get("contract_verifier_agent")
    REVIEWER_AGENT = data.get("reviewer_agent")
    VALIDATOR_AGENT = data.get("validator_agent")

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

CWE_KEYS = Literal[
    "CWE-119", "CWE-416", "CWE-476", "CWE-190", "CWE-362", "CWE-89",
    "CWE-78", "CWE-79", "CWE-94", "CWE-918", "CWE-862", "CWE-863",
    "CWE-639", "CWE-306", "CWE-352", "CWE-384", "CWE-200", "CWE-319",
    "CWE-327", "CWE-502", "CWE-22", "CWE-434", "CWE-770", "CWE-284",
    "CWE-20", "CWE-840", "OTHER_UNCATEGORIZED"
]


class VulnerabilityRecord(BaseModel):
    vuln_id: Optional[str] = None

    # Lifecycle tracking
    status: Literal["hypothesis", "unreachable", "confirmed", "exploitable", "false_positive"] = "hypothesis"

    # Core details (from Explorer/Verifier)
    node_id: str
    cwe_id: str = Field(
        description=(
            "The exact CWE ID. Mapping:\n"
            "\n".join([f"{k}: {v}" for k, v in cwes.items()])
        ),
        json_schema_extra={"enum": list(cwes.keys())}
    )
    vulnerability_type: str = "Code Defect"
    description: str
    demand_id: Optional[str] = None
    source_cve: Optional[str] = Field(
        default=None,
        description="The CVE ID a dependency-internal vulnerability was derived from. Set only on hypotheses emitted directly by the CVE analyzer (upgrade_only CVEs)."
    )
    vulnerable_component: Optional[str] = Field(
        default=None,
        description="The structural anchor from the Explorer hypothesis."
    )

    # Reviewer additions
    reviewer_reasoning: Optional[str] = None

    # Validator additions
    poc_payload: Optional[str] = None
    execution_logs: Optional[str] = None

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
        if not self.vuln_id:
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

            self.vuln_id = f"{self.node_id}:{self.cwe_id}:{anchor}"
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
    import_namespace: str = Field(
        description="The SINGLE top-level root module name used to import this package (e.g., 'bs4' for beautifulsoup4, 'flask' for Flask). You MUST output exactly one word. Required for BOTH fix categories to locate usage sites in application code.",
        pattern=r"^[a-zA-Z0-9_\-]+$"
    )
    required_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Strict, machine-readable list of code-level triggers that MUST appear verbatim in the "
            "application's source code for this vulnerability to be triggerable: specific function/method "
            "names (e.g. 'yaml.load', 'jwt.verify'), option/flag names (e.g. 'parseNested', 'remotePatterns'), "
            "import paths, file paths, or directives (e.g. 'next/image', '/_next/image'). "
            "REQUIRED for BOTH fix categories. NEVER use generic terms "
            "like 'import', 'request', 'file', 'input', or 'data'. Use the bare package name "
            "ONLY if no finer-grained method/option trigger exists."
        )
    )
    security_assumption: Optional[str] = Field(
        default=None,
        description="REQUIRED iff fix_category is 'application_mitigation'. The specific demand or configuration requirement that must be verified in the code."
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
    entry_point_url: Optional[str] = Field(description="The specific HTTP route or URI path required to reach the source node (e.g., '/dashboard').")
    http_method: Optional[str] = Field(description="The HTTP method required (e.g., 'POST', 'GET').")
    required_parameters: Optional[list[str]] = Field(description="List of expected input names, headers, or form fields.")
    auth_required: bool = Field(description="True if the route is protected by an authentication middleware.")
    original_report: list[dict]

class ReviewerOutput(BaseModel):
    vulnerabilities: list[VulnerabilityEvaluation]

class ValidatorOutput(BaseModel):
    reports: list[ValidationResult]

class ValidationResult(BaseModel):
    # report_id: str = Field(description="The ID/title of the vulnerability being tested.")
    is_confirmed: bool = Field(description="True if the exploit successfully triggered in the sandbox.")
    poc_payload: Optional[str] = Field(description="The exact payload, script, or HTTP request that triggered the vulnerability.")
    execution_logs: str = Field(description="Relevant logs or output from the sandbox confirming the exploit.")

class PackageCheck(BaseModel):
    name: str = Field(description="The name of the package")
    version: str = Field(description="The exact version string")

class UpstreamDemand(BaseModel):
    target: str = Field(description="Exact parameter name (e.g., 'query', 'user_id').")
    description: str = Field(description="The security contract.")

class DownstreamDemand(BaseModel):
    target: str = Field(description="Format as 'module::symbol' (e.g., 'app.auth::get_jobs').")
    description: str = Field(description="The security contract.")

class Hypothesis(BaseModel):
    cwe: CWE_KEYS = Field(description="The matching CWE ID from the provided list.")
    component: str = Field(description="The exact parameter, state transition, or function call that is flawed.")

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
    cwe_id: Optional[str] = Field(
        default=None, 
        description=(
            "If status is FAILED, provide the exact CWE ID that best represents this broken assumption. Mapping:\n"
            "\n".join([f"{k}: {v}" for k, v in cwes.items()])
        ),
        json_schema_extra={"enum": list(cwes.keys())}
    )

class VerifierOutput(BaseModel):
    evaluations: list[DemandEvaluation]

# ==========================================
# Tools
# ==========================================

class EvaluationToolInput(BaseModel):
    is_exploitable: bool = Field(
        description="True if the vulnerability has a realistic path to exploitation. False if it is a false positive, purely theoretical, or blocked by mitigations, implemented by the application."
    )
    reasoning: str = Field(description="Brief technical explanation for the decision.")
    entry_point_url: Optional[str] = Field(
        description="The specific HTTP route or URI path required to reach the source node (e.g., '/dashboard').",
        default=None
    )
    http_method: Optional[str] = Field(
        description="The HTTP method required (e.g., 'POST', 'GET').",
        default=None
    )
    required_parameters: Optional[list[str]] = Field(
        description="List of expected input names, query parameters, headers, or form fields.",
        default_factory=list
    )
    auth_required: bool = Field(
        description="True if the route is protected by an authentication middleware.",
        default=False
    )

class ValidationToolInput(BaseModel):
    is_confirmed: bool = Field(description="True if the exploit successfully triggered in the sandbox.")
    poc_payload: Optional[str] = Field(description="The exact payload, script, or HTTP request that triggered the vulnerability.")
    execution_logs: str = Field(description="Relevant logs or output from the sandbox confirming the exploit.")
