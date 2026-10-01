from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional
import json
import yaml


with open("agents.yaml", "r") as f:
    data = yaml.safe_load(f)
    TOOLS = data.get("tools")
    MANAGER_AGENT = data.get("manager_agent")
    EXPERT_AGENTS = data.get("expert_agents")
    CVE_ANALYZER_AGENT = data.get("cve_analyzer")
    VERIFIER_AGENT = data.get("contract_verifier_agent")
    REVIEWER_AGENT = data.get("reviewer_agent")
    VALIDATOR_AGENT = data.get("validator_agent")

MANIFEST_NAMES = {
    "package.json", "pyproject.toml", "Pipfile", "setup.py", "setup.cfg",
    "environment.yml", "conda.yaml", "Gemfile", "composer.json", "pom.xml", 
    "build.gradle", "build.gradle.kts", "Cargo.toml", "go.mod", "pubspec.yaml",
    "mix.exs", "Podfile", "Package.swift", "conanfile.txt", "conanfile.py",
    "vcpkg.json", "CMakeLists.txt"
}

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


class VulnerabilityRecord(BaseModel):
    vuln_id: Optional[str] = None

    # Lifecycle tracking
    status: Literal["hypothesis", "confirmed", "exploitable", "false_positive"] = "hypothesis"

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

class ManagerOutput(BaseModel):
    strategic_overview: str = Field(description="The manager's brief (max 200 words) reasoning on the app's attack surface.")
    tasks: List[ExpertTask] = Field(description="List of tasks matching predefined roles.")

class CVEDemand(BaseModel):
    security_assumption: str = Field(
        description="The specific demand or configuration requirement that must be verified in the code to prevent the vulnerability."
    )
    trigger_condition: Optional[str] = Field(
        default=None,
        description="The explicit data flow, function call, or execution sink required for the vulnerability to trigger. If the CVE description does not explicitly state how the payload is executed, leave empty."
    )
    import_namespace: str = Field(
        description="The actual module name used in the source code to import this package (e.g., if the package is 'beautifulsoup4', the import is 'bs4')."
    )

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

class SecurityDemand(BaseModel):
    direction: Literal["upstream", "downstream"] = Field(
        description="'upstream' (caller must fulfill) or 'downstream' (callee must fulfill)."
    )
    target: str = Field(
        description=(
            "STRICT FORMATTING REQUIRED for programmatic parsing. "
            "If direction is 'downstream': Format as 'module::symbol' (e.g., 'app.auth::get_jobs' or 'self::validate'). "
            "If direction is 'upstream': Format as the exact parameter name (e.g., 'query', 'user_id', or 'context')."
        )
    )
    description: str = Field(
        description="The exact security contract."
    )

class VulnerabilityHypothesis(BaseModel):
    cwe_id: str = Field(..., json_schema_extra={"enum": list(cwes.keys())})
    vulnerable_component: str = Field(
        description="The exact parameter, state transition, or function call that is flawed."
    )

class AnalysisNote(BaseModel):
    #role_in_system: str = Field(..., description="One sentence summarizing the node's purpose.")
    business_interfaces: list[str] = Field(
        default_factory=list, 
        description="List of boundaries. Prefix with [SOURCE] or [SINK] (e.g., '[SOURCE] Kafka consumer'). Leave empty if standard flow."
    )
    demands: list[SecurityDemand] = Field(
        default_factory=list, 
        description="Upstream and downstream security assumptions."
    )
    vulnerability_hypotheses: list[VulnerabilityHypothesis] = Field(
        default_factory=list,
        description="Flag localized vulnerabilities visible in this snippet."
    )

class DemandEvaluation(BaseModel):
    demand_id: str = Field(description="The exact ID extracted from the [ID: ...] tag provided in the demand description.")
    # demand_description: str = Field(description="The exact demand being evaluated")
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
