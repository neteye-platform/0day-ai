from pydantic import BaseModel, Field
from typing import List, Literal, Optional
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


# cwes = {
#     # --- MEMORY SAFETY (C / C++ / Rust-unsafe) ---
#     "CWE-119": "Improper Restriction of Operations within the Bounds of a Memory Buffer",
#     "CWE-416": "Use After Free",
#     "CWE-476": "NULL Pointer Dereference",
#     "CWE-190": "Integer Overflow or Wraparound",
#     # --- CONCURRENCY & EXECUTION (Go / Java / C# / Python) ---
#     "CWE-362": "Concurrent Execution using Shared Resource with Improper Synchronization (Race Condition)",
#     # --- INJECTION (Web / Cloud / DB) ---
#     "CWE-89": "SQL Injection",
#     "CWE-78": "OS Command Injection",
#     "CWE-79": "Cross-Site Scripting (XSS)",
#     "CWE-94": "Code Injection",
#     "CWE-918": "Server-Side Request Forgery (SSRF)",
#         # --- SPECIFIC ACCESS CONTROL (Flat Tier) ---
#     "CWE-862": "Missing Authorization",
#     "CWE-863": "Incorrect Authorization",
#     "CWE-639": "Authorization Bypass Through User-Controlled Key (IDOR)",
#     "CWE-306": "Missing Authentication for Critical Function",
#     # --- STATE & SESSION (Web / API) ---
#     "CWE-352": "Cross-Site Request Forgery (CSRF)",
#     "CWE-384": "Session Fixation",
#     # --- DATA & CRYPTOGRAPHY ---
#     "CWE-200": "Exposure of Sensitive Information to an Unauthorized Actor",
#     "CWE-319": "Cleartext Transmission of Sensitive Information",
#     "CWE-327": "Use of a Broken or Risky Cryptographic Algorithm",
#     "CWE-502": "Deserialization of Untrusted Data",
#     # --- CONFIGURATION & FILE SYSTEM ---
#     "CWE-22": "Path Traversal",
#     "CWE-434": "Unrestricted Upload of File with Dangerous Type",
#     "CWE-770": "Allocation of Resources Without Limits or Throttling",
#     # --- ESCAPE HATCHES (Broad Parent Categories) ---
#     "CWE-284": "Improper Access Control (Use ONLY if no specific access control CWE fits)",
#     "CWE-20": "Improper Input Validation (Use ONLY if no specific injection CWE fits)",
#     "CWE-840": "Business Logic Errors",
#     "OTHER_UNCATEGORIZED": "Use ONLY if no other CWE fits"
# }


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

# class VulnerabilityReport(BaseModel):
#     cwe_class: Literal[cwes.keys()] = Field(description=(
#         "The precise CWE ID. Mapping:\n"
#         "\n".join([f"{k}: {v}" for k, v in cwes.items()])
#     ))
#     source_node: str = Field(description="The exact Node ID where the untrusted data enters the application (e.g., the API endpoint or input parameter).")
#     sink_node: str = Field(description="The exact Node ID, form the assigned nodes list, where the vulnerability triggers. DO NOT append code snippets, explanations, or function calls to this string.")
#     trace_nodes: list[str] = Field(description="List of EXACT Node IDs representing the execution path from the source to the sink.")
#     details: str = Field(description="Technical explanation of the vulnerability.")

class EvaluationToolInput(BaseModel):
    is_exploitable: bool = Field(
        description="True if the vulnerability has a realistic path to exploitation. False if it is a false positive, purely theoretical, or blocked by standard mitigations, implemented by the application."
    )
    confidence_score: int = Field(description="Confidence in this assessment from 1 to 10.")
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

class CVEDemand(BaseModel):
    security_assumption: str = Field(
        description="The specific demand or configuration requirement that must be verified in the code to prevent the vulnerability."
    )
    import_namespace: str = Field(
        description="The actual module name used in the source code to import this package (e.g., if the package is 'beautifulsoup4', the import is 'bs4')."
    )

class VulnerabilityEvaluation(BaseModel):
    report_id: str = Field(description="The unique identifier or title of the vulnerability report.")
    is_exploitable: bool = Field(
        description="True if the vulnerability has a realistic path to exploitation. False if it is a false positive, purely theoretical, or blocked by standard mitigations."
    )
    confidence_score: int = Field(description="Confidence in this assessment from 1 to 10.")
    reasoning: str = Field(description="Brief technical explanation for the decision.")
    entry_point_url: Optional[str] = Field(description="The specific HTTP route or URI path required to reach the source node (e.g., '/dashboard').")
    http_method: Optional[str] = Field(description="The HTTP method required (e.g., 'POST', 'GET').")
    required_parameters: Optional[List[str]] = Field(description="List of expected input names, headers, or form fields.")
    auth_required: bool = Field(description="True if the route is protected by an authentication middleware.")
    original_report: list[dict]

class ReviewerOutput(BaseModel):
    reports: List[VulnerabilityEvaluation]

class ValidationToolInput(BaseModel):
    is_confirmed: bool = Field(description="True if the exploit successfully triggered in the sandbox.")
    poc_payload: Optional[str] = Field(description="The exact payload, script, or HTTP request that triggered the vulnerability.")
    execution_logs: str = Field(description="Relevant logs or output from the sandbox confirming the exploit.")

class ValidationResult(BaseModel):
    report_id: str = Field(description="The ID/title of the vulnerability being tested.")
    is_confirmed: bool = Field(description="True if the exploit successfully triggered in the sandbox.")
    poc_payload: Optional[str] = Field(description="The exact payload, script, or HTTP request that triggered the vulnerability.")
    execution_logs: str = Field(description="Relevant logs or output from the sandbox confirming the exploit.")

class PackageCheck(BaseModel):
    name: str = Field(description="The name of the package")
    version: str = Field(description="The exact version string")

class SecurityAssumption(BaseModel):
    description: str = Field(..., description="The exact security contract this node expects the target to fulfill (e.g., 'Must verify that the caller owns job_id before returning data').")
    module: str = Field(..., description="The module the symbol is imported from (e.g., 'utils', 'app.auth').")
    symbol: str = Field(..., description="The specific function, decorator, or class relied upon (e.g., 'login_required', 'get_jobs').")
    target_parameter: str = Field(..., description="The EXACT variable name, parameter, or HTTP header this assumption applies to (e.g., 'user_id', 'Vary'). If it cannot be tied to a specific variable, you must reconsider if this assumption is valid here.")

class VulnerabilityHypothesis(BaseModel):
    vulnerability_type: str = Field(..., description="The class of vulnerability (e.g., 'IDOR', 'State Machine Bypass', 'Privilege Escalation').")
    description: str = Field(..., description="The suspected flaw.")
    vulnerable_component: str = Field(..., description="The specific parameter, function call, or state transition that is flawed (e.g., 'req.query.id').")

class BusinessInterface(BaseModel):
    interface_type: Literal["source", "sink"] = Field(..., description="Strictly 'source' (untrusted data enters) or 'sink' (sensitive state changes).")
    description: str = Field(..., description="What the interface does (e.g., 'Kafka consumer for order events', 'Upgrades user role'). Ignore standard HTTP/DB flows; focus on business logic boundaries.")

class AnalysisNote(BaseModel):
    node_id: str = Field(..., description="The exact ID of the node analyzed (e.g. 'src_main_login').")
    role_in_system: str = Field(..., description="One sentence summarizing what this node does and its security context.")
    business_interfaces: List[BusinessInterface] = Field(
        default_factory=list,
        description="High-level architectural entry and exit points."
    )
    assumptions_to_verify: List[SecurityAssumption] = Field(
        default_factory=list,
        description="Security demands this node makes of its dependencies."
    )
    vulnerability_hypothesis: List[VulnerabilityHypothesis] = Field(
        default_factory=list,
        description="Suspected localized vulnerabilities. Leave empty if no anomalies are found."
    )
    imports: list[str] = Field(
        default_factory=list,
        description=(
            "A list of external modules or namespaces explicitly imported in this node's source code.\n"
            "CRITICAL: Extract ONLY the base root module name. Never include keywords like 'import', 'from', or aliases. "
            "(e.g. 'from bs4 import BeautifulSoup' -> 'bs4', 'import django.conf' -> 'django')"
        )
    )

class DemandEvaluation(BaseModel):
    demand_description: str = Field(description="The exact demand being evaluated")
    status: Literal["MET", "FAILED", "OUT_OF_SCOPE"] = Field(description="MET if the code fulfills the demand, FAILED if it does not, OUT_OF_SCOPE if the demand describes a responsibility that belongs to a different architectural layer.")
    reasoning: str = Field(description="Brief explanation referencing specific lines of code.")

class VerifierOutput(BaseModel):
    evaluations: List[DemandEvaluation]
