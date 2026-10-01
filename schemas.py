from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import json
import yaml


with open("agents.yaml", "r") as f:
    data = yaml.safe_load(f)
    EXPERT_AGENTS = data.get("expert_agents")
    TOOLS = data.get("tools")
    REVIEWER_AGENT = data.get("reviewer_agent")
    VALIDATOR_AGENT = data.get("validator_agent")

CWEs = Literal[
    # --- MEMORY SAFETY (C / C++ / Rust-unsafe) ---
    "CWE-119: Improper Restriction of Operations within the Bounds of a Memory Buffer",
    "CWE-416: Use After Free",
    "CWE-476: NULL Pointer Dereference",
    "CWE-190: Integer Overflow or Wraparound",

    # --- CONCURRENCY & EXECUTION (Go / Java / C# / Python) ---
    "CWE-362: Concurrent Execution using Shared Resource with Improper Synchronization (Race Condition)",

    # --- INJECTION (Web / Cloud / DB) ---
    "CWE-89: SQL Injection",
    "CWE-78: OS Command Injection",
    "CWE-79: Cross-Site Scripting (XSS)",
    "CWE-94: Code Injection",
    "CWE-918: Server-Side Request Forgery (SSRF)",

    # --- SPECIFIC ACCESS CONTROL (Flat Tier) ---
    "CWE-862: Missing Authorization",
    "CWE-863: Incorrect Authorization",
    "CWE-639: Authorization Bypass Through User-Controlled Key (IDOR)",
    "CWE-306: Missing Authentication for Critical Function",

    # --- STATE & SESSION (Web / API) ---
    "CWE-352: Cross-Site Request Forgery (CSRF)",
    "CWE-384: Session Fixation",

    # --- DATA & CRYPTOGRAPHY ---
    "CWE-200: Exposure of Sensitive Information to an Unauthorized Actor",
    "CWE-319: Cleartext Transmission of Sensitive Information",
    "CWE-327: Use of a Broken or Risky Cryptographic Algorithm",
    "CWE-502: Deserialization of Untrusted Data",

    # --- CONFIGURATION & FILE SYSTEM ---
    "CWE-22: Path Traversal",
    "CWE-434: Unrestricted Upload of File with Dangerous Type",
    "CWE-770: Allocation of Resources Without Limits or Throttling",

    # --- ESCAPE HATCHES (Broad Parent Categories) ---
    "CWE-284: Improper Access Control (Use ONLY if no specific access control CWE fits)",
    "CWE-20: Improper Input Validation (Use ONLY if no specific injection CWE fits)",
    "CWE-840: Business Logic Errors",
    "OTHER_UNCATEGORIZED (Use ONLY if no other CWE fits)"
]

class ExpertTask(BaseModel):
    agent_role: str = Field(
        description="The PREDEFINED role best suited for this specific architectural area.",
        json_schema_extra={"enum": list(EXPERT_AGENTS.keys())}
    )
    target_communities: List[str] = Field(
        description="List of EXACT community IDs this agent must focus on (e.g., ['1', '3', '4']). Extract these exact IDs from the summary. It MUST NOT be empty."
    )
    task_description: str = Field(
        description="Detailed instructions on what specific vulnerability classes, architectural risks, or cross-component interactions to investigate within this subgraph."
    )

class ManagerOutput(BaseModel):
    strategic_overview: str = Field(description="The manager's brief (max 200 words) reasoning on the app's attack surface.")
    tasks: List[ExpertTask] = Field(description="List of tasks matching predefined roles.")

class VulnerabilityReport(BaseModel):
    cwe_class: CWEs = Field(description="The precise CWE category.")
    source_node: str = Field(description="The exact Node ID where the untrusted data enters the application (e.g., the API endpoint or input parameter).")
    sink_node: str = Field(description="The exact Node ID, form the assigned nodes list, where the vulnerability triggers. DO NOT append code snippets, explanations, or function calls to this string.")
    trace_nodes: list[str] = Field(description="List of EXACT Node IDs representing the execution path from the source to the sink.")
    details: str = Field(description="Technical explanation of the vulnerability.")

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
    required_parameters: Optional[List[str]] = Field(
        description="List of expected input names, query parameters, headers, or form fields.",
        default_factory=list
    )
    auth_required: bool = Field(
        description="True if the route is protected by an authentication middleware.",
        default=False
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

class ValidationResult(BaseModel):
    report_id: str = Field(description="The ID/title of the vulnerability being tested.")
    is_confirmed: bool = Field(description="True if the exploit successfully triggered in the sandbox.")
    poc_payload: Optional[str] = Field(description="The exact payload, script, or HTTP request that triggered the vulnerability.")
    execution_logs: str = Field(description="Relevant logs or output from the sandbox confirming the exploit.")

