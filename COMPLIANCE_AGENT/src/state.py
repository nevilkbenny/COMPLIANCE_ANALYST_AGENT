"""
src/state.py
============
Shared LangGraph state schema for the SEBI Regulatory Change Impact Monitor (COMPLIANCE_AGENT).
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional

from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class ObligationsOutput(BaseModel):
    """Structured output from the Obligation Extraction LLM call."""
    obligations: list[str] = Field(
        description=(
            "List of concrete compliance obligations extracted from the circular. "
            "Each item is a short plain-English statement (≤ 25 words)."
        )
    )


class RelevanceOutput(BaseModel):
    """Structured output from the Relevance Judgment LLM call."""
    score: float = Field(
        ge=0.0, le=1.0,
        description="Relevance score: 0 = circular has no bearing on Tata Motors policies, 1 = directly impacts multiple policies."
    )
    reason: str = Field(
        description="One-paragraph explanation of the score."
    )


class GapFinding(BaseModel):
    """Result of one obligation vs policy clause comparison."""
    obligation: str = Field(description="The SEBI obligation being checked.")
    policy_doc: str = Field(description="Name of the Tata Motors policy document checked.")
    policy_clause: str = Field(description="The matched policy clause text (≤ 200 chars).")
    status: str = Field(description="One of: 'compliant', 'non_compliant'.")
    severity: str = Field(description="One of: 'high', 'medium', 'low'.")
    explanation: str = Field(description="Why this gap exists or why the policy is compliant.")


class ActionPlan(BaseModel):
    """Draft amendment for one gap finding with Before vs After diff."""
    policy_doc: str = Field(description="Name of the Tata Motors policy document to amend.")
    gap_summary: str = Field(description="Brief summary of the gap being addressed.")
    change_type: str = Field(description="'Addition' if inserting a new clause, or 'Modification' if amending an existing clause.")
    existing_clause: str = Field(description="Existing policy clause text before amendment (or 'None / New Clause' if currently absent).")
    amended_clause: str = Field(description="Full proposed replacement or new clause text (After).")
    diff_summary: str = Field(description="Clear summary of what was added, removed, or modified between Before and After.")
    owner: str = Field(description="Proposed responsible owner (e.g. 'Company Secretary', 'CCO', 'Audit Committee').")
    deadline: str = Field(description="Recommended compliance deadline (e.g. '30 days from circular date').")
    rationale: str = Field(description="Why this amendment satisfies the SEBI obligation.")


class SEBIMonitorState(TypedDict):
    # ── Watcher outputs ──────────────────────────────────────────────────────
    circular_id: str
    circular_url: str
    pdf_path: str
    already_processed: bool

    # ── Document Loader outputs ──────────────────────────────────────────────
    circular_text: str
    total_pages: Optional[int]
    chunks: Optional[list]

    # ── Document Validator outputs (Jev / TypeSafe) ───────────────────────────
    is_valid_document: Optional[bool]
    document_validation_reason: Optional[str]
    document_type: Optional[str]

    # ── Obligation Extraction outputs ────────────────────────────────────────
    obligations: list[str]

    # ── Retrieval outputs ────────────────────────────────────────────────────
    matched_policies: list[dict]

    # ── Relevance Judgment outputs ───────────────────────────────────────────
    relevance_score: float
    relevance_reason: str

    # ── Gap Analysis outputs (reducer: list concat for parallel fan-out) ─────
    gap_findings: Annotated[list[dict], operator.add]

    # ── Action Plan outputs ──────────────────────────────────────────────────
    action_plans: list[dict]

    # ── Human Review ─────────────────────────────────────────────────────────
    human_decision: str
    human_edits: Optional[str]

    # ── Audit ────────────────────────────────────────────────────────────────
    audit_id: Optional[int]
    run_timestamp: str

    # ── Error tracking ───────────────────────────────────────────────────────
    error: Optional[str]
