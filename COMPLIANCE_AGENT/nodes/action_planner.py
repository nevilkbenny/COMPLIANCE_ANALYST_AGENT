"""
nodes/action_planner.py
=======================
Action Plan node – 1–3 LLM calls (one per affected policy document).

Takes the gap findings from gap_analyzer and groups them by policy document.
For each affected document, calls Gemini 1.5 Flash once to produce a coherent
set of drafted amendments (amended clause text, responsible owner, deadline).

Grouping by document keeps each draft coherent — avoiding the fragmentation
that would occur if every gap finding triggered a separate, context-free call.
"""

from __future__ import annotations

import itertools
import logging
import os
from collections import defaultdict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from COMPLIANCE_AGENT.utilities.llm import get_llm

from COMPLIANCE_AGENT.prompts.prompts import ACTION_PLAN_HUMAN, ACTION_PLAN_SYSTEM
from COMPLIANCE_AGENT.src.state import ActionPlan, SEBIMonitorState

log = logging.getLogger(__name__)


class ActionPlansOutput(BaseModel):
    """Wrapper so the LLM can return multiple action plans in one call."""
    action_plans: list[ActionPlan]


def _build_llm():
    return get_llm(temperature=0.2)


def _format_gap_findings(findings: list[dict]) -> str:
    lines = []
    for i, f in enumerate(findings, 1):
        lines.append(
            f"  [{i}] Obligation : {f.get('obligation', '')}\n"
            f"       Policy clause:\n{f.get('policy_clause', '(none)')}\n"
            f"       Status      : {f.get('status', 'unknown')}\n"
            f"       Severity    : {f.get('severity', 'unknown')}\n"
            f"       Explanation : {f.get('explanation', '')}\n"
        )
    return "\n".join(lines)


def action_planner_node(state: SEBIMonitorState) -> dict:
    """
    LangGraph node: Action Planner  (1–3 LLM calls)

    Groups gap findings by policy document, then makes one LLM call per
    document to produce coherent draft amendments.  Only non-compliant
    findings (status = 'outdated' or 'silent') trigger action plans.
    """
    if state.get("error"):
        log.warning("Skipping action planner — upstream error.")
        return {}

    gap_findings = state.get("gap_findings", [])
    circular_id  = state.get("circular_id", "unknown")
    human_edits  = state.get("human_edits")

    # Filter to only gaps that need action (non-compliant)
    actionable = [
        f for f in gap_findings
        if f.get("status") in ("non_compliant", "non-compliant", "outdated", "silent")
    ]

    if not actionable:
        log.info("No actionable gaps found — no action plans needed.")
        return {"action_plans": []}

    # ── Group findings by policy document ───────────────────────────────────
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for finding in actionable:
        by_doc[finding.get("policy_doc", "unknown")].append(finding)

    log.info(
        "Drafting action plans for %d affected policy document(s): %s",
        len(by_doc),
        list(by_doc.keys()),
    )

    llm = _build_llm()
    structured_llm = llm.with_structured_output(ActionPlansOutput)
    prompt = ChatPromptTemplate.from_messages([
        ("system", ACTION_PLAN_SYSTEM),
        ("human",  ACTION_PLAN_HUMAN),
    ])
    chain = prompt | structured_llm

    all_plans: list[dict] = []

    for policy_doc, findings in by_doc.items():
        gap_findings_text = _format_gap_findings(findings)

        # Append CS edit instructions if this is a re-draft
        system_extra = ""
        if human_edits:
            gap_findings_text += (
                f"\n\nCS EDIT INSTRUCTIONS (incorporate these changes):\n{human_edits}"
            )

        log.info("  → Drafting amendment for: %s", policy_doc)
        try:
            result: ActionPlansOutput = chain.invoke({
                "policy_doc":        policy_doc,
                "gap_findings_text": gap_findings_text,
                "circular_id":       circular_id,
            })
            for plan in result.action_plans:
                all_plans.append(plan.model_dump())
                log.info(
                    "    Draft: %s | owner=%s | deadline=%s",
                    plan.policy_doc[:40], plan.owner, plan.deadline,
                )

        except Exception as exc:
            log.error("Action plan LLM call failed for '%s': %s", policy_doc, exc)
            # Stub plan so the run doesn't silently drop the document
            all_plans.append({
                "policy_doc":      policy_doc,
                "gap_summary":     "LLM call failed",
                "change_type":     "Modification",
                "existing_clause": "(Unavailable due to error)",
                "amended_clause":  f"[Draft generation failed: {exc}]",
                "diff_summary":    f"Error during drafting: {exc}",
                "owner":           "Company Secretary",
                "deadline":        "To be determined",
                "rationale":       f"Error during drafting: {exc}",
            })

    log.info("Action planner complete: %d draft amendment(s).", len(all_plans))
    return {"action_plans": all_plans}
