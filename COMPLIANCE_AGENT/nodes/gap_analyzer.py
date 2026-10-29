"""
nodes/gap_analyzer.py
=====================
Gap Analysis node using Jev (TypeSafe AI) as the primary model for compliant vs non-compliant
classification, with Gemini structured output fallback.

For each (obligation, matched_policy_clauses) pair, Jev classifies:
  - compliant     : existing policy satisfies the SEBI requirement
  - non_compliant : policy does not satisfy the requirement (outdated, missing, or silent)

Parallelism strategy:
  asyncio.gather() fans out all obligation calls simultaneously via TypeSafeClassifier.ainvoke(),
  reducing wall-clock time from ~(N * LLM_latency) to ~max(LLM_latency).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from COMPLIANCE_AGENT.utilities.llm import get_llm
from COMPLIANCE_AGENT.prompts.prompts import (
    GAP_ANALYSIS_HUMAN,
    GAP_ANALYSIS_JEV_CHOICE_CRITERIA,
    GAP_ANALYSIS_JEV_INSTRUCTIONS,
    GAP_ANALYSIS_JEV_SEVERITY_CRITERIA,
    GAP_ANALYSIS_JEV_SEVERITY_INSTRUCTIONS,
    GAP_ANALYSIS_SYSTEM,
)
from COMPLIANCE_AGENT.src.state import GapFinding, SEBIMonitorState

log = logging.getLogger(__name__)


def _build_fallback_llm():
    return get_llm(temperature=0.0)


def _format_policy_excerpts(
    obligation: str,
    matched_policies: list[dict],
) -> str:
    """Filter matched_policies to those retrieved for this obligation without capping."""
    relevant = [
        m for m in matched_policies
        if m.get("obligation", "") == obligation
    ]
    if not relevant:
        return "  (no policy clauses found for this obligation)"

    lines = []
    for match in relevant:
        lines.append(
            f"  [Policy doc: {match['policy_doc']} | Page: {match['metadata'].get('page', '?')}]\n"
            f"  {match['policy_chunk']}\n"
        )
    return "\n".join(lines)


async def _fallback_llm_single_obligation(
    llm: Optional[ChatGoogleGenerativeAI],
    obligation: str,
    policy_excerpts: str,
    default_policy_doc: str,
    default_policy_clause: str,
) -> dict:
    """Fallback to Gemini structured output for gap analysis."""
    if llm is None:
        llm = _build_fallback_llm()

    structured_llm = llm.with_structured_output(GapFinding)
    prompt = ChatPromptTemplate.from_messages([
        ("system", GAP_ANALYSIS_SYSTEM),
        ("human",  GAP_ANALYSIS_HUMAN),
    ])
    chain = prompt | structured_llm

    try:
        result: GapFinding = await chain.ainvoke({
            "obligation":      obligation,
            "policy_excerpts": policy_excerpts,
        })
        log.info(
            "  [Fallback LLM Gap] [%s] %s | status=%s | severity=%s",
            result.policy_doc[:40], obligation[:50], result.status, result.severity
        )
        return result.model_dump()
    except Exception as exc:
        log.error("Gap analysis fallback failed for obligation '%s': %s", obligation[:60], exc)
        return {
            "obligation":   obligation,
            "policy_doc":   default_policy_doc or "unknown",
            "policy_clause": default_policy_clause or "",
            "status":       "non_compliant",
            "severity":     "high",
            "explanation":  f"Analysis failed: {exc}",
        }


async def _analyze_single_obligation(
    classifier: Optional[Any],
    llm: Optional[ChatGoogleGenerativeAI],
    obligation: str,
    matched_policies: list[dict],
    circular_id: str,
    model_name: str,
) -> dict:
    """
    Perform one gap-analysis classification for a single obligation using Jev,
    falling back to Gemini if Jev is unavailable or encounters an error.
    """
    policy_excerpts = _format_policy_excerpts(obligation, matched_policies)
    relevant = [m for m in matched_policies if m.get("obligation", "") == obligation]
    primary_match = relevant[0] if relevant else {}
    policy_doc = primary_match.get("policy_doc", "Tata Motors Internal Policies")
    policy_clause = primary_match.get("policy_chunk", "").strip()

    # ── Attempt 1: Jev (TypeSafeClassifier) for compliant vs non_compliant ──
    if classifier is not None:
        try:
            from langchain_typesafe import Choice

            state_payload = (
                f"SEBI Obligation:\n{obligation}\n\n"
                f"Matched Tata Motors Policy Excerpts:\n{policy_excerpts}"
            )
            response = await classifier.ainvoke(
                {
                    "state": state_payload,
                    "questions": {
                        "status": Choice(
                            instructions=GAP_ANALYSIS_JEV_INSTRUCTIONS,
                            criteria=GAP_ANALYSIS_JEV_CHOICE_CRITERIA,
                        ),
                        "severity": Choice(
                            instructions=GAP_ANALYSIS_JEV_SEVERITY_INSTRUCTIONS,
                            criteria=GAP_ANALYSIS_JEV_SEVERITY_CRITERIA,
                        ),
                    },
                }
            )

            status_choice = response.choices.get("status")
            sev_choice = response.choices.get("severity")

            if status_choice:
                status = status_choice.choice
                status_conf = getattr(status_choice, "confidence", 1.0)
                severity = sev_choice.choice if (status == "non_compliant" and sev_choice) else "low"
                if status == "compliant":
                    explanation = (
                        f"Jev ({model_name}) verified existing policy '{policy_doc}' satisfies "
                        f"this obligation without amendment (confidence: {status_conf:.2f})."
                    )
                else:
                    explanation = (
                        f"Jev ({model_name}) identified compliance gap in '{policy_doc}' "
                        f"(confidence: {status_conf:.2f}, severity: {severity}). Existing policy does not satisfy mandate."
                    )

                log.info(
                    "  [Jev Gap] %s | status=%s | severity=%s | conf=%.2f",
                    obligation[:40], status, severity, status_conf
                )
                return {
                    "obligation":   obligation,
                    "policy_doc":   policy_doc,
                    "policy_clause": policy_clause,
                    "status":       status,
                    "severity":     severity,
                    "explanation":  explanation,
                }
            else:
                log.warning("Jev response missing status choice for obligation '%s'; falling back to LLM.", obligation[:40])
        except Exception as exc:
            log.warning("Jev gap classification failed for '%s': %s. Falling back to LLM.", obligation[:40], exc)

    # ── Fallback: Gemini LLM ─────────────────────────────────────────────────
    return await _fallback_llm_single_obligation(llm, obligation, policy_excerpts, policy_doc, policy_clause)


async def _run_parallel_gap_analysis(
    obligations: list[str],
    matched_policies: list[dict],
    circular_id: str,
) -> list[dict]:
    """Fan out all obligations concurrently and collect findings via Jev / Gemini."""
    load_dotenv()
    typesafe_api_key = (
        os.getenv("TYPESAFE_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
    ).strip()
    base_url = os.getenv("TYPESAFE_BASE_URL", "").strip()
    default_model = "~typesafe/jev-latest" if (base_url and "openrouter" in base_url) else "jev-latest"
    model_name = os.getenv("TYPESAFE_MODEL", "").strip() or default_model

    classifier = None
    if typesafe_api_key:
        try:
            from langchain_typesafe import TypeSafeClassifier

            classifier_kwargs: dict[str, str] = {
                "api_key": typesafe_api_key,
                "model": model_name,
            }
            if base_url:
                classifier_kwargs["base_url"] = base_url

            classifier = TypeSafeClassifier(**classifier_kwargs)
            log.info("Initialized Jev TypeSafeClassifier (model=%s, base_url=%s) for gap analysis.", model_name, base_url or "default")
        except Exception as exc:
            log.warning("Could not initialize TypeSafeClassifier: %s. Using LLM directly.", exc)

    llm = _build_fallback_llm()
    tasks = [
        _analyze_single_obligation(classifier, llm, obl, matched_policies, circular_id, model_name)
        for obl in obligations
    ]
    results = await asyncio.gather(*tasks)
    return list(results)


def gap_analyzer_node(state: SEBIMonitorState) -> dict:
    """
    LangGraph node: Gap Analyzer (parallel Jev / Gemini calls).

    Runs asyncio.gather internally so all obligation→policy comparisons
    execute in parallel, classifying compliant vs non-compliant, then returns
    the merged list of gap findings.
    """
    if state.get("error"):
        log.warning("Skipping gap analyzer — upstream error.")
        return {}

    obligations = state.get("obligations", [])
    matched_policies = state.get("matched_policies", [])
    circular_id = state.get("circular_id", "unknown")

    if not obligations:
        log.warning("No obligations to analyse.")
        return {"gap_findings": []}

    log.info(
        "Starting parallel gap analysis for %d obligations with Jev primary / Gemini fallback …",
        len(obligations),
    )

    # Run async work safely in any execution context
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        findings = loop.run_until_complete(
            _run_parallel_gap_analysis(obligations, matched_policies, circular_id)
        )
    except RuntimeError:
        findings = asyncio.run(
            _run_parallel_gap_analysis(obligations, matched_policies, circular_id)
        )

    log.info("Gap analysis complete: %d findings.", len(findings))
    return {"gap_findings": findings}
