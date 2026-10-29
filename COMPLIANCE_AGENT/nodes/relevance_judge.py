"""
nodes/relevance_judge.py
========================
Relevance Judgment node using Jev (TypeSafe AI) as primary model with Gemini fallback.

Presents the circular's obligations and the retrieved policy matches to
Jev (TypeSafeClassifier), which scores how materially the circular affects
Tata Motors' internal policies on a 0–1 scale.

The score is used by the conditional edge in graph.py:
  score < RELEVANCE_THRESHOLD  → archive (audit_logger only)
  score ≥ RELEVANCE_THRESHOLD  → full gap analysis pipeline
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from COMPLIANCE_AGENT.utilities.llm import get_llm
from COMPLIANCE_AGENT.prompts.prompts import (
    RELEVANCE_JUDGMENT_HUMAN,
    RELEVANCE_JUDGMENT_JEV_CHOICE_CRITERIA,
    RELEVANCE_JUDGMENT_JEV_INSTRUCTIONS,
    RELEVANCE_JUDGMENT_JEV_SCORE_CRITERIA,
    RELEVANCE_JUDGMENT_SYSTEM,
)
from COMPLIANCE_AGENT.src.state import RelevanceOutput, SEBIMonitorState

log = logging.getLogger(__name__)

# Default threshold; override via RELEVANCE_THRESHOLD in .env
_DEFAULT_THRESHOLD = 0.4


def _build_fallback_llm():
    return get_llm(temperature=0.0)


def _format_obligations(obligations: list[str]) -> str:
    return "\n".join(f"  {i}. {obl}" for i, obl in enumerate(obligations, 1))


def _format_policy_matches(matched_policies: list[dict]) -> str:
    """Format matched policy chunks as a readable block for the prompt without capping."""
    lines = []
    for match in matched_policies:
        lines.append(
            f"  [Policy: {match['policy_doc']}]\n"
            f"  Obligation matched: {match['obligation']}\n"
            f"  Excerpt:\n{match['policy_chunk']}\n"
        )
    return "\n".join(lines) or "  (no policy clauses retrieved)"


def _fallback_llm_relevance(obligations_text: str, policy_matches_text: str) -> tuple[float, str]:
    """Fallback relevance judgment using Gemini structured output."""
    llm = _build_fallback_llm()
    structured_llm = llm.with_structured_output(RelevanceOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", RELEVANCE_JUDGMENT_SYSTEM),
        ("human",  RELEVANCE_JUDGMENT_HUMAN),
    ])

    chain = prompt | structured_llm
    result: RelevanceOutput = chain.invoke({
        "obligations_text":    obligations_text,
        "policy_matches_text": policy_matches_text,
    })
    return result.score, result.reason


def relevance_judge_node(state: SEBIMonitorState) -> dict:
    """
    LangGraph node: Relevance Judgment (Jev / TypeSafe AI with Gemini fallback).

    Produces relevance_score (float 0–1) and relevance_reason (string).
    """
    load_dotenv()

    if state.get("error"):
        log.warning("Skipping relevance judge — upstream error.")
        return {}

    if state.get("already_processed"):
        log.info("Circular already processed — skipping relevance judge.")
        return {}

    obligations = state.get("obligations", [])
    matched_policies = state.get("matched_policies", [])
    circular_id = state.get("circular_id", "unknown")

    if not obligations:
        log.warning("No obligations — defaulting relevance score to 0.")
        return {"relevance_score": 0.0, "relevance_reason": "No obligations extracted."}

    obligations_text = _format_obligations(obligations)
    policy_matches_text = _format_policy_matches(matched_policies)

    typesafe_api_key = (
        os.getenv("TYPESAFE_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
    ).strip()
    base_url = os.getenv("TYPESAFE_BASE_URL", "").strip()
    default_model = "~typesafe/jev-latest" if (base_url and "openrouter" in base_url) else "jev-latest"
    model_name = os.getenv("TYPESAFE_MODEL", "").strip() or default_model

    score: float = 0.0
    reason: str = ""

    # ── Attempt 1: Jev (TypeSafe AI / OpenRouter) if configured ───────────────
    if typesafe_api_key:
        try:
            log.info(
                "Judging relevance for circular '%s' with Jev (model: %s, base_url: %s)...",
                circular_id,
                model_name,
                base_url or "https://api.typesafe.ai",
            )
            from langchain_typesafe import Choice, Score, TypeSafeClassifier

            classifier_kwargs: dict[str, str] = {
                "api_key": typesafe_api_key,
                "model": model_name,
            }
            if base_url:
                classifier_kwargs["base_url"] = base_url

            classifier = TypeSafeClassifier(**classifier_kwargs)
            state_input = (
                f"SEBI Circular Obligations:\n{obligations_text}\n\n"
                f"Retrieved Tata Motors Policy Clauses:\n{policy_matches_text}"
            )
            response = classifier.invoke(
                {
                    "state": state_input,
                    "questions": {
                        "relevance_impact": Choice(
                            instructions=RELEVANCE_JUDGMENT_JEV_INSTRUCTIONS,
                            criteria=RELEVANCE_JUDGMENT_JEV_CHOICE_CRITERIA,
                        ),
                        "relevance_score": Score(
                            instructions="Score the material impact level of this SEBI circular on Tata Motors policies from 0 (no impact) to 3 (critical impact).",
                            criteria=RELEVANCE_JUDGMENT_JEV_SCORE_CRITERIA,
                        ),
                    },
                }
            )

            score_ans = response.scores.get("relevance_score")
            choice_ans = response.choices.get("relevance_impact")

            if score_ans:
                # 4 rubric levels (0 to 3) -> normalized to [0.0, 1.0]
                raw_score = float(score_ans.score)
                score = round(min(1.0, max(0.0, raw_score / 3.0)), 2)
                confidence = getattr(score_ans, "confidence", 1.0)
                category = choice_ans.choice if choice_ans else "assessed"
                reason = (
                    f"Jev ({model_name}) assessed circular relevance as '{category}' "
                    f"with score {score:.2f} (confidence: {confidence:.2f})."
                )
                log.info("Jev relevance result: score=%.2f, category=%s, confidence=%.2f", score, category, confidence)
            elif choice_ans:
                category = choice_ans.choice
                score_map = {"not_relevant": 0.1, "peripheral": 0.3, "moderate": 0.6, "high": 0.9}
                score = score_map.get(category, 0.5)
                confidence = getattr(choice_ans, "confidence", 1.0)
                reason = f"Jev ({model_name}) categorized circular impact as '{category}' with score {score:.2f} (confidence: {confidence:.2f})."
                log.info("Jev relevance result: category=%s, score=%.2f", category, score)
            else:
                log.warning("Jev response missing relevance score/choice; falling back to LLM.")
                score, reason = _fallback_llm_relevance(obligations_text, policy_matches_text)

        except Exception as exc:
            log.warning("Jev (TypeSafeClassifier) call encountered an issue: %s. Falling back to LLM relevance judge.", exc)
            try:
                score, reason = _fallback_llm_relevance(obligations_text, policy_matches_text)
            except Exception as fallback_exc:
                log.error("Fallback LLM relevance judge failed: %s", fallback_exc)
                score = 1.0
                reason = f"Relevance judgment failed ({fallback_exc}); defaulting to relevant for safety."
    else:
        log.info("Neither TYPESAFE_API_KEY nor OPENROUTER_API_KEY configured. Using fallback LLM for relevance judgment.")
        try:
            score, reason = _fallback_llm_relevance(obligations_text, policy_matches_text)
        except Exception as fallback_exc:
            log.error("Fallback LLM relevance judge failed: %s", fallback_exc)
            score = 1.0
            reason = f"Relevance judgment failed ({fallback_exc}); defaulting to relevant for safety."

    log.info("Final relevance score: %.2f | Reason: %s", score, reason[:120])
    return {
        "relevance_score": score,
        "relevance_reason": reason,
    }
