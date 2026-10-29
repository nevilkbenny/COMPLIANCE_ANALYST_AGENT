"""
nodes/document_validator.py
===========================
Document Validator node using Jev (TypeSafe AI).

Validates whether an uploaded document is relevant to the SEBI regulatory
compliance pipeline immediately after it is loaded by document_loader.

If an uploaded document is completely unrelated (e.g. invoice, receipt,
payment request, marketing slide, internal voucher), this validator catches
it and halts pipeline execution with an informative error message before
wasting tokens on retrieval, LLM obligation extraction, or policy gap analysis.

Powered by Jev (langchain-typesafe / TypeSafeClassifier).
Falls back gracefully to Gemini structured classification if TYPESAFE_API_KEY
is not yet configured in the environment.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from COMPLIANCE_AGENT.prompts.prompts import (
    DOCUMENT_VALIDATOR_HUMAN,
    DOCUMENT_VALIDATOR_JEV_CRITERIA,
    DOCUMENT_VALIDATOR_JEV_INSTRUCTIONS,
    DOCUMENT_VALIDATOR_SYSTEM,
)
from COMPLIANCE_AGENT.src.state import SEBIMonitorState

log = logging.getLogger(__name__)


class FallbackValidationOutput(BaseModel):
    """Fallback structured output schema for document validation."""
    is_regulatory_document: bool = Field(
        description=(
            "True if the document is an official regulatory circular, notification, "
            "statutory order, or direction from a financial/corporate regulator "
            "(such as SEBI, RBI, MCA, IRDAI, or stock exchanges). False if it is an unrelated "
            "document such as an invoice, bill, receipt, payment request, voucher, marketing deck, or CV."
        )
    )
    document_type: str = Field(
        description="Detected document type (e.g. 'SEBI Circular', 'Commercial Invoice', 'Payment Request', 'Internal Voucher', 'Marketing Deck', 'Resume', etc.)."
    )
    reason: str = Field(
        description="Concise rationale explaining why this document is or is not relevant to the SEBI regulatory compliance pipeline."
    )


def _fallback_llm_validation(sample_text: str) -> tuple[bool, str, str]:
    """Fallback validation using Groq / LLM with structured output."""
    from langchain_core.prompts import ChatPromptTemplate
    from COMPLIANCE_AGENT.utilities.llm import get_llm

    llm = get_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", DOCUMENT_VALIDATOR_SYSTEM),
        ("human", DOCUMENT_VALIDATOR_HUMAN),
    ])

    structured_llm = llm.with_structured_output(FallbackValidationOutput)
    chain = prompt | structured_llm
    result: FallbackValidationOutput = chain.invoke({"sample_text": sample_text})

    return result.is_regulatory_document, result.document_type, result.reason


def document_validator_node(state: SEBIMonitorState) -> dict:
    """
    LangGraph node: Document Validator (Jev / TypeSafe AI).

    Runs immediately after document_loader to filter out unrelated documents.
    """
    load_dotenv()

    # If previous node set an error, propagate
    if state.get("error"):
        return {}

    circular_text = state.get("circular_text", "")
    circular_id = state.get("circular_id", "unknown")

    if not circular_text or not circular_text.strip():
        err_msg = f"Document '{circular_id}' has no readable text. It may be empty or a scanned image."
        log.error(err_msg)
        return {
            "is_valid_document": False,
            "document_type": "Empty Document",
            "document_validation_reason": err_msg,
            "error": err_msg,
        }

    # Use first ~3500 characters for validation (sufficient to establish document identity)
    sample_text = circular_text.strip()
    # Support either TYPESAFE_API_KEY or OPENROUTER_API_KEY
    typesafe_api_key = (
        os.getenv("TYPESAFE_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
    ).strip()
    base_url = os.getenv("TYPESAFE_BASE_URL", "").strip()
    default_model = "~typesafe/jev-latest" if (base_url and "openrouter" in base_url) else "jev-latest"
    model_name = os.getenv("TYPESAFE_MODEL", "").strip() or default_model

    is_valid = False
    doc_type = "Unknown"
    reason = ""

    # ── Attempt 1: Jev (TypeSafe AI / OpenRouter) if configured ───────────────
    if typesafe_api_key:
        try:
            log.info(
                "Validating document '%s' with Jev (model: %s, base_url: %s)...",
                circular_id,
                model_name,
                base_url or "https://api.typesafe.ai",
            )
            from langchain_typesafe import Choice, TypeSafeClassifier

            classifier_kwargs: dict[str, str] = {
                "api_key": typesafe_api_key,
                "model": model_name,
            }
            if base_url:
                classifier_kwargs["base_url"] = base_url

            classifier = TypeSafeClassifier(**classifier_kwargs)
            response = classifier.invoke(
                {
                    "state": sample_text,
                    "questions": {
                        "document_classification": Choice(
                            instructions=DOCUMENT_VALIDATOR_JEV_INSTRUCTIONS,
                            criteria=DOCUMENT_VALIDATOR_JEV_CRITERIA,
                        )
                    },
                }
            )

            choice_ans = response.choices.get("document_classification")
            if choice_ans:
                selected = choice_ans.choice
                confidence = getattr(choice_ans, "confidence", 1.0)
                is_valid = (selected == "regulatory_circular")
                doc_type = "Regulatory Circular" if is_valid else "Unrelated Document"
                reason = f"Jev ({model_name}) classified document as '{selected}' with confidence {confidence:.2f}."
                log.info("Jev classification result: is_valid=%s, type=%s, confidence=%.2f", is_valid, selected, confidence)
            else:
                log.warning("Jev response did not contain document_classification answer, falling back to LLM.")
                is_valid, doc_type, reason = _fallback_llm_validation(sample_text)

        except Exception as exc:
            log.warning("Jev (TypeSafeClassifier) call encountered an issue: %s. Falling back to LLM validator.", exc)
            is_valid, doc_type, reason = _fallback_llm_validation(sample_text)
    else:
        log.info("Neither TYPESAFE_API_KEY nor OPENROUTER_API_KEY configured. Using fallback LLM validator for document classification.")
        is_valid, doc_type, reason = _fallback_llm_validation(sample_text)

    if not is_valid:
        error_message = (
            f"Document validation failed: The uploaded document is not a SEBI/regulatory circular. "
            f"Detected type: '{doc_type}'. Reason: {reason}"
        )
        log.warning("Exiting graph early: %s", error_message)
        return {
            "is_valid_document": False,
            "document_type": doc_type,
            "document_validation_reason": reason,
            "error": error_message,
        }

    log.info("Document '%s' validated successfully as relevant regulatory document (type: %s).", circular_id, doc_type)
    return {
        "is_valid_document": True,
        "document_type": doc_type,
        "document_validation_reason": reason,
        "error": None,
    }
