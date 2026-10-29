"""
nodes/obligation_extractor.py
==============================
Obligation Extraction node – LLM call #1.

Reads the full circular text (reassembled from chunks) and asks Gemini 1.5
Flash to extract every concrete compliance obligation as a short plain-English
statement.  These statements become the retrieval queries in the next node.

Why obligations instead of raw chunks?
  Short, focused semantic queries retrieve far more relevant policy clauses
  than raw legal prose, which tends to be verbose and heavily nominalized.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from COMPLIANCE_AGENT.utilities.llm import get_llm
from COMPLIANCE_AGENT.prompts.prompts import (
    OBLIGATION_EXTRACTION_HUMAN,
    OBLIGATION_EXTRACTION_SYSTEM,
)
from COMPLIANCE_AGENT.src.state import ObligationsOutput, SEBIMonitorState

log = logging.getLogger(__name__)


def _build_llm():
    return get_llm(temperature=0.0)


def obligation_extractor_node(state: SEBIMonitorState) -> dict:
    """
    LangGraph node: Obligation Extraction  (LLM call #1)

    Concatenates all chunk texts to reconstruct the circular, then asks the
    LLM to enumerate its compliance obligations.  Falls back to using chunk
    contents directly if the LLM call fails.
    """
    if state.get("error"):
        log.warning("Skipping obligation extractor — upstream error.")
        return {}

    if state.get("already_processed"):
        log.info("Circular already processed — skipping obligation extractor.")
        return {}

    # ── Retrieve circular text (directly from document_loader, or fallback to chunks) ──
    circular_text = state.get("circular_text", "").strip()
    if not circular_text:
        chunks = state.get("chunks", [])
        if chunks:
            circular_text = "\n\n".join(c.get("page_content", "") for c in chunks).strip()

    if not circular_text:
        log.warning("No circular text in state — cannot extract obligations.")
        return {"obligations": []}

    circular_id = state.get("circular_id", "unknown")
    log.info(
        "Extracting obligations from circular '%s' (%d chars) …",
        circular_id, len(circular_text),
    )

    # ── LLM call with structured output ─────────────────────────────────────
    try:
        llm = _build_llm()
        structured_llm = llm.with_structured_output(ObligationsOutput)

        prompt = ChatPromptTemplate.from_messages([
            ("system", OBLIGATION_EXTRACTION_SYSTEM),
            ("human",  OBLIGATION_EXTRACTION_HUMAN),
        ])

        chain = prompt | structured_llm
        result: ObligationsOutput = chain.invoke({
            "circular_id":   circular_id,
            "circular_text": circular_text,
        })

        obligations = result.obligations
        log.info("Extracted %d obligations.", len(obligations))
        for i, obl in enumerate(obligations, 1):
            log.info("  [%d] %s", i, obl)

        return {"obligations": obligations}

    except Exception as exc:
        log.error("Obligation extraction LLM call failed: %s", exc)
        return {"obligations": "obligation node failed"}
