"""
nodes/document_loader.py
========================
Document Loader node – directly loads the circular PDF into clean, continuous text.

Responsibilities (no LLM involved):
  1. Load the circular PDF using LangChain's PyPDFLoader.
  2. Extract full page text across all pages into a clean, un-chunked document string.
  3. Store 'circular_text' and 'total_pages' in the LangGraph state.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from COMPLIANCE_AGENT.src.state import SEBIMonitorState

log = logging.getLogger(__name__)


def document_loader_node(state: SEBIMonitorState) -> dict:
    """
    LangGraph node: Document Loader

    Loads the circular PDF using PyPDFLoader and extracts the complete text
    without arbitrary character splits or chunk overlaps.
    Returns {'circular_text': str, 'total_pages': int}.
    """
    if state.get("error"):
        log.warning("Skipping document loader — upstream error: %s", state["error"])
        return {}

    if state.get("already_processed"):
        log.info("Circular already processed — skipping document loader.")
        return {}

    pdf_path_str = state.get("pdf_path")
    if not pdf_path_str:
        log.error("No pdf_path in state — cannot load circular PDF.")
        return {"error": "Missing pdf_path in state.", "circular_text": ""}

    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        log.error("Circular PDF does not exist at path: %s", pdf_path)
        return {"error": f"File not found: {pdf_path}", "circular_text": ""}

    circular_id = state.get("circular_id", pdf_path.stem)
    log.info("Loading circular PDF from '%s' (id=%s) …", pdf_path.name, circular_id)

    try:
        loader = PyPDFLoader(str(pdf_path.resolve()))
        pages = loader.load()

        if not pages:
            log.warning("No pages loaded from '%s'.", pdf_path.name)
            return {"circular_text": "", "total_pages": 0}

        # Join pages with standard double newline, preserving natural page flow
        circular_text = "\n\n".join(
            page.page_content.strip() for page in pages if page.page_content and page.page_content.strip()
        )

        log.info(
            "Document loader complete: loaded %d pages (%d chars) from circular '%s'.",
            len(pages),
            len(circular_text),
            circular_id,
        )

        return {
            "circular_text": circular_text,
            "total_pages": len(pages),
        }

    except Exception as exc:
        log.error("Document loader failed for '%s': %s", pdf_path, exc)
        return {
            "error": f"Failed to load PDF: {exc}",
            "circular_text": "",
        }
