"""
nodes/watcher.py
================
Watcher node – entry point of the LangGraph pipeline.

Responsibilities (no LLM involved):
  1. Read CIRCULAR_PDF_PATH from the environment (.env file).
  2. Check the SEBI download manifest to detect already-processed circulars.
  3. Populate the initial state fields: circular_id, pdf_path, already_processed.

The manifest (sebi_download_manifest.json) is a dict keyed by SEBI page URL.
Each value contains: title, date, pdf_url, saved_as.
We check if the PDF filename appears as any `saved_as` value to detect duplicates.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from COMPLIANCE_AGENT.src.state import SEBIMonitorState

log = logging.getLogger(__name__)

# Absolute path to the manifest file (in project root)
_BASE_DIR = Path(__file__).parent.parent.parent          # → c:\Nevil_P\Agentic AI
MANIFEST_PATH = _BASE_DIR / "sebi_download_manifest.json"


def _load_manifest() -> dict:
    """Load and return the download manifest, or {} if it doesn't exist."""
    if not MANIFEST_PATH.exists():
        log.warning("Manifest not found at %s — skipping duplicate check.", MANIFEST_PATH)
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not read manifest (%s) — skipping duplicate check.", exc)
        return {}


def _is_already_processed(manifest: dict, pdf_filename: str) -> bool:
    """Return True if *pdf_filename* appears as a `saved_as` value in the manifest."""
    for entry in manifest.values():
        if isinstance(entry, dict):
            if entry.get("saved_as", "") == pdf_filename:
                return True
    return False


def watcher_node(state: SEBIMonitorState) -> dict:
    """
    LangGraph node: Watcher

    Reads CIRCULAR_PDF_PATH from the environment, validates the file exists,
    checks the manifest for duplicate processing, and seeds the shared state.
    """
    load_dotenv()

    pdf_path_str = os.getenv("CIRCULAR_PDF_PATH", "").strip()
    if not pdf_path_str:
        msg = "CIRCULAR_PDF_PATH is not set in the environment / .env file."
        log.error(msg)
        return {"error": msg, "run_timestamp": datetime.now(timezone.utc).isoformat()}

    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        msg = f"PDF file not found: {pdf_path}"
        log.error(msg)
        return {"error": msg, "run_timestamp": datetime.now(timezone.utc).isoformat()}

    pdf_filename = pdf_path.name
    circular_id  = pdf_path.stem   # filename without extension → used as run identifier

    # ── Duplicate detection via manifest ────────────────────────────────────
    manifest = _load_manifest()
    force_reprocess = os.getenv("FORCE_REPROCESS", "false").lower() in ("true", "1", "yes")
    if force_reprocess:
        already_processed = False
        log.info("FORCE_REPROCESS=True — bypassing manifest check for '%s'.", pdf_filename)
    else:
        already_processed = _is_already_processed(manifest, pdf_filename)

    if already_processed:
        log.info(
            "Circular '%s' is already in the manifest — marking as processed. "
            "The pipeline will archive it without re-running analysis.",
            pdf_filename,
        )
    elif not force_reprocess:
        log.info("New circular detected: '%s' (id=%s)", pdf_filename, circular_id)

    return {
        "circular_id":       circular_id,
        "circular_url":      "",          # not available when loading from local path
        "pdf_path":          str(pdf_path.resolve()),
        "already_processed": already_processed,
        "run_timestamp":     datetime.now(timezone.utc).isoformat(),
        "error":             None,
        # Initialise fields so downstream nodes always see the correct types
        "circular_text":     "",
        "total_pages":       0,
        "chunks":            [],
        "obligations":       [],
        "matched_policies":  [],
        "gap_findings":      [],
        "action_plans":      [],
        "human_decision":    "",
        "human_edits":       None,
        "audit_id":          None,
        "relevance_score":   0.0,
        "relevance_reason":  "",
    }
