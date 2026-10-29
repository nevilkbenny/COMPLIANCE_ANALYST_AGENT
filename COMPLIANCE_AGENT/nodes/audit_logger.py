"""
nodes/audit_logger.py
=====================
Audit Logger node – final node in every execution path (no LLM).

Writes a complete, immutable audit record to SQLite covering:
  - The circular metadata
  - Relevance score and reason
  - All extracted obligations
  - All gap findings
  - All action plan drafts
  - The human decision (approved | rejected | archived)
  - Timestamps

The SQLite database is created automatically at AUDIT_DB_PATH
(default: sebi_monitor_db.sqlite in the project root).

Schema
------
Table: audit_runs
  id              INTEGER PRIMARY KEY AUTOINCREMENT
  circular_id     TEXT
  circular_url    TEXT
  pdf_path        TEXT
  run_timestamp   TEXT
  relevance_score REAL
  relevance_reason TEXT
  obligations     TEXT   (JSON array)
  gap_findings    TEXT   (JSON array)
  action_plans    TEXT   (JSON array)
  human_decision  TEXT
  human_edits     TEXT
  already_processed INTEGER (0/1)
  created_at      TEXT
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from COMPLIANCE_AGENT.src.state import SEBIMonitorState

log = logging.getLogger(__name__)

_PROJECT_ROOT  = Path(__file__).parent.parent.parent
_DEFAULT_DB    = _PROJECT_ROOT / "sebi_monitor_db.sqlite"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    circular_id       TEXT    NOT NULL,
    circular_url      TEXT,
    pdf_path          TEXT,
    run_timestamp     TEXT,
    relevance_score   REAL,
    relevance_reason  TEXT,
    obligations       TEXT,   -- JSON
    gap_findings      TEXT,   -- JSON
    action_plans      TEXT,   -- JSON
    human_decision    TEXT,
    human_edits       TEXT,
    already_processed INTEGER DEFAULT 0,
    created_at        TEXT    NOT NULL
);
"""

_INSERT_SQL = """
INSERT INTO audit_runs (
    circular_id, circular_url, pdf_path, run_timestamp,
    relevance_score, relevance_reason,
    obligations, gap_findings, action_plans,
    human_decision, human_edits, already_processed, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _get_db_path() -> Path:
    load_dotenv()
    return Path(os.getenv("AUDIT_DB_PATH", str(_DEFAULT_DB)))


def audit_logger_node(state: SEBIMonitorState) -> dict:
    """
    LangGraph node: Audit Logger  (no LLM)

    Writes the full run record to SQLite and returns audit_id.
    This is always the final node regardless of the execution path taken.
    """
    db_path = _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    circular_id      = state.get("circular_id",      "unknown")
    human_decision   = state.get("human_decision",   "archived")
    already_processed = state.get("already_processed", False)

    # For invalid documents, irrelevant circulars, or already-processed items:
    if state.get("error") or not state.get("is_valid_document", True):
        human_decision = f"rejected_invalid_document"
    elif not human_decision or human_decision == "archived":
        if already_processed:
            human_decision = "already_processed"
        elif state.get("relevance_score", 1.0) < float(
            os.getenv("RELEVANCE_THRESHOLD", "0.4")
        ):
            human_decision = "archived_irrelevant"

    relevance_reason = state.get("relevance_reason", "")
    if not relevance_reason and state.get("document_validation_reason"):
        relevance_reason = f"[Validation] {state.get('document_validation_reason')}"
    elif not relevance_reason and state.get("error"):
        relevance_reason = f"[Error] {state.get('error')}"

    record = (
        circular_id,
        state.get("circular_url",      ""),
        state.get("pdf_path",          ""),
        state.get("run_timestamp",     ""),
        state.get("relevance_score",   0.0),
        state.get("relevance_reason",  ""),
        json.dumps(state.get("obligations",    []), ensure_ascii=False),
        json.dumps(state.get("gap_findings",   []), ensure_ascii=False),
        json.dumps(state.get("action_plans",   []), ensure_ascii=False),
        human_decision,
        state.get("human_edits"),
        int(already_processed),
        datetime.now(timezone.utc).isoformat(),
    )

    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute(_CREATE_TABLE_SQL)
        cursor = conn.execute(_INSERT_SQL, record)
        audit_id = cursor.lastrowid
        conn.commit()
        conn.close()

        log.info(
            "Audit record written: id=%d | circular='%s' | decision='%s' | db='%s'",
            audit_id, circular_id, human_decision, db_path,
        )
        return {"audit_id": audit_id}

    except Exception as exc:
        log.error("Audit logger failed: %s", exc)
        return {"audit_id": None}
