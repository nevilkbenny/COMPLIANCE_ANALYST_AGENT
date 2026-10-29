"""
streamlit_app.py
================
Streamlit UI for the SEBI Regulatory Change Impact Monitor.

Provides a professional browser-based interface with TWO sections:

  Tab 1 — 📄 Circular Impact Review
  ----------------------------------
  The Company Secretary (CS) uploads a SEBI circular PDF (via CIRCULAR_PDF_PATH
  in .env) and runs the full LangGraph pipeline:
    Watcher → Document Loader → Obligation Extraction → Retrieval →
    Relevance Judgment → Gap Analysis → Action Planning → Human Review (interrupt).
  The CS then approves / rejects / requests edits on draft policy amendments.

  Tab 2 — 💬 Document Chat
  ------------------------
  An interactive conversational interface where the user can ask free-form
  questions about SEBI Circulars and/or Tata Motors internal policies.
  Backed by a separate, lightweight LangGraph (chat_graph.py) with:
    Query Rewriter → Multi-Corpus ChromaDB Retriever → Grounded Answer Generator.
  All answers are grounded in retrieved document chunks and include citations.

Session state keys used
-----------------------
  Pipeline (Tab 1):
    thread_id      : str  — unique run identifier for LangGraph checkpointing
    run_status     : str  — "idle" | "running" | "interrupted" | "done" | "error"
    interrupt_data : dict — snapshot of state at interrupt (for display)
    audit_id       : int  — returned after the run completes

  Chat (Tab 2):
    chat_messages_<corpus> : list[dict] — per-corpus conversation display history
    chat_thread_id         : str        — LangGraph thread ID for chat checkpointing

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SEBI Regulatory Impact Monitor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# ---------------------------------------------------------------------------
# Logger Configuration (writes to streamlit_app.log)
# ---------------------------------------------------------------------------
LOG_FILE = Path(__file__).resolve().parent / "streamlit_app.log"

logger = logging.getLogger("streamlit_app")
logger.setLevel(logging.INFO)

# Formatter with timestamp, level, component name, and message
_log_format = logging.Formatter(
    "%(asctime)s [%(levelname)-7s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# File handler for streamlit_app logger
if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(LOG_FILE) for h in logger.handlers):
    fh = logging.FileHandler(str(LOG_FILE), encoding="utf-8", mode="a")
    fh.setFormatter(_log_format)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

# Also attach to the root logger so all pipeline nodes write step logs to streamlit_app.log
_root_logger = logging.getLogger()
if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(LOG_FILE) for h in _root_logger.handlers):
    _rfh = logging.FileHandler(str(LOG_FILE), encoding="utf-8", mode="a")
    _rfh.setFormatter(_log_format)
    _rfh.setLevel(logging.INFO)
    _root_logger.addHandler(_rfh)

logger.info("Streamlit app loaded / refreshed.")

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    color: #e2e8f0;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: #e2e8f0 !important;
}

/* Main area */
.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 100%);
    color: white;
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 24px rgba(15,39,68,0.4);
}
.main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
.main-header p  { margin: 0.3rem 0 0; opacity: 0.75; font-size: 0.95rem; }

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-idle       { background: #334155; color: #94a3b8; }
.badge-running    { background: #1d4ed8; color: #bfdbfe; }
.badge-interrupted{ background: #d97706; color: #fef3c7; }
.badge-done       { background: #059669; color: #d1fae5; }
.badge-error      { background: #dc2626; color: #fee2e2; }

/* Cards */
.card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.card-header {
    font-weight: 600;
    color: #1e293b;
    font-size: 0.95rem;
    margin-bottom: 0.6rem;
    border-bottom: 2px solid #3b82f6;
    padding-bottom: 0.4rem;
}

/* Severity & Status chips */
.chip-high          { background:#fee2e2; color:#dc2626; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
.chip-medium        { background:#fef9c3; color:#ca8a04; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
.chip-low           { background:#dcfce7; color:#16a34a; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
.chip-compliant     { background:#dcfce7; color:#15803d; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; border:1px solid #86efac; }
.chip-non-compliant { background:#fee2e2; color:#b91c1c; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; border:1px solid #fca5a5; }

/* Diff boxes */
.diff-container {
    display: flex;
    gap: 1rem;
    margin: 0.8rem 0;
    flex-wrap: wrap;
}
.diff-box-before {
    flex: 1;
    min-width: 280px;
    background: #fff1f2;
    border: 1px solid #fecdd3;
    border-radius: 8px;
    padding: 0.9rem 1rem;
}
.diff-box-after {
    flex: 1;
    min-width: 280px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    padding: 0.9rem 1rem;
}
.diff-title-before {
    font-size: 0.76rem;
    font-weight: 700;
    color: #be123c;
    margin-bottom: 0.4rem;
    letter-spacing: 0.03em;
}
.diff-title-after {
    font-size: 0.76rem;
    font-weight: 700;
    color: #15803d;
    margin-bottom: 0.4rem;
    letter-spacing: 0.03em;
}
.diff-summary-bar {
    background: #eff6ff;
    border-left: 3px solid #3b82f6;
    padding: 0.6rem 0.9rem;
    border-radius: 0 6px 6px 0;
    font-size: 0.85rem;
    color: #1e3a8a;
    margin-top: 0.6rem;
}

/* Score gauge */
.relevance-score {
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    line-height: 1;
}

/* ── Chat Tab Styles ──────────────────────────────────────────────── */
.chat-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 20px rgba(15,52,96,0.4);
}
.chat-header h2 { margin: 0; font-size: 1.4rem; font-weight: 700; }
.chat-header p  { margin: 0.3rem 0 0; opacity: 0.72; font-size: 0.88rem; }

.citation-card {
    background: #f1f5f9;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.9rem;
    margin: 0.4rem 0;
    font-size: 0.83rem;
}
.citation-sebi   { border-left-color: #f59e0b; }
.citation-tata   { border-left-color: #10b981; }
.citation-web    { border-left-color: #0284c7; background: #f0f9ff; }
.citation-file   { font-weight: 600; color: #1e293b; }
.citation-meta   { color: #64748b; font-size: 0.75rem; }
.citation-snippet{ color: #475569; margin-top: 0.3rem; font-style: italic; }

.corpus-badge-sebi { background:#fef3c7; color:#b45309; padding:2px 10px;
    border-radius:12px; font-size:0.78rem; font-weight:600; }
.corpus-badge-tata { background:#d1fae5; color:#065f46; padding:2px 10px;
    border-radius:12px; font-size:0.78rem; font-weight:600; }
.corpus-badge-both { background:#ede9fe; color:#5b21b6; padding:2px 10px;
    border-radius:12px; font-size:0.78rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
# Pipeline (Tab 1)
if "thread_id"      not in st.session_state: st.session_state.thread_id      = None
if "run_status"     not in st.session_state: st.session_state.run_status     = "idle"
if "interrupt_data" not in st.session_state: st.session_state.interrupt_data = None
if "audit_id"       not in st.session_state: st.session_state.audit_id       = None
if "run_error"      not in st.session_state: st.session_state.run_error      = None
# Chat (Tab 2)
if "chat_thread_id"           not in st.session_state: st.session_state.chat_thread_id           = str(uuid.uuid4())
if "chat_messages_both"       not in st.session_state: st.session_state.chat_messages_both       = []
if "chat_messages_sebi"       not in st.session_state: st.session_state.chat_messages_sebi       = []
if "chat_messages_tata"       not in st.session_state: st.session_state.chat_messages_tata       = []


# ---------------------------------------------------------------------------
# Graph loaders (cached so the embedding model loads only once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading SEBI Monitor pipeline graph…")
def load_graph():
    logger.info("Initializing SEBI Monitor graph via COMPLIANCE_AGENT.src.graph.build_graph()…")
    from COMPLIANCE_AGENT.src.graph import build_graph
    app = build_graph()
    logger.info("SEBI Monitor graph initialized successfully.")
    return app


@st.cache_resource(show_spinner="Loading Document Chat graph & embeddings…")
def load_chat_graph():
    logger.info("Initializing Chat Q&A graph via CONVERSATIONAL_AGENT.src.graph.build_chat_graph()…")
    from CONVERSATIONAL_AGENT.src.graph import build_chat_graph
    from CONVERSATIONAL_AGENT.nodes.chat_retriever import _get_embeddings
    chat_app = build_chat_graph()
    try:
        _get_embeddings()
    except Exception as exc:
        logger.warning("Could not pre-warm embeddings: %s", exc)
    logger.info("Chat Q&A graph initialized successfully.")
    return chat_app


# ---------------------------------------------------------------------------
# Pipeline execution runner (synchronous with real-time UI status updates)
# ---------------------------------------------------------------------------
def _extract_interrupt_data(app, thread_cfg: dict) -> dict:
    """Pull the interrupt payload from the checkpointed state."""
    try:
        state = app.get_state(thread_cfg)
        for task in (state.tasks or []):
            for interrupt in (getattr(task, "interrupts", []) or []):
                return interrupt.value or {}
        return dict(state.values)
    except Exception as exc:
        logger.warning("Could not extract interrupt value: %s", exc)
        return {}


def execute_pipeline(app) -> None:
    """Execute the pipeline steps with live UI updates."""
    thread_id = str(uuid.uuid4())
    thread_cfg = {"configurable": {"thread_id": thread_id}}
    pdf_path = os.getenv("CIRCULAR_PDF_PATH", "(not set)")

    logger.info("▶ Launching analysis run. thread_id=%s, circular=%s", thread_id, pdf_path)
    st.session_state.thread_id = thread_id
    st.session_state.interrupt_data = None
    st.session_state.audit_id = None
    st.session_state.run_error = None

    step_labels = {
        "watcher": "1️⃣ Checking circular in Watcher…",
        "document_loader": "2️⃣ Loading PDF text via Document Loader…",
        "document_validator": "🔍 Validating document relevance via Jev (TypeSafe AI)…",
        "obligation_extractor": "3️⃣ Extracting listed entity compliance obligations via Gemini…",
        "retriever": "4️⃣ Retrieving matching Tata Motors policy clauses from ChromaDB…",
        "relevance_judge": "5️⃣ Evaluating relevance score & impact…",
        "gap_analyzer": "6️⃣ Conducting parallel gap analysis across obligations…",
        "action_planner": "7️⃣ Drafting Before vs After policy amendments…",
        "human_review": "8️⃣ Pausing for Company Secretary review…",
        "audit_logger": "9️⃣ Writing audit record to SQLite database…",
    }

    with st.status("🚀 Running SEBI Regulatory Impact Pipeline…", expanded=True) as status_box:
        try:
            # Pass {} as the initial input to __start__ (do not pass None!)
            for event in app.stream({}, config=thread_cfg, stream_mode="updates"):
                if isinstance(event, dict):
                    for node_name in event.keys():
                        lbl = step_labels.get(node_name, f"Completed: {node_name}")
                        status_box.write(f"✔️ {lbl}")
                        logger.info("[Thread %s] Step completed: [%s]", thread_id, node_name)

            final = app.get_state(thread_cfg)
            if final.next:
                logger.info("[Thread %s] Graph reached interrupt at node: %s", thread_id, final.next)
                idata = _extract_interrupt_data(app, thread_cfg)
                st.session_state.interrupt_data = idata
                st.session_state.run_status = "interrupted"
                status_box.update(label="⏸ Awaiting Company Secretary Review", state="complete")
            else:
                audit_id = final.values.get("audit_id")
                err = final.values.get("error")
                if err or final.values.get("is_valid_document") is False:
                    err_msg = err or "Uploaded document is not a valid SEBI regulatory circular."
                    logger.warning("[Thread %s] Document validation rejected: %s", thread_id, err_msg)
                    st.session_state.audit_id = audit_id
                    st.session_state.run_error = err_msg
                    st.session_state.run_status = "error"
                    status_box.update(label=f"⛔ Document Validation Rejected", state="error")
                else:
                    logger.info("[Thread %s] Graph execution complete. Audit ID: %s", thread_id, audit_id)
                    st.session_state.audit_id = audit_id
                    st.session_state.run_status = "done"
                    status_box.update(label="✅ Analysis Complete!", state="complete")

        except Exception as exc:
            logger.error("[Thread %s] Pipeline execution failed: %s", thread_id, exc, exc_info=True)
            st.session_state.run_error = str(exc)
            st.session_state.run_status = "error"
            status_box.update(label=f"❌ Pipeline Failed: {exc}", state="error")


def _resume_graph(app, decision: str, edits: str | None) -> None:
    """Resume the graph after human review."""
    thread_id = st.session_state.thread_id
    thread_cfg = {"configurable": {"thread_id": thread_id}}
    logger.info("[Thread %s] Resuming graph with human decision='%s' (has_edits=%s)", thread_id, decision, bool(edits))
    try:
        with st.spinner("Submitting review decision and saving audit record…"):
            for event in app.stream(
                Command(resume={"decision": decision, "edits": edits or None}),
                config=thread_cfg,
                stream_mode="updates",
            ):
                if isinstance(event, dict):
                    for node_name in event.keys():
                        logger.info("[Thread %s] Resume step completed: [%s]", thread_id, node_name)

            final = app.get_state(thread_cfg)
            if final.next:
                logger.info("[Thread %s] Re-interrupted after review at: %s", thread_id, final.next)
                idata = _extract_interrupt_data(app, thread_cfg)
                st.session_state.interrupt_data = idata
                st.session_state.run_status     = "interrupted"
            else:
                audit_id = final.values.get("audit_id")
                logger.info("[Thread %s] Run finalized. Audit ID: %s", thread_id, audit_id)
                st.session_state.audit_id   = audit_id
                st.session_state.run_status = "done"

    except Exception as exc:
        logger.error("[Thread %s] Error resuming graph: %s", thread_id, exc, exc_info=True)
        st.session_state.run_error  = str(exc)
        st.session_state.run_status = "error"


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _severity_chip(severity: str) -> str:
    cls = {"high": "chip-high", "medium": "chip-medium", "low": "chip-low"}.get(
        severity.lower(), "chip-medium"
    )
    return f'<span class="{cls}">{severity.upper()}</span>'


def _status_chip(status: str) -> str:
    s = status.lower().replace("-", "_")
    if s == "compliant":
        return '<span class="chip-compliant">COMPLIANT</span>'
    else:
        return '<span class="chip-non-compliant">NON-COMPLIANT</span>'


def _change_type_chip(change_type: str) -> str:
    ctype = change_type.lower()
    bg = "#e0e7ff" if "addition" in ctype else "#fef3c7"
    color = "#3730a3" if "addition" in ctype else "#92400e"
    return f'<span style="background:{bg};color:{color};padding:2px 8px;border-radius:10px;font-size:0.75rem;font-weight:700;">{change_type.upper()}</span>'


def _badge(label: str, status: str) -> str:
    return f'<span class="status-badge badge-{status}">{label}</span>'


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------

def render_header(run_status: str) -> None:
    status_map = {
        "idle":        ("● Idle",        "idle"),
        "running":     ("⏳ Running…",   "running"),
        "interrupted": ("⏸ Awaiting CS Review", "interrupted"),
        "done":        ("✅ Complete",   "done"),
        "error":       ("❌ Error",      "error"),
    }
    label, css = status_map.get(run_status, ("● Unknown", "idle"))

    st.markdown(f"""
    <div class="main-header">
        <h1>⚖️ SEBI Regulatory Impact Monitor</h1>
        <p>Automated compliance gap analysis &amp; policy amendment drafting for Tata Motors Limited</p>
        <div style="margin-top:0.8rem">{_badge(label, css)}</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(app) -> None:
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")
        st.markdown("---")

        pdf_path = os.getenv("CIRCULAR_PDF_PATH", "")
        pdf_name = Path(pdf_path).name if pdf_path else "(not set)"
        st.markdown(f"**Circular PDF**\n\n`{pdf_name}`")
        st.caption("Set `CIRCULAR_PDF_PATH` in `.env` to change.")
        st.markdown("---")

        status = st.session_state.run_status
        btn_disabled = status in ("running", "interrupted")

        if st.button("▶ Run Analysis", disabled=btn_disabled, use_container_width=True, type="primary"):
            st.session_state.run_status = "running"
            st.rerun()

        st.markdown("---")
        st.markdown("**Thresholds & Settings**")

        threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.4"))
        st.metric("Relevance Threshold", f"{threshold:.0%}")

        force_val = os.getenv("FORCE_REPROCESS", "true").lower() in ("true", "1")
        st.caption(f"Force Reprocess: **{'Enabled' if force_val else 'Disabled'}**")

        if st.session_state.audit_id:
            st.success(f"Audit ID: **{st.session_state.audit_id}**")

        st.markdown("---")
        # Live Step Execution Log Expander in Sidebar
        with st.expander("📜 Step Execution Log", expanded=False):
            if LOG_FILE.exists():
                lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
                st.code("\n".join(lines) if lines else "(No log entries yet)", language="log")
            else:
                st.caption("No log file found yet.")

        st.markdown("---")
        st.caption("SEBI Monitor v2.0 · Direct Loader + Diff Drafting")


def render_run_result(data: dict) -> None:
    """Display circular summary, gap findings, and action plans for CS review."""

    circular_id     = data.get("circular_id",     "Unknown")
    total_pages     = data.get("total_pages")
    obligations     = data.get("obligations",     [])
    gap_findings    = data.get("gap_findings",    [])
    action_plans    = data.get("action_plans",    [])
    rel_score       = data.get("relevance_score", 0.0)
    rel_reason      = data.get("relevance_reason", "")

    # ── Circular Summary ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown(f"### 📄 Circular: `{circular_id}`")
        if total_pages:
            st.caption(f"Loaded directly via PyPDFLoader ({total_pages} pages).")
    with col2:
        score_color = "#dc2626" if rel_score > 0.7 else "#d97706" if rel_score > 0.4 else "#16a34a"
        st.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:0.75rem;color:#64748b;margin-bottom:0.2rem'>Relevance Score</div>"
            f"<div class='relevance-score' style='color:{score_color}'>{rel_score:.0%}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col3:
        # Non-compliant findings represent actionable gaps requiring amendment
        actionable = [
            f for f in gap_findings
            if f.get("status", "").lower().replace("-", "_") in ("non_compliant", "outdated", "silent")
        ]
        high_count = sum(1 for f in actionable if f.get("severity") == "high")
        st.metric("Actionable Gaps", len(actionable), delta=f"{high_count} high severity")

    # ── Relevance reason ───────────────────────────────────────────────────
    with st.expander("📊 Relevance Analysis & Scoring Reason", expanded=True):
        st.markdown(rel_reason or "(No relevance reason recorded)")

    # ── Obligations ────────────────────────────────────────────────────────
    with st.expander(f"📋 Listed Entity Compliance Obligations ({len(obligations)})", expanded=False):
        st.caption("Extracted mandates specifically applicable to listed companies / listed entities.")
        for i, obl in enumerate(obligations, 1):
            st.markdown(f"**{i}.** {obl}")

    # ── Gap Findings ────────────────────────────────────────────────────────
    st.markdown("### 🔍 Gap Findings (Compliant vs Non-Compliant)")
    if not gap_findings:
        st.info("No gap findings — circular may not affect existing policies.")
    else:
        for i, finding in enumerate(gap_findings):
            raw_status  = finding.get("status", "non_compliant")
            status_norm = raw_status.lower().replace("-", "_")
            severity    = finding.get("severity", "medium")
            obl         = finding.get("obligation", "")
            policy      = finding.get("policy_doc", "")
            clause      = finding.get("policy_clause", "")
            explanation = finding.get("explanation", "")

            if status_norm == "compliant":
                with st.expander(f"✅ [{i+1}] {obl[:80]}… — COMPLIANT", expanded=False):
                    st.markdown(
                        f"{_status_chip(raw_status)} {_severity_chip(severity)}",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Policy doc:** `{policy}`")
                    st.markdown(f"**Satisfying clause:** {clause or '(None quoted)'}")
                    st.markdown(f"**Compliance analysis:** {explanation}")
            else:
                with st.expander(
                    f"⚠️ [{i+1}] {obl[:80]}… — **NON-COMPLIANT** ({severity.upper()})",
                    expanded=True,
                ):
                    st.markdown(
                        f"{_status_chip(raw_status)} {_severity_chip(severity)}",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Affected policy doc:** `{policy}`")
                    if clause:
                        st.markdown(f"**Current clause snippet:** _{clause}_")
                    st.markdown(f"**Gap rationale:** {explanation}")

    # ── Action Plans (Before vs After Diff) ──────────────────────────────────
    st.markdown("### 📝 Draft Action Plans & Policy Amendments")
    if not action_plans:
        st.info("No policy amendments required.")
    else:
        for i, plan in enumerate(action_plans):
            doc         = plan.get("policy_doc", "unknown")
            gap_sum     = plan.get("gap_summary", "")
            change_type = plan.get("change_type", "Modification")
            existing    = plan.get("existing_clause", "None / New Clause")
            amended     = plan.get("amended_clause", "")
            diff_sum    = plan.get("diff_summary", "")
            owner       = plan.get("owner", "Company Secretary")
            deadline    = plan.get("deadline", "Within 30 days")
            rationale   = plan.get("rationale", "")

            with st.expander(f"📄 [{i+1}] {doc} — {change_type.upper()}", expanded=True):
                col_a, col_b, col_c = st.columns([1, 1, 1])
                with col_a:
                    st.markdown(f"**Change Type:** {_change_type_chip(change_type)}", unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"**Proposed Owner:** `{owner}`")
                with col_c:
                    st.markdown(f"**Target Deadline:** `{deadline}`")

                st.markdown(f"**Gap Addressed:** {gap_sum}")

                # Before vs After Diff Presentation
                st.markdown("""
                <div class="diff-container">
                    <div class="diff-box-before">
                        <div class="diff-title-before">🔴 BEFORE (EXISTING POLICY CLAUSE)</div>
                        <div style="font-size:0.87rem;color:#4c0519;white-space:pre-wrap;">""" + existing + """</div>
                    </div>
                    <div class="diff-box-after">
                        <div class="diff-title-after">🟢 AFTER (PROPOSED AMENDED CLAUSE)</div>
                        <div style="font-size:0.87rem;color:#14532d;white-space:pre-wrap;">""" + amended + """</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if diff_sum:
                    st.markdown(f"""
                    <div class="diff-summary-bar">
                        <strong>Diff Summary:</strong> {diff_sum}
                    </div>
                    """, unsafe_allow_html=True)

                with st.expander("⚖️ Compliance Drafting Rationale", expanded=False):
                    st.markdown(rationale)


def render_review_form() -> tuple[str | None, str | None]:
    """
    Render the CS review form.
    Returns (decision, edits) when submitted, else (None, None).
    """
    st.markdown("---")
    st.markdown("### ✍️ Your Review Decision (Company Secretary Approval)")

    with st.form(key="review_form", clear_on_submit=False):
        decision = st.radio(
            "Decision",
            options=["approved", "rejected", "edit_requested"],
            format_func=lambda x: {
                "approved":       "✅ Approve — finalize action plans and write to SQLite audit log",
                "rejected":       "❌ Reject — discard and archive this analysis",
                "edit_requested": "✏️ Request Edits — re-draft action plans with instructions",
            }[x],
            horizontal=True,
            key="cs_decision",
        )

        edits = st.text_area(
            "Edit instructions (required if 'Request Edits' selected)",
            placeholder="Describe what needs to change in the draft amendments (e.g., change deadline to 45 days, assign CFO instead)…",
            height=100,
            key="cs_edits",
        )

        submitted = st.form_submit_button("Submit Review Decision", type="primary", use_container_width=True)

    if submitted:
        if decision == "edit_requested" and not edits.strip():
            st.error("Please provide edit instructions before submitting.")
            return None, None
        return decision, edits.strip() or None

    return None, None


# ---------------------------------------------------------------------------
# Chat Tab renderer
# ---------------------------------------------------------------------------

def render_chat_tab(chat_app) -> None:
    """Render the conversational Document Chat UI."""

    # Header
    st.markdown("""
    <div class="chat-header">
        <h2>💬 Document Chat Assistant</h2>
        <p>Ask questions about SEBI Circulars and Tata Motors Governance Policies.
        All answers are grounded in the loaded documents and include citations.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Corpus selector ─────────────────────────────────────────────────────
    col_corpus, col_clear = st.columns([3, 1])
    with col_corpus:
        corpus_choice = st.radio(
            "🔍 Search in:",
            options=["Both (SEBI + Policies)", "SEBI Circulars Only", "Tata Motors Policies Only"],
            horizontal=True,
            key="chat_corpus_radio",
        )
    with col_clear:
        st.markdown("<div style='margin-top:1.8rem'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat", key="clear_chat_btn", use_container_width=True):
            for k in ("chat_messages_both", "chat_messages_sebi", "chat_messages_tata"):
                st.session_state[k] = []
            st.session_state.chat_thread_id = str(uuid.uuid4())
            st.rerun()

    corpus_map = {
        "Both (SEBI + Policies)":       "both",
        "SEBI Circulars Only":          "sebi",
        "Tata Motors Policies Only":    "tata",
    }
    corpus = corpus_map[corpus_choice]
    history_key = f"chat_messages_{corpus}"

    # Corpus badge
    badge_cls = {"both": "corpus-badge-both", "sebi": "corpus-badge-sebi", "tata": "corpus-badge-tata"}[corpus]
    badge_label = {"both": "SEBI + Tata Policies", "sebi": "SEBI Circulars", "tata": "Tata Motors Policies"}[corpus]
    st.markdown(
        f'<span class="{badge_cls}">{badge_label}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Suggested questions ─────────────────────────────────────────────────
    if not st.session_state[history_key]:
        st.markdown("**💡 Try asking:**")
        suggestions = [
            "What are the new SEBI thresholds for Related Party Transaction approval?",
            "Does Tata Motors' Materiality Policy align with SEBI Regulation 30?",
            "What are the disclosure timelines required under the BRSR Core framework?",
            "Who is the designated compliance authority under Tata Motors' RPT Policy?",
        ]
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                    st.session_state[history_key].append({"role": "user", "content": suggestion})
                    st.rerun()

    # ── Helper to render sources & trace ─────────────────────────────────────
    def _render_meta_blocks(msg_data: dict) -> None:
        traces = msg_data.get("decision_trace", [])
        if traces:
            with st.expander("⚡ Jev Agentic Decisions", expanded=False):
                for step in traces:
                    st.markdown(f"- {step}")

        citations = msg_data.get("citations", [])
        if citations:
            with st.expander(f"📄 Internal Sources ({len(citations)} excerpts)", expanded=False):
                for c in citations:
                    border_cls = "citation-sebi" if "SEBI" in c.get("source_type", "") else "citation-tata"
                    st.markdown(
                        f"<div class='citation-card {border_cls}'>"
                        f"<div class='citation-file'>📄 {c.get('file', 'Unknown')} "
                        f"— Page {c.get('page', 'N/A')}</div>"
                        f"<div class='citation-meta'>{c.get('source_type', '')}</div>"
                        f"<div class='citation-snippet'>{c.get('snippet', '')[:280]}…</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        web_results = msg_data.get("web_search_results", [])
        if web_results:
            with st.expander(f"🌐 External Web Sources ({len(web_results)} results)", expanded=False):
                for w in web_results:
                    st.markdown(
                        f"<div class='citation-card citation-web'>"
                        f"<div class='citation-file'>🌐 <a href='{w.get('url', '#')}' target='_blank' style='color:#0369a1;text-decoration:underline;'>{w.get('title', 'Web Result')}</a></div>"
                        f"<div class='citation-meta'>{w.get('url', '')}</div>"
                        f"<div class='citation-snippet'>{w.get('snippet', '')[:280]}…</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # ── Render message history ───────────────────────────────────────────────
    for msg in st.session_state[history_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_meta_blocks(msg)

    # ── Chat input ───────────────────────────────────────────────────────────
    placeholder = {
        "both": "Ask about SEBI rules, Tata Motors policies, or compare both…",
        "sebi": "Ask about a SEBI circular, regulation, or notification…",
        "tata": "Ask about a Tata Motors governance or compliance policy…",
    }[corpus]

    if user_question := st.chat_input(placeholder, key="chat_input_box"):
        # Append and display user message immediately
        st.session_state[history_key].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Build LangChain message history for the graph (exclude the new question,
        # which will be the last element we send separately)
        display_history = st.session_state[history_key]
        lc_messages = []
        for m in display_history[:-1]:   # all prior turns
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            else:
                lc_messages.append(AIMessage(content=m["content"]))
        lc_messages.append(HumanMessage(content=user_question))

        # Invoke chat graph
        with st.chat_message("assistant"):
            with st.spinner("Analyzing query, evaluating documents, and generating verified response…"):
                try:
                    result = chat_app.invoke(
                        {
                            "messages":           lc_messages,
                            "corpus":             corpus,
                            "web_search_results": [],
                            "retrieved_docs":     [],
                            "needs_web_search":   False,
                            "decision_trace":     [],
                            "docs_are_relevant":  False,
                        },
                        config={"configurable": {"thread_id": st.session_state.chat_thread_id}},
                    )
                    answer      = result.get("answer", "⚠️ No answer was generated.")
                    citations   = result.get("retrieved_docs", []) if result.get("docs_are_relevant", True) else []
                    web_res     = result.get("web_search_results", [])
                    dec_trace   = result.get("decision_trace", [])
                    logger.info(
                        "[Chat] corpus=%s | question='%s' | answer_chars=%d | citations=%d | web_results=%d",
                        corpus, user_question[:60], len(answer), len(citations), len(web_res),
                    )
                except Exception as exc:
                    answer    = f"⚠️ Error during document search: {exc}"
                    citations = []
                    web_res   = []
                    dec_trace = []
                    logger.error("[Chat] Error invoking chat graph: %s", exc, exc_info=True)

            st.markdown(answer)
            meta_data = {
                "decision_trace":     dec_trace,
                "citations":          citations,
                "web_search_results": web_res,
            }
            _render_meta_blocks(meta_data)

        # Persist assistant message to display history
        st.session_state[history_key].append({
            "role":               "assistant",
            "content":            answer,
            "citations":          citations,
            "web_search_results": web_res,
            "decision_trace":     dec_trace,
        })


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    pipeline_app = load_graph()
    chat_app     = load_chat_graph()

    render_header(st.session_state.run_status)
    render_sidebar(pipeline_app)

    # ── Two-tab layout ───────────────────────────────────────────────────────
    tab_review, tab_chat = st.tabs(["📄 Circular Impact Review", "💬 Document Chat"])

    # ── Tab 1: Existing pipeline UI ──────────────────────────────────────────
    with tab_review:
        status = st.session_state.run_status

        # ── Idle
        if status == "idle":
            st.info(
                "👋 Welcome to the SEBI Regulatory Impact Monitor.\n\n"
                "Set `CIRCULAR_PDF_PATH` in your `.env` file to point to the circular "
                "you want to analyse, then click **▶ Run Analysis** in the sidebar."
            )

        # ── Running
        elif status == "running":
            execute_pipeline(pipeline_app)
            st.rerun()

        # ── Interrupted: awaiting CS review
        elif status == "interrupted":
            data = st.session_state.interrupt_data or {}
            render_run_result(data)
            decision, edits = render_review_form()
            if decision is not None:
                _resume_graph(pipeline_app, decision, edits)
                st.rerun()

        # ── Done
        elif status == "done":
            st.success(
                f"✅ Analysis complete! Audit record saved "
                f"(ID: {st.session_state.audit_id or 'N/A'})."
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Analyse Another Circular", use_container_width=True):
                    for key in ("thread_id", "run_status", "interrupt_data", "audit_id", "run_error"):
                        st.session_state.pop(key, None)
                    st.rerun()
            with col2:
                if st.button("📊 View Audit Log Records", use_container_width=True):
                    import sqlite3
                    db_path = Path(os.getenv("AUDIT_DB_PATH", "sebi_monitor_db.sqlite"))
                    if db_path.exists():
                        conn = sqlite3.connect(str(db_path))
                        rows = conn.execute("SELECT * FROM audit_runs ORDER BY id DESC LIMIT 10").fetchall()
                        cols = [d[0] for d in conn.execute("PRAGMA table_info(audit_runs)").fetchall()]
                        conn.close()
                        if rows:
                            st.dataframe(
                                [{c: r[i] for i, c in enumerate(cols)} for r in rows],
                                use_container_width=True,
                            )
                        else:
                            st.info("Audit runs table exists but is currently empty.")
                    else:
                        st.warning("Audit database file does not exist yet.")

            with st.expander("📄 Pipeline Execution Log Details", expanded=False):
                if LOG_FILE.exists():
                    lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
                    st.code("\n".join(lines), language="log")

        # ── Error
        elif status == "error":
            st.error(f"❌ Pipeline error: {st.session_state.run_error}")
            if st.button("🔄 Retry"):
                st.session_state.run_status = "idle"
                st.rerun()

    # ── Tab 2: Conversational Document Chat ──────────────────────────────────
    with tab_chat:
        render_chat_tab(chat_app)


if __name__ == "__main__":
    main()
