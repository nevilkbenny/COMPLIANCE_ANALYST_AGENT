"""
nodes/human_review.py
=====================
Human Review node – uses LangGraph's interrupt() for a true pause.

When the graph reaches this node it raises an Interrupt exception.
LangGraph catches it, serialises the state to the checkpointer, and
suspends execution.  The Streamlit UI reads the suspended state, presents
the findings to the CS, and on form submission calls:

    app.invoke(Command(resume={"decision": "approved", "edits": None}), config=thread_config)

The node then resumes from the line after interrupt() with the CS's response
stored in `decision`.

Routing after human_review is handled by the conditional edge
`route_after_human` in graph.py:
    "approved"       → audit_logger
    "rejected"       → audit_logger
    "edit_requested" → action_planner  (re-draft loop)
"""

from __future__ import annotations

import logging

from langgraph.types import interrupt

from COMPLIANCE_AGENT.src.state import SEBIMonitorState

log = logging.getLogger(__name__)


def human_review_node(state: SEBIMonitorState) -> dict:
    """
    LangGraph node: Human Review  (no LLM — interrupt-based pause)

    Suspends the graph and waits for the CS to approve, reject, or request
    edits via the Streamlit UI.  When resumed, updates human_decision and
    human_edits in the shared state.
    """
    circular_id  = state.get("circular_id", "unknown")
    gap_findings = state.get("gap_findings",  [])
    action_plans = state.get("action_plans",  [])
    obligations  = state.get("obligations",   [])

    log.info(
        "Pausing for human review of circular '%s' "
        "(%d gaps | %d action plan drafts) …",
        circular_id, len(gap_findings), len(action_plans),
    )

    # ── Interrupt: suspend graph, hand control to Streamlit UI ───────────────
    # The dict passed to interrupt() becomes the "interrupt value" readable
    # via app.get_state(config).tasks[0].interrupts[0].value in the UI.
    decision: dict = interrupt({
        "circular_id":   circular_id,
        "obligations":   obligations,
        "gap_findings":  gap_findings,
        "action_plans":  action_plans,
        "relevance_score":  state.get("relevance_score", 0.0),
        "relevance_reason": state.get("relevance_reason", ""),
        "message": (
            "Please review the gap findings and draft action plans above. "
            "Choose: approve | reject | edit_requested"
        ),
    })

    # ── Resumed: decision is the dict passed via Command(resume=...) ─────────
    human_decision = decision.get("decision", "rejected")
    human_edits    = decision.get("edits")

    log.info(
        "Human review complete for '%s': decision=%s",
        circular_id, human_decision,
    )

    return {
        "human_decision": human_decision,
        "human_edits":    human_edits,
    }
