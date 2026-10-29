"""
src/graph.py
============
LangGraph StateGraph assembly for the SEBI Regulatory Change Impact Monitor (COMPLIANCE_AGENT).

Graph topology
--------------

  watcher
    │
    ▼
  document_loader
    │
    ▼
  document_validator ──(invalid)──► audit_logger ──► END
    │
  obligation_extractor
    │
    ▼
  retriever
    │
    ▼
  relevance_judge
    │
    ├─ score < threshold ──► audit_logger ──► END
    │
    └─ score ≥ threshold ──► gap_analyzer
                                │
                                ▼
                          action_planner
                                │
                                ▼
                          human_review
                                │
                   ┌────────────┼─────────────┐
                   ▼            ▼             ▼
               approved      rejected   edit_requested
                   │            │             │
                   └─────┬──────┘     action_planner (re-draft loop)
                         ▼
                   audit_logger ──► END
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from COMPLIANCE_AGENT.nodes.action_planner import action_planner_node
from COMPLIANCE_AGENT.nodes.audit_logger import audit_logger_node
from COMPLIANCE_AGENT.nodes.document_loader import document_loader_node
from COMPLIANCE_AGENT.nodes.document_validator import document_validator_node
from COMPLIANCE_AGENT.nodes.gap_analyzer import gap_analyzer_node
from COMPLIANCE_AGENT.nodes.human_review import human_review_node
from COMPLIANCE_AGENT.nodes.obligation_extractor import obligation_extractor_node
from COMPLIANCE_AGENT.nodes.relevance_judge import relevance_judge_node
from COMPLIANCE_AGENT.nodes.retriever import retriever_node
from COMPLIANCE_AGENT.nodes.watcher import watcher_node
from COMPLIANCE_AGENT.src.state import SEBIMonitorState

log = logging.getLogger(__name__)


def _get_relevance_threshold() -> float:
    load_dotenv()
    return float(os.getenv("RELEVANCE_THRESHOLD", "0.4"))


def route_after_watcher(state: SEBIMonitorState) -> str:
    """Skip straight to audit_logger if the circular was already processed."""
    if state.get("error"):
        log.warning("Routing to audit_logger due to error: %s", state["error"])
        return "audit_logger"
    if state.get("already_processed"):
        log.info("Circular already processed — short-circuiting to audit_logger.")
        return "audit_logger"
    return "document_loader"


def route_after_validator(state: SEBIMonitorState) -> str:
    if state.get("error") or not state.get("is_valid_document", True):
        log.warning(
            "Document rejected by validator: %s — short-circuiting to audit_logger.",
            state.get("error", "Invalid or unrelated document"),
        )
        return "audit_logger"
    log.info("Document validated successfully — proceeding to obligation_extractor.")
    return "obligation_extractor"


def route_on_relevance(state: SEBIMonitorState) -> str:
    score     = state.get("relevance_score", 0.0)
    threshold = _get_relevance_threshold()
    if score < threshold:
        log.info("Relevance score %.2f < threshold %.2f — archiving circular.", score, threshold)
        return "audit_logger"
    log.info("Relevance score %.2f ≥ threshold %.2f — proceeding to gap analysis.", score, threshold)
    return "gap_analyzer"


def route_after_human(state: SEBIMonitorState) -> str:
    decision = state.get("human_decision", "rejected")
    if decision == "edit_requested":
        log.info("CS requested edits — looping back to action_planner.")
        return "action_planner"
    log.info("Human decision: '%s' — routing to audit_logger.", decision)
    return "audit_logger"


def build_graph(checkpointer=None) -> "CompiledStateGraph":
    """Build and compile the SEBI Monitor StateGraph."""
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = StateGraph(SEBIMonitorState)

    graph.add_node("watcher",              watcher_node)
    graph.add_node("document_loader",      document_loader_node)
    graph.add_node("document_validator",   document_validator_node)
    graph.add_node("obligation_extractor", obligation_extractor_node)
    graph.add_node("retriever",            retriever_node)
    graph.add_node("relevance_judge",      relevance_judge_node)
    graph.add_node("gap_analyzer",         gap_analyzer_node)
    graph.add_node("action_planner",       action_planner_node)
    graph.add_node("human_review",         human_review_node)
    graph.add_node("audit_logger",         audit_logger_node)

    graph.set_entry_point("watcher")

    graph.add_conditional_edges("watcher", route_after_watcher, {
        "document_loader": "document_loader",
        "audit_logger":    "audit_logger",
    })

    graph.add_edge("document_loader", "document_validator")

    graph.add_conditional_edges("document_validator", route_after_validator, {
        "obligation_extractor": "obligation_extractor",
        "audit_logger":         "audit_logger",
    })

    graph.add_edge("obligation_extractor", "retriever")
    graph.add_edge("retriever",            "relevance_judge")

    graph.add_conditional_edges("relevance_judge", route_on_relevance, {
        "gap_analyzer": "gap_analyzer",
        "audit_logger": "audit_logger",
    })

    graph.add_edge("gap_analyzer",   "action_planner")
    graph.add_edge("action_planner", "human_review")

    graph.add_conditional_edges("human_review", route_after_human, {
        "action_planner": "action_planner",
        "audit_logger":   "audit_logger",
    })

    graph.add_edge("audit_logger", END)

    app = graph.compile(checkpointer=checkpointer)
    log.info("Compliance pipeline graph compiled successfully.")
    return app


_APP = None


def get_app(checkpointer=None):
    """Return the compiled graph app, building it once."""
    global _APP
    if _APP is None:
        _APP = build_graph(checkpointer=checkpointer)
    return _APP
