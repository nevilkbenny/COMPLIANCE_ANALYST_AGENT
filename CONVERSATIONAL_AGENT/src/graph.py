"""
src/graph.py
============
Conversational Q&A LangGraph (CONVERSATIONAL_AGENT).

Graph Topology
--------------

               query_rewriter
                     │
                     ▼
              retrieval_router  (Jev: Is internal retrieval needed?)
                 │          │
        (needs=False)    (needs=True)
                 │          │
                 │          ▼
                 │    chat_retriever (ChromaDB search)
                 │          │
                 │          ▼
                 │    doc_relevance_evaluator (Jev: Are retrieved docs relevant?)
                 │       │               │
                 │  (relevant=True) (relevant=False)
                 │       │               │
                 │       │               ▼
                 │       │     web_search_decider (Jev: Should we search the web?)
                 │       │         │           │
                 │       │  (search=True) (search=False)
                 │       │         │           │
                 │       │         ▼           │
                 │       │   web_search_node   │
                 │       │         │           │
                 ▼       ▼         ▼           ▼
                         answer_generator
                                 │
                                 ▼
                                END
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from CONVERSATIONAL_AGENT.src.state import ChatState
from CONVERSATIONAL_AGENT.nodes.answer_generator import answer_generator_node
from CONVERSATIONAL_AGENT.nodes.chat_retriever import chat_retriever_node
from CONVERSATIONAL_AGENT.nodes.doc_relevance_evaluator import doc_relevance_evaluator_node
from CONVERSATIONAL_AGENT.nodes.query_rewriter import query_rewriter_node
from CONVERSATIONAL_AGENT.nodes.retrieval_router import retrieval_router_node
from CONVERSATIONAL_AGENT.nodes.web_search_decider import web_search_decider_node
from CONVERSATIONAL_AGENT.nodes.web_search_node import web_search_node

log = logging.getLogger(__name__)

_CHAT_APP = None


def _route_after_query_analysis(state: ChatState) -> str:
    intent = state.get("query_intent", "")
    if intent == "internal_search":
        return "chat_retriever"
    elif intent == "external_search":
        return "web_search_decider"
    return "answer_generator"


def _route_after_doc_eval(state: ChatState) -> str:
    if state.get("docs_are_relevant", True):
        return "answer_generator"
    return "web_search_decider"


def _route_after_web_search_decider(state: ChatState) -> str:
    if state.get("needs_web_search", False):
        return "web_search_node"
    return "answer_generator"


def build_chat_graph(checkpointer=None):
    """Build and compile the conversational Q&A StateGraph."""
    if checkpointer is None:
        checkpointer = MemorySaver()

    g = StateGraph(ChatState)

    g.add_node("query_rewriter",          query_rewriter_node)
    g.add_node("retrieval_router",        retrieval_router_node)
    g.add_node("chat_retriever",          chat_retriever_node)
    g.add_node("doc_relevance_evaluator", doc_relevance_evaluator_node)
    g.add_node("web_search_decider",      web_search_decider_node)
    g.add_node("web_search_node",         web_search_node)
    g.add_node("answer_generator",        answer_generator_node)

    g.set_entry_point("query_rewriter")
    g.add_edge("query_rewriter", "retrieval_router")

    g.add_conditional_edges("retrieval_router", _route_after_query_analysis, {
        "chat_retriever":     "chat_retriever",
        "web_search_decider": "web_search_decider",
        "answer_generator":   "answer_generator",
    })

    g.add_edge("chat_retriever", "doc_relevance_evaluator")

    g.add_conditional_edges("doc_relevance_evaluator", _route_after_doc_eval, {
        "answer_generator":   "answer_generator",
        "web_search_decider": "web_search_decider",
    })

    g.add_conditional_edges("web_search_decider", _route_after_web_search_decider, {
        "web_search_node":  "web_search_node",
        "answer_generator": "answer_generator",
    })

    g.add_edge("web_search_node",  "answer_generator")
    g.add_edge("answer_generator", END)

    app = g.compile(checkpointer=checkpointer)
    log.info("Chat Q&A graph compiled successfully.")
    return app


def get_chat_app():
    """Return the compiled chat graph app, building it once per process."""
    global _CHAT_APP
    if _CHAT_APP is None:
        _CHAT_APP = build_chat_graph()
    return _CHAT_APP
