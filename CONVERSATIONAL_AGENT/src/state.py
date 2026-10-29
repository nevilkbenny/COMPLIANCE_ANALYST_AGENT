"""
src/state.py
============
LangGraph state schema for the conversational Document Q&A graph (CONVERSATIONAL_AGENT).
"""

from __future__ import annotations

from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ChatState(TypedDict, total=False):
    """
    State for the conversational Q&A graph with Jev decision-making nodes.

    Fields
    ------
    messages:           Full conversation history (LangChain message objects).
    corpus:             Which ChromaDB collection(s) to search ("both" | "sebi" | "tata").
    needs_retrieval:    Bool decided by Jev: whether retrieval is required.
    retrieved_docs:     List of doc-chunk dicts returned by ChromaDB retriever:
                        [{file, page, source_type, snippet}, ...]
    docs_are_relevant:  Bool decided by Jev: whether retrieved docs are relevant.
    needs_web_search:   Bool decided by Jev: whether external web search is needed.
    web_search_results: List of web search result dicts:
                        [{title, snippet, url}, ...]
    decision_trace:     List of strings logging Jev decisions made along the path.
    answer:             The final plain-text answer string (also appended to messages).
    """

    messages:           Annotated[Sequence[BaseMessage], add_messages]
    corpus:             str
    query_intent:       str   # "internal_search" | "external_search" | "conversational"
    needs_retrieval:    bool
    retrieved_docs:     list[dict]
    docs_are_relevant:  bool
    needs_web_search:   bool
    web_search_results: list[dict]
    decision_trace:     list[str]
    answer:             str
