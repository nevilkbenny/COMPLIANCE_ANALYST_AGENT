"""
nodes/query_rewriter.py
=======================
Query Rewriter node for the conversational Q&A graph.

On follow-up questions, users often ask short, ambiguous queries that rely on
prior context (e.g., "What are the thresholds?" after asking about RPT rules).
This node rewrites such questions into fully self-contained search queries,
dramatically improving retrieval quality from ChromaDB.

On the very first message (no history), the node is a no-op.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from CONVERSATIONAL_AGENT.src.state import ChatState
from CONVERSATIONAL_AGENT.utilities.llm import get_fast_llm

log = logging.getLogger(__name__)

_REWRITE_SYSTEM = (
    "You are a search query optimizer for a SEBI regulatory compliance assistant.\n"
    "Given the conversation history and the user's latest question, rewrite the "
    "question into a standalone, self-contained search query that fully captures "
    "the user's intent without needing any prior context.\n"
    "Return ONLY the rewritten query — no explanation, no preamble, no quotes."
)


def query_rewriter_node(state: ChatState) -> dict:
    """
    LangGraph node: Query Rewriter (Entry Node)

    1. Resets per-turn transient state keys (web_search_results, retrieved_docs,
       needs_web_search, etc.) so prior turn data does not leak into the current turn.
    2. If there is only one message in history (first turn), messages are returned unchanged.
       Otherwise, the LLM rewrites the latest user question into a standalone query
       and replaces the last message in state so downstream nodes use it.
    """
    reset_state: dict = {
        "web_search_results": [],
        "retrieved_docs":     [],
        "needs_web_search":   False,
        "docs_are_relevant":  False,
        "decision_trace":     [],
    }

    messages = list(state.get("messages", []))
    if len(messages) <= 1:
        # First turn — nothing to rewrite
        log.debug("Query rewriter: first turn, skipping rewrite.")
        return reset_state

    # Build full history summary for context
    history_msgs = messages[:-1]
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in history_msgs
    )
    latest_question = messages[-1].content

    llm = get_fast_llm(temperature=0.0)

    rewrite_prompt = [
        SystemMessage(content=_REWRITE_SYSTEM),
        HumanMessage(content=(
            f"Conversation history:\n{history_text}\n\n"
            f"Latest user question: {latest_question}\n\n"
            f"Rewritten standalone search query:"
        )),
    ]

    try:
        rewritten = llm.invoke(rewrite_prompt)
        content = rewritten.content
        if isinstance(content, list):
            content_parts = []
            for item in content:
                if isinstance(item, str):
                    content_parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    content_parts.append(item["text"])
                elif hasattr(item, "text"):
                    content_parts.append(item.text)
                else:
                    content_parts.append(str(item))
            rewritten_text = "".join(content_parts).strip()
        else:
            rewritten_text = str(content).strip()
        log.info("Query rewriter: '%s' → '%s'", latest_question[:60], rewritten_text[:60])

        # Replace the latest HumanMessage with the rewritten version
        # (preserve id so the add_messages reducer deduplicates correctly)
        original_msg = messages[-1]
        updated_msg = HumanMessage(content=rewritten_text, id=getattr(original_msg, "id", None))
        return {
            **reset_state,
            "messages": [updated_msg],
        }

    except Exception as exc:
        log.warning("Query rewriter failed (%s) — using original question.", exc)
        return reset_state
