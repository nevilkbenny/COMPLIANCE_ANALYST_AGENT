"""
nodes/doc_relevance_evaluator.py
================================
Decision-making node using Jev (TypeSafe AI) to grade retrieved document relevance.

Evaluates whether the documents retrieved from ChromaDB actually contain relevant
clauses, facts, or regulatory context to address the user's question.
If YES: passes to the LLM answer_generator.
If NO: routes to the web_search_decider node to check if external web search is needed.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from CONVERSATIONAL_AGENT.src.state import ChatState
from CONVERSATIONAL_AGENT.utilities.llm import get_fast_llm

log = logging.getLogger(__name__)

DOC_RELEVANCE_INSTRUCTIONS = (
    "Evaluate whether the retrieved document excerpts contain relevant, pertinent information "
    "or context that helps directly address or answer the user query. "
    "Even if the excerpts only partially address the query or contain the underlying regulation, "
    "mark as 'relevant'. If the excerpts discuss completely unrelated topics or lack any meaningful "
    "connection to the query, mark as 'not_relevant'."
)

DOC_RELEVANCE_CRITERIA = {
    "relevant": (
        "The retrieved document chunks contain terms, regulations, policy clauses, guidelines, "
        "or context that are substantively related and useful to answer the user query."
    ),
    "not_relevant": (
        "The retrieved document chunks are completely unrelated, off-topic, superficial, or contain "
        "no substantive information that addresses the user query."
    ),
}


def _format_docs_summary(docs: list[dict]) -> str:
    """Format doc snippets into a clean string for evaluation without truncation."""
    if not docs:
        return "(No documents retrieved)"
    snippets = []
    for d in docs:
        file_name = d.get("file", "Doc")
        page = d.get("page", "N/A")
        snippet = d.get("snippet", "").strip()
        snippets.append(f"• [{file_name} | Page {page}]:\n{snippet}")
    return "\n\n".join(snippets)


def _fallback_llm_relevance(query: str, docs_summary: str) -> bool:
    """Fallback relevance evaluation using Groq / LLM."""
    try:
        from langchain_core.messages import HumanMessage
        llm = get_fast_llm(temperature=0.0)
        prompt = (
            "Evaluate whether these document excerpts are relevant to the user query.\n\n"
            f"User Query: {query}\n\n"
            f"Retrieved Excerpts:\n{docs_summary}\n\n"
            "Reply with ONLY 'RELEVANT' if the excerpts provide useful context, or 'NOT_RELEVANT' if unrelated."
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = str(resp.content).strip().upper()
        return "RELEVANT" in text and "NOT_RELEVANT" not in text
    except Exception as exc:
        log.warning("Fallback LLM relevance evaluation error (%s) — defaulting to True.", exc)
        return True


def doc_relevance_evaluator_node(state: ChatState) -> dict:
    """
    LangGraph node: Document Relevance Evaluator (Jev Decision Node)

    Checks if state["retrieved_docs"] are relevant to state["messages"][-1].
    Sets state["docs_are_relevant"] to True or False.
    """
    load_dotenv()

    docs = state.get("retrieved_docs", [])
    if not docs:
        log.info("No documents retrieved — marked as not relevant.")
        trace = list(state.get("decision_trace", []))
        trace.append("🤖 Doc Grader: No documents found in database.")
        return {
            "docs_are_relevant": False,
            "decision_trace": trace,
        }

    # Extract user query
    messages = state.get("messages", [])
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            query = str(msg.content)
            break
    if not query and messages:
        query = str(messages[-1].content)

    docs_summary = _format_docs_summary(docs)

    typesafe_api_key = (
        os.getenv("TYPESAFE_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
    ).strip()
    base_url = os.getenv("TYPESAFE_BASE_URL", "").strip()
    default_model = "~typesafe/jev-latest" if (base_url and "openrouter" in base_url) else "jev-latest"
    model_name = os.getenv("TYPESAFE_MODEL", "").strip() or default_model

    docs_are_relevant = True
    reason = ""

    if typesafe_api_key:
        try:
            log.info("Evaluating document relevance with Jev (model: %s)...", model_name)
            from langchain_typesafe import Choice, TypeSafeClassifier

            classifier_kwargs: dict[str, str] = {
                "api_key": typesafe_api_key,
                "model": model_name,
            }
            if base_url:
                classifier_kwargs["base_url"] = base_url

            import concurrent.futures

            classifier = TypeSafeClassifier(**classifier_kwargs)
            state_text = f"User Query: {query}\n\nRetrieved Document Excerpts:\n{docs_summary}"
            payload = {
                "state": state_text,
                "questions": {
                    "doc_relevance": Choice(
                        instructions=DOC_RELEVANCE_INSTRUCTIONS,
                        criteria=DOC_RELEVANCE_CRITERIA,
                    )
                },
            }

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(classifier.invoke, payload)
                response = future.result(timeout=10)

            choice_ans = response.choices.get("doc_relevance")
            if choice_ans:
                selected = choice_ans.choice.lower()
                docs_are_relevant = (selected == "relevant")
                confidence = getattr(choice_ans, "confidence", 1.0)
                reason = f"Jev ({model_name}) decided: documents are {'RELEVANT' if docs_are_relevant else 'NOT RELEVANT'} (confidence: {confidence:.2f})"
                log.info(reason)
            else:
                log.warning("Jev response missing doc_relevance; falling back to LLM.")
                docs_are_relevant = _fallback_llm_relevance(query, docs_summary)
                reason = f"Fallback LLM decided: documents are {'RELEVANT' if docs_are_relevant else 'NOT RELEVANT'}"
        except Exception as exc:
            log.warning("Jev doc relevance call failed: %s — falling back to LLM.", exc)
            docs_are_relevant = _fallback_llm_relevance(query, docs_summary)
            reason = f"Fallback LLM decided: documents are {'RELEVANT' if docs_are_relevant else 'NOT RELEVANT'} (Jev error: {exc})"
    else:
        docs_are_relevant = _fallback_llm_relevance(query, docs_summary)
        reason = f"Fallback LLM decided: documents are {'RELEVANT' if docs_are_relevant else 'NOT RELEVANT'}"

    trace = list(state.get("decision_trace", []))
    trace.append(f"🤖 Doc Grader: {reason}")

    return {
        "docs_are_relevant": docs_are_relevant,
        "decision_trace": trace,
    }
