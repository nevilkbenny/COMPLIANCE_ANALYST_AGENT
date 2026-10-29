"""
nodes/retrieval_router.py
=========================
Decision-making node using Jev (TypeSafe AI) to classify user query intent.

Classifies incoming queries into three distinct operational paths:
  1. "internal_search" -> Questions regarding SEBI regulations, circulars, compliance rules,
                         or Tata Motors corporate governance policies (routes to ChromaDB).
  2. "external_search" -> Questions asking for live stock prices, market data, recent news,
                         or external factual info not in internal policy PDFs (routes to web search).
  3. "conversational"  -> Greetings, chit-chat, assistant identity or capability questions
                         (routes directly to LLM answer generator).
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from CONVERSATIONAL_AGENT.src.state import ChatState
from CONVERSATIONAL_AGENT.utilities.llm import get_fast_llm

log = logging.getLogger(__name__)

# Jev Prompt & Rubric Criteria
QUERY_INTENT_INSTRUCTIONS = (
    "Classify the intent of the user query into one of three categories: "
    "'internal_search', 'external_search', or 'conversational'."
)

QUERY_INTENT_CRITERIA = {
    "internal_search": (
        "The query asks about SEBI regulations, circulars, compliance requirements, thresholds, "
        "deadlines, penalties, listing obligations (LODR), insider trading, related party transactions, "
        "or Tata Motors corporate policies, board governance, or code of conduct."
    ),
    "external_search": (
        "The query asks for live stock prices, current market share data, real-time financial market prices, "
        "recent company/industry news, or general external facts that are not contained in internal policy PDFs."
    ),
    "conversational": (
        "The query is a greeting (e.g., 'hi', 'hello', 'how are you', 'good morning'), a polite closing "
        "(e.g., 'thanks', 'bye'), an assistant capability/identity question (e.g., 'who are you', 'what can you do'), "
        "or casual conversational chit-chat."
    ),
}


def _fallback_llm_intent(query: str) -> str:
    """Fallback classifier using Groq / LLM when Jev/OpenRouter is unavailable."""
    try:
        from langchain_core.messages import HumanMessage
        llm = get_fast_llm(temperature=0.0)
        prompt = (
            "Classify this query into one of: INTERNAL_SEARCH, EXTERNAL_SEARCH, or CONVERSATIONAL.\n\n"
            f"Query: {query}\n\n"
            "Reply with ONLY one of the three labels."
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = str(resp.content).strip().upper()
        if "CONVERSATIONAL" in text:
            return "conversational"
        if "EXTERNAL" in text:
            return "external_search"
        return "internal_search"
    except Exception as exc:
        log.warning("Fallback LLM intent router error (%s) — defaulting to internal_search.", exc)
        return "internal_search"


def retrieval_router_node(state: ChatState) -> dict:
    """
    LangGraph node: Retrieval Router (Jev Decision Node)

    Analyzes the latest user message in state["messages"].
    Sets state["query_intent"] to 'internal_search', 'external_search', or 'conversational',
    and state["needs_retrieval"] to True/False.
    """
    load_dotenv()

    messages = state.get("messages", [])
    if not messages:
        return {
            "query_intent": "conversational",
            "needs_retrieval": False,
            "decision_trace": ["Empty message — conversational direct answer"],
        }

    # Extract latest user query text
    latest_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            latest_query = str(msg.content)
            break

    if not latest_query:
        latest_query = str(messages[-1].content)

    typesafe_api_key = (
        os.getenv("TYPESAFE_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
    ).strip()
    base_url = os.getenv("TYPESAFE_BASE_URL", "").strip()
    default_model = "~typesafe/jev-latest" if (base_url and "openrouter" in base_url) else "jev-latest"
    model_name = os.getenv("TYPESAFE_MODEL", "").strip() or default_model

    intent = "internal_search"
    reason = ""

    # Attempt Jev (TypeSafe AI)
    if typesafe_api_key:
        try:
            log.info("Classifying query intent with Jev (model: %s)...", model_name)
            from langchain_typesafe import Choice, TypeSafeClassifier

            classifier_kwargs: dict[str, str] = {
                "api_key": typesafe_api_key,
                "model": model_name,
            }
            if base_url:
                classifier_kwargs["base_url"] = base_url

            import concurrent.futures

            classifier = TypeSafeClassifier(**classifier_kwargs)
            payload = {
                "state": f"User Query: {latest_query}",
                "questions": {
                    "intent": Choice(
                        instructions=QUERY_INTENT_INSTRUCTIONS,
                        criteria=QUERY_INTENT_CRITERIA,
                    )
                },
            }

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(classifier.invoke, payload)
                response = future.result(timeout=10)

            choice_ans = response.choices.get("intent")
            if choice_ans:
                intent = choice_ans.choice.lower()
                confidence = getattr(choice_ans, "confidence", 1.0)
                reason = f"Jev ({model_name}) classified intent as '{intent}' (confidence: {confidence:.2f})"
                log.info(reason)
            else:
                log.warning("Jev response missing intent choice; falling back to LLM.")
                intent = _fallback_llm_intent(latest_query)
                reason = f"Fallback LLM classified intent as '{intent}'"
        except Exception as exc:
            log.warning("Jev intent classifier failed: %s — falling back to LLM.", exc)
            intent = _fallback_llm_intent(latest_query)
            reason = f"Fallback LLM classified intent as '{intent}' (Jev error: {exc})"
    else:
        log.info("No Jev API key configured. Using fallback LLM for intent classification.")
        intent = _fallback_llm_intent(latest_query)
        reason = f"Fallback LLM classified intent as '{intent}'"

    needs_retrieval = (intent == "internal_search")
    trace = list(state.get("decision_trace", []))
    trace.append(f"🤖 Query Analyzer: {reason}")

    return {
        "query_intent": intent,
        "needs_retrieval": needs_retrieval,
        "decision_trace": trace,
    }
