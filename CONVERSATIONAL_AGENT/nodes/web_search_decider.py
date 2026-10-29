"""
nodes/web_search_decider.py
===========================
Decision-making node using Jev (TypeSafe AI) to decide if external web search is needed.

Called when ChromaDB document retrieval returned no relevant documents.
Evaluates whether searching the public web (SEBI official portal, regulatory news,
public market regulations) can answer the user's question, or whether the query
is strictly internal/confidential or off-topic.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from CONVERSATIONAL_AGENT.src.state import ChatState
from CONVERSATIONAL_AGENT.utilities.llm import get_fast_llm

log = logging.getLogger(__name__)

WEB_SEARCH_DECISION_INSTRUCTIONS = (
    "Determine whether conducting an external public web search (e.g. for live stock prices, "
    "market data, SEBI notifications, public announcements, news, or regulatory rules) "
    "is appropriate and useful to answer the user query."
)

WEB_SEARCH_DECISION_CRITERIA = {
    "search": (
        "The query is about current stock/market prices, financial market data, public company news, "
        "external public regulations, SEBI circulars, stock exchange guidelines, or public financial topics "
        "that can be found on public websites or official search."
    ),
    "no_search": (
        "The query is exclusively about confidential internal Tata Motors internal policies, "
        "proprietary internal workflows, non-public employee matters, nonsensical text, "
        "or questions where public web search would be irrelevant or inappropriate."
    ),
}


def _fallback_llm_web_search_decision(query: str) -> bool:
    """Fallback classifier using Groq / LLM when Jev/OpenRouter is unavailable."""
    try:
        from langchain_core.messages import HumanMessage
        llm = get_fast_llm(temperature=0.0)
        prompt = (
            "Internal documents do not contain the answer. Should we perform a public web search "
            "for this regulatory query?\n\n"
            f"Query: {query}\n\n"
            "Reply with ONLY 'SEARCH' if public web search is appropriate, or 'NO_SEARCH' if not."
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = str(resp.content).strip().upper()
        return "SEARCH" in text and "NO_SEARCH" not in text
    except Exception as exc:
        log.warning("Fallback LLM web search decision error (%s) — defaulting to False.", exc)
        return False


def web_search_decider_node(state: ChatState) -> dict:
    """
    LangGraph node: Web Search Decider (Jev Decision Node)

    Evaluates if public web search is appropriate for state["messages"][-1].
    Sets state["needs_web_search"] to True or False.
    """
    load_dotenv()

    messages = state.get("messages", [])
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            query = str(msg.content)
            break
    if not query and messages:
        query = str(messages[-1].content)

    typesafe_api_key = (
        os.getenv("TYPESAFE_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
    ).strip()
    base_url = os.getenv("TYPESAFE_BASE_URL", "").strip()
    default_model = "~typesafe/jev-latest" if (base_url and "openrouter" in base_url) else "jev-latest"
    model_name = os.getenv("TYPESAFE_MODEL", "").strip() or default_model

    needs_web_search = False
    reason = ""

    if typesafe_api_key:
        try:
            log.info("Deciding web search necessity with Jev (model: %s)...", model_name)
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
                "state": f"User Query: {query}",
                "questions": {
                    "web_search": Choice(
                        instructions=WEB_SEARCH_DECISION_INSTRUCTIONS,
                        criteria=WEB_SEARCH_DECISION_CRITERIA,
                    )
                },
            }

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(classifier.invoke, payload)
                response = future.result(timeout=10)

            choice_ans = response.choices.get("web_search")
            if choice_ans:
                selected = choice_ans.choice.lower()
                needs_web_search = (selected == "search")
                confidence = getattr(choice_ans, "confidence", 1.0)
                reason = f"Jev ({model_name}) decided: web search {'NEEDED' if needs_web_search else 'NOT NEEDED'} (confidence: {confidence:.2f})"
                log.info(reason)
            else:
                log.warning("Jev response missing web_search; falling back to LLM.")
                needs_web_search = _fallback_llm_web_search_decision(query)
                reason = f"Fallback LLM decided: web search {'NEEDED' if needs_web_search else 'NOT NEEDED'}"
        except Exception as exc:
            log.warning("Jev web search decision call failed: %s — falling back to LLM.", exc)
            needs_web_search = _fallback_llm_web_search_decision(query)
            reason = f"Fallback LLM decided: web search {'NEEDED' if needs_web_search else 'NOT NEEDED'} (Jev error: {exc})"
    else:
        needs_web_search = _fallback_llm_web_search_decision(query)
        reason = f"Fallback LLM decided: web search {'NEEDED' if needs_web_search else 'NOT NEEDED'}"

    trace = list(state.get("decision_trace", []))
    trace.append(f"🤖 Web Search Decider: {reason}")

    return {
        "needs_web_search": needs_web_search,
        "decision_trace": trace,
    }
