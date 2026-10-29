"""
nodes/web_search_node.py
========================
Web Search execution node for the conversational Q&A graph.

Executes a targeted search via Tavily AI (with DuckDuckGo fallback) when
internal document retrieval is insufficient and Jev determines that an
external regulatory or market search is required.
Stores structured search results into state["web_search_results"].
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from CONVERSATIONAL_AGENT.src.state import ChatState

log = logging.getLogger(__name__)


def web_search_node(state: ChatState) -> dict:
    """
    LangGraph node: Web Search Execution

    Searches using Tavily AI (or DuckDuckGo fallback) and saves top results
    to state["web_search_results"].
    """
    load_dotenv()

    messages = state.get("messages", [])
    query = ""
    for msg in reversed(messages):
        is_human = (
            isinstance(msg, HumanMessage)
            or getattr(msg, "type", "") in ("human", "user")
            or (isinstance(msg, dict) and msg.get("type") in ("human", "user") or (isinstance(msg, dict) and msg.get("role") in ("human", "user")))
        )
        if is_human:
            query = getattr(msg, "content", "") or (msg.get("content", "") if isinstance(msg, dict) else "")
            if query:
                break
    if not query and messages:
        last = messages[-1]
        query = getattr(last, "content", "") or (last.get("content", "") if isinstance(last, dict) else str(last))

    results: list[dict] = []
    search_query = query.strip()
    provider_used = "Tavily AI"
    tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()

    log.info("Executing web search for query: '%s'", search_query)

    # 1. Primary: Tavily AI Search
    if tavily_api_key:
        try:
            from tavily import TavilyClient

            search_depth = os.getenv("TAVILY_SEARCH_DEPTH", "advanced").strip().lower()
            client = TavilyClient(api_key=tavily_api_key)
            response = client.search(
                query=search_query,
                search_depth=search_depth,
                max_results=4,
            )

            for r in response.get("results", []):
                results.append({
                    "title":   r.get("title", "Web Source"),
                    "snippet": r.get("content", ""),
                    "url":     r.get("url", ""),
                })
            log.info("Tavily search succeeded with %d results.", len(results))
        except Exception as exc:
            log.warning("Tavily search error (%s) — falling back to DuckDuckGo.", exc)
            results = []
    else:
        log.info("TAVILY_API_KEY not configured in .env — using DuckDuckGo fallback.")

    # 2. Fallback: DuckDuckGo (if Tavily key missing or search returned empty/failed)
    if not results:
        provider_used = "DuckDuckGo (fallback)"
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                raw_results = list(ddgs.text(search_query, max_results=4))

            for r in raw_results:
                results.append({
                    "title":   r.get("title", "Web Source"),
                    "snippet": r.get("body", ""),
                    "url":     r.get("href", ""),
                })
            log.info("DuckDuckGo fallback search succeeded with %d results.", len(results))
        except Exception as exc:
            log.error("DuckDuckGo fallback search failed: %s", exc)

    trace = list(state.get("decision_trace", []))
    trace.append(
        f"🌐 Web Search ({provider_used}): executed search ('{search_query[:40]}…') — found {len(results)} external results."
    )

    return {
        "web_search_results": results,
        "decision_trace": trace,
    }
