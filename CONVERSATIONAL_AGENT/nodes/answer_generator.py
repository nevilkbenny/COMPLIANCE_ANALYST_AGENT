"""
nodes/answer_generator.py
=========================
Answer Generator node for the conversational Q&A graph.

Assembles a grounded, citation-backed answer using:
  - Internal ChromaDB document chunks (when available and verified relevant by Jev)
  - External Web Search results (when internal docs were insufficient and web search was performed)
  - Direct conversational response (when Jev determined no retrieval was needed)
  - Full conversation message history (for multi-turn coherence)
  - Strict system prompt that prevents hallucination of regulations / dates
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from CONVERSATIONAL_AGENT.src.state import ChatState
from CONVERSATIONAL_AGENT.utilities.llm import get_llm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_DOCUMENT_SYSTEM_TEMPLATE = """\
You are an expert SEBI Regulatory and Corporate Governance Legal Assistant for Tata Motors Limited.

Your knowledge base consists of:
1. Official SEBI Circulars and Regulatory Notifications
2. Tata Motors' internal Corporate Governance and Compliance Policy documents

Answer the user's question based STRICTLY on the provided context excerpts below.

CITATION RULES (mandatory):
- After every factual claim, regulation threshold, date, or policy clause, cite its source
  using the format: [Source: <file_name>, Page <page>]
- Example: "The threshold for RPT approval is ₹1,000 crore or 10% of annual turnover
  [Source: SEBI Circular on RPT, Page 3]."

GROUNDING RULES:
- If the context does not contain enough information to answer the question, respond:
  "The loaded SEBI circulars and Tata Motors policy documents do not contain sufficient
   information to answer this question. Please consult the primary source documents directly."
- Do NOT invent, extrapolate, or assume any regulation, date, threshold, or policy clause
  that is not explicitly present in the context.

FORMATTING RULES:
- Use clear, professional headings, bullet points, and numbered lists where helpful.
- Keep responses concise and actionable for a Company Secretary / Compliance audience.

---
CONTEXT (Retrieved Internal Document Excerpts):
{context}
---
"""

_WEB_SEARCH_SYSTEM_TEMPLATE = """\
You are an expert SEBI Regulatory and Corporate Governance Legal Assistant for Tata Motors Limited.

An external public web search was conducted to answer the user's question.
Answer the user's question directly using the external web search results below.

GUIDELINES:
1. State clearly that this response is synthesized from public web search results.
2. Directly answer the question based on the provided web excerpts, news, or exchange data.
3. For market / stock price queries: summarize any available figures, recent price trends, or trading details mentioned in the excerpts, and provide direct, helpful links to the live quote portals (e.g. NSE, BSE, Groww, Investing.com) from the sources.
4. CITATION RULES: Cite the web sources with markdown links using the format: [Source: <title>](<url>)
   Example: "...according to recent market listings [Source: Groww - Tata Motors Live Quote](https://groww.in/...)".
5. Keep the tone professional, objective, and clear.

---
CONTEXT (External Web Search Results):
{context}
---
"""

_DIRECT_CONVERSATIONAL_SYSTEM = """\
You are an expert SEBI Regulatory and Corporate Governance Legal Assistant for Tata Motors Limited.

The user's query is conversational, a greeting, or an inquiry about your capabilities.
Answer politely, professionally, and clearly.

Briefly explain how you can help:
- Reviewing compliance impact of new SEBI circulars against Tata Motors policies.
- Answering questions regarding SEBI LODR, PIT (insider trading), RPT rules, and board disclosures.
- Searching internal regulatory circulars, governance policies, or conducting external regulatory web searches.

Keep your response friendly, concise, and professional.
"""


def _format_docs_context(docs: list[dict]) -> str:
    """Format internal doc chunks into context block."""
    if not docs:
        return "(No relevant internal document excerpts available.)"
    blocks = []
    for i, d in enumerate(docs, 1):
        blocks.append(
            f"[{i}] {d.get('file', 'Unknown')} | Page {d.get('page', 'N/A')} "
            f"| Type: {d.get('source_type', 'General')}\n"
            f"{d.get('snippet', '')}"
        )
    return "\n\n---\n\n".join(blocks)


def _format_web_context(results: list[dict]) -> str:
    """Format web search results into context block."""
    if not results:
        return "(No web search results available.)"
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(
            f"[{i}] Title: {r.get('title', 'Web Result')}\n"
            f"URL: {r.get('url', 'N/A')}\n"
            f"Excerpt: {r.get('snippet', '')}"
        )
    return "\n\n---\n\n".join(blocks)


def _extract_text(content: object) -> str:
    """Safely extract plain text from str or list response.content."""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(getattr(item, "text", ""))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


def answer_generator_node(state: ChatState) -> dict:
    """
    LangGraph node: Answer Generator

    Synthesizes the response based on the active path:
      - Direct reply (needs_retrieval == False)
      - Internal docs (docs_are_relevant == True)
      - Web search (web_search_results present)
      - Fallback grounded notice (unsupported / off-topic)
    """
    load_dotenv()

    needs_retrieval    = state.get("needs_retrieval", True)
    docs_are_relevant  = state.get("docs_are_relevant", True)
    web_results        = state.get("web_search_results", [])
    internal_docs      = state.get("retrieved_docs", [])
    messages           = list(state.get("messages", []))

    # Determine which prompt mode to use
    if web_results:
        # Path 1: External web search answer (must come before not needs_retrieval!)
        context_str = _format_web_context(web_results)
        system_prompt = _WEB_SEARCH_SYSTEM_TEMPLATE.format(context=context_str)
    elif docs_are_relevant and internal_docs:
        # Path 2: Grounded internal document answer
        context_str = _format_docs_context(internal_docs)
        system_prompt = _DOCUMENT_SYSTEM_TEMPLATE.format(context=context_str)
    elif not needs_retrieval:
        # Path 3: Conversational / greeting — direct LLM answer
        system_prompt = _DIRECT_CONVERSATIONAL_SYSTEM
    else:
        # Path 4: Neither internal docs nor web search applied
        answer_text = (
            "ℹ️ The internal SEBI circulars and Tata Motors policy documents do not contain "
            "information relevant to your query, and external web search was not needed. "
            "Please refine your query or consult the primary legal sources directly."
        )
        return {
            "messages": [AIMessage(content=answer_text)],
            "answer":   answer_text,
        }

    llm_messages = [SystemMessage(content=system_prompt)]
    # Keep recent multi-turn conversation (last 8 messages = 4 turns) for rich dialogue context
    recent_messages = messages[-8:] if len(messages) > 8 else messages
    llm_messages.extend(recent_messages)

    llm = get_llm(temperature=0.1, timeout=30)

    try:
        response = llm.invoke(llm_messages)
        answer_text = _extract_text(response.content)
        log.info("Answer generator: produced %d-char response.", len(answer_text))
    except Exception as exc:
        log.error("Answer generator: LLM call failed: %s", exc)
        answer_text = (
            "⚠️ An error occurred while generating the answer. "
            "Please check the logs and try again."
        )

    return {
        "messages": [AIMessage(content=answer_text)],
        "answer":   answer_text,
    }
