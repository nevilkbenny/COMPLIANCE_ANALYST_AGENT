"""
utilities/llm.py
================
Unified LLM factory providing Groq with intelligent fallback.

Primary Provider: Groq (using ChatGroq)
  - Models:
    * `llama-3.3-70b-versatile` for detailed extraction, reasoning, structured output, and Q&A.
    * `llama-3.1-8b-instant` for ultra-fast query classification, rewriting, and routing.
  - High limits, low latency (< 1s per query).

Fallback Provider: Google Gemini (ChatGoogleGenerativeAI)
  - Model: `gemini-1.5-flash` with fail-fast timeouts (avoids preview quota issues).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)


def get_llm(
    temperature: float = 0.1,
    model: str | None = None,
    timeout: int = 30,
    **kwargs: Any,
):
    """
    Return the primary LLM instance (ChatGroq if GROQ_API_KEY is available,
    otherwise ChatGoogleGenerativeAI).
    """
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

    if groq_api_key:
        from langchain_groq import ChatGroq

        model_name = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        log.debug("Using Groq LLM: model=%s, temperature=%.2f", model_name, temperature)
        return ChatGroq(
            model=model_name,
            groq_api_key=groq_api_key,
            temperature=temperature,
            timeout=timeout,
            max_retries=2,
            **kwargs,
        )

    # Fallback to Gemini with robust timeout and 1.5-flash model
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        log.warning("GROQ_API_KEY not found; falling back to Google Gemini (%s)", model_name)
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_key,
            temperature=temperature,
            request_timeout=timeout,
            max_retries=1,
            **kwargs,
        )

    raise ValueError(
        "No LLM API key found. Please set GROQ_API_KEY in your .env file."
    )


def get_fast_llm(
    temperature: float = 0.0,
    model: str | None = None,
    timeout: int = 15,
    **kwargs: Any,
):
    """
    Return an ultra-fast LLM instance for routers, classifiers, and short rewrites.
    Defaults to `llama-3.1-8b-instant` on Groq.
    """
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

    if groq_api_key:
        from langchain_groq import ChatGroq

        model_name = model or os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
        log.debug("Using Fast Groq LLM: model=%s", model_name)
        return ChatGroq(
            model=model_name,
            groq_api_key=groq_api_key,
            temperature=temperature,
            timeout=timeout,
            max_retries=2,
            **kwargs,
        )

    return get_llm(temperature=temperature, model=model, timeout=timeout, **kwargs)


# Convenience alias for high-reasoning nodes
get_reasoning_llm = get_llm
