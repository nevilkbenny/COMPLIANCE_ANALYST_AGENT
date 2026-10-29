"""
nodes/chat_retriever.py
=======================
Multi-corpus retrieval node for the conversational Q&A graph.

Reads the latest user query from state["messages"] and searches one or both
ChromaDB collections based on state["corpus"]:
  "sebi"  → sebi_circulars only
  "tata"  → tata_motors_policies only
  "both"  → both collections (default)

Caches embedding model and ChromaDB collection instances in memory so
searches execute in < 0.5s instead of reloading 400MB models on every turn.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from CONVERSATIONAL_AGENT.utilities.embeddings import (  # noqa: E402
    COLLECTION_SEBI,
    COLLECTION_TATA,
    DEFAULT_CHROMA_DIR,
    build_embedding_model,
    get_vectorstore,
)
from CONVERSATIONAL_AGENT.src.state import ChatState  # noqa: E402

log = logging.getLogger(__name__)

# Number of chunks to retrieve per collection
_K = 5


def _get_embeddings():
    """Return cached embedding model singleton."""
    return build_embedding_model(device="cpu")


def _get_vectorstore(collection_name: str, chroma_path: str | None = None) -> Chroma:
    """Return cached Chroma vectorstore instance."""
    return get_vectorstore(collection_name, chroma_path or DEFAULT_CHROMA_DIR)


def _latest_user_query(messages: list) -> str:
    """Return the text of the most recent HumanMessage."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
        if getattr(msg, "type", "") == "human":
            return str(getattr(msg, "content", ""))
    return ""


def chat_retriever_node(state: ChatState) -> dict:
    """
    LangGraph node: Chat Retriever

    Performs fast semantic similarity search against cached ChromaDB collections.
    """
    messages = list(state.get("messages", []))
    query = _latest_user_query(messages)
    corpus = state.get("corpus", "both")

    if not query:
        log.warning("Chat retriever: no user query found in state.")
        return {"retrieved_docs": []}

    log.info("Chat retriever: querying corpus='%s' for: '%s'", corpus, query[:80])
    chroma_path = str(Path(DEFAULT_CHROMA_DIR).resolve())

    retrieved: list[dict] = []

    def _search(collection_name: str, source_type_label: str) -> None:
        try:
            vs = _get_vectorstore(collection_name, chroma_path)
            docs = vs.similarity_search(query, k=_K)
            for d in docs:
                retrieved.append({
                    "file":        d.metadata.get("file_name", d.metadata.get("source", "Unknown")),
                    "page":        d.metadata.get("page", "N/A"),
                    "source_type": source_type_label,
                    "snippet":     d.page_content,
                })
            log.info("Chat retriever: %d chunks from '%s'.", len(docs), collection_name)
        except Exception as exc:
            log.error("Chat retriever: error querying collection '%s': %s", collection_name, exc)

    if corpus in ("sebi", "both"):
        _search(COLLECTION_SEBI, "SEBI Circular")
    if corpus in ("tata", "both"):
        _search(COLLECTION_TATA, "Tata Motors Policy")

    log.info("Chat retriever: total %d chunks retrieved.", len(retrieved))
    return {"retrieved_docs": retrieved}
