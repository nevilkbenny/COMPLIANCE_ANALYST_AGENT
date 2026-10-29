"""
nodes/retriever.py
==================
Retrieval node – no LLM involved.

For each extracted obligation, queries the Tata Motors policy ChromaDB
collection using the MMR retriever from embed_and_store.py.  Results are
merged, deduplicated, and stored in matched_policies.

ChromaDB collection used: tata_motors_policies
  (built by: python embed_and_store.py --source tata)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── Ensure project root is importable ───────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from COMPLIANCE_AGENT.utilities.embeddings import (       # noqa: E402
    COLLECTION_TATA,
    DEFAULT_CHROMA_DIR,
    build_embedding_model,
    build_retriever,
    get_or_create_vectorstore,
)

from COMPLIANCE_AGENT.src.state import SEBIMonitorState  # noqa: E402

log = logging.getLogger(__name__)

# Number of policy chunks to retrieve per obligation
_K = int(5)
# MMR candidate pool size
_FETCH_K = int(20)


def _build_tata_retriever():
    """Initialise the embedding model and return an MMR retriever over Tata policies."""
    load_dotenv()
    embeddings  = build_embedding_model()
    vectorstore = get_or_create_vectorstore(
        embeddings=embeddings,
        persist_dir=DEFAULT_CHROMA_DIR,
        collection_name=COLLECTION_TATA,
        reset=False,
    )
    return build_retriever(
        vectorstore=vectorstore,
        search_type="mmr",
        k=_K,
        fetch_k=_FETCH_K,
        lambda_mult=0.6,
        filter_meta={"source_type": "corporate_policy"},
    )


def retriever_node(state: SEBIMonitorState) -> dict:
    """
    LangGraph node: Retriever  (no LLM)

    Runs each obligation through the Tata Motors policy vector store and
    collects the top-k matched policy chunks.  Results are stored as a list
    of plain dicts for JSON-serialisable state.
    """
    if state.get("error"):
        log.warning("Skipping retriever — upstream error.")
        return {}

    if state.get("already_processed"):
        log.info("Circular already processed — skipping retriever.")
        return {}

    obligations = state.get("obligations", [])
    if not obligations:
        log.warning("No obligations in state — cannot retrieve policy matches.")
        return {"matched_policies": []}

    log.info(
        "Retrieving Tata Motors policy matches for %d obligations …", len(obligations)
    )

    try:
        retriever = _build_tata_retriever()
    except Exception as exc:
        log.error(
            "Could not initialise Tata Motors retriever: %s\n"
            "Have you run 'python embed_and_store.py --source tata' yet?",
            exc,
        )
        return {"matched_policies": []}

    seen_ids: set[str] = set()
    matched_policies: list[dict] = []

    for obligation in obligations:
        try:
            docs = retriever.invoke(obligation)
            for doc in docs:
                # Build a stable dedup key from file + chunk index
                dedup_key = (
                    doc.metadata.get("file_name", "")
                    + str(doc.metadata.get("chunk_index", ""))
                )
                if dedup_key in seen_ids:
                    continue
                seen_ids.add(dedup_key)

                matched_policies.append({
                    "obligation":   obligation,
                    "policy_doc":   doc.metadata.get("file_name", "unknown"),
                    "policy_chunk": doc.page_content.strip(),
                    "metadata":     {
                        k: v for k, v in doc.metadata.items()
                        if k in ("file_name", "page", "source_type", "chunk_index")
                    },
                })
        except Exception as exc:
            log.error("Retrieval failed for obligation '%s': %s", obligation[:60], exc)

    log.info(
        "Retrieval complete: %d unique policy chunks matched across %d obligations.",
        len(matched_policies),
        len(obligations),
    )
    return {"matched_policies": matched_policies}
