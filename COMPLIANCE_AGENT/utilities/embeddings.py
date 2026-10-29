"""
utilities/embeddings.py
=======================
Self-contained ChromaDB and sentence-transformer embedding helpers for COMPLIANCE_AGENT.
Independent of external scripts.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

log = logging.getLogger(__name__)

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv()

# Collections
COLLECTION_SEBI = "sebi_circulars"
COLLECTION_TATA = "tata_motors_policies"
DEFAULT_COLLECTION = "regulatory_impact_monitor"
DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"


def resolve_chroma_dir() -> Path:
    """Find Chroma persist directory, checking env var then Data/chroma_db then chroma_db."""
    env_dir = os.getenv("CHROMA_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)
    data_dir = _PROJECT_ROOT / "Data" / "chroma_db"
    if data_dir.exists():
        return data_dir
    local_dir = _PROJECT_ROOT / "chroma_db"
    return local_dir


DEFAULT_CHROMA_DIR = resolve_chroma_dir()

# Cached embedding model singleton
_EMBEDDINGS: HuggingFaceEmbeddings | None = None


def build_embedding_model(
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
) -> HuggingFaceEmbeddings:
    """Load and cache locally a HuggingFace sentence-transformer model."""
    global _EMBEDDINGS
    if _EMBEDDINGS is not None:
        return _EMBEDDINGS

    log.info("Loading embedding model (%s, %s)...", model_name, device)
    _EMBEDDINGS = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    return _EMBEDDINGS


def get_or_create_vectorstore(
    embeddings: HuggingFaceEmbeddings,
    persist_dir: Path | str | None = None,
    collection_name: str = COLLECTION_TATA,
    reset: bool = False,
) -> Chroma:
    """Open or create a persistent ChromaDB vector store."""
    target_dir = Path(persist_dir) if persist_dir else resolve_chroma_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    if reset:
        import chromadb
        client = chromadb.PersistentClient(path=str(target_dir))
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(target_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_retriever(
    vectorstore: Chroma,
    search_type: str = "similarity",
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.6,
    filter_meta: dict | None = None,
):
    """Build a LangChain retriever from a Chroma vectorstore."""
    search_kwargs: dict = {"k": k}
    if search_type == "mmr":
        search_kwargs["fetch_k"] = fetch_k
        search_kwargs["lambda_mult"] = lambda_mult
    if filter_meta:
        search_kwargs["filter"] = filter_meta

    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
