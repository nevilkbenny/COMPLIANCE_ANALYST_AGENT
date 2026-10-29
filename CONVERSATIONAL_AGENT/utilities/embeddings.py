"""
utilities/embeddings.py
=======================
Self-contained ChromaDB and sentence-transformer embedding helpers for CONVERSATIONAL_AGENT.
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
_VECTORSTORES: dict[str, Chroma] = {}


def build_embedding_model(
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
) -> HuggingFaceEmbeddings:
    """Load and cache locally a HuggingFace sentence-transformer model."""
    global _EMBEDDINGS
    if _EMBEDDINGS is not None:
        return _EMBEDDINGS

    log.info("Chat pipeline: Initializing embedding model singleton (%s, %s)...", model_name, device)
    _EMBEDDINGS = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    return _EMBEDDINGS


def get_vectorstore(
    collection_name: str,
    persist_dir: Path | str | None = None,
) -> Chroma:
    """Return cached Chroma vectorstore instance."""
    global _VECTORSTORES
    target_dir = Path(persist_dir) if persist_dir else resolve_chroma_dir()
    cache_key = f"{target_dir}::{collection_name}"
    if cache_key not in _VECTORSTORES:
        emb = build_embedding_model(device="cpu")
        _VECTORSTORES[cache_key] = Chroma(
            collection_name=collection_name,
            embedding_function=emb,
            persist_directory=str(target_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _VECTORSTORES[cache_key]
