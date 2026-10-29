"""
embed_and_store.py
==================
Embedding and vector-store ingestion pipeline for the Regulatory Impact Monitor.

Takes the LangChain Document chunks produced by text_splitter.py, embeds them
with a local HuggingFace sentence-transformer model, and persists the vectors
to a ChromaDB collection on disk.

Embedding Model  : sentence-transformers/all-mpnet-base-v2  (768-dim, free, local)
Vector Database  : ChromaDB  (embedded, persistent, metadata-filterable)

Why these choices?
------------------
Embedding model:
  - `all-mpnet-base-v2` scores higher than `all-MiniLM-L6-v2` on the MTEB
    semantic-similarity benchmark while still running efficiently on CPU.
  - No API key or cost – consistent with the local-first philosophy of the
    rest of this project.
  - 768-dim vectors give a good balance between retrieval quality and storage.

Vector DB:
  - ChromaDB runs as an embedded process (zero infrastructure).
  - Data is persisted to disk and survives interpreter restarts.
  - First-class LangChain integration via `langchain-chroma`.
  - Supports server-side metadata filtering on fields like `source_type`,
    `regulator`, and `company` that are already present in your chunk metadata.

Usage
-----
    python embed_and_store.py                          # full pipeline (default dirs)
    python embed_and_store.py --source sebi            # only SEBI circulars
    python embed_and_store.py --source tata            # only Tata Motors policies
    python embed_and_store.py --splitter token         # use token splitter
    python embed_and_store.py --collection my_col      # custom Chroma collection name
    python embed_and_store.py --reset                  # wipe & rebuild the collection
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# LangChain + ChromaDB imports
# ---------------------------------------------------------------------------
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Reuse the pipeline utilities from text_splitter.py
# ---------------------------------------------------------------------------
from text_splitter import (
    SEBI_DIR,
    TATA_DIR,
    SAMPLE,
    load_pdfs_from_directory,
    split_documents,
    print_chunk_stats,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BASE_DIR             = Path(__file__).resolve().parent.parent
DEFAULT_CHROMA_DIR   = (BASE_DIR / "Data" / "chroma_db") if (BASE_DIR / "Data" / "chroma_db").exists() else (BASE_DIR / "chroma_db")
COLLECTION_SEBI      = "sebi_circulars"                    # SEBI regulatory circulars
COLLECTION_TATA      = "tata_motors_policies"              # Tata Motors policy docs
DEFAULT_COLLECTION   = "regulatory_impact_monitor"         # fallback single collection
DEFAULT_MODEL        = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE           = 128   # how many chunks to upsert per Chroma call

# all-mpnet-base-v2 uses a BERT WordPiece tokenizer.
# Hard limit: 512 content tokens + [CLS] + [SEP] = 514 positions total.
# Anything beyond token 512 is SILENTLY TRUNCATED by sentence-transformers.
MODEL_MAX_TOKENS     = 512   # exact token limit – NOT chars


# ---------------------------------------------------------------------------
# Step 1 – Build the embedding model
# ---------------------------------------------------------------------------

def build_embedding_model(
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
) -> HuggingFaceEmbeddings:
    """
    Load (and cache locally) a HuggingFace sentence-transformer model.

    Parameters
    ----------
    model_name : HuggingFace model identifier.
                 Default: 'sentence-transformers/all-mpnet-base-v2'
                   - 768-dim embeddings
                   - Highest quality among free local sentence-transformer models
                   - Consistently outperforms all-MiniLM-L6-v2 on MTEB benchmarks
    device     : 'cpu' or 'cuda' – use 'cuda' if a GPU is available.

    Returns
    -------
    HuggingFaceEmbeddings instance ready to embed texts.
    """
    log.info("Loading embedding model: %s  (device=%s) …", model_name, device)
    t0 = time.perf_counter()

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,   # cosine similarity ≡ dot-product
            "batch_size": 32,               # tune based on available RAM
        },
    )

    log.info("Embedding model loaded in %.2fs", time.perf_counter() - t0)
    return embeddings


# ---------------------------------------------------------------------------
# Step 2 – Get or create a persistent ChromaDB collection
# ---------------------------------------------------------------------------

def get_or_create_vectorstore(
    embeddings: HuggingFaceEmbeddings,
    persist_dir: Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    reset: bool = False,
) -> Chroma:
    """
    Open (or create) a persistent ChromaDB vector store.

    Why ChromaDB?
    -------------
    - Embedded process, zero server setup.
    - Persists data to disk automatically.
    - Native LangChain integration for add / similarity_search / MMR.
    - Supports `filter` dicts on metadata (e.g. source_type, regulator).

    Parameters
    ----------
    embeddings      : Embedding model used to encode queries at retrieval time.
    persist_dir     : Directory where ChromaDB will write its SQLite + binary data.
    collection_name : Logical namespace inside ChromaDB.
    reset           : If True, delete and recreate the collection (full rebuild).

    Returns
    -------
    Chroma vectorstore object.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)

    if reset:
        log.warning("--reset flag set: wiping existing ChromaDB collection '%s'.", collection_name)
        import chromadb
        client = chromadb.PersistentClient(path=str(persist_dir))
        try:
            client.delete_collection(collection_name)
            log.info("Collection '%s' deleted.", collection_name)
        except Exception:
            log.info("Collection '%s' did not exist yet – nothing to delete.", collection_name)

    log.info(
        "Opening ChromaDB  |  dir='%s'  |  collection='%s'",
        persist_dir,
        collection_name,
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_metadata={
            "hnsw:space": "cosine",     # use cosine similarity (embeddings are L2-normalised)
        },
    )
    return vectorstore


# ---------------------------------------------------------------------------
# Step 2b – Token-length guard
# ---------------------------------------------------------------------------

def validate_chunk_token_lengths(
    chunks: list[Document],
    model_name: str = DEFAULT_MODEL,
    max_tokens: int = MODEL_MAX_TOKENS,
) -> None:
    """
    Validate every chunk against the embedding model's EXACT token limit.

    Uses the model's own AutoTokenizer (the same WordPiece tokenizer that
    sentence-transformers uses internally) so the token counts are precise –
    not a character-count heuristic.

    all-mpnet-base-v2 hard limit: 512 content tokens (+ [CLS] + [SEP]).
    Sentence-transformers SILENTLY TRUNCATES anything beyond token 512.
    No exception is raised – the tail of the chunk simply disappears, and
    the stored vector only represents the first ~512 tokens.

    Parameters
    ----------
    chunks     : LangChain Document chunks to inspect.
    model_name : HuggingFace model name whose tokenizer to load.
    max_tokens : Token limit to enforce (default: 512 for all-mpnet-base-v2).
    """
    # Load the model's own tokenizer – already available as a transitive
    # dependency of sentence-transformers (no extra install needed).
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        log.info("Token validator: loaded tokenizer for '%s'.", model_name)
    except Exception as exc:
        log.warning(
            "Could not load tokenizer for '%s' (%s). "
            "Skipping exact token validation.",
            model_name, exc,
        )
        return

    log.info("Counting tokens for %d chunks …", len(chunks))

    oversized: list[tuple[int, int, str]] = []   # (chunk_idx, token_count, file_name)
    max_seen  = 0

    for i, chunk in enumerate(chunks):
        # encode() returns the token IDs including [CLS] and [SEP]
        token_ids  = tokenizer.encode(
            chunk.page_content,
            add_special_tokens=True,   # mirrors sentence-transformers behaviour
            truncation=False,          # count ALL tokens, don't truncate here
        )
        token_count = len(token_ids)
        if token_count > max_seen:
            max_seen = token_count

        if token_count > max_tokens:
            oversized.append(
                (i, token_count, chunk.metadata.get("file_name", "?"))
            )

    if not oversized:
        log.info(
            "✅  All %d chunks are within the %d-token limit. "
            "(Longest chunk: %d tokens)",
            len(chunks),
            max_tokens,
            max_seen,
        )
        return

    log.warning(
        "⚠️  %d / %d chunks EXCEED the %d-token limit.\n"
        "    Sentence-transformers will SILENTLY TRUNCATE these chunks.\n"
        "    The stored vector will only represent the FIRST %d tokens.\n"
        "    Longest chunk seen: %d tokens.\n"
        "    Fix: use --splitter token --chunk_size 512  (token-exact)\n"
        "         or  --chunk_size 1000 with recursive splitter (~200 tokens).",
        len(oversized), len(chunks), max_tokens, max_tokens, max_seen,
    )
    for idx, tok_count, fname in oversized[:5]:
        log.warning(
            "    chunk #%d  |  %d tokens  |  %d over limit  |  file: %s",
            idx, tok_count, tok_count - max_tokens, fname,
        )
    if len(oversized) > 5:
        log.warning("    … and %d more.", len(oversized) - 5)


# ---------------------------------------------------------------------------
# Step 3 – Embed chunks and upsert into ChromaDB
# ---------------------------------------------------------------------------

def embed_and_store(
    chunks: list[Document],
    vectorstore: Chroma,
    batch_size: int = BATCH_SIZE,
) -> int:
    """
    Embed *chunks* and upsert them into *vectorstore* in batches.

    Uses deterministic IDs derived from (file_name, chunk_index) so that
    re-running the pipeline with the same documents is idempotent – existing
    vectors are updated rather than duplicated.

    Parameters
    ----------
    chunks      : LangChain Document chunks from split_documents().
    vectorstore : Persistent ChromaDB instance.
    batch_size  : Number of chunks per upsert call (tune for memory).

    Returns
    -------
    Total number of vectors stored.
    """
    if not chunks:
        log.warning("No chunks to embed – aborting.")
        return 0

    # Guard: warn before embedding if any chunk risks silent truncation
    validate_chunk_token_lengths(chunks)

    log.info("Embedding & storing %d chunks (batch_size=%d) …", len(chunks), batch_size)
    t0 = time.perf_counter()

    total_stored = 0
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]

        # Build deterministic IDs to make upserts idempotent
        ids = [
            f"{chunk.metadata.get('file_name', 'doc')}__"
            f"p{chunk.metadata.get('page', 0)}__"
            f"c{chunk.metadata.get('chunk_index', batch_start + i)}"
            for i, chunk in enumerate(batch)
        ]

        # Chroma's add_documents handles embedding internally via the
        # embedding_function provided at construction time.
        vectorstore.add_documents(documents=batch, ids=ids)
        total_stored += len(batch)

        log.info(
            "  Stored batch %d–%d  (%d/%d chunks done)",
            batch_start,
            batch_start + len(batch) - 1,
            total_stored,
            len(chunks),
        )

    elapsed = time.perf_counter() - t0
    log.info(
        "✅  Embedding complete: %d vectors stored in %.2fs  (%.1f chunks/sec)",
        total_stored,
        elapsed,
        total_stored / elapsed if elapsed > 0 else 0,
    )
    return total_stored


# ---------------------------------------------------------------------------
# Step 4 – Quick sanity-check: similarity search
# ---------------------------------------------------------------------------

def sanity_check(
    vectorstore: Chroma,
    query: str = "What are the compliance requirements for listed companies?",
    k: int = 3,
    filter_meta: dict | None = None,
) -> None:
    """
    Run a sample similarity search against the vector store and print results.

    Parameters
    ----------
    vectorstore  : Populated ChromaDB instance.
    query        : Test question to embed and search.
    k            : Number of top results to return.
    filter_meta  : Optional metadata filter dict, e.g. {"source_type": "regulatory_circular"}.
                   Useful to narrow retrieval to SEBI vs Tata Motor documents.
    """
    log.info("\n%s\n  Sanity-check query: %r\n%s", "─" * 60, query, "─" * 60)

    kwargs: dict = {"k": k}
    if filter_meta:
        kwargs["filter"] = filter_meta

    results: list[tuple[Document, float]] = vectorstore.similarity_search_with_score(
        query, **kwargs
    )

    if not results:
        log.warning("No results returned – the collection may be empty.")
        return

    for rank, (doc, score) in enumerate(results, start=1):
        print(f"\n  [{rank}] Score: {score:.4f}")
        print(f"       File  : {doc.metadata.get('file_name', 'N/A')}")
        print(f"       Page  : {doc.metadata.get('page', 'N/A')}")
        print(f"       Type  : {doc.metadata.get('source_type', 'N/A')}")
        preview = doc.page_content[:300].replace("\n", " ")
        print(f"       Text  : {preview!r} …")

    print()


# ---------------------------------------------------------------------------
# Step 5 – Retriever factory (for use in LangGraph / RAG chains)
# ---------------------------------------------------------------------------

def build_retriever(
    vectorstore: Chroma,
    search_type: str = "mmr",           # 'similarity' | 'mmr'
    k: int = 5,
    fetch_k: int = 20,                  # MMR candidate pool size
    lambda_mult: float = 0.6,           # MMR diversity weight (0=diverse, 1=relevant)
    filter_meta: dict | None = None,
):
    """
    Wrap the ChromaDB vectorstore as a LangChain retriever.

    Uses Maximal Marginal Relevance (MMR) by default to avoid returning
    near-duplicate chunks from the same document section.

    Parameters
    ----------
    vectorstore  : Populated ChromaDB instance.
    search_type  : 'similarity' for pure cosine search, 'mmr' for diversity-aware.
    k            : Number of chunks to return per query.
    fetch_k      : (MMR only) Candidate pool size before MMR re-ranking.
    lambda_mult  : (MMR only) Trade-off between relevance and diversity.
    filter_meta  : Optional server-side metadata filter, e.g.:
                     {"source_type": "regulatory_circular"}
                     {"regulator": "SEBI"}
                     {"company": "Tata Motors Limited"}

    Returns
    -------
    LangChain BaseRetriever, ready to plug into RetrievalQA or LangGraph nodes.
    """
    search_kwargs: dict = {"k": k}

    if search_type == "mmr":
        search_kwargs["fetch_k"] = fetch_k
        search_kwargs["lambda_mult"] = lambda_mult

    if filter_meta:
        search_kwargs["filter"] = filter_meta

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    log.info(
        "Retriever built  |  search=%s  |  k=%d  |  filter=%s",
        search_type,
        k,
        filter_meta,
    )
    return retriever


# ---------------------------------------------------------------------------
# Utility – collection statistics
# ---------------------------------------------------------------------------

def print_collection_stats(vectorstore: Chroma) -> None:
    """Print the number of vectors currently stored in the collection."""
    count = vectorstore._collection.count()   # internal Chroma client call
    log.info("ChromaDB collection contains %d vectors.", count)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed text chunks and store them in a ChromaDB vector store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["sebi", "tata", "sample", "both"],
        default="sample",
        help="Which document source to ingest.",
    )
    parser.add_argument(
        "--splitter",
        choices=["recursive", "token", "semantic"],
        default="recursive",
        help="Text splitting strategy.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", 1000)),
        help="Max chunk size (chars for recursive/semantic, tokens for token).",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP", 150)),
        help="Overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL),
        help="HuggingFace embedding model name.",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("EMBEDDING_DEVICE", "cpu"),
        choices=["cpu", "cuda"],
        help="Device to run the embedding model on.",
    )
    parser.add_argument(
        "--chroma_dir",
        type=Path,
        default=Path(os.getenv("CHROMA_DIR", str(DEFAULT_CHROMA_DIR))),
        help="Directory to persist ChromaDB data (shared by both collections).",
    )
    parser.add_argument(
        "--sebi_collection",
        default=os.getenv("CHROMA_SEBI_COLLECTION", COLLECTION_SEBI),
        help="ChromaDB collection name for SEBI circulars.",
    )
    parser.add_argument(
        "--tata_collection",
        default=os.getenv("CHROMA_TATA_COLLECTION", COLLECTION_TATA),
        help="ChromaDB collection name for Tata Motors policies.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the ChromaDB collections before ingestion.",
    )
    parser.add_argument(
        "--no_sanity_check",
        action="store_true",
        help="Skip the post-ingestion similarity search sanity check.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-source ingestion helper
# ---------------------------------------------------------------------------

def ingest_source(
    source_dir: Path,
    source_key: str,
    collection_name: str,
    embeddings,
    args,
    sanity_query: str,
) -> int:
    """
    Full load → split → validate → embed → store pipeline for ONE source.

    Parameters
    ----------
    source_dir      : Directory containing the PDFs for this source.
    source_key      : 'sebi' or 'tata' – selects metadata tags.
    collection_name : ChromaDB collection to write into.
    embeddings      : Pre-loaded HuggingFaceEmbeddings (shared across sources).
    args            : Parsed CLI args (splitter, chunk_size, etc.).
    sanity_query    : Domain-relevant sample query for the post-ingest check.

    Returns
    -------
    Number of vectors stored (0 if no documents found).
    """
    log.info(
        "\n%s\n  Ingesting: %-20s  →  collection: '%s'\n%s",
        "=" * 65, source_key.upper(), collection_name, "=" * 65,
    )

    # 1. Load
    raw_docs = load_pdfs_from_directory(source_dir, source_key)
    if not raw_docs:
        log.warning("No documents found in '%s' – skipping collection '%s'.",
                    source_dir, collection_name)
        return 0

    # 2. Split
    chunks = split_documents(
        raw_docs,
        splitter_type=args.splitter,   # type: ignore[arg-type]
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print_chunk_stats(chunks, label=f"{source_key.upper()} chunks")

    # 3. Open / create the dedicated ChromaDB collection
    vs = get_or_create_vectorstore(
        embeddings=embeddings,
        persist_dir=args.chroma_dir,
        collection_name=collection_name,
        reset=args.reset,
    )

    # 4. Validate token lengths → embed → store
    total = embed_and_store(chunks, vs)
    print_collection_stats(vs)

    # 5. Sanity check
    if not args.no_sanity_check and total > 0:
        sanity_check(vs, query=sanity_query)

    log.info(
        "✅  '%s' → %d vectors stored in collection '%s'.",
        source_key.upper(), total, collection_name,
    )
    return total


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    args = parse_args()

    # ── Shared embedding model (loaded ONCE, reused for both collections) ────
    embeddings = build_embedding_model(model_name=args.model, device=args.device)

    totals: dict[str, int] = {}

    # ── Collection 1: SEBI Circulars ─────────────────────────────────────────
    if args.source in ("sebi", "both"):
        totals["sebi"] = ingest_source(
            source_dir=SEBI_DIR,
            source_key="sebi",
            collection_name=args.sebi_collection,
            embeddings=embeddings,
            args=args,
            sanity_query="What are the disclosure requirements for listed companies under SEBI?",
        )

    # ── Collection 2: Tata Motors Policies ───────────────────────────────────
    if args.source in ("tata", "both"):
        totals["tata"] = ingest_source(
            source_dir=TATA_DIR,
            source_key="tata",
            collection_name=args.tata_collection,
            embeddings=embeddings,
            args=args,
            sanity_query="What are the employee conduct and ethics policies at Tata Motors?",
        )

    # ── Fallback: sample directory (dev / testing) ───────────────────────────
    if args.source == "sample":
        totals["sample"] = ingest_source(
            source_dir=SAMPLE,
            source_key="sample",
            collection_name=DEFAULT_COLLECTION,
            embeddings=embeddings,
            args=args,
            sanity_query="What are the key compliance obligations for listed entities?",
        )

    if not totals:
        log.error("No documents ingested. Check your --source flag and PDF directories.")
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n%s", "=" * 65)
    log.info("  INGESTION COMPLETE  |  ChromaDB: '%s'", args.chroma_dir)
    log.info("  %-34s  %s", "Collection", "Vectors stored")
    log.info("  %s", "-" * 55)
    for source, count in totals.items():
        col = (args.sebi_collection if source == "sebi"
               else args.tata_collection if source == "tata"
               else DEFAULT_COLLECTION)
        log.info("  %-34s  %d", col, count)
    log.info("%s\n", "=" * 65)

    # ── Retrievers (plug into LangGraph nodes) ────────────────────────────────
    if totals.get("sebi", 0) > 0:
        sebi_vs = Chroma(
            collection_name=args.sebi_collection,
            embedding_function=embeddings,
            persist_directory=str(args.chroma_dir),
        )
        sebi_retriever = build_retriever(sebi_vs, search_type="mmr", k=5)
        log.info("SEBI retriever ready: %s", sebi_retriever)

    if totals.get("tata", 0) > 0:
        tata_vs = Chroma(
            collection_name=args.tata_collection,
            embedding_function=embeddings,
            persist_directory=str(args.chroma_dir),
        )
        tata_retriever = build_retriever(tata_vs, search_type="mmr", k=5)
        log.info("Tata Motors retriever ready: %s", tata_retriever)


if __name__ == "__main__":
    main()
