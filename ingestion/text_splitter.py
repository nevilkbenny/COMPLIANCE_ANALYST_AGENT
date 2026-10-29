"""
text_splitter.py
================
Document loading and text splitting pipeline for the Regulatory Impact Monitor.

Loads SEBI circulars and Tata Motors policy PDFs using LangChain document loaders,
then splits them into overlapping chunks suitable for embedding and vector-store
ingestion in a LangGraph RAG workflow.

Supported splitters
-------------------
1. RecursiveCharacterTextSplitter  – general-purpose, hierarchy-aware splitting
2. TokenTextSplitter               – token-budget-aware splitting (for LLM context limits)
3. SemanticChunker                 – embedding-driven, semantically coherent chunks
                                    (requires sentence-transformers / OpenAI embeddings)

Usage
-----
    python text_splitter.py                        # uses defaults
    python text_splitter.py --splitter recursive   # choose splitter type
    python text_splitter.py --chunk_size 800 --chunk_overlap 100
    python text_splitter.py --source sebi          # only SEBI circulars
    python text_splitter.py --source tata          # only Tata Motors policies
    python text_splitter.py --output chunks.json   # persist chunks to JSON

Install
-------
    pip install langchain langchain-community langchain-experimental \
                pypdf sentence-transformers tiktoken
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# LangChain imports
# ---------------------------------------------------------------------------
from langchain_community.document_loaders import PyPDFLoader,UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

# SemanticChunker lives in langchain_experimental – import lazily so the rest
# of the module still works even if the package is not installed.
try:
    from langchain_experimental.text_splitter import SemanticChunker

    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False

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
# Default paths  (relative to this file's directory)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data" if (BASE_DIR / "Data").exists() else BASE_DIR

SEBI_DIR = (DATA_DIR / "SEBI CIRCULARS") if (DATA_DIR / "SEBI CIRCULARS").exists() else (BASE_DIR / "SEBI CIRCULARS")
TATA_DIR = (DATA_DIR / "TATA MOTORS POLICIES") if (DATA_DIR / "TATA MOTORS POLICIES").exists() else (BASE_DIR / "TATA MOTORS POLICIES")
SAMPLE   = (DATA_DIR / "sample") if (DATA_DIR / "sample").exists() else (BASE_DIR / "SAMPLE")

# ---------------------------------------------------------------------------
# Source metadata tags – propagated into each chunk's metadata
# ---------------------------------------------------------------------------
SOURCE_META = {
    "sebi": {
        "source_type": "regulatory_circular",
        "regulator": "SEBI",
        "jurisdiction": "India",
    },
    "tata": {
        "source_type": "corporate_policy",
        "company": "Tata Motors Limited",
        "jurisdiction": "India",
    },
}


# ---------------------------------------------------------------------------
# Step 1 – Document Loading
# ---------------------------------------------------------------------------

def load_pdfs_from_directory(
    directory: Path,
    source_key: str,
    *,
    page_metadata: bool = True,
) -> list[Document]:
    """
    Recursively load all PDF files from *directory* using PyPDFLoader.

    Each returned Document carries rich metadata:
      - file_name      : basename of the PDF
      - file_path      : absolute path
      - page           : 0-indexed page number (added by PyPDFLoader)
      - source_type    : 'regulatory_circular' | 'corporate_policy'
      - regulator / company / jurisdiction : domain tags

    Parameters
    ----------
    directory   : Path to the folder containing PDFs.
    source_key  : 'sebi' or 'tata' – selects the metadata tag set.
    page_metadata : If True, keep per-page Document objects (default).
                    If False, concatenate pages into a single Document per file.
    """
    if not directory.exists():
        log.warning("Directory does not exist, skipping: %s", directory)
        return []

    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        log.warning("No PDF files found in: %s", directory)
        return []

    log.info("Loading %d PDFs from '%s' …", len(pdf_files), directory.name)
    all_docs: list[Document] = []
    extra_meta = SOURCE_META.get(source_key, {})

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            #loader = UnstructuredPDFLoader(str(pdf_path),mode="elements", strategy="hi_res")
            pages: list[Document] = loader.load()

            for doc in pages:
                # Enrich metadata
                doc.metadata.update(
                    {
                        "file_name": pdf_path.name,
                        "file_path": str(pdf_path.resolve()),
                        **extra_meta,
                    }
                )

            if page_metadata:
                all_docs.extend(pages)
            else:
                # Merge all pages into a single Document per file
                combined_text = "\n\n".join(d.page_content for d in pages)
                merged = Document(
                    page_content=combined_text,
                    metadata={
                        "file_name": pdf_path.name,
                        "file_path": str(pdf_path.resolve()),
                        "total_pages": len(pages),
                        **extra_meta,
                    },
                )
                all_docs.append(merged)

            log.info("  ✓  %-72s  (%d pages)", pdf_path.name[:72], len(pages))

        except Exception as exc:  # noqa: BLE001
            log.error("  ✗  Failed to load '%s': %s", pdf_path.name, exc)

    log.info("Loaded %d document-page objects from %s\n", len(all_docs), source_key.upper())
    print(all_docs)
    return all_docs


# ---------------------------------------------------------------------------
# Step 2 – Text Splitting
# ---------------------------------------------------------------------------

def get_recursive_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> RecursiveCharacterTextSplitter:
    """
    RecursiveCharacterTextSplitter – tries to split on paragraphs → newlines →
    sentences → words → characters, preserving as much semantic context as
    possible at each level.

    Best for: general regulatory text, policy documents, mixed layouts.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        add_start_index=True,  # adds 'start_index' to each chunk's metadata
    )


def get_token_splitter(
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    encoding_name: str = "cl100k_base",  # GPT-4 / text-embedding-3 tokeniser
) -> TokenTextSplitter:
    """
    TokenTextSplitter – splits on token boundaries instead of character count.
    Guarantees chunks never exceed the model's context window.

    Best for: feeding directly into OpenAI embeddings or LLM calls.
    Requires: pip install tiktoken
    """
    return TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding_name,
    )


def get_semantic_chunker(embeddings=None) -> "SemanticChunker":
    """
    SemanticChunker – uses embedding cosine-similarity to detect topic shifts
    and places boundaries only when the content truly changes subject.

    Best for: long, heterogeneous documents (e.g. Tata Motors policy PDFs).
    Requires: pip install langchain-experimental sentence-transformers
    """
    if not _SEMANTIC_AVAILABLE:
        raise ImportError(
            "SemanticChunker requires 'langchain-experimental'.\n"
            "Install with: pip install langchain-experimental"
        )

    if embeddings is None:
        # Default: free, local HuggingFace sentence-transformers model
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )
            log.info("Using HuggingFace 'all-MiniLM-L6-v2' for semantic chunking.")
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",  # 'percentile' | 'standard_deviation' | 'interquartile'
        breakpoint_threshold_amount=95,
    )


def split_documents(
    docs: list[Document],
    splitter_type: Literal["recursive", "token", "semantic"] = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """
    Split a list of LangChain Documents into smaller chunks.

    Parameters
    ----------
    docs          : Raw documents returned by load_pdfs_from_directory().
    splitter_type : Which splitting strategy to use.
    chunk_size    : Max characters (recursive/semantic) or tokens (token).
    chunk_overlap : Overlap between consecutive chunks to preserve context.

    Returns
    -------
    List of chunk Documents, each with all original metadata plus
    'chunk_index' and (for recursive) 'start_index'.
    """
    if not docs:
        log.warning("No documents to split.")
        return []

    log.info(
        "Splitting %d document objects using '%s' splitter …",
        len(docs),
        splitter_type,
    )
    t0 = time.perf_counter()

    if splitter_type == "recursive":
        splitter = get_recursive_splitter(chunk_size, chunk_overlap)
        chunks = splitter.split_documents(docs)

    elif splitter_type == "token":
        splitter = get_token_splitter(chunk_size, chunk_overlap)
        chunks = splitter.split_documents(docs)

    elif splitter_type == "semantic":
        splitter = get_semantic_chunker()
        chunks = splitter.split_documents(docs)

    else:
        raise ValueError(f"Unknown splitter_type: '{splitter_type}'")

    # Tag each chunk with its index (useful for vector store deduplication)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    elapsed = time.perf_counter() - t0
    log.info(
        "Splitting complete: %d chunks produced in %.2fs\n",
        len(chunks),
        elapsed,
    )
    return chunks


# ---------------------------------------------------------------------------
# Step 3 – Inspection / Statistics
# ---------------------------------------------------------------------------

def print_chunk_stats(chunks: list[Document], label: str = "") -> None:
    """Print a summary of chunk sizes for quick sanity-checking."""
    if not chunks:
        print("No chunks to display.")
        return

    lengths = [len(c.page_content) for c in chunks]
    total_chars = sum(lengths)
    avg_len = total_chars / len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)

    header = f" {label} " if label else " Chunk Stats "
    print(f"\n{'─' * 60}")
    print(f"{header:^60}")
    print(f"{'─' * 60}")
    print(f"  Total chunks   : {len(chunks):>8,}")
    print(f"  Total chars    : {total_chars:>8,}")
    print(f"  Avg chunk size : {avg_len:>8.0f} chars")
    print(f"  Min chunk size : {min_len:>8,} chars")
    print(f"  Max chunk size : {max_len:>8,} chars")
    print(f"{'─' * 60}\n")

    print("  Sample chunk #0:")
    print(f"  Metadata : {chunks[0].metadata}")
    print(f"  Content  : {chunks[0].page_content[:300]!r} …")
    print()

    if len(chunks) > 1:
        print("  Sample chunk #-1 (last):")
        print(f"  Metadata : {chunks[-1].metadata}")
        print(f"  Content  : {chunks[-1].page_content[:300]!r} …")
        print()


# ---------------------------------------------------------------------------
# Step 4 – Persistence (optional)
# ---------------------------------------------------------------------------

def save_chunks_to_json(chunks: list[Document], output_path: Path) -> None:
    """
    Persist all chunks to a JSON file for inspection or offline processing.

    Schema per entry:
      {
        "chunk_index": <int>,
        "metadata"  : { … },
        "content"   : "<text>"
      }
    """
    records = [
        {
            "chunk_index": chunk.metadata.get("chunk_index", i),
            "metadata": chunk.metadata,
            "content": chunk.page_content,
        }
        for i, chunk in enumerate(chunks)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Saved %d chunks → %s", len(records), output_path)


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    # Read config from environment variables
    input_path_str = os.getenv("INPUT_PATH", str(SAMPLE))
    output_path_str = os.getenv("OUTPUT_PATH", "chunks.json")
    splitter_type = os.getenv("SPLITTER_TYPE", "recursive")
    
    # Validation
    if splitter_type not in ["recursive", "token", "semantic"]:
        log.warning("Unknown splitter type '%s', falling back to 'recursive'.", splitter_type)
        splitter_type = "recursive"
        
    input_dir = Path(input_path_str)
    output_path = Path(output_path_str) if output_path_str else None

    # ── 1. Load ──────────────────────────────────────────────────────────────
    raw_docs: list[Document] = []
    
    raw_docs.extend(
        load_pdfs_from_directory(input_dir, "sample", page_metadata=True)
    )

    log.info("Total raw document objects loaded: %d", len(raw_docs))

    # ── 2. Split ─────────────────────────────────────────────────────────────
    # Optional env vars for advanced tweaking
    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 150))

    chunks = split_documents(
        raw_docs,
        # We know splitter_type is one of the valid literals because of validation
        splitter_type=splitter_type, # type: ignore
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # ── 3. Inspect ───────────────────────────────────────────────────────────
    print_chunk_stats(chunks, label="All Chunks")

    # ── 4. Persist (optional) ────────────────────────────────────────────────
    if output_path:
        save_chunks_to_json(chunks, output_path)

    log.info("Pipeline complete. %d total chunks ready for embedding.", len(chunks))


if __name__ == "__main__":
    main()
