"""
RAG SYSTEM — embedding/vector_store.py
========================================
Builds and loads the FAISS vector index using
OpenAI's text-embedding-3-large model.

Why text-embedding-3-large for multilingual?
  - OpenAI's best embedding model (3072 dimensions)
  - Natively multilingual — trained on 100+ languages
  - German + English work out of the box, no special config
  - Significantly better than text-embedding-ada-002
  - text-embedding-3-small is cheaper but lower quality

Cost estimate:
  ~100 pages × ~10 chunks × 400 chars ≈ 1000 chunks
  1000 chunks × 400 chars ÷ 4 chars/token ≈ 100k tokens
  text-embedding-3-large: $0.00013 / 1k tokens → ~$0.013 total

Index is saved to disk — OpenAI is only called when BUILDING
the index. Loading and searching are completely free/local.
"""

import os
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ── Config ────────────────────────────────────────────────────

FAISS_INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", "data/faiss_index"))

# text-embedding-3-large: best quality, multilingual, 3072 dims
# text-embedding-3-small: cheaper, still multilingual, 1536 dims
OPENAI_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# Batch size — OpenAI allows up to 2048 inputs per request
# Keep lower to avoid rate limits on large indexes
EMBED_BATCH_SIZE = 100

# ── In-memory caches ──────────────────────────────────────────

_embedding_model:    OpenAIEmbeddings | None = None
_vector_store_cache: FAISS | None            = None


# ── Embedding model (lazy singleton) ─────────────────────────

def get_embedding_model() -> OpenAIEmbeddings:
    """
    Load OpenAI embedding model once and cache in memory.
    No download needed — API calls happen at embed time only.
    """
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set.\n"
            "Add OPENAI_API_KEY=sk-... to your .env file."
        )

    print(f"\n📦 OpenAI embedding model : {OPENAI_EMBEDDING_MODEL}")
    print(f"   Languages             : multilingual (100+ langs, German + English ✓)")
    print(f"   Dimensions            : 3072")
    print(f"   Note                  : API calls only during index BUILD, not retrieval\n")

    _embedding_model = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        openai_api_key=api_key,
        chunk_size=EMBED_BATCH_SIZE,   # batch size per API request
    )

    return _embedding_model


# ── Build ─────────────────────────────────────────────────────

def build_vector_store(
    chunks: list[Document],
    force_rebuild: bool = False,
) -> FAISS:
    """
    Embed all chunks via OpenAI and build a FAISS index.
    Saves the index to disk — subsequent loads are free (no API calls).

    Args:
        chunks        — from ingest/chunker.py
        force_rebuild — delete existing index and rebuild from scratch
    """
    global _vector_store_cache

    if force_rebuild and FAISS_INDEX_DIR.exists():
        print(f"\n🗑️  Force rebuild — deleting {FAISS_INDEX_DIR}")
        shutil.rmtree(FAISS_INDEX_DIR)
        _vector_store_cache = None

    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    embedder = get_embedding_model()

    web_count = sum(1 for c in chunks if c.metadata.get("source_type") == "web")
    pdf_count = sum(1 for c in chunks if c.metadata.get("source_type") == "pdf")

    print(f"🔨 Building FAISS index")
    print(f"   Chunks   : {len(chunks)} total ({web_count} web + {pdf_count} pdf)")
    print(f"   Model    : {OPENAI_EMBEDDING_MODEL}")
    print(f"   Batch    : {EMBED_BATCH_SIZE} chunks per API call")
    print(f"   ⚠️  OpenAI API will be called — check your usage dashboard\n")

    t0 = time.time()

    # FAISS.from_documents handles batching via chunk_size set on embedder
    _vector_store_cache = FAISS.from_documents(chunks, embedder)

    elapsed = time.time() - t0

    _vector_store_cache.save_local(str(FAISS_INDEX_DIR))

    print(f"\n✅ Index built in {elapsed:.1f}s")
    print(f"   Vectors  : {_vector_store_cache.index.ntotal}")
    print(f"   Saved to : {FAISS_INDEX_DIR}\n")

    return _vector_store_cache


# ── Load ──────────────────────────────────────────────────────

def load_vector_store() -> FAISS:
    """
    Load existing FAISS index from disk. No API calls.
    Uses in-memory cache — safe to call on every request.
    """
    global _vector_store_cache

    if _vector_store_cache is not None:
        return _vector_store_cache

    index_file = FAISS_INDEX_DIR / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_DIR}\n"
            f"Build it first: python embedding/vector_store.py --build"
        )

    print(f"\n📂 Loading FAISS index from {FAISS_INDEX_DIR} ...")
    embedder = get_embedding_model()

    t0 = time.time()
    _vector_store_cache = FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embedder,
        allow_dangerous_deserialization=True,  # safe — our own index
    )
    elapsed = time.time() - t0

    print(f"   ✅ {_vector_store_cache.index.ntotal} vectors loaded in {elapsed:.1f}s\n")
    return _vector_store_cache


# ── Convenience: load or build ────────────────────────────────

def get_or_build_vector_store(
    chunks: list[Document] | None = None,
    force_rebuild: bool = False,
) -> FAISS:
    """
    Load existing index or build a new one.
    Called by api/app.py on startup.
    """
    global _vector_store_cache

    # Return cached if available and not forcing rebuild
    if _vector_store_cache is not None and not force_rebuild:
        print("\n📂 Using cached in-memory FAISS index")
        return _vector_store_cache

    if force_rebuild:
        if chunks is None:
            raise ValueError("chunks required when force_rebuild=True")
        return build_vector_store(chunks, force_rebuild=True)

    try:
        return load_vector_store()
    except FileNotFoundError:
        if chunks is None:
            raise ValueError(
                "No FAISS index found and no chunks provided.\n"
                "Run: python embedding/vector_store.py --build"
            )
        return build_vector_store(chunks)


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from ingest.chunker import load_documents_from_disk, chunk_documents

    args  = sys.argv[1:]
    force = "--build" in args or "--rebuild" in args

    # Always load + chunk before building or testing
    docs   = load_documents_from_disk()
    chunks = chunk_documents(docs)
    vs     = get_or_build_vector_store(chunks, force_rebuild=force)

    print(f"✅ FAISS index ready: {vs.index.ntotal} vectors  |  model: {OPENAI_EMBEDDING_MODEL}")

    # Test searches
    if "--test" in args:
        idx   = args.index("--test")
        query = args[idx + 1] if idx + 1 < len(args) else "book appointment"
        queries = [query]
    else:
        # Default: test both English and German
        queries = [
            "How can I book an appointment?",
            "Wie kann ich einen Termin buchen?",
            "Physiotherapie Öffnungszeiten",
        ]

    for q in queries:
        results = vs.similarity_search(q, k=3)
        print(f"\n🔍 '{q}'")
        for i, doc in enumerate(results, 1):
            stype = doc.metadata.get("source_type", "?")
            name  = doc.metadata.get("page_name", "?")
            print(f"   {i}. [{stype}] {name}")
            print(f"      {doc.page_content[:120].replace(chr(10), ' ')}...")