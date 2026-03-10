"""
RAG SYSTEM — STEP 4: Hybrid Retrieval
=======================================
Combines two different retrieval methods and merges their results.

Why hybrid?
  Semantic search (FAISS) is great for meaning-based queries:
    "How do I recover from surgery?" → finds physio pages
  
  Keyword search (BM25) is great for exact terms:
    "Dr. Schmidt" → finds the exact doctor's name
    "Öffnungszeiten" → finds the exact German word
  
  Together they cover each other's weaknesses.

Pipeline:
    1. FAISS semantic search  → top-K by cosine similarity
    2. BM25 keyword search    → top-K by term frequency
    3. Reciprocal Rank Fusion → merge both lists smartly
    4. Deduplicate
    5. Return top-N results

Reciprocal Rank Fusion (RRF):
    For each document, score = Σ  1 / (rank + 60)
    The constant 60 dampens the impact of very high ranks.
    Documents appearing in both result lists score higher.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever

from embedding.vector_store import get_or_build_vector_store

load_dotenv()

# ── Config ────────────────────────────────────────────────────

TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "10"))

# Number of candidates to fetch from each retriever before fusion
# More candidates = better fusion but slower
CANDIDATE_MULTIPLIER = 3

# RRF constant (60 is the standard value from the original paper)
RRF_K = 60

# ── BM25 cache ────────────────────────────────────────────────

_bm25_retriever: BM25Retriever | None = None
_bm25_docs_hash: int = 0


def get_bm25_retriever(chunks: list[Document]) -> BM25Retriever:
    """
    Build BM25 retriever from chunks.
    Cached — rebuilds only if chunks change.
    """
    global _bm25_retriever, _bm25_docs_hash

    # Simple hash to detect if chunks changed
    current_hash = hash(tuple(c.page_content[:50] for c in chunks[:10]))

    if _bm25_retriever is None or current_hash != _bm25_docs_hash:
        print("📦 Building BM25 index...")
        _bm25_retriever = BM25Retriever.from_documents(
            chunks,
            bm25_variant="plus",   # BM25+ slightly better than classic BM25
        )
        _bm25_docs_hash = current_hash
        print(f"   ✅ BM25 index ready ({len(chunks)} documents)")

    return _bm25_retriever


# ── Reciprocal Rank Fusion ────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    k: int = RRF_K,
) -> list[tuple[Document, float]]:
    """
    Merge multiple ranked lists of documents using Reciprocal Rank Fusion.

    Args:
        result_lists — list of ranked document lists (e.g. [faiss_results, bm25_results])
        k            — RRF constant (default 60)

    Returns:
        List of (Document, rrf_score) sorted by score descending.
    """
    # Use page_content hash as unique document identifier
    scores: dict[str, float]    = {}
    doc_map: dict[str, Document] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = _doc_id(doc)
            rrf_score = 1.0 / (rank + k)
            scores[doc_id]  = scores.get(doc_id, 0.0) + rrf_score
            doc_map[doc_id] = doc

    # Sort by combined RRF score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[doc_id], score) for doc_id, score in ranked]


def _doc_id(doc: Document) -> str:
    """Unique ID for deduplication — based on content hash."""
    return str(hash(doc.page_content))


# ── Main Retriever ────────────────────────────────────────────

class HybridRetriever:
    """
    Combines FAISS semantic search + BM25 keyword search
    via Reciprocal Rank Fusion.

    Usage:
        retriever = HybridRetriever(chunks)
        results = retriever.retrieve("book physiotherapy appointment", top_n=5)
    """

    def __init__(self, chunks: list[Document]):
        self.chunks       = chunks
        self.vector_store = get_or_build_vector_store(chunks)
        self.bm25         = get_bm25_retriever(chunks)
        self._n_candidates = TOP_K_RETRIEVAL * CANDIDATE_MULTIPLIER

    def retrieve(
        self,
        query: str,
        top_n: int = TOP_K_RETRIEVAL,
        source_filter: str | None = None,   # "web" | "pdf" | None
    ) -> list[Document]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query         — user's question (English or German)
            top_n         — how many chunks to return
            source_filter — optionally filter to only web or pdf sources

        Returns:
            List of Document chunks, most relevant first.
        """
        print(f"\n🔍 Retrieving: '{query[:60]}...' " if len(query) > 60
              else f"\n🔍 Retrieving: '{query}'")

        n = min(self._n_candidates, len(self.chunks))

        # ── FAISS semantic search ─────────────────────────────
        faiss_results = self.vector_store.similarity_search(query, k=n)
        print(f"   FAISS: {len(faiss_results)} results")

        # ── BM25 keyword search ───────────────────────────────
        self.bm25.k = n
        bm25_results = self.bm25.invoke(_normalize_query(query))
        print(f"   BM25 : {len(bm25_results)} results")

        # ── Reciprocal Rank Fusion ────────────────────────────
        fused = reciprocal_rank_fusion([faiss_results, bm25_results])
        print(f"   Fused: {len(fused)} unique results")

        # ── Optional source filter ────────────────────────────
        if source_filter:
            fused = [
                (doc, score) for doc, score in fused
                if doc.metadata.get("source_type") == source_filter
            ]

        # ── Return top N ──────────────────────────────────────
        top_docs = [doc for doc, _ in fused[:top_n]]

        # Debug output
        for i, (doc, score) in enumerate(fused[:top_n], 1):
            name = doc.metadata.get("page_name", "?")[:40]
            stype = doc.metadata.get("source_type", "?")
            print(f"   {i:>2}. [{stype:>3}] score={score:.4f} | {name}")

        return top_docs

    def retrieve_with_scores(
        self,
        query: str,
        top_n: int = TOP_K_RETRIEVAL,
    ) -> list[tuple[Document, float]]:
        """Same as retrieve() but returns (doc, score) tuples."""
        n = min(self._n_candidates, len(self.chunks))

        faiss_results = self.vector_store.similarity_search(query, k=n)
        self.bm25.k   = n
        bm25_results  = self.bm25.invoke(_normalize_query(query))

        fused = reciprocal_rank_fusion([faiss_results, bm25_results])
        return fused[:top_n]


# ── Query Normalization ───────────────────────────────────────

def _normalize_query(query: str) -> str:
    """
    Normalize query for BM25 (case folding, whitespace).
    Keep special characters — German umlauts matter for BM25.
    """
    q = query.strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q


# ── Singleton retriever ───────────────────────────────────────
# The API uses this shared instance — loaded once at startup.

_retriever: HybridRetriever | None = None


def get_retriever(chunks: list[Document] | None = None) -> HybridRetriever:
    """
    Get or initialize the global HybridRetriever instance.

    Args:
        chunks — required on first call (loaded from chunker)
    """
    global _retriever

    if _retriever is None:
        if chunks is None:
            raise ValueError(
                "Retriever not initialized. Pass chunks on first call."
            )
        _retriever = HybridRetriever(chunks)

    return _retriever


# ── CLI Test ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from ingest.chunker import load_documents_from_disk, chunk_documents

    print("Loading documents and building retriever...")
    docs   = load_documents_from_disk()
    chunks = chunk_documents(docs)

    retriever = HybridRetriever(chunks)

    # Test queries
    test_queries = [
        "How can I book an appointment?",
        "Wie kann ich einen Termin buchen?",
        "Physiotherapy treatment options",
        "Öffnungszeiten",
        "doctors at functiomed",
    ]

    if len(sys.argv) > 1:
        test_queries = [" ".join(sys.argv[1:])]

    for query in test_queries:
        results = retriever.retrieve(query, top_n=3)
        print(f"\n  Top result preview:")
        if results:
            print(f"  {results[0].page_content[:300]}\n")
        print("─" * 60)