"""
RAG SYSTEM — STEP 7: Test Suite
=================================
Validates every layer of the RAG pipeline independently.
Run this after setup to confirm everything works before connecting
to the voice agent.

Usage:
    python tests/test_pipeline.py                  # run all tests
    python tests/test_pipeline.py --step scraper   # test one step
    python tests/test_pipeline.py --step chunker
    python tests/test_pipeline.py --step embedding
    python tests/test_pipeline.py --step retrieval
    python tests/test_pipeline.py --step llm
    python tests/test_pipeline.py --step api
    python tests/test_pipeline.py --queries        # test with real queries
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "
INFO = "ℹ️ "

results = []


def log(status, name, detail=""):
    icons = {"pass": PASS, "fail": FAIL, "warn": WARN, "info": INFO}
    icon = icons.get(status, "?")
    line = f"  {icon} {name}"
    if detail:
        line += f"\n     {detail}"
    print(line)
    results.append((status, name))


def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}\n")


# ── Step 1: Scraper output check ──────────────────────────────

def test_scraper():
    section("STEP 1: Scraper Output")

    clean_dir = Path(os.getenv("CLEAN_TEXT_DIR", "data/clean_text"))

    if not clean_dir.exists():
        log("fail", "data/clean_text/ exists",
            "Directory missing. Run: python scraper/scraper.py")
        return

    json_files = [f for f in clean_dir.glob("*.json") if not f.name.startswith("_")]

    if not json_files:
        log("fail", "JSON files present",
            "No .json files found. Run: python scraper/scraper.py")
        return

    log("pass", f"data/clean_text/ has {len(json_files)} files")

    # Check content quality
    total_words = 0
    empty = 0
    types = {"web": 0, "pdf": 0}

    for f in json_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            words = data.get("word_count", 0)
            total_words += words
            if words < 20:
                empty += 1
            src_type = "pdf" if f.name.startswith("pdf__") else "web"
            types[src_type] += 1
        except Exception:
            pass

    log("pass", f"Web pages: {types['web']}, PDF files: {types['pdf']}")
    log("pass", f"Total words: {total_words:,}")

    if empty > 0:
        log("warn", f"{empty} near-empty files found (< 20 words)")

    # Check for key Functiomed pages
    expected_keywords = ["angebot", "kontakt", "physiotherap", "termin"]
    for kw in expected_keywords:
        matches = [f for f in json_files if kw in f.name.lower()]
        if matches:
            log("pass", f"'{kw}' pages found", f"{len(matches)} file(s)")
        else:
            log("warn", f"No '{kw}' pages found",
                "These pages may not have been scraped yet")


# ── Step 2: Chunker ───────────────────────────────────────────

def test_chunker():
    section("STEP 2: Document Chunker")

    try:
        from ingest.chunker import load_documents_from_disk, chunk_documents
    except ImportError as e:
        log("fail", "Import chunker", str(e))
        return

    try:
        docs = load_documents_from_disk()
        log("pass", f"load_documents_from_disk()", f"{len(docs)} documents loaded")
    except Exception as e:
        log("fail", "load_documents_from_disk()", str(e))
        return

    try:
        t0     = time.time()
        chunks = chunk_documents(docs)
        elapsed = time.time() - t0

        log("pass", f"chunk_documents()", f"{len(chunks)} chunks in {elapsed:.2f}s")

        if chunks:
            avg_len = sum(len(c.page_content) for c in chunks) // len(chunks)
            min_len = min(len(c.page_content) for c in chunks)
            max_len = max(len(c.page_content) for c in chunks)
            log("pass", f"Chunk sizes",
                f"avg={avg_len} chars, min={min_len}, max={max_len}")

            # Check metadata
            sample = chunks[0]
            has_meta = all(k in sample.metadata
                           for k in ["source", "source_type", "page_name"])
            log("pass" if has_meta else "warn",
                "Chunk metadata complete",
                str(dict(list(sample.metadata.items())[:4])))

        return chunks

    except Exception as e:
        log("fail", "chunk_documents()", str(e))
        return None


# ── Step 3: Embedding model ───────────────────────────────────

def test_embedding():
    section("STEP 3: Embedding Model")

    try:
        from embedding.vector_store import get_embedding_model
    except ImportError as e:
        log("fail", "Import embedding module", str(e))
        return

    try:
        t0      = time.time()
        embedder = get_embedding_model()
        elapsed = time.time() - t0
        log("pass", "Embedding model loaded", f"in {elapsed:.1f}s")
    except Exception as e:
        log("fail", "Load embedding model", str(e))
        return

    # Test embedding a sentence
    try:
        test_texts = [
            "book a physiotherapy appointment",
            "Termin bei Physiotherapie buchen",  # German equivalent
        ]
        embeddings = embedder.embed_documents(test_texts)

        log("pass", "embed_documents() works",
            f"Embedding dimension: {len(embeddings[0])}")

        # Check semantic similarity: EN and DE versions should be close
        import numpy as np
        en_vec = np.array(embeddings[0])
        de_vec = np.array(embeddings[1])
        cosine_sim = float(np.dot(en_vec, de_vec) / (np.linalg.norm(en_vec) * np.linalg.norm(de_vec)))

        log(
            "pass" if cosine_sim > 0.7 else "warn",
            f"EN/DE semantic similarity",
            f"Cosine similarity: {cosine_sim:.4f} "
            f"({'good — multilingual model working' if cosine_sim > 0.7 else 'low — check model'})"
        )
    except Exception as e:
        log("fail", "Test embedding", str(e))


# ── Step 4: FAISS index ───────────────────────────────────────

def test_vector_store(chunks=None):
    section("STEP 4: FAISS Vector Store")

    try:
        from embedding.vector_store import get_or_build_vector_store, FAISS_INDEX_DIR
    except ImportError as e:
        log("fail", "Import vector_store module", str(e))
        return

    index_file = FAISS_INDEX_DIR / "index.faiss"

    if not index_file.exists():
        if chunks is None:
            log("warn", "FAISS index exists",
                "Index not found. Run: python embedding/vector_store.py --build")
            return
        else:
            try:
                from embedding.vector_store import build_vector_store
                t0 = time.time()
                build_vector_store(chunks)
                log("pass", "Built FAISS index", f"in {time.time()-t0:.1f}s")
            except Exception as e:
                log("fail", "Build FAISS index", str(e))
                return
    else:
        log("pass", "FAISS index exists", str(index_file))

    try:
        from embedding.vector_store import load_vector_store
        vs = load_vector_store()
        n_vectors = vs.index.ntotal
        log("pass", f"FAISS index loaded", f"{n_vectors} vectors")

        # Test a similarity search
        results = vs.similarity_search("appointment booking", k=3)
        log("pass", "similarity_search() works",
            f"Returns {len(results)} results for 'appointment booking'")

    except Exception as e:
        log("fail", "Load/search FAISS index", str(e))


# ── Step 5: Retrieval ─────────────────────────────────────────

def test_retrieval(chunks=None):
    section("STEP 5: Hybrid Retrieval (FAISS + BM25)")

    try:
        from retrieval.retriever import HybridRetriever
        from ingest.chunker import load_documents_from_disk, chunk_documents
    except ImportError as e:
        log("fail", "Import retriever", str(e))
        return

    if chunks is None:
        try:
            docs   = load_documents_from_disk()
            chunks = chunk_documents(docs)
        except Exception as e:
            log("fail", "Load chunks for retrieval test", str(e))
            return

    try:
        t0 = time.time()
        retriever = HybridRetriever(chunks)
        log("pass", "HybridRetriever created", f"in {time.time()-t0:.1f}s")
    except Exception as e:
        log("fail", "Create HybridRetriever", str(e))
        return

    test_queries = [
        ("How can I book an appointment?",       "appointment|book|termin|kontakt"),
        ("Wie kann ich einen Termin buchen?",     "termin|buchen|kontakt|appointment"),
        ("physiotherapy treatments",              "physio|behandlung|therapie"),
        ("Öffnungszeiten",                        "öffnungszeit|open|uhr|hour"),
    ]

    for query, expected_keywords in test_queries:
        try:
            results = retriever.retrieve(query, top_n=5)

            if not results:
                log("fail", f"Query: '{query[:35]}'", "No results returned")
                continue

            combined = " ".join(r.page_content.lower() for r in results)
            keywords = expected_keywords.split("|")
            found    = [kw for kw in keywords if kw in combined]

            if found:
                log("pass", f"Query: '{query[:35]}'",
                    f"Found: {found} in top results")
            else:
                log("warn", f"Query: '{query[:35]}'",
                    f"Expected keywords not found. "
                    f"Top result: {results[0].page_content[:100]}")

        except Exception as e:
            log("fail", f"Query: '{query[:35]}'", str(e))

    return retriever


# ── Step 6: LLM generation ────────────────────────────────────

def test_llm(retriever=None):
    section("STEP 6: LLM Generation")

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or groq_key == "your_groq_api_key_here":
        log("fail", "GROQ_API_KEY set", "Set GROQ_API_KEY in your .env file")
        return

    log("pass", "GROQ_API_KEY is set")

    try:
        from llm.generator import get_llm, detect_language, generate_answer
    except ImportError as e:
        log("fail", "Import generator", str(e))
        return

    # Test language detection
    tests = [
        ("How can I book?",           "en"),
        ("Wie buche ich einen Termin?", "de"),
        ("Öffnungszeiten Physiotherapie", "de"),
        ("contact information",       "en"),
    ]
    all_correct = True
    for text, expected in tests:
        detected = detect_language(text)
        ok = detected == expected
        if not ok:
            all_correct = False
        log("pass" if ok else "warn",
            f"Language detect: '{text[:30]}'",
            f"Expected: {expected}, Got: {detected}")

    # Test LLM response
    try:
        from langchain.schema import Document
        dummy_chunk = Document(
            page_content="Functiomed is located at Fraumünsterstrasse 17, 8001 Zurich. "
                          "Phone: +41 44 000 00 00. Open Monday-Friday 8:00-18:00.",
            metadata={"source": "test", "page_name": "Contact"},
        )
        t0     = time.time()
        answer = generate_answer("What is the address?", [dummy_chunk], language="en")
        elapsed = time.time() - t0

        if answer and len(answer) > 20 and "error" not in answer.lower():
            log("pass", "LLM generates answer", f"in {elapsed:.1f}s: '{answer[:80]}...'")
        else:
            log("warn", "LLM answer quality", f"Short or error response: '{answer[:100]}'")

    except Exception as e:
        log("fail", "LLM generate_answer()", str(e))


# ── Step 7: API endpoints ─────────────────────────────────────

def test_api(base_url: str = "http://localhost:8000"):
    section(f"STEP 7: API Endpoints ({base_url})")

    import requests

    try:
        r = requests.get(f"{base_url}/", timeout=10)
        data = r.json()
        log("pass" if r.status_code == 200 else "fail",
            "GET / health check",
            f"status={r.status_code}, pipeline_ready={data.get('status')}")
    except Exception as e:
        log("fail", "API reachable", f"{e}\n     Is the server running? "
            "Run: python api/app.py")
        return

    # Test /ask
    try:
        r = requests.post(
            f"{base_url}/ask",
            json={"question": "What services does functiomed offer?", "top_n": 5},
            timeout=30,
        )
        data = r.json()
        if r.status_code == 200 and data.get("answer"):
            log("pass", "POST /ask works",
                f"Answer preview: '{data['answer'][:80]}...'")
        else:
            log("fail", "POST /ask failed", str(data))
    except Exception as e:
        log("fail", "POST /ask", str(e))

    # Test /retrieve
    try:
        r = requests.post(
            f"{base_url}/retrieve",
            json={"query": "book appointment", "k": 3},
            timeout=15,
        )
        data = r.json()
        if r.status_code == 200 and data.get("results"):
            log("pass", "POST /retrieve works",
                f"{len(data['results'])} chunks returned")
        else:
            log("fail", "POST /retrieve failed", str(data))
    except Exception as e:
        log("fail", "POST /retrieve", str(e))

    # Test /status
    try:
        r = requests.get(f"{base_url}/status", timeout=10)
        data = r.json()
        log("pass", "GET /status works",
            f"chunks={data.get('chunks_indexed')}, model={data.get('embedding_model')}")
    except Exception as e:
        log("fail", "GET /status", str(e))


# ── Real query test ───────────────────────────────────────────

def test_real_queries(retriever=None):
    section("REAL-WORLD QUERIES — Full Pipeline")

    try:
        from llm.generator import ask as rag_ask
    except ImportError as e:
        log("fail", "Import generator", str(e))
        return

    queries = [
        "How can I book an appointment at Functiomed?",
        "What physiotherapy services are available?",
        "Wie kann ich einen Termin buchen?",
        "Was kostet eine Massage bei Functiomed?",
        "What are the opening hours?",
        "Öffnungszeiten für Physiotherapie",
    ]

    for q in queries:
        print(f"\n  Q: {q}")
        try:
            t0 = time.time()
            result = rag_ask(q, retriever=retriever, top_n=6)
            elapsed = time.time() - t0

            answer_preview = result["answer"][:150].replace("\n", " ")
            print(f"  A [{result['language']} | {elapsed:.1f}s]: {answer_preview}...")
            print(f"     Sources: {[s['title'][:30] for s in result['sources'][:3]]}")

        except Exception as e:
            print(f"  ❌ Error: {e}")


# ── Summary ───────────────────────────────────────────────────

def print_summary():
    print(f"\n{'='*55}")
    print("  SUMMARY")
    print(f"{'='*55}\n")

    passes = sum(1 for s, _ in results if s == "pass")
    warns  = sum(1 for s, _ in results if s == "warn")
    fails  = sum(1 for s, _ in results if s == "fail")

    print(f"  {PASS} Passed : {passes}")
    print(f"  {WARN} Warned : {warns}")
    print(f"  {FAIL} Failed : {fails}")

    if fails == 0:
        print(f"\n  🎉 All checks passed! RAG system is ready.\n")
    else:
        print(f"\n  🔧 {fails} failure(s) need fixing. See details above.\n")

    return fails == 0


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline Test Suite")
    parser.add_argument("--step", choices=["scraper","chunker","embedding","retrieval","llm","api"],
                        help="Test only a specific step")
    parser.add_argument("--queries", action="store_true",
                        help="Run real-world query tests (requires full pipeline)")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="API base URL for API tests")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print("  FUNCTIOMED RAG — FULL PIPELINE TEST")
    print(f"{'='*55}")

    chunks   = None
    retriever = None

    step = args.step

    if step is None or step == "scraper":
        test_scraper()

    if step is None or step == "chunker":
        chunks = test_chunker()

    if step is None or step == "embedding":
        test_embedding()

    if step is None or step in ("retrieval", "llm"):
        if chunks is None:
            try:
                from ingest.chunker import load_documents_from_disk, chunk_documents
                docs   = load_documents_from_disk()
                chunks = chunk_documents(docs)
            except Exception:
                pass

    if step is None or step == "retrieval":
        test_vector_store(chunks)
        retriever = test_retrieval(chunks)

    if step is None or step == "llm":
        test_llm(retriever)

    if step == "api":
        test_api(args.api_url)

    if args.queries:
        test_real_queries(retriever)

    if step is None:
        print_summary()