"""
RAG Backend — api/app.py
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

from ingest.chunker import load_documents_from_disk, chunk_documents
from embedding.vector_store import get_or_build_vector_store
from retrieval.retriever import HybridRetriever
from llm.generator import ask, detect_language

from api.bookings_router import router as bookings_router
from api.clinic_router    import router as clinic_router
from database.models      import init_db, close_db   # ← close_db added

_pipeline_ready = False
_pipeline_error = None
_chunks         = []
_retriever      = None
_startup_time   = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline_ready, _pipeline_error, _chunks, _retriever, _startup_time

    print("\n" + "="*60)
    print("  FUNCTIOMED RAG SYSTEM — STARTING UP")
    print("="*60)

    t0   = time.time()
    loop = asyncio.get_event_loop()

    try:
        print("\n🗄️  Initializing database...")
        await init_db()

        print("\n📚 Loading documents from disk...")
        docs    = await loop.run_in_executor(None, load_documents_from_disk)
        _chunks = await loop.run_in_executor(None, chunk_documents, docs)

        print("\n🔧 Building vector store...")
        await loop.run_in_executor(None, get_or_build_vector_store, _chunks)

        print("\n🔧 Initializing retriever...")
        _retriever = HybridRetriever(_chunks)

        elapsed         = time.time() - t0
        _pipeline_ready = True
        _startup_time   = elapsed

        print(f"\n{'='*60}")
        print(f"  ✅ READY in {elapsed:.1f}s  |  {len(_chunks)} chunks indexed")
        print(f"{'='*60}\n")

    except Exception as e:
        _pipeline_error = str(e)
        print(f"\n❌ STARTUP FAILED: {e}")
        import traceback; traceback.print_exc()

    yield

    print("\n👋 Shutting down...")
    await close_db()   # ← close PostgreSQL pool on shutdown


app = FastAPI(
    title="Functiomed RAG API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

app.include_router(bookings_router)
app.include_router(clinic_router)

from api.livekit_router import router as livekit_router
app.include_router(livekit_router)

try:
    from api.pdf_router import router as pdf_router
    app.include_router(pdf_router)
except ImportError:
    pass


# ── Models ────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str  = Field(..., min_length=1, max_length=1000)
    top_n:    int  = Field(default=8, ge=1, le=20)
    voice:    bool = Field(default=False)
    language: str | None = None


class AskResponse(BaseModel):
    question:    str
    answer:      str
    language:    str
    sources:     list[dict]
    chunks_used: int
    latency_ms:  float


class RetrieveRequest(BaseModel):
    query: str
    k:     int = Field(default=5, ge=1, le=20)


def require_pipeline():
    if not _pipeline_ready:
        raise HTTPException(status_code=503, detail=_pipeline_error or "Pipeline loading")


# ── Core endpoints ────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service":   "Functiomed RAG API",
        "status":    "ready" if _pipeline_ready else "loading",
        "chunks":    len(_chunks),
        "startup_s": round(_startup_time or 0, 2),
    }


@app.get("/status")
def status():
    index_path   = Path(os.getenv("FAISS_INDEX_DIR", "data/faiss_index"))
    index_exists = (index_path / "index.faiss").exists()
    return {
        "pipeline_ready":  _pipeline_ready,
        "pipeline_error":  _pipeline_error,
        "chunks_indexed":  len(_chunks),
        "index_on_disk":   index_exists,
        "startup_s":       round(_startup_time or 0, 2),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2"),
        "llm_model":       os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    require_pipeline()
    t0     = time.time()
    result = ask(question=req.question, retriever=_retriever, top_n=req.top_n, voice_mode=req.voice)
    if req.language in ("en", "de"):
        result["language"] = req.language
    return AskResponse(
        question    = result["question"],
        answer      = result["answer"],
        language    = result["language"],
        sources     = result["sources"],
        chunks_used = result["chunks_used"],
        latency_ms  = round((time.time() - t0) * 1000, 1),
    )


@app.post("/retrieve")
def retrieve_chunks(req: RetrieveRequest):
    require_pipeline()
    results = _retriever.retrieve_with_scores(req.query, top_n=req.k)
    return {
        "query":   req.query,
        "results": [
            {
                "rank":      i + 1,
                "score":     round(score, 6),
                "page_name": doc.metadata.get("page_name"),
                "content":   doc.page_content,
            }
            for i, (doc, score) in enumerate(results)
        ],
    }


# ── Scrape endpoint ───────────────────────────────────────────

_scrape_status = {"running": False, "last_run": None, "scraped": 0, "failed": 0, "error": None}


def _run_scrape(urls: list[str] | None, crawl: bool):
    """Runs in a background thread via BackgroundTasks."""
    import asyncio as _asyncio
    from scraper.scraper import WebScraper

    _scrape_status["running"] = True
    _scrape_status["error"]   = None

    try:
        scraper = WebScraper(urls=urls, dry_run=False, crawl=crawl)
        result  = _asyncio.run(scraper.scrape_all())
        _scrape_status["scraped"]  = len(scraper.scraped)
        _scrape_status["failed"]   = len(scraper.failed)
        _scrape_status["last_run"] = __import__("datetime").datetime.now().isoformat()
    except Exception as e:
        _scrape_status["error"] = str(e)
    finally:
        _scrape_status["running"] = False


@app.post("/scrape")
def trigger_scrape(
    background_tasks: BackgroundTasks,
    crawl: bool = False,
    urls: list[str] | None = None,
):
    """
    Trigger a web scrape in the background.

    - Uses the predefined URL list by default (all functiomed.ch pages)
    - Pass `urls` body to scrape specific pages instead
    - Pass `crawl=true` to also discover and follow links from scraped pages
    - Poll GET /scrape/status to track progress

    After scraping completes, call POST /pdfs/ingest to rebuild the RAG index.
    """
    if _scrape_status["running"]:
        raise HTTPException(status_code=409, detail="A scrape is already running")

    background_tasks.add_task(_run_scrape, urls, crawl)

    url_count = len(urls) if urls else "all predefined (83 URLs)"
    return {
        "message":   "Scrape started in background",
        "urls":      url_count,
        "crawl":     crawl,
        "poll":      "GET /scrape/status",
        "next_step": "After scraping, call POST /pdfs/ingest?rebuild_index=true",
    }


@app.get("/scrape/status")
def scrape_status():
    """Poll scrape progress."""
    return _scrape_status


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    print(f"\n🚀 Starting on http://{host}:{port}\n")
    uvicorn.run("api.app:app", host=host, port=port, reload=True)