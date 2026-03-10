"""
RAG Backend — api/pdf_router.py
=================================
PDF ingestion endpoints.

Add to your api/app.py:
    from api.pdf_router import router as pdf_router
    app.include_router(pdf_router)

Endpoints:
    POST /pdfs/upload     — upload a PDF file → saves to pdf_data/files/ → parses to .txt
    POST /pdfs/ingest     — re-parse all PDFs in pdf_data/files/ (no upload)
    GET  /pdfs            — list all PDFs currently in pdf_data/files/
    DELETE /pdfs/{name}   — delete a PDF and its .txt file
"""

import os
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/pdfs", tags=["PDFs"])

PDF_DIR   = "pdf_data/files"
CLEAN_DIR = "data/clean_text"
PDF_PREFIX = "pdf__"


# ── GET /pdfs — list all PDFs ─────────────────────────────────

@router.get("/")
def list_pdfs():
    """
    List all PDF files currently in pdf_data/files/.
    Shows whether each one has been parsed to .txt yet.
    """
    Path(PDF_DIR).mkdir(parents=True, exist_ok=True)

    pdf_files = [
        f for f in os.listdir(PDF_DIR)
        if f.lower().endswith(".pdf")
    ]

    result = []
    for filename in sorted(pdf_files):
        txt_name   = f"{PDF_PREFIX}{Path(filename).stem}.txt"
        txt_exists = (Path(CLEAN_DIR) / txt_name).exists()
        pdf_size   = Path(PDF_DIR, filename).stat().st_size

        result.append({
            "filename":  filename,
            "size_kb":   round(pdf_size / 1024, 1),
            "parsed":    txt_exists,
            "txt_file":  txt_name if txt_exists else None,
        })

    return {
        "pdfs":  result,
        "count": len(result),
        "pdf_dir":   PDF_DIR,
        "clean_dir": CLEAN_DIR,
    }


# ── POST /pdfs/upload — upload + parse a PDF ─────────────────

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    rebuild_index: bool = False,
    background_tasks: BackgroundTasks = None,
):
    """
    Upload a PDF file, save it to pdf_data/files/, and parse it to .txt.

    Args:
        file          — the PDF file to upload
        rebuild_index — if True, also rebuilds the FAISS index after parsing
                        (slow — runs in background)

    Returns the parsed .txt filename and character count.
    """
    # Validate it's a PDF
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Only .pdf files accepted. Got: {file.filename}"
        )

    Path(PDF_DIR).mkdir(parents=True, exist_ok=True)
    Path(CLEAN_DIR).mkdir(parents=True, exist_ok=True)

    pdf_path = Path(PDF_DIR) / file.filename
    txt_name = f"{PDF_PREFIX}{Path(file.filename).stem}.txt"
    txt_path = Path(CLEAN_DIR) / txt_name

    # Save uploaded PDF to disk
    try:
        content = await file.read()
        pdf_path.write_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Parse PDF → .txt
    try:
        char_count = _parse_single_pdf(str(pdf_path), str(txt_path))
    except Exception as e:
        # Clean up the saved PDF if parsing fails
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"PDF parsing failed: {e}")

    response = {
        "success":      True,
        "pdf_file":     file.filename,
        "txt_file":     txt_name,
        "chars_parsed": char_count,
        "message":      f"Parsed and saved to {CLEAN_DIR}/{txt_name}",
    }

    # Optionally rebuild the FAISS index in background
    if rebuild_index and background_tasks:
        background_tasks.add_task(_rebuild_index)
        response["index_rebuild"] = "started in background — poll GET /status"

    return response


# ── POST /pdfs/ingest — (re)parse all PDFs ───────────────────

@router.post("/ingest")
def ingest_all_pdfs(rebuild_index: bool = False, background_tasks: BackgroundTasks = None):
    """
    Parse all PDFs in pdf_data/files/ that haven't been parsed yet.
    Use force=true as a query param to re-parse already parsed files.

    After this, call POST /ingest to rebuild the FAISS index,
    or pass rebuild_index=true to do it automatically in the background.
    """
    from pdf_data.pdf_data import save_pdfs_to_clean_text
    result = save_pdfs_to_clean_text()

    response = {
        "saved":   result["saved"],
        "skipped": result["skipped"],
        "failed":  result["failed"],
        "message": (
            f"Parsed {len(result['saved'])} new PDF(s). "
            f"{len(result['skipped'])} already existed. "
            f"{len(result['failed'])} failed."
        ),
    }

    if rebuild_index and background_tasks:
        background_tasks.add_task(_rebuild_index)
        response["index_rebuild"] = "started in background — poll GET /status"

    return response


# ── DELETE /pdfs/{filename} — remove a PDF ───────────────────

@router.delete("/{filename}")
def delete_pdf(filename: str):
    """
    Delete a PDF file and its parsed .txt file.
    Note: you must rebuild the FAISS index (POST /ingest) after deletion.
    """
    if not filename.lower().endswith(".pdf"):
        filename = filename + ".pdf"

    pdf_path = Path(PDF_DIR) / filename
    txt_name = f"{PDF_PREFIX}{Path(filename).stem}.txt"
    txt_path = Path(CLEAN_DIR) / txt_name

    deleted = []

    if pdf_path.exists():
        pdf_path.unlink()
        deleted.append(str(pdf_path))
    else:
        raise HTTPException(status_code=404, detail=f"{filename} not found in {PDF_DIR}/")

    if txt_path.exists():
        txt_path.unlink()
        deleted.append(str(txt_path))

    return {
        "success": True,
        "deleted": deleted,
        "note":    "Run POST /ingest to rebuild the FAISS index.",
    }


# ── Internal helpers ──────────────────────────────────────────

def _parse_single_pdf(pdf_path: str, txt_path: str) -> int:
    """
    Parse one PDF file to plain text.
    Returns character count of the output text.
    Uses PyPDFLoader (same as pdf_data.py).
    """
    import re
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()

    full_text = "\n\n".join(
        re.sub(r"\s+", " ", page.page_content).strip()
        for page in pages
        if page.page_content and page.page_content.strip()
    )

    if not full_text.strip():
        raise ValueError("PDF appears to be empty or unreadable (scanned image?)")

    Path(txt_path).write_text(full_text, encoding="utf-8")
    return len(full_text)


def _rebuild_index():
    """Background task: rebuild FAISS index after new PDFs are added."""
    try:
        from ingest.chunker import load_documents_from_disk, chunk_documents
        from embedding.vector_store import build_vector_store

        docs   = load_documents_from_disk()
        chunks = chunk_documents(docs)
        build_vector_store(chunks, force_rebuild=True)
        print("✅ FAISS index rebuilt after PDF ingestion")
    except Exception as e:
        print(f"❌ Index rebuild failed: {e}")