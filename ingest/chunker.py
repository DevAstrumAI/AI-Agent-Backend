"""
RAG SYSTEM — ingest/chunker.py
================================
Loads ALL .txt files from data/clean_text/ and chunks them.

Every file in clean_text/ is plain text:
  • web scraper (scraper/scraper.py) → saves  www.functiomed.ch_angebot.txt
  • pdf_data.py                      → saves  pdf__<name>.txt

Naming convention (matches your existing pdf_data.py):
  pdf__<name>.txt   → source_type = "pdf"
  anything_else.txt → source_type = "web"

Chunking matches your existing web_data.py exactly:
  chunk_size=400, chunk_overlap=200, same separators.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

CLEAN_DIR     = os.getenv("CLEAN_TEXT_DIR", "data/clean_text")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
PDF_PREFIX    = "pdf__"
MIN_LENGTH    = 50


def load_documents_from_disk() -> list[Document]:
    """Load every .txt file from data/clean_text/. Returns one Document per file."""
    clean_path = Path(CLEAN_DIR)

    if not clean_path.exists():
        raise FileNotFoundError(
            f"data/clean_text/ not found.\n"
            f"Run scraper:    python scraper/scraper.py\n"
            f"Run PDF ingest: python -c \"from pdf_data.pdf_data import save_pdfs_to_clean_text; save_pdfs_to_clean_text()\""
        )

    txt_files = sorted(
        f for f in clean_path.iterdir()
        if f.suffix == ".txt"
        and not f.name.startswith("_")
        and not f.name.startswith(".")
    )

    if not txt_files:
        raise ValueError(f"No .txt files found in {CLEAN_DIR}.")

    print(f"\n📂 Found {len(txt_files)} .txt files in {CLEAN_DIR}")

    docs, skipped = [], 0

    for file in txt_files:
        text = file.read_text(encoding="utf-8").strip()

        if len(text) < MIN_LENGTH:
            skipped += 1
            continue

        is_pdf      = file.stem.startswith(PDF_PREFIX)
        source_type = "pdf" if is_pdf else "web"
        page_name   = file.stem[len(PDF_PREFIX):] if is_pdf else file.stem

        docs.append(Document(
            page_content=text,
            metadata={
                "source_type": source_type,
                "page_name":   page_name,
                "filename":    file.name,
            },
        ))

    web_docs = sum(1 for d in docs if d.metadata["source_type"] == "web")
    pdf_docs = sum(1 for d in docs if d.metadata["source_type"] == "pdf")
    print(f"  ✅ Loaded {len(docs)} documents ({web_docs} web + {pdf_docs} pdf) | skipped: {skipped}")
    return docs


def chunk_documents(
    docs: Optional[list[Document]] = None,
    chunk_size:    int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into overlapping chunks. Identical to web_data.py."""
    if docs is None:
        docs = load_documents_from_disk()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    print(f"\n✂️  Chunking {len(docs)} documents (size={chunk_size}, overlap={chunk_overlap})")
    chunks = splitter.split_documents(docs)

    web_chunks = sum(1 for c in chunks if c.metadata["source_type"] == "web")
    pdf_chunks = sum(1 for c in chunks if c.metadata["source_type"] == "pdf")
    print(f"  ✅ {len(chunks)} chunks total ({web_chunks} web + {pdf_chunks} pdf)\n")
    return chunks


if __name__ == "__main__":
    docs   = load_documents_from_disk()
    chunks = chunk_documents(docs)

    web = [c for c in chunks if c.metadata["source_type"] == "web"]
    pdf = [c for c in chunks if c.metadata["source_type"] == "pdf"]

    if web:
        print(f"Sample web [{web[0].metadata['page_name']}]: {web[0].page_content[:200]}\n")
    if pdf:
        print(f"Sample pdf [{pdf[0].metadata['page_name']}]: {pdf[0].page_content[:200]}\n")