"""
pdf_data/pdf_data.py
=====================
Reads all PDFs from pdf_data/files/ and saves each as a
plain .txt file in data/clean_text/ using the pdf__ prefix.

Output filename convention (matches ingest/chunker.py):
    Tarife.pdf  →  data/clean_text/pdf__Tarife.txt

Usage:
    # Run directly
    python pdf_data/pdf_data.py

    # Or import in your code
    from pdf_data.pdf_data import save_pdfs_to_clean_text
    save_pdfs_to_clean_text()
"""

import os
import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# ── Paths ─────────────────────────────────────────────────────

PDF_DIR   = "pdf_data/files"      # put your .pdf files here
CLEAN_DIR = "data/clean_text"     # plain .txt files saved here
PDF_PREFIX = "pdf__"              # must match ingest/chunker.py


# ── Helpers ───────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Normalize whitespace and strip."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _pdf_to_txt_name(pdf_filename: str) -> str:
    """
    Tarife.pdf  →  pdf__Tarife.txt
    """
    stem = Path(pdf_filename).stem
    return f"{PDF_PREFIX}{stem}.txt"


# ── Main function ─────────────────────────────────────────────

def save_pdfs_to_clean_text() -> dict:
    """
    Parse every PDF in pdf_data/files/ and save as plain .txt
    in data/clean_text/.

    Skips files that have already been converted (idempotent).

    Returns:
        {
            "saved":   ["pdf__Tarife.txt", ...],
            "skipped": ["pdf__OldFile.txt", ...],
            "failed":  ["BadFile.pdf: error msg", ...]
        }
    """
    Path(PDF_DIR).mkdir(parents=True, exist_ok=True)
    Path(CLEAN_DIR).mkdir(parents=True, exist_ok=True)

    pdf_files = [
        f for f in os.listdir(PDF_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"⚠️  No PDFs found in {PDF_DIR}/")
        print(f"   Put your .pdf files in {PDF_DIR}/ and run again.")
        return {"saved": [], "skipped": [], "failed": []}

    print(f"\n{'='*55}")
    print(f"  PDF INGESTION — {len(pdf_files)} file(s) found")
    print(f"{'='*55}\n")

    saved, skipped, failed = [], [], []

    for pdf_filename in sorted(pdf_files):
        txt_filename = _pdf_to_txt_name(pdf_filename)
        txt_path     = os.path.join(CLEAN_DIR, txt_filename)
        pdf_path     = os.path.join(PDF_DIR, pdf_filename)

        # Skip if already converted
        if os.path.exists(txt_path):
            print(f"  ⏭️  Skip (exists): {txt_filename}")
            skipped.append(txt_filename)
            continue

        try:
            loader = PyPDFLoader(pdf_path)
            pages  = loader.load()

            # Join all pages into one clean text block
            full_text = "\n\n".join(
                _clean_text(page.page_content)
                for page in pages
                if page.page_content and page.page_content.strip()
            )

            if not full_text.strip():
                print(f"  ⚠️  Empty content: {pdf_filename}")
                skipped.append(txt_filename)
                continue

            Path(txt_path).write_text(full_text, encoding="utf-8")

            print(f"  ✅ {pdf_filename} → {txt_filename} ({len(full_text):,} chars)")
            saved.append(txt_filename)

        except Exception as e:
            msg = f"{pdf_filename}: {e}"
            print(f"  ❌ Failed: {msg}")
            failed.append(msg)

    print(f"\n{'='*55}")
    print(f"  Done — saved: {len(saved)}, skipped: {len(skipped)}, failed: {len(failed)}")
    print(f"{'='*55}\n")

    return {"saved": saved, "skipped": skipped, "failed": failed}


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    result = save_pdfs_to_clean_text()

    if result["saved"]:
        print("Saved files:")
        for f in result["saved"]:
            print(f"  {CLEAN_DIR}/{f}")

    if result["failed"]:
        print("\nFailed:")
        for f in result["failed"]:
            print(f"  {f}")