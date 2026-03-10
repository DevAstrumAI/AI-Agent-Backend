"""
RAG SYSTEM — STEP 1: Web Scraper
==================================
Scrapes every page of a website using Playwright (handles JS-rendered pages).
Saves raw HTML + clean extracted text to disk.

Why Playwright and not requests?
  Modern clinic websites use React/Vue — requests only gets the empty shell.
  Playwright runs a real browser, waits for JS to finish, then grabs the content.

Usage:
    python scraper/scraper.py                        # scrape default TARGET_URL
    python scraper/scraper.py --url https://...      # scrape a custom URL
    python scraper/scraper.py --dry-run              # just print URLs, don't scrape
"""

import asyncio
import argparse
import hashlib
import os
import re
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.async_api import async_playwright, Browser, Page

load_dotenv()

# ── Config ────────────────────────────────────────────────────

# BASE_URL       = os.getenv("TARGET_URL", "https://www.functiomed.ch")

BASE_URL       = os.getenv("TARGET_URL", "https://www.functiomed.ch/angebot/")

RAW_HTML_DIR   = Path(os.getenv("RAW_HTML_DIR", "data/raw_html"))
CLEAN_TEXT_DIR = Path(os.getenv("CLEAN_TEXT_DIR", "data/clean_text"))

# File extensions that are never web pages
SKIP_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".zip", ".tar", ".gz", ".mp4", ".mp3", ".woff", ".woff2",
    ".ico", ".xml", ".json", ".css", ".js",
}

# URL patterns to skip (dynamic/pagination/auth pages)
SKIP_PATTERNS = [
    r"/wp-admin", r"/login", r"/logout", r"/register",
    r"\?.*page=\d+", r"#",  # anchors
    r"/cdn-cgi/",           # Cloudflare internals
]

# Maximum pages to scrape (safety limit)
MAX_PAGES = 500

# How long to wait for page to fully load (ms)
PAGE_TIMEOUT_MS = 30_000

# CSS selectors for content — tries each in order, uses first match
CONTENT_SELECTORS = ["main", "article", "#content", ".content", "body"]

# Tags to strip before extracting text (navigation, ads, etc.)
STRIP_TAGS = ["script", "style", "nav", "footer", "header",
              "noscript", "aside", "iframe", ".cookie-banner"]


# ── URL Utilities ─────────────────────────────────────────────

def normalize_url(url: str) -> str:
    """Remove fragments, trailing slashes, lowercasing path."""
    parsed = urlparse(url.strip())
    # Remove fragment
    clean = parsed._replace(fragment="", query="")
    path = clean.path.rstrip("/") or "/"
    clean = clean._replace(path=path)
    return urlunparse(clean).lower()


def is_same_domain(url: str, base: str) -> bool:
    return urlparse(url).netloc == urlparse(base).netloc


def should_skip(url: str) -> bool:
    """Return True if this URL should NOT be scraped."""
    parsed = urlparse(url)

    # Skip non-http
    if parsed.scheme not in ("http", "https"):
        return True

    # Skip known file extensions
    path_lower = parsed.path.lower()
    if any(path_lower.endswith(ext) for ext in SKIP_EXTENSIONS):
        return True

    # Skip known bad patterns
    full = url.lower()
    if any(re.search(p, full) for p in SKIP_PATTERNS):
        return True

    return False


def url_to_filename(url: str, ext: str) -> str:
    """
    Convert URL to a safe filename.
    'https://www.functiomed.ch/angebot/massage' → 'functiomed.ch_angebot_massage.html'
    """
    parsed = urlparse(url)
    # Remove www. prefix
    host = parsed.netloc.replace("www.", "")
    path = parsed.path.strip("/").replace("/", "_") or "index"
    # Truncate very long filenames
    name = f"{host}_{path}"[:180]
    return f"{name}{ext}"


# ── Text Extraction ───────────────────────────────────────────

def extract_clean_text(html: str, url: str) -> dict:
    """
    Extract clean text + metadata from HTML.

    Returns dict with:
        text      — clean body text
        title     — page <title>
        url       — source URL
        h1        — first H1 heading (often most descriptive)
        word_count — approximate word count
    """
    soup = BeautifulSoup(html, "lxml")

    # Pull metadata before stripping
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    h1_tag = soup.find("h1")
    h1 = h1_tag.get_text(" ").strip() if h1_tag else ""

    # Strip noise tags
    for selector in STRIP_TAGS:
        if selector.startswith("."):
            for tag in soup.select(selector):
                tag.decompose()
        else:
            for tag in soup.find_all(selector):
                tag.decompose()

    # Find main content area
    content_el = None
    for selector in CONTENT_SELECTORS:
        content_el = soup.select_one(selector)
        if content_el:
            break

    raw_text = content_el.get_text(separator=" ") if content_el else soup.get_text(separator=" ")

    # Clean up whitespace
    text = re.sub(r"[ \t]+", " ", raw_text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return {
        "text": text,
        "title": title,
        "url": url,
        "h1": h1,
        "word_count": len(text.split()),
    }


# ── Scraper ───────────────────────────────────────────────────

class WebScraper:
    """
    Breadth-first website scraper using Playwright.

    Algorithm:
        1. Start with the base URL in the queue
        2. Visit each URL: render with Playwright, extract text, save files
        3. Find all <a href> links on the page
        4. Add same-domain, not-yet-visited links to the queue
        5. Repeat until queue empty or MAX_PAGES reached
    """

    def __init__(self, base_url: str = BASE_URL, dry_run: bool = False):
        self.base_url    = normalize_url(base_url)
        self.dry_run     = dry_run
        self.visited:    set[str] = set()
        self.queue:      list[str] = [self.base_url]
        self.failed:     list[dict] = []
        self.scraped:    list[dict] = []

        RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
        CLEAN_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    async def scrape_all(self):
        """Entry point — scrape all pages."""
        print(f"\n{'='*60}")
        print(f"  WEB SCRAPER")
        print(f"{'='*60}")
        print(f"  Base URL : {self.base_url}")
        print(f"  Max pages: {MAX_PAGES}")
        print(f"  Dry run  : {self.dry_run}")
        print(f"{'='*60}\n")

        if self.dry_run:
            print("DRY RUN — discovering URLs only, not saving files.\n")

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )

            try:
                # Use a single browser context for all pages (faster, shared cookies)
                context = await browser.new_context(
                    viewport={"width": 1280, "height": 800},
                    user_agent="Mozilla/5.0 (compatible; FunctiomedBot/1.0)",
                )

                while self.queue and len(self.visited) < MAX_PAGES:
                    url = self.queue.pop(0)
                    url = normalize_url(url)

                    if url in self.visited:
                        continue
                    if should_skip(url):
                        continue
                    if not is_same_domain(url, self.base_url):
                        continue

                    self.visited.add(url)
                    await self._scrape_page(context, url)

                print(f"\n{'='*60}")
                print(f"  SCRAPING COMPLETE")
                print(f"  Pages scraped : {len(self.scraped)}")
                print(f"  Pages failed  : {len(self.failed)}")
                print(f"  Queue remaining: {len(self.queue)}")
                print(f"{'='*60}\n")

            finally:
                await browser.close()

        # Save a manifest of all scraped pages
        self._save_manifest()
        return self.scraped

    async def _scrape_page(self, context, url: str):
        """Scrape a single page: render → extract → save."""
        n = len(self.visited)
        print(f"  [{n:>3}] Scraping: {url}")

        if self.dry_run:
            return

        page = await context.new_page()
        try:
            # Navigate and wait for network to go quiet
            await page.goto(url, timeout=PAGE_TIMEOUT_MS, wait_until="networkidle")

            # Extra wait for any lazy-loaded content
            await page.wait_for_timeout(500)

            html = await page.content()

            # Extract text + metadata
            data = extract_clean_text(html, url)
            data["scraped_at"] = datetime.now().isoformat()

            # Skip pages with almost no content
            if data["word_count"] < 20:
                print(f"         ⏭  Skip (too short: {data['word_count']} words)")
                return

            # Save raw HTML
            html_filename = url_to_filename(url, ".html")
            (RAW_HTML_DIR / html_filename).write_text(html, encoding="utf-8")

            # Save clean text as plain .txt (same format as pdf_data.py)
            txt_filename = url_to_filename(url, ".txt")
            (CLEAN_TEXT_DIR / txt_filename).write_text(
                data["text"],
                encoding="utf-8",
            )

            print(f"         ✅ {data['word_count']} words | {data['title'][:50]}")
            self.scraped.append(data)

            # Discover links on this page
            links = await page.eval_on_selector_all(
                "a[href]", "els => els.map(e => e.href)"
            )
            for link in links:
                normalized = normalize_url(link)
                if (
                    normalized not in self.visited
                    and not should_skip(normalized)
                    and is_same_domain(normalized, self.base_url)
                    and normalized not in self.queue
                ):
                    self.queue.append(normalized)

        except Exception as e:
            print(f"         ❌ Failed: {e}")
            self.failed.append({"url": url, "error": str(e)})
        finally:
            await page.close()

    def _save_manifest(self):
        """Save a JSON manifest of all scraped pages for debugging."""
        manifest = {
            "scraped_at": datetime.now().isoformat(),
            "base_url": self.base_url,
            "total_pages": len(self.scraped),
            "total_failed": len(self.failed),
            "pages": [
                {"url": p["url"], "title": p["title"], "words": p["word_count"]}
                for p in self.scraped
            ],
            "failed": self.failed,
        }
        manifest_path = CLEAN_TEXT_DIR / "_scrape_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  📋 Manifest saved: {manifest_path}")


# ── PDF Ingestion ─────────────────────────────────────────────

def ingest_pdfs(pdf_dir: str = None):
    """
    Extract text from all PDF files in pdf_dir.
    Saves each as a .json file in CLEAN_TEXT_DIR alongside web pages.

    Uses PyMuPDF (fitz) — handles both digital PDFs and
    provides much better text extraction than PyPDF2/pdfplumber.
    """
    import fitz  # PyMuPDF

    pdf_path = Path(pdf_dir or os.getenv("PDF_DIR", "data/pdfs"))
    pdf_path.mkdir(parents=True, exist_ok=True)
    CLEAN_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdf_path.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  No PDFs found in {pdf_path}")
        return []

    print(f"\n{'='*60}")
    print(f"  PDF INGESTION — {len(pdf_files)} file(s)")
    print(f"{'='*60}\n")

    results = []
    for pdf_file in pdf_files:
        out_name = f"pdf__{pdf_file.stem}.json"
        out_path = CLEAN_TEXT_DIR / out_name

        if out_path.exists():
            print(f"  ⏭  Skip (exists): {out_name}")
            continue

        try:
            doc = fitz.open(str(pdf_file))
            pages_text = []

            for page_num, page in enumerate(doc):
                text = page.get_text("text")  # "text" = plain text extraction
                text = re.sub(r"[ \t]+", " ", text)
                text = re.sub(r"\n{3,}", "\n\n", text).strip()
                if text and len(text) > 30:
                    pages_text.append(text)

            full_text = "\n\n".join(pages_text)
            doc.close()

            if not full_text.strip():
                print(f"  ⚠️  Empty: {pdf_file.name}")
                continue

            data = {
                "text": full_text,
                "title": pdf_file.stem.replace("_", " ").replace("-", " "),
                "url": f"local://{pdf_file.name}",
                "h1": "",
                "word_count": len(full_text.split()),
                "source_type": "pdf",
                "scraped_at": datetime.now().isoformat(),
            }

            out_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  ✅ {pdf_file.name} → {out_name} ({data['word_count']} words)")
            results.append(data)

        except Exception as e:
            print(f"  ❌ Failed {pdf_file.name}: {e}")

    return results


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Functiomed Web Scraper")
    parser.add_argument("--url",     default=BASE_URL, help="Base URL to scrape")
    parser.add_argument("--dry-run", action="store_true", help="Discover URLs only")
    parser.add_argument("--pdfs",    action="store_true", help="Ingest PDFs only")
    args = parser.parse_args()

    if args.pdfs:
        ingest_pdfs()
    else:
        scraper = WebScraper(base_url=args.url, dry_run=args.dry_run)
        asyncio.run(scraper.scrape_all())

        if not args.dry_run:
            # Also ingest any PDFs in data/pdfs/
            ingest_pdfs()