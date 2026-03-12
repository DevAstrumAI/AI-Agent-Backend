"""
RAG SYSTEM — STEP 1: Web Scraper
==================================
Scrapes a predefined list of URLs using Playwright (handles JS-rendered pages).
Saves raw HTML + clean extracted text to disk.

Usage:
    python scraper/scraper.py                        # scrape all predefined URLs
    python scraper/scraper.py --dry-run              # just print URLs, don't scrape
    python scraper/scraper.py --url https://...      # scrape a single custom URL
    python scraper/scraper.py --crawl                # crawl mode (discover links)
"""

import asyncio
import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.async_api import async_playwright

load_dotenv()

# ── Config ────────────────────────────────────────────────────

BASE_URL       = os.getenv("TARGET_URL", "https://www.functiomed.ch")
RAW_HTML_DIR   = Path(os.getenv("RAW_HTML_DIR",   "data/raw_html"))
CLEAN_TEXT_DIR = Path(os.getenv("CLEAN_TEXT_DIR", "data/clean_text"))

PAGE_TIMEOUT_MS    = 30_000
MIN_WORD_COUNT     = 20
CONTENT_SELECTORS  = ["main", "article", "#content", ".content", "body"]
STRIP_TAGS         = ["script", "style", "nav", "footer", "header",
                      "noscript", "aside", "iframe", ".cookie-banner"]
SKIP_EXTENSIONS    = {
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".zip", ".tar", ".gz", ".mp4", ".mp3", ".woff", ".woff2",
    ".ico", ".xml", ".json", ".css", ".js",
}

# ── Full predefined URL list ───────────────────────────────────

PREDEFINED_URLS = [
    "https://www.functiomed.ch",
    "https://www.functiomed.ch/en",
    "https://www.functiomed.ch/abteilung/geschaeftsleitung",
    "https://www.functiomed.ch/abteilung/empfang",
    "https://www.functiomed.ch/abteilung/physiotherapie",
    "https://www.functiomed.ch/angebot/physiotherapie",
    "https://www.functiomed.ch/en/angebot/physiotherapie-kinderphysiotherapie",
    "https://www.functiomed.ch/angebot/mental-coaching",
    "https://www.functiomed.ch/en/angebot/mentaltraining",
    "https://www.functiomed.ch/angebot/massage",
    "https://www.functiomed.ch/en/angebot/massage",
    "https://www.functiomed.ch/angebot/ergotherapie",
    "https://www.functiomed.ch/angebot/orthopaedie-und-traumatologie",
    "https://www.functiomed.ch/en/angebot/orthopaedie-und-traumatologie_sportmedizin",
    "https://www.functiomed.ch/angebot/stammzellen",
    "https://www.functiomed.ch/angebot/integrative_medizin",
    "https://www.functiomed.ch/en/angebot/integrative_medizin",
    "https://www.functiomed.ch/angebot/infusionstherapie",
    "https://www.functiomed.ch/angebot/erspe-institut-ernaehrungsdiagnostik",
    "https://www.functiomed.ch/en/angebot/ernaehrungsberatung",
    "https://www.functiomed.ch/angebot/numo",
    "https://www.functiomed.ch/en/angebot/numo",
    "https://www.functiomed.ch/angebot/fitamara-praxis-fuer-ernaehrung-und-gesundheit-in-zuerich",
    "https://www.functiomed.ch/angebot/homoeopathie",
    "https://www.functiomed.ch/en/angebot/homoeopathie",
    "https://www.functiomed.ch/angebot/akupunktur",
    "https://www.functiomed.ch/en/angebot/akupunktur",
    "https://www.functiomed.ch/angebot/osteophatie-etiopathie",
    "https://www.functiomed.ch/en/angebot/osteophatie-etiopathie",
    "https://www.functiomed.ch/angebot/sport-osteopathie",
    "https://www.functiomed.ch/en/angebot/sport-osteopathie",
    "https://www.functiomed.ch/angebot/kinderosteopathie",
    "https://www.functiomed.ch/en/angebot/kinderosteopathie",
    "https://www.functiomed.ch/angebot/functiotraining",
    "https://www.functiomed.ch/en/angebot/functiotraining",
    "https://www.functiomed.ch/angebot/functiokurse",
    "https://www.functiomed.ch/en/angebot/functiokurse",
    "https://www.functiomed.ch/angebot/schwangerschaft",
    "https://www.functiomed.ch/en/angebot/schwangerschaft",
    "https://www.functiomed.ch/termin-buchen",
    "https://www.functiomed.ch/en/book-appointment",
    "https://www.functiomed.ch/news",
    "https://www.functiomed.ch/news/spezielle-oeffnungszeiten",
    "https://www.functiomed.ch/tarife",
    "https://www.functiomed.ch/unsere-partner",
    "https://www.functiomed.ch/unsere-functiosportler",
    "https://www.functiomed.ch/news/bilderausstellung-von-leoarta-rushiti",
    "https://www.functiomed.ch/ausstellungen/bilderausstellung-von-manu-ueltschi",
    "https://www.functiomed.ch/gesundheitstipps/uebung-des-monats-juni",
    "https://www.functiomed.ch/gesundheitstipps/uebung-des-monats-juli",
    "https://www.functiomed.ch/gesundheitstipps/uebung-des-monats-mai",
    "https://www.functiomed.ch/gesundheitstipps/uebung-des-monats-april",
    "https://www.functiomed.ch/gesundheitstipps/uebung-des-monats-maerz",
    "https://www.functiomed.ch/gesundheitstipps/uebung-des-monats-februar",
    "https://www.functiomed.ch/gesundheitstipps/uebung-des-monats-januar",
    "https://www.functiomed.ch/ausstellungen/bilderausstellung-von-claudia-kircher",
    "https://www.functiomed.ch/shop",
    "https://www.functiomed.ch/angebot/rheumatologie-innere-medizin",
    "https://www.functiomed.ch/angebot/colon-hydro-therapie",
    "https://www.functiomed.ch/kontakt",
    "https://www.functiomed.ch/events/leichtathletik-em-rom-2024",
    "https://www.functiomed.ch/notfall",
    "https://www.functiomed.ch/news/world-athletics-relays-bahamas",
    "https://www.functiomed.ch/ueber-die-praxis",
    "https://www.functiomed.ch/news/interview-mit-martin-spring",
    "https://www.functiomed.ch/ausstellungen/bilderausstellung-von-milijana-tanovic",
    "https://www.functiomed.ch/angebot/kiefertherapie",
    "https://www.functiomed.ch/ausstellungen/bilderausstellung-von-annette-k",
    "https://www.functiomed.ch/datenschutz",
    "https://www.functiomed.ch/impressum",
    "https://www.functiomed.ch/ueber-uns",
    "https://www.functiomed.ch/privacy-policy",
    "https://www.functiomed.ch/angebot",
    "https://www.functiomed.ch/videos/trainingsvideo-die-goldenen-uebungen",
    "https://www.functiomed.ch/videos/vortrag-von-tamara-meier-ernaehrungsberatung",
    "https://www.functiomed.ch/videos/vortrag-von-marian-leuthold-naturheilpraktikerin",
    "https://www.functiomed.ch/videos/ein-schmerzfreier-ruecken",
    "https://www.functiomed.ch/videos/trainingsvideo-rueckenturnen",
    "https://www.functiomed.ch/videos/yoga",
    "https://www.functiomed.ch/angebot/unser-engagement-functiosport",
]


# ── URL Utilities ─────────────────────────────────────────────

def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    clean  = parsed._replace(fragment="", query="")
    path   = clean.path.rstrip("/") or "/"
    clean  = clean._replace(path=path)
    return urlunparse(clean).lower()


def should_skip(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return True
    if any(parsed.path.lower().endswith(ext) for ext in SKIP_EXTENSIONS):
        return True
    return False


def url_to_filename(url: str, ext: str) -> str:
    parsed = urlparse(url)
    host   = parsed.netloc.replace("www.", "")
    path   = parsed.path.strip("/").replace("/", "_") or "index"
    name   = f"{host}_{path}"[:180]
    return f"{name}{ext}"


# ── Text Extraction ───────────────────────────────────────────

def extract_clean_text(html: str, url: str) -> dict:
    soup  = BeautifulSoup(html, "lxml")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    h1_tag = soup.find("h1")
    h1    = h1_tag.get_text(" ").strip() if h1_tag else ""

    for selector in STRIP_TAGS:
        for tag in (soup.select(selector) if selector.startswith(".") else soup.find_all(selector)):
            tag.decompose()

    content_el = None
    for selector in CONTENT_SELECTORS:
        content_el = soup.select_one(selector)
        if content_el:
            break

    raw_text = content_el.get_text(separator=" ") if content_el else soup.get_text(separator=" ")
    text = re.sub(r"[ \t]+", " ", raw_text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return {
        "text":       text,
        "title":      title,
        "url":        url,
        "h1":         h1,
        "word_count": len(text.split()),
    }


# ── Scraper ───────────────────────────────────────────────────

class WebScraper:
    def __init__(self, urls: list[str] = None, dry_run: bool = False, crawl: bool = False):
        self.urls     = [normalize_url(u) for u in (urls or PREDEFINED_URLS) if not should_skip(u)]
        self.dry_run  = dry_run
        self.crawl    = crawl   # if True, also discover and follow links
        self.scraped: list[dict] = []
        self.failed:  list[dict] = []
        self.visited: set[str]   = set()

        RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
        CLEAN_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    async def scrape_all(self):
        total = len(self.urls)
        print(f"\n{'='*60}")
        print(f"  FUNCTIOMED WEB SCRAPER")
        print(f"{'='*60}")
        print(f"  URLs to scrape : {total}")
        print(f"  Output dir     : {CLEAN_TEXT_DIR}")
        print(f"  Dry run        : {self.dry_run}")
        print(f"  Crawl mode     : {self.crawl}")
        print(f"{'='*60}\n")

        if self.dry_run:
            for i, url in enumerate(self.urls, 1):
                print(f"  [{i:>3}/{total}] {url}")
            print(f"\n  Total: {total} URLs")
            return []

        queue = list(self.urls)

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            context = await browser.new_context(
                viewport   = {"width": 1280, "height": 800},
                user_agent = "Mozilla/5.0 (compatible; FunctiomedBot/1.0)",
            )
            try:
                i = 0
                while queue:
                    url = queue.pop(0)
                    if url in self.visited:
                        continue
                    self.visited.add(url)
                    i += 1
                    print(f"  [{i:>3}] {url}")

                    new_links = await self._scrape_page(context, url)

                    # In crawl mode, add newly discovered links to queue
                    if self.crawl and new_links:
                        for link in new_links:
                            if link not in self.visited and link not in queue:
                                # Only follow same-domain links
                                if urlparse(link).netloc == urlparse(BASE_URL).netloc:
                                    queue.append(link)

            finally:
                await browser.close()

        self._save_manifest()

        print(f"\n{'='*60}")
        print(f"  DONE  ✅  scraped={len(self.scraped)}  failed={len(self.failed)}")
        print(f"{'='*60}\n")
        return self.scraped

    async def _scrape_page(self, context, url: str) -> list[str]:
        page = await context.new_page()
        new_links = []
        try:
            await page.goto(url, timeout=PAGE_TIMEOUT_MS, wait_until="networkidle")
            await page.wait_for_timeout(500)
            html = await page.content()

            data = extract_clean_text(html, url)
            data["scraped_at"] = datetime.now().isoformat()

            if data["word_count"] < MIN_WORD_COUNT:
                print(f"         ⏭  too short ({data['word_count']} words)")
                return []

            # Save raw HTML
            (RAW_HTML_DIR / url_to_filename(url, ".html")).write_text(html, encoding="utf-8")

            # Save clean text
            (CLEAN_TEXT_DIR / url_to_filename(url, ".txt")).write_text(data["text"], encoding="utf-8")

            print(f"         ✅ {data['word_count']} words — {data['title'][:50]}")
            self.scraped.append(data)

            # Discover links for crawl mode
            if self.crawl:
                new_links = await page.eval_on_selector_all(
                    "a[href]", "els => els.map(e => e.href)"
                )
                new_links = [normalize_url(l) for l in new_links if not should_skip(l)]

        except Exception as e:
            print(f"         ❌ {e}")
            self.failed.append({"url": url, "error": str(e)})
        finally:
            await page.close()

        return new_links

    def _save_manifest(self):
        manifest = {
            "scraped_at":   datetime.now().isoformat(),
            "total_pages":  len(self.scraped),
            "total_failed": len(self.failed),
            "pages":        [{"url": p["url"], "title": p["title"], "words": p["word_count"]} for p in self.scraped],
            "failed":       self.failed,
        }
        path = CLEAN_TEXT_DIR / "_scrape_manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  📋 Manifest: {path}")


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Functiomed Web Scraper")
    parser.add_argument("--url",     default=None,  help="Scrape a single URL instead of predefined list")
    parser.add_argument("--dry-run", action="store_true", help="Print URLs only, no scraping")
    parser.add_argument("--crawl",   action="store_true", help="Also discover and follow links from scraped pages")
    args = parser.parse_args()

    urls = [args.url] if args.url else PREDEFINED_URLS
    scraper = WebScraper(urls=urls, dry_run=args.dry_run, crawl=args.crawl)
    asyncio.run(scraper.scrape_all())