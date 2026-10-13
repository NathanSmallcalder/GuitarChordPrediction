"""Image scraper for building CNN training sets.

NOTE ON GOOGLE: Google Images serves a JS-only page and redirects scripted
requests to google.com/sorry/index with a reCAPTCHA. The `google` backend
handles this by opening a real Chrome window and pausing for YOU to clear the
check by hand, then scraping the session you unlocked. It keeps a browser
profile on disk, so in practice you solve it once and not again for a while.

Backends:
  bing       - Bing image search HTML. No key, full-res URLs, paginates.
               Highest volume, no browser needed; start here.
  openverse  - Openverse API. Real API, CC-licensed images, known provenance.
  commons    - Wikimedia Commons API. Small but very clean.
  google     - Selenium + visible Chrome, human-in-the-loop. Slowest, capped
               around a few hundred per query by how far the page will scroll.

Usage:
    python Scraper.py "g chord guitar" --limit 200 --out data
    python Scraper.py "g chord guitar" "c chord guitar" --engine openverse
    python Scraper.py "g chord guitar" --engine google --limit 150

Build a labelled dataset (one folder per class, ready for ImageFolder):
    python Scraper.py "G chord guitar" "C chord guitar" "D chord guitar" \
        --limit 300 --out data
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
API_UA = "GuitarCNN/1.0 (dataset research)"


# --------------------------------------------------------------------------
# Backends: each yields image URLs, up to `limit` of them
# --------------------------------------------------------------------------

def search_bing(query: str, limit: int, session: requests.Session):
    """Scrape image URLs from Bing's image search HTML.

    Each result tile carries a JSON blob in a data attribute; `murl` is the
    full-resolution source URL. Bing serves this in the initial HTML with no
    JS and no bot check, which is why this works where Google does not.
    """
    seen = set()
    first = 1
    while len(seen) < limit:
        resp = session.get(
            "https://www.bing.com/images/search",
            params={"q": query, "first": first, "count": 35},
            headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"},
            timeout=20,
        )
        resp.raise_for_status()

        urls = re.findall(r"murl&quot;:&quot;(.*?)&quot;", resp.text)
        fresh = 0
        for url in urls:
            url = url.replace("\\/", "/")
            if url in seen:
                continue
            seen.add(url)
            fresh += 1
            yield url
            if len(seen) >= limit:
                return
        if fresh == 0:
            break  # ran out of results for this query
        first += 35
        time.sleep(0.6)


def search_openverse(query: str, limit: int, session: requests.Session):
    """Openverse API - openly licensed images with real provenance."""
    page = 1
    got = 0
    while got < limit and page <= 20:
        resp = session.get(
            "https://api.openverse.org/v1/images/",
            params={"q": query, "page_size": 100, "page": page},
            headers={"User-Agent": API_UA},
            timeout=25,
        )
        if resp.status_code == 404:
            break  # past the last page
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            break
        for item in results:
            url = item.get("url")
            if url:
                yield url
                got += 1
                if got >= limit:
                    return
        page += 1
        time.sleep(0.5)


def search_commons(query: str, limit: int, session: requests.Session):
    """Wikimedia Commons image search via the MediaWiki API."""
    offset = 0
    got = 0
    while got < limit:
        resp = session.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query", "format": "json", "generator": "search",
                "gsrsearch": query, "gsrnamespace": "6", "gsrlimit": "50",
                "gsroffset": offset, "prop": "imageinfo",
                "iiprop": "url", "iiurlwidth": "1024",
            },
            headers={"User-Agent": API_UA},
            timeout=25,
        )
        resp.raise_for_status()
        pages = (resp.json().get("query") or {}).get("pages", {})
        if not pages:
            break
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            url = info.get("thumburl") or info.get("url")
            if url:
                yield url
                got += 1
                if got >= limit:
                    return
        offset += 50
        time.sleep(0.5)


PROFILE_DIR = Path(__file__).parent / ".chrome-profile"


def search_google_selenium(query: str, limit: int, _session: requests.Session):
    """Google Images via a real Chrome window, with you clearing the check.

    Headless Chrome is detected immediately, so this opens a visible window.
    If Google presents its "unusual traffic" page, the script blocks and waits
    for you to solve it in the browser and press Enter here. The profile is
    persisted to .chrome-profile/ so that clearance survives between runs.

    Yields a mix of full-resolution URLs harvested from the page's JS state
    and `data:` thumbnails straight out of the DOM. Thumbnails are typically
    200-400px, which is already above a 224x224 CNN input, so they are worth
    keeping rather than discarding.
    """
    from selenium import webdriver
    from selenium.webdriver.common.by import By

    opts = webdriver.ChromeOptions()
    opts.add_argument(f"--user-data-dir={PROFILE_DIR}")
    opts.add_argument("--window-size=1400,1100")
    opts.add_argument("--log-level=3")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])

    driver = webdriver.Chrome(options=opts)
    try:
        driver.get(
            "https://www.google.com/search"
            f"?q={requests.utils.quote(query)}&udm=2&hl=en"
        )
        time.sleep(3)

        if "/sorry/" in driver.current_url:
            print("\n  Google is showing its 'unusual traffic' check.")
            print("  Solve it in the Chrome window that just opened,")
            print("  then come back here and press Enter.")
            input("  [Enter] once the results are visible: ")

        if "/sorry/" in driver.current_url:
            print("  Still on the check page - skipping this query.")
            return

        # Scroll to pull in more results. Google lazy-loads in batches, so
        # stop early once the page height stops growing.
        last_height = 0
        for _ in range(30):
            driver.execute_script("window.scrollBy(0, 3000)")
            time.sleep(1.2)
            height = driver.execute_script("return document.body.scrollHeight")
            count = len(driver.find_elements(By.TAG_NAME, "img"))
            if count >= limit and height == last_height:
                break
            last_height = height

        seen = set()

        # Full-res URLs sitting in the page's embedded JS state.
        for url in re.findall(
            r'https?://(?!\w*\.?google|\w*\.?gstatic)[^"\\\s]+?'
            r"\.(?:jpg|jpeg|png|webp)",
            driver.page_source,
        ):
            if url not in seen:
                seen.add(url)
                yield url
                if len(seen) >= limit:
                    return

        # DOM thumbnails as a fallback, so a query still yields something
        # when the full-res URLs are not exposed.
        for img in driver.find_elements(By.TAG_NAME, "img"):
            src = img.get_attribute("src") or ""
            if not src.startswith(("data:image", "https://")):
                continue
            if src in seen:
                continue
            seen.add(src)
            yield src
            if len(seen) >= limit:
                return
    finally:
        driver.quit()


BACKENDS = {
    "bing": search_bing,
    "openverse": search_openverse,
    "commons": search_commons,
    "google": search_google_selenium,
}


# --------------------------------------------------------------------------
# Download + clean
# --------------------------------------------------------------------------

def fetch_one(url: str, out_dir: Path, min_px: int, max_px: int,
              seen_hashes: set, session: requests.Session) -> bool:
    """Download, validate, dedupe and normalize one image. True if kept.

    Accepts both http(s) URLs and `data:` URIs, since the Selenium backend
    pulls base64 thumbnails straight out of the DOM.
    """
    try:
        if url.startswith("data:image"):
            raw = base64.b64decode(url.split(",", 1)[1])
        else:
            resp = session.get(url, timeout=15, stream=True,
                               headers={"User-Agent": UA})
            resp.raise_for_status()
            raw = resp.raw.read(15 * 1024 * 1024, decode_content=True)

        # Dedupe on bytes before doing any decode work. Search engines return
        # the same image from many hosts, so this fires a lot.
        digest = hashlib.sha1(raw).hexdigest()
        if digest in seen_hashes:
            return False
        seen_hashes.add(digest)

        img = Image.open(io.BytesIO(raw))
        img.verify()                       # catches truncated/corrupt files
        img = Image.open(io.BytesIO(raw))  # verify() consumes the file object
        img = img.convert("RGB")

        if min(img.size) < min_px:
            return False
        if max(img.size) > max_px:
            img.thumbnail((max_px, max_px), Image.LANCZOS)

        img.save(out_dir / f"{digest[:16]}.jpg", "JPEG", quality=92)
        return True
    except Exception:
        return False  # dead links, HTML error pages, 403s, odd formats


def scrape(query: str, limit: int, out_dir: Path, engine: str = "bing",
           min_px: int = 128, max_px: int = 1024, workers: int = 16) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Over-request: a good third of URLs are dead, too small, or duplicates.
    urls = BACKENDS[engine](query, int(limit * 2.5), requests.Session())

    seen_hashes: set[str] = set()
    kept = 0
    dl = requests.Session()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(fetch_one, u, out_dir, min_px, max_px, seen_hashes, dl)
            for u in urls
        ]
        for fut in futures:
            if fut.result():
                kept += 1
                print(f"\r  {query!r}: {kept} kept", end="", flush=True)
            if kept >= limit:
                break
    print()
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("query", nargs="+", help="search term(s); one folder each")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default="data", help="output root directory")
    ap.add_argument("--engine", choices=list(BACKENDS), default="bing")
    ap.add_argument("--min-px", type=int, default=128,
                    help="discard images whose short side is under this")
    ap.add_argument("--max-px", type=int, default=1024)
    args = ap.parse_args()

    root = Path(args.out)
    total = 0
    for q in args.query:
        slug = re.sub(r"[^a-z0-9]+", "_", q.lower()).strip("_")
        total += scrape(q, args.limit, root / slug, args.engine,
                        args.min_px, args.max_px)
    print(f"done: {total} images under {root}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
