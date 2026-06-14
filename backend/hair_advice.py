"""
hair_advice.py
==============
Fetches dermatologist-approved advice from AAD and Healthline
for each hair type. Scrapes ONCE on first run, saves to cache file,
then serves from cache instantly on every prediction after that.

The frontend never knows where the data came from — app.py just
includes it in the prediction response JSON.

Usage (from app.py):
    from hair_advice import get_advice
    advice = get_advice("hairfall")
    # returns: { "precautions": [...], "tips": [...], "source": "aad.org" }
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import time

# ===== CACHE FILE =====
CACHE_FILE = "../model/hair_advice_cache.json"

# ===== SOURCE URLS PER HAIR TYPE =====
# AAD = gold standard (dermatologist-written, no ads)
# Healthline = medically reviewed, good for hair type conditions
SOURCE_URLS = {
    "hairfall":   "https://www.aad.org/public/diseases/hair-loss/treatment/tips",
    "bald":       "https://www.aad.org/public/diseases/hair-loss/treatment/tips",
    "dry":        "https://www.aad.org/public/everyday-care/hair-scalp-care/hair/healthy-hair-tips",
    "frizzy":     "https://www.aad.org/public/everyday-care/hair-scalp-care/hair/styling-tips",
    "healthy":    "https://www.aad.org/public/everyday-care/hair-scalp-care/hair/healthy-hair-tips",
    "straight":   "https://www.aad.org/public/everyday-care/hair-scalp-care/hair/healthy-hair-tips",
    "wavy":       "https://www.aad.org/public/everyday-care/hair-scalp-care/hair/healthy-hair-tips",
    "curly":      "https://www.aad.org/public/everyday-care/hair-scalp-care/hair/care-african-american",
    "kinky":      "https://www.aad.org/public/everyday-care/hair-scalp-care/hair/care-african-american",
    "dreadlocks": "https://www.aad.org/public/everyday-care/hair-scalp-care/hair/care-african-american",
    "notbald":    "https://www.aad.org/public/everyday-care/hair-scalp-care/hair/healthy-hair-tips",
}

# ===== REQUEST HEADERS =====
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ===== SCRAPE ONE URL =====
def scrape_aad(url: str) -> list[str]:
    """
    Scrapes bullet-point tips from an AAD page.
    Returns a list of tip strings (max 6).
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        tips = []

        # AAD uses <li> tags for their tip lists
        for li in soup.find_all("li"):
            text = li.get_text(strip=True)
            # Filter: must be a real sentence (not nav links, not too short)
            if len(text) > 40 and not text.startswith("Alopecia") and "dermatologist" not in text.lower()[:20]:
                tips.append(text)
            if len(tips) >= 6:
                break

        return tips if tips else []

    except Exception as e:
        print(f"  ⚠️  Could not scrape {url}: {e}")
        return []


# ===== BUILD FULL CACHE =====
def build_cache() -> dict:
    """
    Scrapes all hair types and saves to cache file.
    Only called once (or when cache is missing/outdated).
    """
    print("🌐 Building hair advice cache from AAD...")
    cache = {}

    for hair_type, url in SOURCE_URLS.items():
        print(f"  Fetching: {hair_type} → {url}")
        tips = scrape_aad(url)

        cache[hair_type] = {
            "tips_from_web": tips,
            "source": "aad.org",
            "source_url": url,
            "scraped_at": time.strftime("%Y-%m-%d")
        }

        time.sleep(1)  # polite delay between requests

    # Save cache
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"✅ Cache saved to {CACHE_FILE}")
    return cache


# ===== LOAD CACHE =====
def load_cache() -> dict:
    """
    Loads cache from file. Rebuilds if missing.
    """
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    else:
        print("📭 No cache found — building now...")
        return build_cache()


# ===== MAIN API FUNCTION =====
# Cache is loaded into memory once when the module is first imported
_cache = None

def get_advice(hair_type: str) -> dict:
    """
    Returns web-sourced advice for the given hair type.
    Uses in-memory cache → file cache → scrape (in that order).

    Args:
        hair_type: e.g. "hairfall", "dry", "healthy"

    Returns:
        {
            "tips_from_web": ["tip1", "tip2", ...],
            "source": "aad.org",
            "source_url": "https://...",
            "scraped_at": "2024-01-01"
        }
    """
    global _cache

    # Load into memory if not already loaded
    if _cache is None:
        _cache = load_cache()

    # Normalize key
    key = hair_type.lower().strip()

    # Return cached advice or empty fallback
    return _cache.get(key, {
        "tips_from_web": [],
        "source": "aad.org",
        "source_url": SOURCE_URLS.get(key, "https://www.aad.org"),
        "scraped_at": "N/A"
    })


# ===== FORCE REFRESH =====
def refresh_cache():
    """
    Forces a full re-scrape. Call this manually if you want
    to update the cache with fresh content from AAD.
    Run: python hair_advice.py
    """
    global _cache
    _cache = build_cache()
    print("✅ Cache refreshed!")


# ===== RUN DIRECTLY TO REBUILD CACHE =====
if __name__ == "__main__":
    refresh_cache()
    print("\n📋 Sample output for 'hairfall':")
    print(json.dumps(get_advice("hairfall"), indent=2))