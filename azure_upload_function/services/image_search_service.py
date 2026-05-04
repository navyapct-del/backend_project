"""
services/image_search_service.py

3-level image search pipeline:
  Level 1 (PRIMARY):        SearXNG self-hosted metasearch engine
  Level 2 (FALLBACK 1):     Wikimedia Commons API
  Level 3 (FALLBACK 2):     Wikipedia REST API (single image)

Response format (strict):
  {
    "type": "image",
    "source": "searxng" | "wikimedia_commons" | "wikipedia_fallback",
    "data": [{"url": "", "thumbnail": "", "title": ""}]
  }

Environment variables:
  SEARXNG_BASE_URL  - Base URL of your SearXNG instance
                      e.g. http://your-searxng-server:8080
                      If not set, SearXNG is skipped and Wikimedia is tried first.
  PIXABAY_API_KEY   - Optional Pixabay key (kept for backward compatibility, not used in pipeline)
"""

from __future__ import annotations

import logging
import os
import urllib.parse

import requests

MAX_RESULTS = 5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; DataOrchBot/1.0; "
        "+https://azure.microsoft.com)"
    )
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_images(query: str) -> list[dict]:
    """
    Return up to 5 images for *query* using a 3-level fallback pipeline.

    Level 1: SearXNG (if SEARXNG_BASE_URL is set)
    Level 2: Wikimedia Commons
    Level 3: Wikipedia REST API
    """
    clean = _clean_query(query)

    # ── Level 1: SearXNG ─────────────────────────────────────────────────
    searxng_url = os.environ.get("SEARXNG_BASE_URL", "").strip().rstrip("/")
    if searxng_url:
        try:
            results = _search_searxng(clean, searxng_url)
            if results:
                logging.info(
                    "search_images: SearXNG returned %d results for %r",
                    len(results), clean
                )
                return results
            logging.info(
                "search_images: SearXNG returned 0 results for %r — trying Wikimedia",
                clean
            )
        except Exception as exc:
            logging.warning(
                "search_images: SearXNG failed (%s) — trying Wikimedia", exc
            )
    else:
        logging.info(
            "search_images: SEARXNG_BASE_URL not set — skipping to Wikimedia"
        )

    # ── Level 2: Wikimedia Commons ───────────────────────────────────────
    try:
        results = _search_wikimedia_commons(clean)
        if results:
            logging.info(
                "search_images: Wikimedia Commons returned %d results for %r",
                len(results), clean
            )
            return results
        logging.info(
            "search_images: Wikimedia Commons returned 0 results for %r — trying Wikipedia",
            clean
        )
    except Exception as exc:
        logging.warning(
            "search_images: Wikimedia Commons failed (%s) — trying Wikipedia", exc
        )

    # ── Level 3: Wikipedia REST API ──────────────────────────────────────
    try:
        results = _search_wikipedia(clean)
        if results:
            logging.info(
                "search_images: Wikipedia returned %d results for %r",
                len(results), clean
            )
            return results
    except Exception as exc:
        logging.warning("search_images: Wikipedia failed (%s)", exc)

    return []


# ---------------------------------------------------------------------------
# Level 1: SearXNG
# ---------------------------------------------------------------------------

def _search_searxng(query: str, base_url: str) -> list[dict]:
    """
    Query a self-hosted SearXNG instance for image results.
    Docs: https://docs.searxng.org/dev/search_api.html
    """
    response = requests.get(
        f"{base_url}/search",
        params={
            "q":          query,
            "categories": "images",
            "format":     "json",
            "language":   "en",
        },
        headers=HEADERS,
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("results", [])[:MAX_RESULTS]:
        url   = item.get("img_src") or item.get("url", "")
        thumb = item.get("thumbnail_src") or item.get("thumbnail") or url
        title = item.get("title") or item.get("content") or query
        if url:
            results.append({"url": url, "thumbnail": thumb, "title": title})

    return results


# ---------------------------------------------------------------------------
# Level 2: Wikimedia Commons
# ---------------------------------------------------------------------------

def _search_wikimedia_commons(query: str) -> list[dict]:
    """
    Search Wikimedia Commons for images matching the query.

    For person/celebrity queries, appends "portrait" to improve photo hit rate.
    Uses a batch imageinfo call instead of one request per file.
    Returns up to MAX_RESULTS items.
    """
    # Detect person-like queries (two capitalised words or known person signals)
    words = query.split()
    is_person_query = (
        len(words) >= 2
        and all(w[0].isupper() for w in words if w.isalpha())
    )

    # Use a portrait-biased search term for people to surface actual photos
    search_term = f"{query} portrait" if is_person_query else query
    encoded = urllib.parse.quote(search_term)

    search_url = (
        "https://commons.wikimedia.org/w/api.php"
        f"?action=query&list=search&srsearch={encoded}&srnamespace=6"
        f"&srlimit={MAX_RESULTS * 5}&format=json&origin=*"
    )
    res = requests.get(search_url, headers=HEADERS, timeout=10)
    res.raise_for_status()

    # Collect candidate titles (photo extensions only; skip obvious non-photos)
    candidates = []
    for item in res.json().get("query", {}).get("search", []):
        title = item.get("title", "")
        if not title.startswith("File:"):
            continue
        lower = title.lower()
        # Only skip clearly non-photographic file types
        if lower.endswith(".svg"):
            continue
        # For non-person queries keep the stricter keyword filter;
        # for person queries only skip logos/icons (not maps/charts which rarely appear)
        if is_person_query:
            if any(s in lower for s in ["logo", "icon"]):
                continue
        else:
            if any(s in lower for s in ["logo", "icon", "flag", "map", "chart", "diagram"]):
                continue
        if not any(lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]):
            continue
        candidates.append(title)
        if len(candidates) >= MAX_RESULTS * 3:
            break

    if not candidates:
        return []

    # Batch imageinfo request — one API call for all candidates
    titles_param = "|".join(urllib.parse.quote(t) for t in candidates[:MAX_RESULTS * 2])
    info_url = (
        "https://commons.wikimedia.org/w/api.php"
        f"?action=query&titles={titles_param}"
        "&prop=imageinfo&iiprop=url|thumburl&iiurlwidth=400"
        "&format=json&origin=*"
    )
    try:
        info_res = requests.get(info_url, headers=HEADERS, timeout=12)
        info_res.raise_for_status()
        pages = info_res.json().get("query", {}).get("pages", {})
    except Exception:
        return []

    results = []
    for page in pages.values():
        ii = page.get("imageinfo", [{}])[0]
        full_url = ii.get("url", "")
        thumb    = ii.get("thumburl", full_url)
        raw_title = page.get("title", "")
        if full_url:
            results.append({
                "url":       full_url,
                "thumbnail": thumb,
                "title":     raw_title.replace("File:", ""),
            })
        if len(results) >= MAX_RESULTS:
            break

    return results


# ---------------------------------------------------------------------------
# Level 3: Wikipedia REST API
# ---------------------------------------------------------------------------

def _search_wikipedia(query: str) -> list[dict]:
    """
    Search Wikipedia for the query and return up to MAX_RESULTS page thumbnails.
    Tries multiple query variations to maximise hit rate for person queries.
    """
    variations = _build_query_variations(query)
    results: list[dict] = []
    seen_urls: set[str] = set()

    for variation in variations:
        if len(results) >= MAX_RESULTS:
            break
        try:
            search_url = (
                "https://en.wikipedia.org/w/api.php"
                f"?action=query&list=search&srsearch={urllib.parse.quote(variation)}"
                f"&srlimit={MAX_RESULTS}&format=json&origin=*"
            )
            search_res = requests.get(search_url, headers=HEADERS, timeout=10)
            search_res.raise_for_status()
            search_results = search_res.json().get("query", {}).get("search", [])

            for result in search_results:
                if len(results) >= MAX_RESULTS:
                    break
                title = result.get("title", "").replace(" ", "_")
                if not title:
                    continue
                try:
                    summary_url = (
                        f"https://en.wikipedia.org/api/rest_v1/page/summary/"
                        f"{urllib.parse.quote(title)}"
                    )
                    summary_res = requests.get(summary_url, headers=HEADERS, timeout=8)
                    if summary_res.status_code != 200:
                        continue
                    data      = summary_res.json()
                    thumbnail = data.get("thumbnail", {})
                    original  = data.get("originalimage", {})
                    thumb_url = thumbnail.get("source", "")
                    full_url  = original.get("source", thumb_url)
                    if full_url and full_url not in seen_urls:
                        seen_urls.add(full_url)
                        results.append({
                            "url":       full_url,
                            "thumbnail": thumb_url or full_url,
                            "title":     data.get("title", query),
                        })
                except Exception:
                    continue
        except Exception as exc:
            logging.warning(
                "search_images: Wikipedia variation %r failed: %s", variation, exc
            )
            continue

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_query(query: str) -> str:
    """Strip intent signal words so search engines get a clean term."""
    q = query.lower()
    for phrase in [
        "show me a image of", "show me images of", "show image of",
        "show images of", "find image of", "find images of",
        "find pictures of", "find picture of", "display photos of",
        "search for images of", "get image of", "fetch image of",
        "show a image of", "show me a photo of", "show photo of",
        "show me photo of", "show me picture of", "show picture of",
        "i want a image of", "i want image of", "give me image of",
        "show me", "image of", "images of", "photo of", "photos of",
        "picture of", "pictures of", "a image of", "an image of",
    ]:
        q = q.replace(phrase, "")
    # Strip role words that confuse stock photo searches
    for word in ["actor", "actress", "singer", "bollywood", "tollywood", "hollywood"]:
        q = q.replace(word, "")
    return " ".join(q.split()) or query


def _build_query_variations(query: str) -> list[str]:
    """Build Wikipedia search variations for better hit rate."""
    q = query.lower()

    # Strip role/descriptor words for people
    person_role_words = [
        "actor", "actress", "singer", "player", "ceo", "founder",
        "president", "minister", "director", "politician", "businessman",
        "entrepreneur", "scientist", "artist", "musician", "comedian",
        "tollywood", "bollywood", "hollywood", "indian", "american",
    ]
    clean = q
    for word in person_role_words:
        clean = clean.replace(word, "")
    clean = " ".join(clean.split())

    variations = []
    if clean and clean != q:
        variations.append(clean.title())

    if "wind turbine" in q or "wind farm" in q:
        return ["Wind turbine", "Wind farm", query]
    if "solar plant" in q or "solar farm" in q or "solar power" in q:
        return ["Solar power plant", "Photovoltaic power station", query]
    if "manufacturing plant" in q or "factory" in q:
        return ["Manufacturing", "Factory", query]
    if "cnc" in q or "milling machine" in q:
        return ["CNC milling machine", "Milling machine", query]
    if "conveyor belt" in q or "conveyor" in q:
        return ["Conveyor belt", query]
    if "turbine" in q:
        return ["Turbine", "Gas turbine", query]
    if "robot" in q:
        return ["Industrial robot", "Robot", query]

    if variations:
        return variations + [query]

    words = query.split()
    short = " ".join(words[:3]) if len(words) > 3 else query
    return [query, short]


# ---------------------------------------------------------------------------
# Kept for diagnostic endpoint compatibility
# ---------------------------------------------------------------------------

def _search_duckduckgo(query: str) -> list[dict]:
    """DuckDuckGo is blocked from Azure — always returns empty."""
    return []


def _search_pixabay(query: str, api_key: str) -> list[dict]:
    """Pixabay kept for backward compatibility — not used in main pipeline."""
    try:
        response = requests.get(
            "https://pixabay.com/api/",
            params={
                "key":        api_key,
                "q":          query,
                "image_type": "photo",
                "per_page":   3,
                "safesearch": "true",
                "order":      "popular",
            },
            timeout=10,
        )
        response.raise_for_status()
        results = []
        for item in response.json().get("hits", [])[:1]:
            url   = item.get("largeImageURL") or item.get("webformatURL", "")
            thumb = item.get("previewURL") or item.get("webformatURL", url)
            title = item.get("tags", query).split(",")[0].strip().title()
            if url:
                results.append({"url": url, "thumbnail": thumb, "title": title})
        return results
    except Exception:
        return []
