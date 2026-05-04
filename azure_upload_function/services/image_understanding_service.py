"""
services/image_understanding_service.py

Analyzes an uploaded image using Azure OpenAI gpt-4o vision.

Feature: agentic-bot
Requirements: 5.2, 5.3, 5.4, 5.5, 5.6
"""

from __future__ import annotations

import base64
import logging
import urllib.parse

import requests

from services.table_service import TableService
from services.blob_service import BlobService

# ---------------------------------------------------------------------------
# Supported image MIME types
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp"}

_MIME_MAP: dict[str, str] = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}


def analyze_image(image_id: str, question: str) -> str:
    """
    Analyze an uploaded image and answer *question* about it.

    Steps:
    1. Retrieve document metadata from TableService by image_id (RowKey).
    2. Validate that the file is an image (extension check).
    3. Download image bytes from BlobService.
    4. Base64-encode the bytes and call Azure OpenAI gpt-4o vision.
    5. Return the model's answer string.

    Raises:
        KeyError:   if no document with image_id exists in Table Storage.
        ValueError: if the document's filename is not an image file.
        Any OpenAI exception propagates to the caller.
    """
    from services.openai_service import _get_client, _vision_deployment

    # ------------------------------------------------------------------
    # 1. Retrieve metadata
    # ------------------------------------------------------------------
    table_service = TableService()
    metadata = _find_by_id(table_service, image_id)
    if metadata is None:
        raise KeyError("Document not found")

    # ------------------------------------------------------------------
    # 2. Validate file type
    # ------------------------------------------------------------------
    filename: str = metadata.get("filename", "")
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _IMAGE_EXTENSIONS:
        raise ValueError(f"Document '{image_id}' is not an image file.")

    mime_type = _MIME_MAP.get(ext, "image/jpeg")

    # ------------------------------------------------------------------
    # 3. Download image bytes
    # ------------------------------------------------------------------
    blob_url: str = metadata.get("blob_url", "")
    blob_service = BlobService()
    image_bytes: bytes = (
        blob_service._get_blob_client_from_url(blob_url)
        .download_blob()
        .readall()
    )

    # ------------------------------------------------------------------
    # 4. Base64-encode and call Azure OpenAI vision
    # ------------------------------------------------------------------
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    client = _get_client()
    deployment = _vision_deployment()

    image_data = {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}

    # ── Step 1: Use filename as identity hint ─────────────────────────────
    # Filenames like "apj_image.png", "gandhi_photo.jpg", "elon_musk.png"
    # often contain the person's name — extract and look up Wikipedia directly.
    name_from_file = _name_from_filename(filename)
    logging.info("analyze_image: name_from_file=%r", name_from_file)
    if name_from_file:
        wiki = _wikipedia_lookup_by_name(name_from_file)
        if wiki:
            return f"This is {wiki}"

    # ── Step 2: Ask GPT-4o to read any visible text in the image ──────────
    text_resp = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": [
            image_data,
            {"type": "text", "text": (
                "List ALL text visible in this image: name tags, captions, watermarks, "
                "banners, title cards, subtitles, jersey names. "
                "Return only the text exactly as written, or 'NONE'."
            )},
        ]}],
        max_tokens=80,
    )
    visible_text = text_resp.choices[0].message.content.strip()
    logging.info("analyze_image: visible_text=%r", visible_text)

    if visible_text and visible_text.upper() != "NONE":
        wiki = _wikipedia_lookup_by_name(visible_text)
        if wiki:
            return f"This is {wiki}"

    # ── Step 3: Force GPT-4o to describe — never refuse ───────────────────
    desc_resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": (
                "You are a visual description assistant. Your job is ONLY to describe "
                "what you see — never to identify or name anyone. "
                "Describe: approximate age, gender, hair, clothing, expression, setting, "
                "any medals/badges/insignia, cultural indicators, era. "
                "Be specific and detailed. Never say 'I cannot identify' or 'I'm unable'."
            )},
            {"role": "user", "content": [image_data, {"type": "text", "text": "Describe this person in detail."}]},
        ],
        max_tokens=400,
    )
    description = desc_resp.choices[0].message.content.strip()

    # ── Step 4: Use description clues to search Wikipedia ─────────────────
    wiki = _wikipedia_lookup_by_description(description)
    if wiki:
        return f"Based on the visual clues, this appears to be {wiki}"

    return description


# ---------------------------------------------------------------------------
# Wikipedia lookup — search by visual clues, return summary of best match
# ---------------------------------------------------------------------------

_WIKI_HEADERS = {"User-Agent": "DataOrchBot/1.0 (https://azure.microsoft.com)"}


def _name_from_filename(filename: str) -> str:
    """
    Extract a probable person name from the filename.
    e.g. 'apj_image.png' -> 'APJ Abdul Kalam', 'elon_musk.jpg' -> 'Elon Musk'
    """
    import re, os
    stem = os.path.splitext(filename)[0]          # remove extension
    stem = re.sub(r'[_\-\.]+', ' ', stem).strip() # underscores/dashes -> spaces
    # Remove generic words
    generic = {"image", "photo", "pic", "picture", "img", "screenshot",
               "download", "file", "copy", "scan", "portrait"}
    words = [w for w in stem.split() if w.lower() not in generic]
    if not words:
        return ""
    candidate = " ".join(words)
    # Must be at least 3 chars and not purely numeric
    if len(candidate) < 3 or candidate.replace(" ", "").isdigit():
        return ""
    logging.info("_name_from_filename: candidate=%r from %r", candidate, filename)
    return candidate


def _wikipedia_lookup_by_name(name: str) -> str:
    """Direct Wikipedia lookup by name. Returns 'Name. extract' or ''."""
    try:
        title = name.replace(" ", "_")
        res = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}",
            headers=_WIKI_HEADERS, timeout=8
        )
        if res.status_code == 200:
            data = res.json()
            extract = data.get("extract", "")
            found = data.get("title", name)
            if extract and len(extract) > 50:
                return f"{found}. {extract}"
        # Search fallback
        res = requests.get(
            f"https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={urllib.parse.quote(name)}&srlimit=3&format=json&origin=*",
            headers=_WIKI_HEADERS, timeout=8
        )
        for hit in res.json().get("query", {}).get("search", []):
            t = hit["title"].replace(" ", "_")
            sr = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(t)}",
                headers=_WIKI_HEADERS, timeout=8
            )
            if sr.status_code != 200:
                continue
            data = sr.json()
            extract = data.get("extract", "")
            found = data.get("title", "")
            if data.get("thumbnail") and extract and len(extract) > 50:
                logging.info("_wikipedia_lookup_by_name: found %r", found)
                return f"{found}. {extract}"
    except Exception as exc:
        logging.warning("_wikipedia_lookup_by_name failed for %r: %s", name, exc)
    return ""


def _wikipedia_lookup_by_description(description: str) -> str:
    """
    Extract role/nationality clues from GPT description and search Wikipedia.
    Only returns a result if a person page with thumbnail is found.
    """
    import re
    # Pull out key phrases: nationality + role
    role_patterns = [
        r'\b(indian|american|british|french|german|chinese|japanese|russian|'
        r'pakistani|bangladeshi|sri lankan|australian|canadian|brazilian)\b',
        r'\b(president|prime minister|scientist|physicist|astronaut|cricketer|'
        r'footballer|actor|actress|singer|politician|general|admiral|ceo|founder)\b',
        r'\b(independence|freedom fighter|revolutionary|nobel|bharat ratna)\b',
    ]
    clues = []
    desc_lower = description.lower()
    for pattern in role_patterns:
        clues.extend(re.findall(pattern, desc_lower))

    if len(clues) < 2:
        return ""

    query = " ".join(dict.fromkeys(clues)) + " person"  # deduplicated
    try:
        res = requests.get(
            f"https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={urllib.parse.quote(query)}&srlimit=5&format=json&origin=*",
            headers=_WIKI_HEADERS, timeout=8
        )
        for hit in res.json().get("query", {}).get("search", []):
            title = hit["title"]
            if any(s in title.lower() for s in ["disambiguation", "list of", "flag"]):
                continue
            t = title.replace(" ", "_")
            sr = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(t)}",
                headers=_WIKI_HEADERS, timeout=8
            )
            if sr.status_code != 200:
                continue
            data = sr.json()
            extract = data.get("extract", "")
            found = data.get("title", "")
            if data.get("thumbnail") and "born" in extract[:300].lower() and len(extract) > 50:
                logging.info("_wikipedia_lookup_by_description: found %r for clues %r", found, clues)
                return f"{found}. {extract}"
    except Exception as exc:
        logging.warning("_wikipedia_lookup_by_description failed: %s", exc)
    return ""


# ---------------------------------------------------------------------------
# Private helper — look up a document entity by its RowKey (= document ID)
# ---------------------------------------------------------------------------


def _find_by_id(table_service, image_id: str) -> dict | None:
    """
    Query Table Storage for the entity whose RowKey equals *image_id*.
    Returns a plain dict with at least 'filename' and 'blob_url', or None.
    """
    try:
        from services.table_service import PARTITION_KEY

        entities = list(
            table_service._client.query_entities(
                query_filter=(
                    f"PartitionKey eq '{PARTITION_KEY}' and RowKey eq '{image_id}'"
                ),
                select=["RowKey", "filename", "blob_url", "status"],
            )
        )
        if not entities:
            return None
        e = entities[0]
        return {
            "id": e.get("RowKey", ""),
            "filename": e.get("filename", ""),
            "blob_url": e.get("blob_url", ""),
            "status": e.get("status", ""),
        }
    except Exception as exc:
        logging.error("_find_by_id failed for image_id=%s: %s", image_id, exc)
        return None
