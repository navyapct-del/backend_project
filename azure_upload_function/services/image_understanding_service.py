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

    # ── Step 1: Extract any visible text / name from the image ────────────
    text_resp = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": [
            image_data,
            {"type": "text", "text": (
                "Read ALL visible text in this image exactly as written: "
                "captions, watermarks, name tags, banners, jersey names, title cards, subtitles. "
                "Return only the text you see, or 'NONE' if there is no text."
            )},
        ]}],
        max_tokens=60,
    )
    visible_text = text_resp.choices[0].message.content.strip()
    logging.info("analyze_image: visible_text=%r", visible_text)

    # ── Step 2: If text found, search Wikipedia for the name ──────────────
    if visible_text and visible_text.upper() != "NONE" and len(visible_text) > 2:
        wiki = _wikipedia_lookup_by_name(visible_text)
        if wiki:
            return f"This is {wiki}"

    # ── Step 3: Ask GPT-4o to describe the person in detail ───────────────
    # GPT-4o cannot identify faces but can describe appearance, clothing,
    # setting, and any contextual clues that help the user understand the image.
    desc_resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": (
                "You are a visual assistant. When shown an image of a person:\n"
                "1. Describe their physical appearance in detail: age, build, hair, facial features\n"
                "2. Describe their clothing and any visible accessories or medals\n"
                "3. Describe the setting and background\n"
                "4. Note any cultural, professional, or historical context clues\n"
                "5. If the image has any text, captions, or labels, mention them\n"
                "Be specific and detailed. Do NOT say 'I cannot identify' — just describe what you see."
            )},
            {"role": "user", "content": [image_data, {"type": "text", "text": question}]},
        ],
        max_tokens=500,
    )
    answer = desc_resp.choices[0].message.content.strip()
    logging.info("analyze_image: image_id=%s answer_len=%d", image_id, len(answer))
    return answer


# ---------------------------------------------------------------------------
# Wikipedia lookup — search by visual clues, return summary of best match
# ---------------------------------------------------------------------------

_WIKI_HEADERS = {"User-Agent": "DataOrchBot/1.0 (https://azure.microsoft.com)"}


def _wikipedia_lookup_by_name(name: str) -> str:
    """
    Look up a person by name on Wikipedia.
    Returns "Name. [extract]" or "" if not found.
    """
    try:
        title = name.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
        res = requests.get(url, headers=_WIKI_HEADERS, timeout=8)
        if res.status_code == 200:
            data = res.json()
            extract = data.get("extract", "")
            found_name = data.get("title", name)
            if extract and len(extract) > 50:
                logging.info("_wikipedia_lookup_by_name: found %r", found_name)
                return f"{found_name}. {extract}"

        # Fallback: search API if direct lookup missed
        search_url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={urllib.parse.quote(name)}"
            "&srlimit=3&format=json&origin=*"
        )
        res = requests.get(search_url, headers=_WIKI_HEADERS, timeout=8)
        res.raise_for_status()
        hits = res.json().get("query", {}).get("search", [])
        for hit in hits:
            t = hit.get("title", "").replace(" ", "_")
            sr = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(t)}",
                headers=_WIKI_HEADERS, timeout=8
            )
            if sr.status_code != 200:
                continue
            data = sr.json()
            extract = data.get("extract", "")
            found_name = data.get("title", "")
            if data.get("thumbnail") and extract and len(extract) > 50:
                logging.info("_wikipedia_lookup_by_name: search found %r", found_name)
                return f"{found_name}. {extract}"
    except Exception as exc:
        logging.warning("_wikipedia_lookup_by_name failed for %r: %s", name, exc)
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
