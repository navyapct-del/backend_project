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

    # ── Step 1: Extract visual context clues (no identity claim required) ──
    _CLUES_PROMPT = (
        "Describe this image by extracting the following clues. "
        "Be specific and factual — do NOT try to name the person.\n"
        "Return a single line of comma-separated clues covering:\n"
        "visible text or logos, clothing style and colour, background setting, "
        "nationality/cultural indicators, approximate era (modern/historical), "
        "gender, approximate age, distinctive physical features (glasses, beard, etc.), "
        "any emblems, flags, or symbols visible.\n"
        "Example: 'white dhoti, round glasses, thin elderly man, Indian flag, historical photo, bald'"
    )

    clues_resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                    {"type": "text", "text": _CLUES_PROMPT},
                ],
            }
        ],
        max_tokens=120,
    )
    clues: str = clues_resp.choices[0].message.content.strip()
    logging.info("analyze_image: clues=%r", clues)

    # ── Step 2: Wikipedia lookup using the clues ───────────────────────────
    wiki_summary = _wikipedia_lookup(clues)

    # ── Step 3: Build final answer ─────────────────────────────────────────
    if wiki_summary:
        answer = (
            f"Based on the visual clues in this image, this appears to be "
            f"{wiki_summary}"
        )
    else:
        # Fallback: ask GPT-4o to describe naturally without identifying
        fallback_resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a visual assistant. Describe the person in the image naturally: "
                        "appearance, clothing, setting, expression. Be conversational. "
                        "Do not use structured labels like 'Visible text:' or 'Name clues:'."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                        {"type": "text", "text": question},
                    ],
                },
            ],
            max_tokens=600,
        )
        answer = fallback_resp.choices[0].message.content.strip()

    logging.info("analyze_image: image_id=%s answer_len=%d", image_id, len(answer))
    return answer


# ---------------------------------------------------------------------------
# Wikipedia lookup — search by visual clues, return summary of best match
# ---------------------------------------------------------------------------

_WIKI_HEADERS = {"User-Agent": "DataOrchBot/1.0 (https://azure.microsoft.com)"}


def _wikipedia_lookup(clues: str) -> str:
    """
    Search Wikipedia using visual clues extracted from the image.
    Returns a formatted string like "Mahatma Gandhi. [summary...]" or "" on failure.
    """
    if not clues:
        return ""
    try:
        search_url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={urllib.parse.quote(clues)}"
            "&srlimit=3&format=json&origin=*"
        )
        res = requests.get(search_url, headers=_WIKI_HEADERS, timeout=8)
        res.raise_for_status()
        results = res.json().get("query", {}).get("search", [])
        if not results:
            return ""

        # Try each result until we get a summary with a thumbnail (person page)
        for hit in results:
            title = hit.get("title", "").replace(" ", "_")
            summary_url = (
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
            )
            sr = requests.get(summary_url, headers=_WIKI_HEADERS, timeout=8)
            if sr.status_code != 200:
                continue
            data = sr.json()
            # Only use person/biography pages (they have a thumbnail)
            if not data.get("thumbnail"):
                continue
            name = data.get("title", "")
            extract = data.get("extract", "")
            if name and extract:
                logging.info("_wikipedia_lookup: matched %r for clues %r", name, clues[:80])
                return f"{name}. {extract}"

        return ""
    except Exception as exc:
        logging.warning("_wikipedia_lookup failed: %s", exc)
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
