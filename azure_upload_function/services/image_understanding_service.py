"""
services/image_understanding_service.py

Analyzes an uploaded image using Azure OpenAI gpt-4o vision.

Feature: agentic-bot
Requirements: 5.2, 5.3, 5.4, 5.5, 5.6
"""

from __future__ import annotations

import base64
import logging

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
    from services.openai_service import _get_client, _deployment

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
    deployment = _deployment()

    # Detect person-identification queries
    _PERSON_ID_SIGNALS = [
        "who is", "who are", "identify this person", "identify the person",
        "identify this man", "identify this woman", "identify him", "identify her",
        "name this person", "tell me about this person", "who is this",
        "who is that", "which person", "what is his name", "what is her name",
        "whose photo", "whose image", "whose picture",
    ]
    q_lower = question.lower()
    is_person_query = any(sig in q_lower for sig in _PERSON_ID_SIGNALS)

    if is_person_query:
        # Step 1: Ask GPT-4o to extract all visible context clues — NOT to identify by face
        clue_prompt = (
            "Look at this image carefully. Extract ALL visible text, labels, name tags, "
            "watermarks, logos, captions, banners, or any written information present. "
            "Also describe the setting, any visible branding, event names, or organization names. "
            "Do NOT attempt to identify the person by their face. "
            "Return ONLY a JSON object: "
            "{\"visible_text\": \"...\", \"setting\": \"...\", \"name_clues\": \"...\"}"
        )
        clue_resp = client.chat.completions.create(
            model=deployment,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                    {"type": "text", "text": clue_prompt},
                ],
            }],
            max_tokens=400,
        )
        clue_raw = clue_resp.choices[0].message.content.strip()
        logging.info("analyze_image: clue extraction raw=%s", clue_raw[:200])

        # Parse clues
        import re, json as _json
        name_guess = ""
        clue_summary = clue_raw
        try:
            cleaned = re.sub(r"^```(?:json)?\s*", "", clue_raw)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
            clues = _json.loads(cleaned)
            name_clues   = clues.get("name_clues", "").strip()
            visible_text = clues.get("visible_text", "").strip()
            setting      = clues.get("setting", "").strip()
            # Best candidate name = name_clues first, then visible_text
            name_guess   = name_clues or visible_text
            clue_summary = f"Visible text: {visible_text}. Setting: {setting}. Name clues: {name_clues}."
        except Exception:
            pass

        # Step 2: Wikipedia lookup if we have a name candidate
        wiki_info = ""
        if name_guess and len(name_guess.strip()) > 2:
            try:
                import urllib.parse, requests as _req
                search_name = name_guess.strip().split("\n")[0][:80]
                wiki_url = (
                    "https://en.wikipedia.org/api/rest_v1/page/summary/"
                    + urllib.parse.quote(search_name.replace(" ", "_"))
                )
                wiki_resp = _req.get(wiki_url, timeout=6,
                                     headers={"User-Agent": "DataOrchBot/1.0"})
                if wiki_resp.status_code == 200:
                    wiki_data = wiki_resp.json()
                    wiki_info = wiki_data.get("extract", "")[:800]
                    logging.info("analyze_image: Wikipedia hit for '%s'", search_name)
                else:
                    # Try search API
                    search_url = (
                        "https://en.wikipedia.org/w/api.php?action=query&list=search"
                        f"&srsearch={urllib.parse.quote(search_name)}&srlimit=1&format=json&origin=*"
                    )
                    s_resp = _req.get(search_url, timeout=6,
                                      headers={"User-Agent": "DataOrchBot/1.0"})
                    if s_resp.status_code == 200:
                        results = s_resp.json().get("query", {}).get("search", [])
                        if results:
                            title = results[0]["title"].replace(" ", "_")
                            wiki_url2 = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
                            w2 = _req.get(wiki_url2, timeout=6,
                                          headers={"User-Agent": "DataOrchBot/1.0"})
                            if w2.status_code == 200:
                                wiki_info = w2.json().get("extract", "")[:800]
                                logging.info("analyze_image: Wikipedia search hit for '%s'", title)
            except Exception as wiki_exc:
                logging.warning("analyze_image: Wikipedia lookup failed: %s", wiki_exc)

        # Step 3: Compose final answer
        if wiki_info:
            answer = (
                f"Based on visible text and context in the image:\n\n"
                f"**Identified:** {name_guess}\n\n"
                f"**About this person (from Wikipedia):**\n{wiki_info}"
            )
        elif name_guess:
            answer = (
                f"Based on visible text in the image, the person appears to be: **{name_guess}**\n\n"
                f"No additional Wikipedia information was found."
            )
        else:
            answer = (
                f"I could not find a name or identifying text in this image.\n\n"
                f"Image context: {clue_summary}\n\n"
                f"If you know the person's name, try asking: 'Tell me about [Name]'"
            )
    else:
        # Standard image Q&A — not a person identification query
        response = client.chat.completions.create(
            model=deployment,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                    {"type": "text", "text": question},
                ],
            }],
            max_tokens=1000,
        )
        answer = response.choices[0].message.content

    logging.info(
        "analyze_image: image_id=%s ext=%s is_person_query=%s answer_len=%d",
        image_id, ext, is_person_query, len(answer),
    )
    return answer


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
