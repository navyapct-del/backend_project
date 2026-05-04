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

    _VISION_SYSTEM_PROMPT = (
        "You are a visual analysis assistant with broad knowledge of public figures, celebrities, "
        "athletes, politicians, and other well-known people. "
        "When given an image:\n"
        "1. If you recognise a well-known public figure, state their name and provide relevant "
        "information about them (profession, achievements, context).\n"
        "2. If you cannot identify the person with confidence, describe visible attributes: "
        "approximate age range, clothing, expression, posture, scene, and background.\n"
        "3. Always describe the full scene: objects, setting, colours, any visible text.\n"
        "Never refuse to analyse an image. If uncertain about identity, say so and describe "
        "what you can observe."
    )

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": _VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                    },
                    {"type": "text", "text": question},
                ],
            },
        ],
        max_tokens=1000,
    )

    answer: str = response.choices[0].message.content
    logging.info(
        "analyze_image: image_id=%s ext=%s answer_len=%d",
        image_id,
        ext,
        len(answer),
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
