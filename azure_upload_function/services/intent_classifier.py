"""
Intent classifier for the Agentic Bot.

Classifies a user query into one of five intents using a two-stage approach:
  Stage 1: Fast keyword matching (priority order)
  Stage 2: Optional LLM fallback for ambiguous short queries

Valid intents: image_search | image_qa | general_qa | document_qa | followup
"""

from __future__ import annotations

import logging
import os

# ---------------------------------------------------------------------------
# Module-level signal constants
# ---------------------------------------------------------------------------

IMAGE_SEARCH_SIGNALS = [
    "show images of",
    "show image of",
    "show me images",
    "show me image",
    "find images of",
    "find image of",
    "find pictures of",
    "find picture of",
    "display photos of",
    "display images of",
    "show me pictures",
    "show pictures of",
    "search for images",
    "search images",
    "image of",
    "images of",
    "photo of",
    "photos of",
    "picture of",
    "pictures of",
    "show photos",
    "find photos",
    "get images",
    "fetch images",
]

IMAGE_QA_SIGNALS = [
    "what is in this image",
    "what is in the image",
    "describe this image",
    "describe the image",
    "describe this photo",
    "what does this image show",
    "analyze this image",
    "analyse this image",
    "what is in the picture",
    "describe this chart",
    "what is shown in",
    "explain this image",
    "read this image",
    "summarize this image",
    "summarise this image",
    "summarize the image",
    "what is this image",
    "what is this photo",
    "tell me about this image",
    "tell me about the image",
    "what can you see",
    "what do you see",
    "identify this",
    "identify the image",
    "extract text from",
    "read the text in",
    "ocr this",
    "what is happening in",
    "what is shown",
    # Person / scene description signals
    "tell about the person",
    "tell me about the person",
    "describe the person",
    "who is in the image",
    "who is in this image",
    "who is in the photo",
    "who is in this photo",
    "what does the person",
    "describe the scene",
    "what is the person",
    "tell about this photo",
    "tell about this picture",
    "describe this picture",
    "what is in this photo",
    "what is in this picture",
    "analyze this photo",
    "analyse this photo",
    "explain this photo",
    "explain this picture",
    "what do you see in",
    "describe what you see",
]

DOCUMENT_QA_SIGNALS = [
    "in the document",
    "according to the report",
    "in the file",
    "from the uploaded",
    "in the pdf",
    "in the spreadsheet",
    "the document says",
    "based on the document",
    "in the report",
    "from the document",
    "from the report",
    "from the file",
    "from my documents",
    "from the documents",
    "i have uploaded",
    "uploaded documents",
    "in my documents",
    "from uploaded",
    "the uploaded",
]

FOLLOWUP_SIGNALS = [
    "tell me more",
    "elaborate",
    "explain further",
    "what about that",
    "and what about",
    "more details",
    "can you expand",
    "go on",
    "what else",
    "continue",
    "what did you mean",
]

FOLLOWUP_PRONOUNS = {"this", "that", "it", "he", "she", "they", "those", "these"}

# ---------------------------------------------------------------------------
# Valid intent set (used for validation)
# ---------------------------------------------------------------------------

VALID_INTENTS = frozenset(
    {"image_search", "image_qa", "general_qa", "document_qa", "followup"}
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_intent(
    query: str,
    session_context: list[dict],
    image_id: str | None = None,
) -> str:
    """
    Classify a user query into one of five intents.

    Args:
        query:           Raw user query string.
        session_context: Last N session turns [{query, response, timestamp}, ...].
        image_id:        Optional image ID present in the current request.

    Returns:
        One of: "image_search" | "image_qa" | "general_qa" | "document_qa" | "followup"
    """
    result = _stage1_keyword(query, session_context, image_id)

    # Stage 2: optional LLM fallback for ambiguous short general_qa queries
    if result == "general_qa":
        result = _stage2_llm_fallback(query, session_context, result)

    # Safety net — should never be needed, but guarantees the contract
    if result not in VALID_INTENTS:
        logging.warning(
            "classify_intent: unexpected result %r — falling back to general_qa", result
        )
        result = "general_qa"

    return result


# ---------------------------------------------------------------------------
# Stage 1: keyword matching
# ---------------------------------------------------------------------------


def _stage1_keyword(
    query: str,
    session_context: list[dict],
    image_id: str | None,
) -> str:
    """Fast O(n) keyword scan evaluated in priority order."""
    q_lower = query.lower() if query else ""

    # 1. Image search — highest priority
    if any(signal in q_lower for signal in IMAGE_SEARCH_SIGNALS):
        return "image_search"

    # 2. Image Q&A — only when an image is attached
    if image_id is not None and any(signal in q_lower for signal in IMAGE_QA_SIGNALS):
        return "image_qa"

    # 3. Document Q&A
    if any(signal in q_lower for signal in DOCUMENT_QA_SIGNALS):
        return "document_qa"

    # 4. Follow-up — explicit signal OR short pronoun-bearing query with prior context
    if any(signal in q_lower for signal in FOLLOWUP_SIGNALS):
        return "followup"

    words = q_lower.split()
    if (
        len(words) <= 5
        and bool(session_context)
        and any(pronoun in words for pronoun in FOLLOWUP_PRONOUNS)
    ):
        return "followup"

    # 5. If image is attached and no other intent matched — treat as image Q&A
    if image_id is not None:
        return "image_qa"

    # 6. Default
    return "general_qa"


# ---------------------------------------------------------------------------
# Stage 2: LLM fallback (optional, gated by env var)
# ---------------------------------------------------------------------------

_LLM_FALLBACK_PROMPT_SYSTEM = (
    "You are an intent classifier. Classify the user query into exactly one of: "
    "image_search, image_qa, general_qa, document_qa, followup\n"
    "Return ONLY the intent label, nothing else."
)


def _stage2_llm_fallback(
    query: str,
    session_context: list[dict],
    keyword_result: str,
) -> str:
    """
    Call Azure OpenAI for intent classification when:
      - ENABLE_LLM_INTENT_FALLBACK == "true"
      - keyword result is general_qa
      - query word count <= 8

    Falls back to keyword_result on any exception.
    """
    if os.environ.get("ENABLE_LLM_INTENT_FALLBACK", "false").lower() != "true":
        return keyword_result

    words = (query or "").split()
    if len(words) > 8:
        return keyword_result

    try:
        from services.openai_service import _get_client, _deployment  # type: ignore

        last_turn_summary = ""
        if session_context:
            last = session_context[-1]
            last_turn_summary = str(last.get("query", ""))[:200]

        user_message = query
        if last_turn_summary:
            user_message = f"{query}\nContext (last turn): {last_turn_summary}"

        client = _get_client()
        resp = client.chat.completions.create(
            model=_deployment(),
            messages=[
                {"role": "system", "content": _LLM_FALLBACK_PROMPT_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=20,
        )
        llm_intent = resp.choices[0].message.content.strip().lower()

        if llm_intent in VALID_INTENTS:
            logging.info(
                "classify_intent: LLM fallback changed %r → %r for query %r",
                keyword_result,
                llm_intent,
                query[:80],
            )
            return llm_intent

        logging.warning(
            "classify_intent: LLM returned unknown intent %r — keeping keyword result",
            llm_intent,
        )
        return keyword_result

    except Exception:
        logging.exception(
            "classify_intent: LLM fallback failed — using keyword result %r", keyword_result
        )
        return keyword_result
