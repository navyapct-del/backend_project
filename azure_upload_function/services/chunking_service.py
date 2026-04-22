"""
chunking_service.py — 200-300 token chunks with sentence boundaries and 50-token overlap.
"""

import re
import logging

_CHARS_PER_TOKEN = 4
CHUNK_TOKENS     = 400          # target ~400 tokens (~1600 chars) — better context density
OVERLAP_TOKENS   = 80           # ~320 chars overlap — preserves cross-chunk context
CHUNK_SIZE       = CHUNK_TOKENS  * _CHARS_PER_TOKEN   # 1600 chars
OVERLAP_SIZE     = OVERLAP_TOKENS * _CHARS_PER_TOKEN  # 320 chars


def chunk_text(text: str, doc_id: str, filename: str) -> list[dict]:
    if not text or not text.strip():
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current, cur_len, idx = [], [], 0, 0

    for sentence in sentences:
        s_len = len(sentence)

        if s_len > CHUNK_SIZE:
            if current:
                _flush(chunks, current, idx, doc_id, filename)
                idx += 1
                current, cur_len = [], 0
            for start in range(0, s_len, CHUNK_SIZE - OVERLAP_SIZE):
                chunks.append(_make_chunk(sentence[start:start + CHUNK_SIZE], idx, doc_id, filename))
                idx += 1
            continue

        if cur_len + s_len > CHUNK_SIZE and current:
            _flush(chunks, current, idx, doc_id, filename)
            idx += 1
            overlap_text = " ".join(current)[-OVERLAP_SIZE:]
            current  = [overlap_text] if overlap_text.strip() else []
            cur_len  = len(overlap_text)

        current.append(sentence)
        cur_len += s_len + 1

    if current:
        _flush(chunks, current, idx, doc_id, filename)

    logging.info("chunking: '%s' → %d chunks (~%d tokens each)", filename, len(chunks), CHUNK_TOKENS)
    return chunks


def _flush(chunks, current, idx, doc_id, filename):
    text = " ".join(current).strip()
    if text:
        chunks.append(_make_chunk(text, idx, doc_id, filename))


def _make_chunk(text: str, idx: int, doc_id: str, filename: str) -> dict:
    return {
        "chunk_id":       f"{doc_id}_chunk_{idx}",
        "doc_id":         doc_id,
        "filename":       filename,
        "chunk_index":    idx,
        "text":           text,
        "token_estimate": len(text) // _CHARS_PER_TOKEN,
    }
