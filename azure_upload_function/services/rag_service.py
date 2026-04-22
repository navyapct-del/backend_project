"""
rag_service.py — Production-grade grounded answer generation.

Architecture:
  - Intent detection: routes to structured engine (charts/tables) or prose RAG
  - Context window: 2000 chars per chunk × up to 7 chunks = ~14k chars
  - Response format: JSON envelope with type, answer, optional table/chart data
  - Strict grounding: no hallucination, sources always cited
"""

import os
import re
import json
import logging
from openai import AzureOpenAI

_CHUNK_TEXT_LIMIT = 2000   # ~500 tokens per chunk — enough for dense context


class RAGService:
    def __init__(self):
        endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        api_key    = os.getenv("AZURE_OPENAI_API_KEY", "")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini")
        api_ver    = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

        if not endpoint or not api_key:
            raise EnvironmentError(
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set."
            )

        self._deployment = deployment
        self._client     = AzureOpenAI(
            api_key        = api_key,
            api_version    = api_ver,
            azure_endpoint = endpoint,
        )

    def generate_answer(self, query: str, documents: list[dict]) -> str:
        """
        Generate a grounded prose answer from retrieved chunks.
        Used only for conversational / factual questions.
        Returns plain text (not JSON).
        """
        if not documents:
            return "Not enough information in the available documents to answer this question."

        context_parts = []
        citation_list = []
        seen = set()

        for i, chunk in enumerate(documents, start=1):
            text     = (chunk.get("text") or chunk.get("content") or "").strip()[:_CHUNK_TEXT_LIMIT]
            filename = chunk.get("filename", f"Document {i}")
            score    = chunk.get("score", 0)
            if text:
                context_parts.append(f"[Chunk {i} — {filename} (score: {score:.2f})]\n{text}")
            if filename not in seen:
                seen.add(filename)
                citation_list.append(filename)

        if not context_parts:
            return "Not enough information in the available documents to answer this question."

        context = "\n\n".join(context_parts)

        prompt = (
            "You are a precise AI assistant. Answer the question using ONLY the context below.\n"
            "If the answer is not in the context, say exactly: 'Not enough information in the documents.'\n\n"
            "FORMAT:\n"
            "- Use numbered bullet points for multi-part answers\n"
            "- Each point on its own line\n"
            "- Be concise and factual\n"
            f"- End with: Sources: {', '.join(citation_list)}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )

        try:
            response = self._client.chat.completions.create(
                model       = self._deployment,
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.1,
                max_tokens  = 800,
            )
            answer = response.choices[0].message.content.strip()
            answer = re.sub(r'\s+(\d+\.)\s+', r'\n\1 ', answer).strip()
            logging.info("RAGService.generate_answer: %d chars from %d chunks",
                         len(answer), len(documents))
            return answer
        except Exception:
            logging.exception("RAGService.generate_answer failed.")
            raise
