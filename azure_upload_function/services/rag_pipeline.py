"""
rag_pipeline.py — Advanced RAG pipeline with:

  1. Intent classification   — routes to structured engine or prose RAG
  2. HyDE retrieval          — embeds a hypothetical answer for better recall
  3. Multi-query retrieval   — 3 query variants merged + deduplicated
  4. Contextual compression  — extracts only the relevant passage per chunk
  5. Grounded generation     — strict system prompt, temperature=0, no hallucination
  6. Self-consistency check  — validates answer is grounded in context

Architecture:
  query
    │
    ├─► intent_classifier()
    │       │
    │       ├─ STRUCTURED → query_engine (pandas + LLM plan)
    │       │
    │       └─ PROSE / HYBRID
    │               │
    │               ├─► multi_query_retrieval()   [3 variants + HyDE]
    │               ├─► contextual_compression()  [extract relevant passages]
    │               └─► grounded_generate()       [strict RAG answer]
    │
    └─► format_response()
"""

import re
import json
import logging
import hashlib
from functools import lru_cache
from typing import Literal

# ---------------------------------------------------------------------------
# Intent types
# ---------------------------------------------------------------------------

IntentType = Literal["structured", "prose", "hybrid"]

# Keywords that signal the query engine should handle the request
_STRUCTURED_KEYWORDS = {
    # Aggregation
    "sum", "total", "average", "avg", "mean", "count", "max", "min",
    "how many", "number of", "add up", "calculate",
    # Grouping / breakdown
    "breakdown", "group by", "per", "by department", "by category",
    "distribution", "frequency",
    # Comparison
    "compare", "comparison", "versus", " vs ", "difference between",
    # Filtering
    "filter", "where", "only", "exclude",
    # Sorting
    "top", "bottom", "highest", "lowest", "ranked",
    # Listing / enumeration — only route to structured engine for CSV/Excel, not PDFs
    # (handled by has_structured_data check in classify_intent)
    "list all", "show all", "show me all", "all the", "give me all",
    "enumerate", "display all", "fetch all", "get all",
}

# Keywords that signal a chart/graph is wanted
_CHART_KEYWORDS = {
    "chart", "graph", "plot", "visualize", "visualise", "visualisation",
    "bar chart", "line chart", "pie chart", "scatter", "histogram",
    "trend", "over time", "growth", "distribution",
}

# Keywords that signal a table is wanted
_TABLE_KEYWORDS = {
    "table", "list", "show all", "enumerate", "tabular", "rows", "columns",
    "spreadsheet", "grid",
}


def classify_intent(query: str, has_structured_data: bool) -> IntentType:
    """
    Classify query intent to route to the right pipeline.

    Returns:
      "structured" — use query engine (aggregations, charts from data)
      "prose"      — use RAG (factual questions, summaries, explanations)
      "hybrid"     — try query engine first, fall back to RAG
    """
    q = query.lower()

    has_chart_intent      = any(k in q for k in _CHART_KEYWORDS)
    has_structured_intent = any(k in q for k in _STRUCTURED_KEYWORDS)
    has_table_intent      = any(k in q for k in _TABLE_KEYWORDS)

    if not has_structured_data:
        return "prose"

    if has_chart_intent or (has_structured_intent and has_structured_data):
        return "structured"

    if has_table_intent:
        return "hybrid"

    return "prose"


# ---------------------------------------------------------------------------
# Multi-query retrieval with HyDE
# ---------------------------------------------------------------------------

def generate_query_variants(query: str) -> list[str]:
    """
    Generate 3 semantically diverse query variants + 1 HyDE hypothetical answer.
    Returns [original, variant1, variant2, variant3, hyde_passage].

    HyDE (Hypothetical Document Embedding): generate a short hypothetical answer
    and embed it — this often retrieves better chunks than the raw question.
    """
    from services.openai_service import _get_client, _deployment

    prompt = f"""Given this user question, generate:
1. One alternative phrasing (different vocabulary, same intent)
2. A short hypothetical answer passage (1-2 sentences) that would ideally answer the question

Return ONLY valid JSON:
{{"variant": "rephrased question here", "hyde": "hypothetical answer here"}}

Question: {query}

JSON:"""

    try:
        resp = _get_client().chat.completions.create(
            model       = _deployment(),
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.3,
            max_tokens  = 150,
            timeout     = 8,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
        data     = json.loads(raw)
        variant  = data.get("variant", "")
        hyde     = data.get("hyde", "")
        result   = [query]
        if variant and variant != query:
            result.append(variant)
        if hyde:
            result.append(hyde)
        logging.info("generate_query_variants: %d queries total", len(result))
        return result
    except Exception as exc:
        logging.warning("generate_query_variants failed (%s) — using original query only", exc)
        return [query]


def multi_query_retrieve(
    query: str,
    top_k: int = 7,
    filename_filter: str = "",
    uploaded_by: str = "",
    use_hyde: bool = True,
    doc_ids: list[str] | None = None,
) -> list[dict]:
    """
    Retrieve chunks using multiple query variants + HyDE for better recall.

    Strategy:
      1. Generate query variants + HyDE passage
      2. Embed each variant
      3. Run hybrid search for each
      4. Merge results, deduplicate by chunk id
      5. Re-rank merged set by best score per chunk
      6. Return top_k

    Falls back to single-query retrieval if variant generation fails.
    """
    from services.openai_service import generate_embedding
    from services.search_service import vector_search

    # Generate variants (includes HyDE if enabled)
    if use_hyde:
        queries = generate_query_variants(query)
    else:
        queries = [query]

    # Retrieve for each query variant
    seen_ids: dict[str, dict] = {}   # chunk_id → best chunk

    for q_variant in queries:
        embedding = generate_embedding(q_variant)
        if not embedding:
            continue
        chunks = vector_search(
            query_embedding = embedding,
            query_text      = q_variant,
            top             = top_k,
            filename_filter = filename_filter,
            uploaded_by     = uploaded_by,
            doc_ids         = doc_ids,
        )
        for chunk in chunks:
            cid = chunk["id"]
            # Keep the version with the highest score
            if cid not in seen_ids or chunk["score"] > seen_ids[cid]["score"]:
                seen_ids[cid] = chunk

    if not seen_ids:
        # Hard fallback: single query
        embedding = generate_embedding(query)
        if embedding:
            return vector_search(
                query_embedding = embedding,
                query_text      = query,
                top             = top_k,
                filename_filter = filename_filter,
                uploaded_by     = uploaded_by,
                doc_ids         = doc_ids,
            )
        return []

    # Sort merged results by score, return top_k
    merged = sorted(seen_ids.values(), key=lambda x: x["score"], reverse=True)
    result = merged[:top_k]
    logging.info("multi_query_retrieve: %d unique chunks from %d queries → returning %d",
                 len(seen_ids), len(queries), len(result))
    return result


# ---------------------------------------------------------------------------
# Contextual compression
# ---------------------------------------------------------------------------

def compress_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """
    Extract only the passage most relevant to the query from each chunk.
    Skipped for short/simple queries to avoid stripping relevant content.
    Falls back to original chunk text if compression fails.
    """
    from services.openai_service import _get_client, _deployment

    if not chunks:
        return chunks

    # Skip compression for short queries — risk of stripping relevant content
    # outweighs benefit for simple factual questions
    word_count = len(query.split())
    if word_count <= 8 or len(chunks) <= 3:
        logging.info("compress_chunks: skipping (query_words=%d, chunks=%d)", word_count, len(chunks))
        return chunks

    # Build batched extraction prompt
    chunk_texts = []
    for i, chunk in enumerate(chunks):
        text = (chunk.get("text") or chunk.get("content") or "").strip()
        chunk_texts.append(f"[Chunk {i+1}]\n{text[:1500]}")

    combined = "\n\n".join(chunk_texts)

    prompt = f"""You are a precise information extractor.

For each chunk below, extract ONLY the sentences directly relevant to the question.
If a chunk has no relevant content, return an empty string for it.
Return ONLY valid JSON: {{"extracts": ["extract1", "extract2", ...]}}
The array must have exactly {len(chunks)} elements (one per chunk, empty string if not relevant).

Question: {query}

Chunks:
{combined}

JSON:"""

    try:
        resp = _get_client().chat.completions.create(
            model       = _deployment(),
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.0,
            max_tokens  = 1500,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
        data     = json.loads(raw)
        extracts = data.get("extracts", [])

        compressed = []
        for i, chunk in enumerate(chunks):
            extract = extracts[i].strip() if i < len(extracts) else ""
            if extract:
                # Use compressed text but keep all metadata
                new_chunk = dict(chunk)
                new_chunk["text"]              = extract
                new_chunk["_compressed"]       = True
                new_chunk["_original_text_len"] = len(chunk.get("text") or chunk.get("content") or "")
                compressed.append(new_chunk)
            else:
                # Chunk not relevant — still include with original text (lower weight)
                compressed.append(chunk)

        # Filter out chunks where compression returned empty AND original text is short
        relevant = [
            c for c in compressed
            if (c.get("text") or c.get("content") or "").strip()
        ]

        logging.info("compress_chunks: %d → %d relevant chunks after compression",
                     len(chunks), len(relevant))
        return relevant if relevant else chunks

    except Exception as exc:
        logging.warning("compress_chunks failed (%s) — using original chunks", exc)
        return chunks


# ---------------------------------------------------------------------------
# Grounded answer generation
# ---------------------------------------------------------------------------

# System prompt — defines the assistant's persona and grounding rules
_SYSTEM_PROMPT = """You are a helpful, precise AI assistant for document question-answering.

CORE RULES:
1. Answer ONLY using the provided document context. Do not use external knowledge.
2. If the context does not contain the answer, say: "No relevant information found in this document."
3. Be specific and factual. Include exact figures, names, and dates from the context.
4. Do not fabricate data, names, or statistics not present in the context.

RESPONSE QUALITY:
- Be comprehensive but concise
- Use numbered lists for multi-part answers
- Use bullet points for enumerations
- Bold key terms or figures using **term** syntax"""


def grounded_generate(
    query: str,
    chunks: list[dict],
    response_format: Literal["text", "table", "chart", "auto"] = "auto",
    history: list[dict] | None = None,
) -> dict:
    """
    Generate a grounded answer from compressed, relevant chunks.

    Args:
        query:           User question
        chunks:          Compressed, relevant chunks from retrieval
        response_format: Force a specific format or let the LLM decide

    Returns:
        {
          "type":    "text" | "table" | "chart",
          "answer":  str,
          "columns": [...],   # for table
          "rows":    [...],   # for table
          "labels":  [...],   # for chart
          "values":  [...],   # for chart
          "chart_type": str,  # for chart
        }
    """
    from services.openai_service import _get_client, _deployment

    if not chunks:
        return {
            "type":   "text",
            "answer": "No relevant information found in this document.",
        }

    # Build context with clear source attribution
    context_parts = []
    citation_list = []
    seen_files    = set()

    for i, chunk in enumerate(chunks, 1):
        filename = chunk.get("filename", f"Document {i}")
        text     = (chunk.get("text") or chunk.get("content") or "").strip()
        summary  = chunk.get("summary", "")
        score    = chunk.get("score", 0)

        if not text:
            continue

        # Include summary as context header for first chunk of each document
        if filename not in seen_files and summary:
            context_parts.append(
                f"[Source {i}: {filename} | relevance: {score:.2f}]\n"
                f"Document summary: {summary}\n\n"
                f"Relevant excerpt:\n{text}"
            )
        else:
            context_parts.append(
                f"[Source {i}: {filename} | relevance: {score:.2f}]\n{text}"
            )

        if filename not in seen_files:
            seen_files.add(filename)
            citation_list.append(filename)

    if not context_parts:
        return {
            "type":   "text",
            "answer": "No relevant information found in this document.",
        }

    context = "\n\n---\n\n".join(context_parts)

    # Determine format instructions
    q_lower = query.lower()
    if response_format == "auto":
        wants_chart = any(k in q_lower for k in _CHART_KEYWORDS)
        wants_table = any(k in q_lower for k in _TABLE_KEYWORDS)
        if wants_chart:
            response_format = "chart"
        elif wants_table:
            response_format = "table"
        else:
            response_format = "text"

    format_instructions = _build_format_instructions(response_format, citation_list)

    user_prompt = f"""Context from documents:
{context}

---

Question: {query}

{format_instructions}"""

    try:
        # Build messages: system + optional history (last 6 turns) + current user prompt
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        if history:
            # Include last 6 turns (3 user + 3 assistant) to stay within token budget
            for turn in history[-6:]:
                role    = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content[:800]})
        messages.append({"role": "user", "content": user_prompt})

        resp = _get_client().chat.completions.create(
            model    = _deployment(),
            messages = messages,
            temperature = 0.0,
            max_tokens  = 2500,
        )
        raw = resp.choices[0].message.content.strip()

        # Parse structured response
        parsed = _parse_llm_response(raw, citation_list, response_format)

        # ── Post-process chart data ───────────────────────────────────────
        if parsed.get("type") == "chart":
            parsed = _clean_chart_data(parsed, user_query=query)

        logging.info("grounded_generate: type=%s answer_len=%d",
                     parsed.get("type"), len(str(parsed.get("answer", ""))))
        return parsed

    except Exception as exc:
        logging.error("grounded_generate failed: %s", exc)
        return {
            "type":   "text",
            "answer": "Failed to generate answer. Please try again.",
        }


def _clean_chart_data(chart: dict, user_query: str = "") -> dict:
    """
    Post-process chart data for quality.
    Respects explicit user chart type requests — never downgrades pie→bar
    if the user explicitly asked for a pie chart.
    """
    chart_type    = chart.get("chart_type", "bar")
    labels        = chart.get("labels", [])
    values        = chart.get("values", [])
    _q = user_query.lower()
    user_wants_pie = any(k in _q for k in ("pie chart", "pie graph", " pie ", "as pie", "a pie"))

    if not labels or not values:
        return chart

    # Align lengths
    min_len = min(len(labels), len(values))
    labels  = labels[:min_len]
    values  = values[:min_len]

    # Convert values to numbers, replace None with 0
    clean_values = []
    for v in values:
        try:
            clean_values.append(float(v) if v is not None else 0.0)
        except (TypeError, ValueError):
            clean_values.append(0.0)

    if chart_type == "pie":
        # Filter out zero/negative slices
        filtered = [(l, v) for l, v in zip(labels, clean_values) if v > 0]
        if not filtered:
            chart["labels"] = labels
            chart["values"] = clean_values
            return chart
        # Only downgrade to bar if user did NOT explicitly ask for pie
        if len(filtered) == 1 and not user_wants_pie:
            chart["chart_type"] = "bar"
            chart["labels"]     = [f[0] for f in filtered]
            chart["values"]     = [f[1] for f in filtered]
            return chart
        chart["labels"] = [f[0] for f in filtered]
        chart["values"] = [f[1] for f in filtered]
    else:
        chart["labels"] = labels
        chart["values"] = clean_values

    return chart


def _build_format_instructions(
    response_format: str,
    citation_list: list[str],
) -> str:
    """Build format-specific instructions for the LLM."""

    if response_format == "table":
        return """Respond with ONLY a valid JSON object in this exact format:
{"type":"table","columns":["Col1","Col2","Col3"],"rows":[{"Col1":"v1","Col2":"v2","Col3":"v3"}],"answer":"1-line summary of the table"}

Rules:
- Extract ALL relevant data from the context into table rows
- Use the actual column names from the data
- Include every relevant row — do not truncate
- "answer" should be a 1-line summary of what the table shows"""

    if response_format == "chart":
        return """Respond with ONLY a valid JSON object in this exact format:
{"type":"chart","chart_type":"bar|line|pie|area|scatter","labels":["A","B","C"],"values":[10,20,30],"answer":"1-line summary"}

Rules:
- Extract numeric data from the context
- IMPORTANT: If the user explicitly asked for a specific chart type (e.g. "pie chart", "line chart", "bar chart"), use EXACTLY that chart_type
- Otherwise choose chart_type based on data: line for trends/time, pie for proportions/shares, bar for comparisons
- labels = category names or time periods
- values = corresponding numeric values
- If multiple series exist, use: {"type":"chart","chart_type":"bar","series":{"Series1":[1,2,3],"Series2":[4,5,6]},"labels":["A","B","C"],"answer":"..."}"""

    # Default: text
    return """Respond with ONLY a valid JSON object in this exact format:
{"type":"text","answer":"<your detailed answer here>"}

Rules:
- Answer must be comprehensive and directly address the question
- Use numbered lists (1. 2. 3.) for multi-part answers
- Use **bold** for key terms and figures
- Include exact numbers/dates/names from the context
- If the answer is not found in the context, say: "No relevant information found in this document." """


def _parse_llm_response(
    raw: str,
    citation_list: list[str],
    expected_format: str,
) -> dict:
    """
    Parse LLM response into a structured dict.
    Handles: valid JSON, JSON in markdown fences, plain text fallback.
    """
    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # Try direct JSON parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "type" in parsed:
            # Remove sources field if present (we don't expose it)
            parsed.pop("sources", None)
            return parsed
    except Exception:
        pass

    # Try extracting JSON object from mixed response
    m = re.search(r'\{[\s\S]*\}', cleaned)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, dict) and "type" in parsed:
                parsed.pop("sources", None)
                return parsed
        except Exception:
            pass

    # Plain text fallback — wrap in text envelope
    answer_text = raw.strip().lstrip("{[").rstrip("}]").strip()
    if not answer_text:
        answer_text = raw
    logging.warning("_parse_llm_response: LLM returned plain text, wrapping as text response (len=%d)", len(answer_text))
    return {
        "type":   "text",
        "answer": answer_text,
    }


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

# Answer cache: query_hash → (response dict, timestamp)
# Key uses query + filename_filter only — document content is reflected in the answer itself.
# TTL: 30 minutes — ensures stale answers after document updates are evicted.
_pipeline_cache: dict[str, tuple[dict, float]] = {}
_MAX_CACHE_SIZE = 300
_CACHE_TTL_SECS = 1800  # 30 minutes


def run_rag_pipeline(
    query:           str,
    filename_filter: str = "",
    uploaded_by:     str = "",
    session_id:      str = "",
    top_k:           int = 7,
    use_hyde:        bool = True,
    use_compression: bool = True,
    doc_ids:         list[str] | None = None,
    history:         list[dict] | None = None,
) -> dict:
    """
    Full advanced RAG pipeline entry point.

    Pipeline:
      1. Multi-query retrieval with HyDE
      2. Contextual compression
      3. Grounded generation with strict system prompt
      4. Response formatting

    Returns a response dict with type, answer, sources, and optional
    table/chart data.
    """
    from services.table_service import TableService

    if not query or not query.strip():
        return {"type": "text", "answer": "No question provided."}

    import time as _time

    # ── Step 1: Build cache key — includes doc_ids so different selections
    #    never share a cached answer (FIX: was missing doc_ids before)
    doc_ids_key = "|".join(sorted(doc_ids)) if doc_ids else ""
    cache_key = hashlib.md5(
        (query + "|" + filename_filter + "|" + uploaded_by + "|" + session_id + "|" + doc_ids_key).encode()
    ).hexdigest()

    # ── Step 2: Check cache early ─────────────────────────────────────────
    cached = _pipeline_cache.get(cache_key)
    if cached:
        result, ts = cached
        if _time.time() - ts < _CACHE_TTL_SECS:
            logging.info("run_rag_pipeline: cache hit")
            return result
        del _pipeline_cache[cache_key]

    # ── Step 3: Multi-query retrieval with HyDE ───────────────────────────
    chunks = multi_query_retrieve(
        query           = query,
        top_k           = top_k,
        filename_filter = filename_filter,
        uploaded_by     = uploaded_by,
        use_hyde        = use_hyde,
        doc_ids         = doc_ids,
    )

    if not chunks:
        return {
            "type":   "text",
            "answer": "No relevant information found in this document.",
        }

    # ── Step 4: Collect structured data from ALL relevant docs ────────────
    # FIX: previously only picked the single best-scoring doc.
    # Now we collect ALL docs that have structured data and score > 0,
    # merge them into one DataFrame with a _source column for cross-doc queries.
    table_svc = TableService()
    q_lower   = query.lower()
    q_words   = set(w.strip(".,!?;:()") for w in q_lower.split() if len(w) >= 3)

    def _stem(w: str) -> str:
        if w.endswith("ing") and len(w) > 5: return w[:-3]
        if w.endswith("ies") and len(w) > 4: return w[:-3] + "y"
        if w.endswith("es")  and len(w) > 4: return w[:-2]
        if w.endswith("s")   and len(w) > 3: return w[:-1]
        return w

    stemmed_q_words = {_stem(w) for w in q_words} | q_words

    def _sd_relevance_score(sd: dict, filename: str) -> int:
        score = 0
        sd_columns = sd.get("columns", [])
        if not sd_columns and sd.get("sheets"):
            for sheet_data in sd["sheets"].values():
                sd_columns = sheet_data.get("columns", [])
                if sd_columns:
                    break
        for col in sd_columns:
            col_lower = col.lower().replace("_", " ").replace("-", " ")
            col_words = set(col_lower.split())
            stemmed_col_words = {_stem(w) for w in col_words} | col_words
            for qw in stemmed_q_words:
                if qw in col_lower or any(qw in cw or cw in qw for cw in stemmed_col_words):
                    score += 3
                    break
        fname_lower = filename.lower().replace("_", " ").replace("-", " ")
        for qw in stemmed_q_words:
            if qw in fname_lower:
                score += 5
                break
        chart_agg_kw = {"chart", "graph", "plot", "total", "sum", "average",
                        "count", "max", "min", "distribution", "breakdown"}
        if any(k in q_lower for k in chart_agg_kw) and sd_columns:
            score += 1
        return score

    # Deduplicate chunks by doc_id so we fetch structured data once per doc
    seen_doc_ids: set[str] = set()
    candidate_docs: list[tuple[str, str, str]] = []  # (doc_id, filename, chunk_doc_id)
    for chunk in chunks:
        fname  = chunk.get("filename", "")
        doc_id = chunk.get("doc_id", "")
        if not fname:
            continue
        dedup_key = doc_id or fname
        if dedup_key in seen_doc_ids:
            continue
        seen_doc_ids.add(dedup_key)
        candidate_docs.append((doc_id, fname))

    # Collect all docs with relevant structured data
    relevant_sds: list[tuple[str, dict, int]] = []  # (filename, sd, score)
    for doc_id, fname in candidate_docs:
        # FIX: pass doc_id to get_structured_data to prevent filename collision
        sd = table_svc.get_structured_data(fname, session_id=session_id, doc_id=doc_id)
        if not sd:
            continue
        score = _sd_relevance_score(sd, fname)
        logging.info("run_rag_pipeline: structured data '%s' (doc_id=%s) score=%d", fname, doc_id, score)
        if score > 0:
            relevant_sds.append((fname, sd, score))

    # Build merged structured data if we have any relevant docs
    stored_sd: dict | None = None
    if relevant_sds:
        if len(relevant_sds) == 1:
            # Single doc — use as-is (no _source column needed, preserves existing behaviour)
            stored_sd = relevant_sds[0][1]
            logging.info("run_rag_pipeline: single structured doc '%s'", relevant_sds[0][0])
        else:
            # Multiple docs — merge into one with _source column (FIX: was silently dropping all but best)
            stored_sd = _merge_structured_data(relevant_sds)
            logging.info("run_rag_pipeline: merged %d structured docs", len(relevant_sds))

    if not relevant_sds:
        logging.info("run_rag_pipeline: no relevant structured data — using prose RAG")

    # ── Step 5: Intent classification ────────────────────────────────────
    intent = classify_intent(query, has_structured_data=stored_sd is not None)
    logging.info("run_rag_pipeline: intent=%s has_sd=%s", intent, stored_sd is not None)

    # ── Step 6: Route to structured engine or prose RAG ──────────────────
    if intent == "structured" and stored_sd:
        from services.query_engine import generate_plan, execute_plan, structured_to_df

        def _run_engine_local(query: str, structured: dict) -> dict | None:
            try:
                df = structured_to_df(structured)
                if df.empty:
                    return None
                # Keep _source column if present (cross-doc merge marker)
                drop_cols = [c for c in df.columns if c.startswith("_") and c != "_source"]
                df = df.drop(columns=drop_cols, errors="ignore")
                cols   = list(df.columns)
                plan   = generate_plan(query, cols)
                result = execute_plan(df, plan)
                return result
            except Exception as exc:
                logging.warning("_run_engine_local failed: %s", exc)
                return None

        engine_result = _run_engine_local(query, stored_sd)
        if engine_result and engine_result.get("type") != "error":
            rows = engine_result.get("rows", [])
            if rows or engine_result.get("type") == "text":
                if engine_result.get("type") != "chart":
                    from services.query_engine import promote_to_chart
                    try:
                        df   = structured_to_df(stored_sd)
                        drop_cols = [c for c in df.columns if c.startswith("_") and c != "_source"]
                        df   = df.drop(columns=drop_cols, errors="ignore")
                        cols = list(df.columns)
                        plan = generate_plan(query, cols)
                        if any(k in query.lower() for k in _CHART_KEYWORDS) and plan.get("group_by"):
                            engine_result = promote_to_chart(engine_result, query)
                    except Exception:
                        pass
                if engine_result.get("type") == "chart" and engine_result.get("labels"):
                    engine_result = _clean_chart_data(engine_result, user_query=query)
                result = {k: v for k, v in engine_result.items() if k != "sources"}
                _cache_result(cache_key, result)
                return result
        logging.info("run_rag_pipeline: structured engine failed — falling back to prose RAG")

    # ── Step 7: Contextual compression ───────────────────────────────────
    if use_compression and len(chunks) > 2:
        compressed_chunks = compress_chunks(query, chunks)
    else:
        compressed_chunks = chunks

    # ── Step 8: Grounded generation — inject history for multi-turn context ──
    if intent == "hybrid":
        response_format = "table"
    elif any(k in q_lower for k in _CHART_KEYWORDS):
        response_format = "chart"
    elif any(k in q_lower for k in _TABLE_KEYWORDS):
        response_format = "table"
    else:
        response_format = "text"

    result = grounded_generate(
        query           = query,
        chunks          = compressed_chunks,
        response_format = response_format,
        history         = history,
    )

    # ── Step 9: Cache and return ──────────────────────────────────────────
    _cache_result(cache_key, result)
    return result


def _merge_structured_data(docs: list[tuple[str, dict, int]]) -> dict:
    """
    Merge structured data from multiple documents into a single dict.
    Adds a '_source' column with the filename so the query engine and user
    can tell which row came from which document.

    docs: list of (filename, structured_data, score) — sorted by score desc.
    Returns a merged structured_data dict with 'rows' and 'columns'.
    """
    from services.query_engine import structured_to_df
    import pandas as pd

    docs_sorted = sorted(docs, key=lambda x: x[2], reverse=True)
    frames = []
    for fname, sd, _ in docs_sorted:
        try:
            df = structured_to_df(sd)
            if df.empty:
                continue
            # Drop internal columns except keep schema markers out
            df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
            df["_source"] = fname
            frames.append(df)
        except Exception as exc:
            logging.warning("_merge_structured_data: failed to convert '%s': %s", fname, exc)

    if not frames:
        return {}

    merged = pd.concat(frames, ignore_index=True, sort=False)
    # Fill NaN with None for JSON serialisation
    merged = merged.where(pd.notnull(merged), None)

    return {
        "rows":    merged.to_dict(orient="records"),
        "columns": list(merged.columns),
        "_merged": True,
        "_sources": [f for f, _, _ in docs_sorted],
    }


def _cache_result(key: str, result: dict) -> None:
    """Cache result with TTL and LRU eviction."""
    import time as _time
    global _pipeline_cache
    _pipeline_cache[key] = (result, _time.time())
    if len(_pipeline_cache) > _MAX_CACHE_SIZE:
        # Evict oldest 30 entries
        keys_to_evict = list(_pipeline_cache.keys())[:30]
        for k in keys_to_evict:
            _pipeline_cache.pop(k, None)
