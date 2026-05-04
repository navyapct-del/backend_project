import os
import re
import json
import math
import logging
from openai import AzureOpenAI
from services.config import require_env, get_env
from functools import lru_cache
import hashlib

# In-memory caches
_embedding_cache: dict[str, list[float]] = {}
_answer_cache: dict[str, dict] = {}

# Shared client — instantiated once per worker lifetime.
# Endpoint: standard Azure OpenAI (openai.azure.com)
# API version 2024-05-01-preview supports gpt-4o and text-embedding-3-small.
_client: AzureOpenAI | None = None

def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        endpoint = require_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
        # Override via AZURE_OPENAI_API_VERSION env var if needed.
        api_version = get_env("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        _client = AzureOpenAI(
            api_key        = require_env("AZURE_OPENAI_API_KEY"),
            api_version    = api_version,
            azure_endpoint = endpoint,
        )
        logging.info("AzureOpenAI client initialised | endpoint=%s | api_version=%s",
                     endpoint, api_version)
    return _client

def _deployment() -> str:
    return get_env("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini")

def _vision_deployment() -> str:
    """Return the vision-capable deployment name.
    Falls back to the standard deployment (gpt-4.1-mini / gpt-4o both support vision).
    Override with AZURE_OPENAI_VISION_DEPLOYMENT to use a dedicated vision deployment."""
    return get_env("AZURE_OPENAI_VISION_DEPLOYMENT", _deployment())

_EMBED_MODEL = get_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def generate_embedding(text: str) -> list[float]:
    """Cached embedding generation — avoids duplicate API calls for same text."""
    if not text:
        return []
    # text-embedding-3-small supports up to 8191 tokens (~32k chars); cap at 8000 chars
    truncated = text[:8000]
    key = hashlib.md5(truncated.encode()).hexdigest()
    if key in _embedding_cache:
        return _embedding_cache[key]
    try:
        resp = _get_client().embeddings.create(
            model = _EMBED_MODEL,
            input = truncated,
        )
        vec = resp.data[0].embedding
        _embedding_cache[key] = vec
        return vec
    except Exception:
        logging.exception("generate_embedding failed.")
        return []


def expand_query(query: str) -> str:
    """
    Expand user query with synonyms/related terms to improve retrieval recall.
    Returns expanded query string. Falls back to original on failure.
    """
    if not query or not query.strip():
        return query
    prompt = (
        "Expand the following search query with 3-5 related terms or synonyms "
        "to improve document retrieval. Return ONLY the expanded query as a single line, "
        "no explanation.\n\n"
        f"Query: {query}\n\nExpanded query:"
    )
    try:
        resp = _get_client().chat.completions.create(
            model       = _deployment(),
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.0,
            max_tokens  = 60,
        )
        expanded = resp.choices[0].message.content.strip()
        logging.info("expand_query: '%s' → '%s'", query, expanded)
        return expanded
    except Exception:
        logging.warning("expand_query failed — using original query.")
        return query


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def generate_summary(text: str) -> str:
    """Generate a 3-4 line summary of the document."""
    if not text:
        return ""
    prompt = f"Summarize the following document in 3-4 concise lines:\n\n{text[:2000]}"
    try:
        resp = _get_client().chat.completions.create(
            model       = _deployment(),
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.2,
            max_tokens  = 150,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        logging.exception("generate_summary failed.")
        return text[:300].strip()


# ---------------------------------------------------------------------------
# Tags / key phrases
# ---------------------------------------------------------------------------

def generate_tags(text: str) -> str:
    """Extract up to 10 key phrases as a comma-separated string."""
    if not text:
        return ""
    prompt = (
        "Extract up to 10 key topics or phrases from the text below. "
        "Return ONLY a comma-separated list, nothing else.\n\n"
        f"{text[:2000]}"
    )
    try:
        resp = _get_client().chat.completions.create(
            model       = _deployment(),
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.0,
            max_tokens  = 100,
        )
        return resp.choices[0].message.content.strip()[:500]
    except Exception:
        logging.exception("generate_tags failed.")
        return ""


# ---------------------------------------------------------------------------
# RAG answer — production-grade, intent-aware
# ---------------------------------------------------------------------------

def generate_rag_answer(query: str, docs: list[dict]) -> dict:
    """
    Production RAG answer with intent-aware response formatting.

    Routing:
      - Analytical queries (table/chart intent) → structured JSON response
      - Conversational queries → rich prose answer

    Cache: keyed on (query, doc_ids) — evicted LRU at 500 entries.
    """
    if not query or not query.strip():
        return {"type": "text", "answer": "No question provided."}
    if not docs:
        return {"type": "text", "answer": "No relevant documents found. Please upload documents first."}

    # Cache key: query + sorted doc ids (order-independent)
    doc_ids   = sorted(d.get("id", "") for d in docs)
    cache_key = hashlib.md5((query + "|" + ",".join(doc_ids)).encode()).hexdigest()
    if cache_key in _answer_cache:
        logging.info("generate_rag_answer: cache hit")
        return _answer_cache[cache_key]

    # ── Build context ─────────────────────────────────────────────────────
    # Fetch stored summaries for richer context
    stored_summaries: dict[str, str] = {}
    try:
        from services.table_service import TableService
        _ts = TableService()
        for fname in dict.fromkeys(d.get("filename", "") for d in docs if d.get("filename")):
            entity = _ts.find_by_filename(fname)
            if entity and entity.get("summary"):
                stored_summaries[fname] = entity["summary"]
    except Exception as exc:
        logging.warning("RAG: summary fetch failed: %s", exc)

    context_parts = []
    citation_list = []
    seen_files    = set()

    for i, doc in enumerate(docs, 1):
        filename    = doc.get("filename", f"Doc {i}")
        chunk_text  = (doc.get("content") or doc.get("text") or "").strip()[:3000]
        doc_summary = stored_summaries.get(filename, "") or doc.get("summary", "")

        # Prepend summary for richer context
        if doc_summary and chunk_text:
            full_text = f"[Summary]: {doc_summary}\n\n[Excerpt]: {chunk_text}"
        elif doc_summary:
            full_text = f"[Summary]: {doc_summary}"
        else:
            full_text = chunk_text

        if full_text:
            context_parts.append(f"[Source {i}: {filename}]\n{full_text}")

        if filename not in seen_files:
            seen_files.add(filename)
            citation_list.append(filename)

    if not context_parts:
        return {"type": "text", "answer": "No content could be extracted from the retrieved documents."}

    context = "\n\n---\n\n".join(context_parts)[:16000]

    # ── Intent detection ──────────────────────────────────────────────────
    q_lower = query.lower()
    wants_table = any(k in q_lower for k in [
        "table", "list", "show all", "enumerate", "compare", "comparison",
        "breakdown", "details", "columns", "rows", "tabular"
    ])
    wants_chart = any(k in q_lower for k in [
        "chart", "graph", "plot", "visualize", "visualise", "bar", "line",
        "pie", "trend", "distribution", "histogram"
    ])

    sources_str = ", ".join(citation_list)

    # ── Prompt ────────────────────────────────────────────────────────────
    prompt = f"""You are a precise, grounded AI assistant. Answer the question using ONLY the context provided.

STRICT RULES:
1. Answer ONLY from the context. Do NOT use prior knowledge or hallucinate.
2. If the answer is not in the context, respond: {{"type":"text","answer":"The documents do not contain information to answer this question."}}
3. Respond with ONLY a single valid JSON object — no markdown, no explanation outside JSON.
4. Always include "sources" field: {json.dumps(citation_list)}

RESPONSE FORMAT — choose the best type:

For factual/conversational questions:
{{"type":"text","answer":"<detailed answer with numbered points if multi-part>","sources":{json.dumps(citation_list)}}}

For tabular data (lists, comparisons, multi-attribute items):
{{"type":"table","columns":["Col1","Col2","Col3"],"rows":[{{"Col1":"v1","Col2":"v2","Col3":"v3"}}],"answer":"<1-line summary>","sources":{json.dumps(citation_list)}}}

For charts/graphs (numeric trends, distributions, comparisons):
{{"type":"chart","chart_type":"bar|line|pie|area|scatter","labels":["A","B","C"],"values":[10,20,30],"answer":"<1-line summary>","sources":{json.dumps(citation_list)}}}

FORMAT SELECTION RULES:
- Use "table" when: listing items with multiple attributes, comparisons, enumerations
- Use "chart" when: user explicitly asks for chart/graph AND numeric data exists in context
- Use "text" for: all factual questions, explanations, summaries
{"- PREFER table format since user wants tabular data" if wants_table else ""}
{"- PREFER chart format since user wants a visualization" if wants_chart else ""}

Context:
{context}

Question: {query}

JSON response:"""

    try:
        resp = _get_client().chat.completions.create(
            model       = _deployment(),
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.0,
            max_tokens  = 2500,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

        def _extract_json(s: str) -> dict | None:
            # Try direct parse first
            try:
                p = json.loads(s)
                if isinstance(p, dict) and "type" in p:
                    return p
            except Exception:
                pass
            # Extract first JSON object from response
            m = re.search(r'\{[\s\S]*\}', s)
            if m:
                try:
                    p = json.loads(m.group())
                    if isinstance(p, dict) and "type" in p:
                        return p
                except Exception:
                    pass
            return None

        parsed = _extract_json(raw)

        # If LLM returned plain text instead of JSON, wrap it
        if not parsed:
            parsed = {"type": "text", "answer": raw, "sources": citation_list}

        # Ensure sources always present
        if "sources" not in parsed:
            parsed["sources"] = citation_list

        # Reject "insufficient data" responses that are actually answerable
        # (LLM sometimes refuses when context is sufficient)
        answer_text = parsed.get("answer", "")
        if "insufficient" in answer_text.lower() and len(context_parts) > 0:
            logging.warning("RAG: LLM returned insufficient-data despite %d context chunks — retrying",
                            len(context_parts))
            # One retry with a more direct prompt
            retry_prompt = (
                f"Based on this context, answer the question directly and concisely.\n\n"
                f"Context:\n{context[:8000]}\n\n"
                f"Question: {query}\n\n"
                f"Answer (be specific, use bullet points if needed):"
            )
            retry_resp = _get_client().chat.completions.create(
                model       = _deployment(),
                messages    = [{"role": "user", "content": retry_prompt}],
                temperature = 0.1,
                max_tokens  = 1000,
            )
            retry_answer = retry_resp.choices[0].message.content.strip()
            parsed = {"type": "text", "answer": retry_answer, "sources": citation_list}

        # Cache with LRU eviction
        _answer_cache[cache_key] = parsed
        if len(_answer_cache) > 500:
            # Evict oldest 50 entries
            for _ in range(50):
                if _answer_cache:
                    del _answer_cache[next(iter(_answer_cache))]

        logging.info("generate_rag_answer: type=%s answer_len=%d sources=%d",
                     parsed.get("type"), len(str(parsed.get("answer", ""))), len(citation_list))
        return parsed

    except Exception as exc:
        logging.error("generate_rag_answer failed: %s", exc)
        return {"type": "text", "answer": "Failed to generate answer. Please try again.", "sources": citation_list}


def smart_chart_from_structured(query: str, structured: dict) -> dict | None:
    """
    Given stored structured_data {"columns": [...], "rows": [...]} from an
    Excel/CSV upload, intelligently:
      1. Detects entities mentioned in the query (e.g. Bihar, Maharashtra)
      2. Finds the column that contains those entities (e.g. "State")
      3. Filters rows to only those entities
      4. Detects the x-axis column (Year, Date, Month, etc.)
      5. Detects the value column (numeric)
      6. Pivots: rows of (x, entity, value) → columns per entity
      7. Returns chart-ready data + chart_config

    Returns None if the data cannot be meaningfully charted.
    """
    try:
        import pandas as pd

        sheets = structured.get("sheets", {})   # per-sheet data if available

        # ── 1. Build working DataFrame for entity detection ───────────────
        # Prefer flat rows; fall back to merging all sheet rows
        all_rows = structured.get("rows", [])
        if not all_rows and sheets:
            for sname, sd in sheets.items():
                for r in sd.get("rows", []):
                    row = dict(r)
                    row["_sheet"] = sname
                    all_rows.append(row)

        if not all_rows:
            return None

        df_all   = pd.DataFrame(all_rows)
        q_lower  = query.lower()
        entities = _extract_entities_from_query(q_lower, df_all)
        logging.info("smart_chart: entities detected = %s", entities)

        # ── 2. Select the best sheet (if per-sheet data available) ────────
        if sheets:
            best_sheet = _select_best_sheet(q_lower, entities, sheets)
            logging.info("smart_chart: selected sheet = '%s'", best_sheet)
            if best_sheet:
                sd   = sheets[best_sheet]
                cols = sd["columns"]
                rows = sd["rows"]
                df   = pd.DataFrame(rows, columns=cols)
            else:
                # No clear winner — use all rows but skip _sheet column
                df = df_all.drop(columns=["_sheet"], errors="ignore")
        else:
            df = pd.DataFrame(
                structured.get("rows", []),
                columns=structured.get("columns") or None,
            )

        if df.empty:
            return None

        # ── 2. Find entity column (categorical column whose values match entities) ──
        entity_col = _find_entity_column(df, entities)
        logging.info("smart_chart: entity_col = %s", entity_col)

        # ── 3. Filter rows to only the requested entities ─────────────────
        if entity_col and entities:
            mask = df[entity_col].astype(str).str.upper().isin(
                [e.upper() for e in entities]
            )
            df = df[mask]
            if df.empty:
                logging.warning("smart_chart: filter produced empty DataFrame")
                return None

        # ── 4. Find x-axis column (Year, Date, Month, Quarter, Category) ──
        x_col = _find_column(df, ["year","date","month","quarter","period","category","name"])
        if not x_col:
            # fallback: first non-entity, non-numeric column
            for c in df.columns:
                if c != entity_col and df[c].dtype == object:
                    x_col = c
                    break
        if not x_col:
            x_col = df.columns[0]

        # ── 5. Find value column (first numeric column that isn't x or entity) ──
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        value_col    = next(
            (c for c in numeric_cols if c not in (x_col, entity_col)), None
        )
        if not value_col and numeric_cols:
            value_col = numeric_cols[0]
        if not value_col:
            logging.warning("smart_chart: no numeric column found")
            return None

        logging.info("smart_chart: x=%s entity=%s value=%s", x_col, entity_col, value_col)

        # ── 6. Pivot: (x, entity) → wide format ───────────────────────────
        if entity_col and entity_col != x_col:
            pivot = df.pivot_table(
                index   = x_col,
                columns = entity_col,
                values  = value_col,
                aggfunc = "sum",
            ).reset_index()
            pivot.columns = [str(c) for c in pivot.columns]
            series = [c for c in pivot.columns if c != x_col]
            data   = pivot.to_dict(orient="records")
        else:
            # No entity column — simple x vs value
            pivot  = df[[x_col, value_col]].copy()
            pivot.columns = [x_col, value_col]
            series = [value_col]
            data   = pivot.to_dict(orient="records")

        # ── 7. Determine chart type from query ────────────────────────────
        chart_type = "line" if any(k in q_lower for k in ["trend","growth","over time","line"]) else "bar"

        return {
            "data":         data,
            "chart_config": {
                "type":   chart_type,
                "xKey":   x_col,
                "series": series,
            },
            "script": (
                f"SELECT {x_col}, {entity_col}, {value_col} FROM data "
                f"WHERE {entity_col} IN ({', '.join(repr(e) for e in entities)}) "
                f"ORDER BY {x_col};"
            ) if entity_col else f"SELECT {x_col}, {value_col} FROM data ORDER BY {x_col};",
        }

    except Exception:
        logging.exception("smart_chart_from_structured failed.")
        return None


def _extract_entities_from_query(q_lower: str, df) -> list[str]:
    """
    Find words/phrases in the query that match actual values in any categorical column.
    Case-insensitive matching — returns original-cased values from the DataFrame.
    """
    import re

    # Build a lookup: lowercase_value → original_cased_value
    value_map: dict[str, str] = {}
    for col in df.select_dtypes(include="object").columns:
        if col.lower() in {"_sheet", "_file", "_source"}:
            continue
        for val in df[col].dropna().astype(str).unique():
            value_map[val.lower()] = val

    if not value_map:
        return []

    # Try multi-word matches first (e.g. "Tamil Nadu"), then single words
    matched = []
    seen    = set()

    # Multi-word: check every 2-3 word window
    words = re.findall(r"[a-zA-Z]+", q_lower)
    for size in (3, 2, 1):
        for i in range(len(words) - size + 1):
            phrase = " ".join(words[i:i + size])
            if phrase in value_map and value_map[phrase] not in seen:
                matched.append(value_map[phrase])
                seen.add(value_map[phrase])

    return matched


def _find_entity_column(df, entities: list[str]) -> str | None:
    """Find the column whose values contain the queried entities. Skips internal columns."""
    if not entities:
        return None
    entity_lower = {e.lower() for e in entities}
    # Skip internal/metadata columns
    skip = {"_sheet", "_file", "_source"}
    for col in df.select_dtypes(include="object").columns:
        if col.lower() in skip:
            continue
        col_vals = set(df[col].dropna().astype(str).str.lower().unique())
        if entity_lower & col_vals:   # intersection
            return col
    return None


def _find_column(df, candidates: list[str]) -> str | None:
    """Return the first column whose name (lowercased) matches any candidate."""
    lc = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lc:
            return lc[cand]
    return None


def _select_best_sheet(q_lower: str, entities: list[str], sheets: dict) -> str | None:
    """
    Pick the most relevant sheet for the query.

    Strategy (in order):
    1. If any entity value appears in a sheet's rows → prefer that sheet
    2. If any query word appears in the sheet name → prefer that sheet
    3. Return the first sheet as fallback
    """
    import pandas as pd

    entity_lower = {e.lower() for e in entities}

    # Score each sheet
    scores: dict[str, int] = {}
    for sheet_name, sd in sheets.items():
        score = 0
        # Check if sheet name contains query words
        sheet_lower = sheet_name.lower()
        for word in q_lower.split():
            if len(word) > 3 and word in sheet_lower:
                score += 2

        # Check if entity values exist in this sheet's data
        if entity_lower and sd.get("rows"):
            df = pd.DataFrame(sd["rows"])
            for col in df.select_dtypes(include="object").columns:
                if col.lower() in {"_sheet", "_file"}:
                    continue
                col_vals = set(df[col].dropna().astype(str).str.lower().unique())
                matches  = entity_lower & col_vals
                score   += len(matches) * 5   # entity match is high signal

        scores[sheet_name] = score
        logging.info("smart_chart: sheet '%s' score=%d", sheet_name, score)

    if not scores:
        return None

    best       = max(scores, key=lambda k: scores[k])
    best_score = scores[best]

    # Only use best sheet if it has a meaningful score
    if best_score > 0:
        return best

    # Fallback: first sheet
    return next(iter(sheets))


# ---------------------------------------------------------------------------
# Structured data extraction (for table/chart responses)
# ---------------------------------------------------------------------------

def extract_structured_data(query: str, docs: list[dict]) -> list[dict]:
    """
    Ask OpenAI to extract structured rows from document text.
    Returns a list of flat dicts, or [] on failure.
    """
    if not docs:
        return []

    context_parts = []
    for i, doc in enumerate(docs, 1):
        text = (doc.get("content") or doc.get("text") or doc.get("extracted_text") or "")[:2000]
        context_parts.append(f"[Document {i}: {doc.get('filename','')}]\n{text}")
    context = "\n\n".join(context_parts)

    prompt = (
        "You are a data extraction assistant.\n"
        "Extract ALL numerical/tabular data relevant to the question from the context.\n"
        "Return ONLY a valid JSON array of flat objects with consistent keys.\n"
        "Example: [{\"year\": 2018, \"state\": \"Bihar\", \"value\": 6239}]\n"
        "If no structured data exists, return: []\n"
        "Do NOT include any explanation — only the JSON array.\n\n"
        f"Question: {query}\n\nContext:\n{context}\n\nJSON:"
    )
    try:
        resp = _get_client().chat.completions.create(
            model       = _deployment(),
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.0,
            max_tokens  = 1500,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        logging.exception("extract_structured_data failed.")
        return []


# ---------------------------------------------------------------------------
# Short explanation of structured data
# ---------------------------------------------------------------------------

def generate_explanation(query: str, data: list[dict]) -> str:
    if not data:
        return "No structured data could be extracted."
    sample = json.dumps(data[:6], indent=2)
    prompt = f"Summarize this data in 2-3 sentences relevant to: '{query}'\n\nData:\n{sample}\n\nSummary:"
    try:
        resp = _get_client().chat.completions.create(
            model       = _deployment(),
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.2,
            max_tokens  = 150,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        logging.exception("generate_explanation failed.")
        return ""
