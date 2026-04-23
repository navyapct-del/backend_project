import azure.functions as func
import logging
import json
import time
import os
import uuid

try:
    from services.config import log_config_status, require_env
    from services.blob_service    import BlobService
    from services.table_service   import TableService
    from services.extractor       import extract_text, extract_with_structured
    from services.search_service  import ensure_index, index_document, vector_search, delete_index
    from services.openai_service  import (
        generate_summary, generate_tags, generate_embedding,
        generate_rag_answer, extract_structured_data, generate_explanation,
        smart_chart_from_structured,
    )
    from services.query_engine import generate_plan, execute_plan, structured_to_df, get_series_from_data, detect_dual_axis_from_rows, chart_type_from_query, promote_to_chart
    from services.delete_service       import delete_document
except Exception as _import_exc:
    logging.error("STARTUP IMPORT ERROR: %s", _import_exc)
    raise

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# ---------------------------------------------------------------------------
# Custom JSON encoder — handles NaN, Infinity, numpy types from pandas
# ---------------------------------------------------------------------------

import math

class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return None if math.isnan(obj) else float(obj)
            if isinstance(obj, (np.ndarray,)):  return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)

    def encode(self, obj):
        # Replace NaN/Infinity in the final string — catches all edge cases
        result = super().encode(obj)
        return result

def _safe_json(obj) -> str:
    """Serialize to JSON, replacing NaN/Infinity with null."""
    raw = json.dumps(obj, cls=_SafeEncoder)
    # Replace bare NaN and Infinity tokens (not valid JSON)
    import re as _re
    raw = _re.sub(r'\bNaN\b',       'null', raw)
    raw = _re.sub(r'\bInfinity\b',  'null', raw)
    raw = _re.sub(r'\b-Infinity\b', 'null', raw)
    return raw

# ---------------------------------------------------------------------------
# GET /health — validate all required env vars, return status
# ---------------------------------------------------------------------------

@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("GET /health")
    required = [
        "AZURE_STORAGE_CONNECTION_STRING",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_KEY",
        "DOC_INTELLIGENCE_ENDPOINT",
        "DOC_INTELLIGENCE_KEY",
    ]
    status = {}
    all_ok = True
    for k in required:
        present = bool(os.environ.get(k, "").strip())
        status[k] = "OK" if present else "MISSING"
        if not present:
            all_ok = False

    return func.HttpResponse(
        json.dumps({"healthy": all_ok, "config": status}, indent=2),
        status_code=200 if all_ok else 503,
        mimetype="application/json",
    )


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

_CHART_KW = {"plot", "graph", "chart", "trend", "visualize", "growth", "pie", "bar chart", "line chart", "pie chart", "show as chart", "show as graph", "scatter", "histogram", "distribution", "heatmap", "radar", "funnel", "treemap", "area chart"}
_TABLE_KW = {"compare", "comparison", "difference", "versus", " vs ", "year-wise",
             "yearwise", "state-wise", "breakdown", "statewise"}

# Keywords that signal aggregation chart intent even without explicit "plot/chart"
_AGG_CHART_KW = {"average", "avg", "sum", "total", "count", "by", "per",
                 "distribution", "breakdown", "correlation", "relationship"}

def _detect_type(query: str) -> str:
    q = " " + query.lower() + " "
    if any(f" {k} " in q for k in _CHART_KW): return "chart"
    if any(k in q for k in _TABLE_KW): return "table"
    return "text"

def _is_analytical(query: str) -> bool:
    return _detect_type(query) in ("chart", "table")

def _is_chart_intent(query: str) -> bool:
    """True when query explicitly asks for a chart OR implies aggregation visualisation."""
    if _detect_type(query) == "chart":
        return True
    q = query.lower()
    return sum(1 for k in _AGG_CHART_KW if k in q) >= 2   # at least 2 agg signals


def _chart_type_from_query(query: str) -> str:
    from services.query_engine import chart_type_from_query
    return chart_type_from_query(query)


def _promote_to_chart(result: dict, query: str) -> dict:
    from services.query_engine import promote_to_chart
    return promote_to_chart(result, query)


def _run_query_engine(user_query: str, structured: dict) -> dict | None:
    """
    Run the LLM query planner + pandas execution engine against stored structured data.
    Promotes groupby aggregation results to chart format when query intent warrants it.
    """
    import pandas as pd
    import json as _json
    try:
        df = structured_to_df(structured)
        if df.empty:
            logging.warning("[QUERY] Structured data produced empty DataFrame")
            return None

        df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
        columns = list(df.columns)

        logging.info("[QUERY] User query: %s", user_query)
        logging.info("[QUERY] Schema columns: %s", columns)

        plan   = generate_plan(user_query, columns)
        logging.info("[QUERY] Generated plan: op=%s select=%s group_by=%s aggs=%d filters=%d",
                     plan.get("operation"), plan.get("select"),
                     plan.get("group_by"), len(plan.get("aggregations", [])),
                     len(plan.get("filters", [])))

        result = execute_plan(df, plan)
        resp_type = result.get("type")
        row_count = len(result.get("rows", []))

        if resp_type == "error":
            logging.warning("[QUERY ERROR] Invalid columns | answer=%s", result.get("answer"))
        elif not result.get("rows") and resp_type != "text":
            logging.warning("[QUERY] Empty result set | type=%s op=%s", resp_type, plan.get("operation"))
        else:
            logging.info("[QUERY] Result: type=%s rows=%d", resp_type, row_count)

        # Promote to chart if:
        # 1. Query has chart/aggregation intent, AND
        # 2. Result is a table (not already a chart), AND
        # 3. Result has numeric columns suitable for charting
        if (result.get("type") != "chart"
                and result.get("type") != "error"
                and _is_chart_intent(user_query)
                and plan.get("group_by")):
            result = _promote_to_chart(result, user_query)

        return result

    except ValueError as exc:
        # _validate_plan raises ValueError with JSON payload when all requested
        # columns are invalid — surface as a structured error result.
        try:
            detail = _json.loads(str(exc))
            invalid_cols = detail.get("invalid_columns", [])
            available    = detail.get("available_columns", columns if 'columns' in dir() else [])
            suggestions  = detail.get("suggestions", [])
            hint = (f"Did you mean: {', '.join(suggestions)}?" if suggestions
                    else f"Available columns: {', '.join(available)}")
            logging.warning("_run_query_engine: invalid columns requested=%s suggestions=%s",
                            invalid_cols, suggestions)
            return {
                "type":               "error",
                "answer":             (
                    f"The dataset does not contain "
                    f"{'column' if len(invalid_cols) == 1 else 'columns'} "
                    f"{', '.join(repr(c) for c in invalid_cols)}. {hint}"
                ),
                "invalid_columns":    invalid_cols,
                "available_columns":  available,
                "suggestions":        suggestions,
                "columns":            [],
                "rows":               [],
                "chart_config":       None,
                "script":             "",
            }
        except (_json.JSONDecodeError, Exception):
            logging.warning("_run_query_engine schema error: %s", exc)
            return None

    except Exception as exc:
        logging.warning("_run_query_engine failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# POST /upload
# Upload → Blob → OCR → Embedding → Table Storage + AI Search
# ---------------------------------------------------------------------------

@app.route(route="upload", methods=["POST"])
def upload(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("POST /upload")
    try:
        file = req.files.get("file")
        if not file:
            return func.HttpResponse(json.dumps({"error": "No file provided."}),
                                     status_code=400, mimetype="application/json")

        filename    = req.form.get("filename") or file.filename
        description = req.form.get("description", "")
        tags_input  = req.form.get("tags", "")
        temp_flag   = req.form.get("temp", "false").lower() == "true"
        session_id  = req.form.get("session_id", "")

        # Extract uploader identity from JWT (preferred) or form field fallback
        uploaded_by = ""
        auth_header = req.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                import base64 as _b64
                payload_b64 = auth_header.split(".")[1]
                payload_b64 += "=" * (-len(payload_b64) % 4)
                jwt_payload = json.loads(_b64.b64decode(payload_b64))
                uploaded_by = jwt_payload.get("email") or jwt_payload.get("preferred_username") or ""
            except Exception:
                pass
        if not uploaded_by:
            uploaded_by = req.form.get("uploaded_by", "")

        if not filename:
            return func.HttpResponse(json.dumps({"error": "filename is required."}),
                                     status_code=400, mimetype="application/json")

        # ── File type validation ───────────────────────────────────────────
        ALLOWED_MIME_TYPES = {
            # Images
            "image/jpeg", "image/png", "image/gif", "image/svg+xml", "image/webp",
            # PDF
            "application/pdf",
            # CSV — browsers report different MIME types
            "text/csv", "application/csv", "text/plain",
            # Excel
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            # Word
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            # Generic binary (some browsers send this for xlsx/docx)
            "application/octet-stream",
            "application/zip",
        }
        ALLOWED_EXTENSIONS = {
            "jpg", "jpeg", "png", "gif", "svg", "webp",
            "pdf",
            "csv",
            "xls", "xlsx",
            "doc", "docx",
            "txt",
        }

        content_type = (file.content_type or "").lower().split(";")[0].strip()
        file_ext     = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        # Validate by extension first (more reliable than MIME type)
        if file_ext not in ALLOWED_EXTENSIONS and content_type not in ALLOWED_MIME_TYPES:
            return func.HttpResponse(
                json.dumps({"error": f"Unsupported file type '.{file_ext}'. Allowed: jpg, jpeg, png, pdf, csv, xls, xlsx, doc, docx, txt."}),
                status_code=400, mimetype="application/json")

        # ── Temp upload validation ─────────────────────────────────────────
        if temp_flag and not session_id:
            return func.HttpResponse(
                json.dumps({"error": "session_id is required for temporary uploads."}),
                status_code=400, mimetype="application/json")

        file_bytes = file.read()

        # Guard against oversized uploads (default cap: 50 MB)
        MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            return func.HttpResponse(
                json.dumps({"error": f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024*1024)} MB."}),
                status_code=413, mimetype="application/json")

        # ── 1. Blob Storage ───────────────────────────────────────────────
        # ── 1. Blob Storage ───────────────────────────────────────────────
        # Check for duplicate filename (non-temp uploads only)
        if not temp_flag:
            table_svc_check = TableService()
            existing = table_svc_check.find_by_filename(filename)
            if existing:
                logging.info("Duplicate upload rejected: '%s' already exists (id=%s)", filename, existing.get("id"))
                return func.HttpResponse(
                    json.dumps({
                        "error": f"A file named '{filename}' already exists. Please delete it first or rename your file.",
                        "duplicate": True,
                        "existing_id": existing.get("id", ""),
                    }),
                    status_code=409, mimetype="application/json")

        blob_svc = BlobService()
        # Use temp prefix when temp flag is set
        if temp_flag:
            custom_blob_name = f"temp/{session_id}/{uuid.uuid4().hex}_{filename}"
        else:
            custom_blob_name = ""  # BlobService will generate UUID prefix
        blob_url = blob_svc.upload(filename, file_bytes,
                                   file.content_type or "application/octet-stream",
                                   blob_name=custom_blob_name)
        logging.info("Blob uploaded: %s", blob_url)

        # ── 2. Table Storage placeholder (status=processing) ──────────────
        table_svc = TableService()
        record_id = table_svc.insert_entity(filename, blob_url, description, tags_input,
                                            temp=temp_flag, session_id=session_id,
                                            uploaded_by=uploaded_by)

        # ── 3. Text extraction (universal — PDF, CSV, Excel, Word, TXT, Image) ──
        t0   = time.time()
        try:
            text, structured_data = extract_with_structured(file_bytes, filename)
        except RuntimeError as extraction_err:
            # Images with no detectable text raise RuntimeError("too little text").
            # Fall back to filename-based text so the upload can still complete.
            logging.warning("Extraction warning for '%s': %s — using filename fallback", filename, extraction_err)
            text            = filename
            structured_data = None

        logging.info("Text extracted: %d chars in %.2fs from '%s' | structured=%s",
                     len(text), time.time() - t0, filename, structured_data is not None)

        # For images/files with no extractable text, use filename as minimal content
        # so the record is still searchable by filename.
        name_lower = filename.lower()
        is_image   = name_lower.endswith((".jpg", ".jpeg", ".png", ".svg", ".gif", ".bmp", ".webp"))
        if len(text.strip()) < 10:
            if is_image:
                logging.info("Image '%s' produced little/no OCR text — using filename as content", filename)
                text = f"Image file: {filename}"
            else:
                return func.HttpResponse(
                    json.dumps({"error": f"Extraction returned too little text from '{filename}'."}),
                    status_code=422, mimetype="application/json")

        # ── 4. OpenAI: summary + tags + embedding ─────────────────────────
        t1 = time.time()

        # generate_summary with error handling (Task 9.3)
        try:
            summary = generate_summary(text)
            if not summary:
                logging.warning("generate_summary returned empty for '%s'", filename)
                summary = ""
        except Exception as sum_exc:
            logging.warning("generate_summary failed for '%s': %s — storing empty summary", filename, sum_exc)
            summary = ""

        # generate_tags with error handling (Task 9.3)
        # Always generate AI tags; merge with any user-provided tags
        try:
            ai_tags_str = generate_tags(text)
            if not ai_tags_str:
                logging.warning("generate_tags returned empty for '%s'", filename)
                ai_tags_str = ""
        except Exception as tag_exc:
            logging.warning("generate_tags failed for '%s': %s — storing empty tags", filename, tag_exc)
            ai_tags_str = ""

        # Merge user-provided tags with AI-generated tags (deduplicated)
        user_tags = [t.strip() for t in tags_input.split(",") if t.strip()]
        ai_tags   = [t.strip() for t in ai_tags_str.split(",") if t.strip()]
        merged    = list(dict.fromkeys(user_tags + ai_tags))  # preserve order, deduplicate
        tags_str  = ", ".join(merged)

        tags_list = [t.strip() for t in tags_str.split(",") if t.strip()]

        embedding = generate_embedding(text)
        logging.info("Summary+tags+embedding in %.2fs | Embedding size: %d",
                     time.time() - t1, len(embedding) if embedding else 0)

        if not embedding:
            return func.HttpResponse(
                json.dumps({"error": "Embedding generation failed. Check AZURE_OPENAI_API_KEY."}),
                status_code=502, mimetype="application/json")

        # ── 5. Upload text + structured_data to Blob, store URLs in Table ──
        t_blob   = time.time()
        text_url = ""
        sd_url   = ""
        try:
            text_url = blob_svc.upload_text(record_id, text)
            logging.info("Text blob uploaded in %.2fs: %s", time.time() - t_blob, text_url)
        except Exception as exc:
            logging.warning("Text blob upload failed (will use inline fallback): %s", exc)

        if structured_data:
            try:
                sd_url = blob_svc.upload_structured_data(record_id, structured_data)
                logging.info("Structured data blob uploaded: %s", sd_url)
            except Exception as exc:
                logging.warning("Structured data blob upload failed: %s", exc)

        table_svc.update_ai_fields(
            filename, text, summary, tags_str,
            structured_data     = structured_data if not sd_url else None,  # inline only if no URL
            text_url            = text_url,
            structured_data_url = sd_url,
        )
        logging.info("Table Storage updated: %s → completed", filename)

        # ── 6. Chunk text + index each chunk with embedding ───────────────────
        t_search = time.time()
        from services.chunking_service import chunk_text
        from services.table_service import store_chunk_embedding
        ensure_index()
        chunks = chunk_text(text, record_id, filename)
        if not chunks:
            chunks = [{"chunk_id": record_id, "doc_id": record_id,
                       "filename": filename, "chunk_index": 0, "text": text[:32000]}]

        for chunk in chunks:
            chunk_emb = generate_embedding(chunk["text"])
            index_document(
                doc_id      = record_id,
                filename    = filename,
                content     = chunk["text"],
                summary     = summary,
                tags        = tags_list,
                blob_url    = blob_url,
                embedding   = chunk_emb,
                chunk_index = chunk["chunk_index"],
                chunk_id    = chunk["chunk_id"],
                uploaded_by = uploaded_by,
            )
            if chunk_emb:
                store_chunk_embedding(chunk["chunk_id"], record_id, chunk_emb)

        logging.info("Indexed %d chunks in %.2fs: doc_id=%s",
                     len(chunks), time.time() - t_search, record_id)

        return func.HttpResponse(
            json.dumps({"id": record_id, "filename": filename,
                        "blob_url": blob_url, "message": "Upload successful."}),
            status_code=201, mimetype="application/json")

    except Exception as exc:
        logging.exception("Upload error.")
        return func.HttpResponse(json.dumps({"error": "Internal server error.", "detail": str(exc)}),
                                 status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# POST /reset-index — delete and recreate AI Search index with correct schema
# ---------------------------------------------------------------------------

@app.route(route="reset-index", methods=["POST"])
def reset_index(req: func.HttpRequest) -> func.HttpResponse:
    """
    Deletes the existing AI Search index and recreates it with the correct schema.
    Call this ONCE if the index was created with a broken schema.
    After calling this, re-upload all documents.
    """
    logging.info("POST /reset-index")
    try:
        deleted = delete_index()
        if not deleted:
            return func.HttpResponse(
                json.dumps({"error": "Failed to delete index."}),
                status_code=500, mimetype="application/json")
        ensure_index()
        return func.HttpResponse(
            json.dumps({"message": "Index reset successfully. Re-upload your documents."}),
            status_code=200, mimetype="application/json")
    except Exception as exc:
        logging.exception("reset-index error.")
        return func.HttpResponse(json.dumps({"error": str(exc)}),
                                 status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# POST /reprocess — auto-reprocess stale documents (schema version upgrade)
# ---------------------------------------------------------------------------

@app.route(route="reprocess", methods=["POST"])
def reprocess(req: func.HttpRequest) -> func.HttpResponse:
    """
    Finds all documents with outdated schema_version and reprocesses them
    by re-downloading from Blob Storage and re-extracting structured data.

    Call this after any backend logic change that affects structured_data format.
    No manual re-upload required.
    """
    logging.info("POST /reprocess")
    try:
        from services.table_service import SCHEMA_VERSION
        from services.extractor     import extract_with_structured
        from azure.storage.blob     import BlobServiceClient

        table_svc = TableService()
        stale     = table_svc.get_stale_documents()

        if not stale:
            return func.HttpResponse(
                json.dumps({"message": f"All documents are up to date (v{SCHEMA_VERSION}).",
                            "updated": 0}),
                status_code=200, mimetype="application/json")

        conn_str   = require_env("AZURE_STORAGE_CONNECTION_STRING")
        blob_svc_c = BlobServiceClient.from_connection_string(conn_str)

        updated = 0
        failed  = []

        for doc in stale:
            filename = doc["filename"]
            blob_url = doc["blob_url"]
            try:
                # Download file bytes from Blob Storage using the SDK's URL parser
                # (avoids fragile split("/") which breaks on blob names containing "/")
                from azure.storage.blob import BlobClient
                blob_client = BlobClient.from_blob_url(
                    blob_url   = blob_url,
                    credential = blob_svc_c.credential,
                )
                file_bytes  = blob_client.download_blob().readall()

                # Re-extract with current logic
                text, structured_data = extract_with_structured(file_bytes, filename)

                # Update Table Storage with new schema
                table_svc.update_ai_fields(
                    filename, text,
                    summary         = "",   # keep existing summary (don't re-call OpenAI)
                    tags            = "",
                    structured_data = structured_data,
                )
                logging.info("Reprocessed: %s → v%d", filename, SCHEMA_VERSION)
                updated += 1

            except Exception as exc:
                logging.exception("Reprocess failed for %s", filename)
                failed.append({"filename": filename, "error": str(exc)})

        return func.HttpResponse(
            json.dumps({
                "message":        f"Reprocess complete. Schema v{SCHEMA_VERSION}.",
                "updated":        updated,
                "failed":         len(failed),
                "failed_details": failed,
            }),
            status_code=200, mimetype="application/json")

    except Exception as exc:
        logging.exception("Reprocess endpoint error.")
        return func.HttpResponse(json.dumps({"error": str(exc)}),
                                 status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# POST /backfill-uploaded-by — one-time migration to add uploaded_by to existing search index chunks
# ---------------------------------------------------------------------------

@app.route(route="backfill-uploaded-by", methods=["POST"])
def backfill_uploaded_by_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    """
    One-time migration: reads all docs from Table Storage and patches
    the search index chunks with the correct uploaded_by value.
    """
    logging.info("POST /backfill-uploaded-by")
    try:
        from services.table_service  import TableService
        from services.search_service import backfill_uploaded_by

        docs = TableService().list_documents()
        doc_id_to_owner = {d["id"]: d["uploaded_by"] for d in docs if d.get("uploaded_by")}
        updated = backfill_uploaded_by(doc_id_to_owner)
        return func.HttpResponse(
            json.dumps({"updated_chunks": updated, "docs_processed": len(doc_id_to_owner)}),
            status_code=200, mimetype="application/json")
    except Exception:
        logging.exception("/backfill-uploaded-by error")
        return func.HttpResponse(
            json.dumps({"error": "Backfill failed"}),
            status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# POST /register
# ---------------------------------------------------------------------------

@app.route(route="register", methods=["POST"])
def register(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body       = req.get_json()
        email      = (body.get("email") or "").strip().lower()
        password   = body.get("password") or ""
        first_name = body.get("first_name", "")
        last_name  = body.get("last_name", "")
        if not email or not password:
            return func.HttpResponse(json.dumps({"error": "email and password required"}),
                                     status_code=400, mimetype="application/json")
        import re, bcrypt
        if (len(password) < 8 or not re.search(r'[A-Z]', password) or
            not re.search(r'[a-z]', password) or not re.search(r'[0-9]', password) or
            not re.search(r'[!@#$%^&*(),.?":{}|<>]', password)):
            return func.HttpResponse(
                json.dumps({"error": "Password must be 8+ chars with uppercase, lowercase, number and special character"}),
                status_code=400, mimetype="application/json")
        from services.table_service import create_user
        pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        created = create_user(email, pw_hash, first_name, last_name)
        if not created:
            return func.HttpResponse(json.dumps({"error": "User already exists"}),
                                     status_code=409, mimetype="application/json")
        return func.HttpResponse(json.dumps({"message": "User created"}),
                                 status_code=201, mimetype="application/json")
    except Exception:
        logging.exception("/register error")
        return func.HttpResponse(json.dumps({"error": "Registration failed"}),
                                 status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# POST /login
# ---------------------------------------------------------------------------

@app.route(route="login", methods=["POST"])
def login(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body     = req.get_json()
        email    = (body.get("email") or "").strip().lower()
        password = body.get("password") or ""
        if not email or not password:
            return func.HttpResponse(json.dumps({"error": "email and password required"}),
                                     status_code=400, mimetype="application/json")
        import bcrypt, jwt as pyjwt
        from services.table_service import get_user
        user = get_user(email)
        if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
            return func.HttpResponse(json.dumps({"error": "Invalid credentials"}),
                                     status_code=401, mimetype="application/json")
        import datetime as dt
        secret = require_env("JWT_SECRET")
        token = pyjwt.encode({
            "email":      email,
            "name":       f"{user['first_name']} {user['last_name']}".strip(),
            "first_name": user["first_name"],
            "last_name":  user["last_name"],
            "exp":        dt.datetime.utcnow() + dt.timedelta(hours=8),
        }, secret, algorithm="HS256")
        return func.HttpResponse(json.dumps({"access_token": token}),
                                 status_code=200, mimetype="application/json")
    except Exception:
        logging.exception("/login error")
        return func.HttpResponse(json.dumps({"error": "Login failed"}),
                                 status_code=500, mimetype="application/json")


# GET /documents — list from Table Storage (lightweight, for UI polling)
# ---------------------------------------------------------------------------

@app.route(route="documents", methods=["GET"])
def documents(req: func.HttpRequest) -> func.HttpResponse:
    t0 = time.time()
    try:
        uploaded_by = req.params.get("uploaded_by", "").strip()
        svc = TableService()
        docs = svc.list_documents_by_user(uploaded_by) if uploaded_by else svc.list_documents()
        logging.info("/documents: %d docs in %.3fs", len(docs), time.time() - t0)
        return func.HttpResponse(json.dumps(docs), status_code=200, mimetype="application/json")
    except Exception as exc:
        logging.exception("/documents error.")
        return func.HttpResponse(json.dumps({"error": str(exc)}),
                                 status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# GET /diagnose — raw Table Storage state for debugging
# ---------------------------------------------------------------------------

@app.route(route="diagnose", methods=["GET"])
def diagnose(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("GET /diagnose")
    try:
        table_svc = TableService()
        docs      = table_svc.list_documents()
        # Augment with text_chars + has_summary by fetching full entities via search
        from azure.data.tables import TableServiceClient
        from services.config import require_env as _re
        conn_str = _re("AZURE_STORAGE_CONNECTION_STRING")
        raw_client = TableServiceClient.from_connection_string(conn_str).get_table_client("documentsmetadata")
        entities   = list(raw_client.query_entities(query_filter="PartitionKey eq 'documents'"))
        entity_map = {e.get("RowKey", ""): e for e in entities}
        report = []
        for d in docs:
            e = entity_map.get(d["id"], {})
            report.append({
                "filename":    d["filename"],
                "status":      d["status"],
                "text_chars":  len(e.get("text", "")),
                "has_summary": bool(d["summary"]),
                "RowKey":      d["id"],
            })
        return func.HttpResponse(json.dumps(report, indent=2),
                                 status_code=200, mimetype="application/json")
    except Exception as exc:
        logging.exception("diagnose error.")
        return func.HttpResponse(json.dumps({"error": str(exc)}),
                                 status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# GET|POST /query — Advanced RAG pipeline
#
# Architecture:
#   1. Multi-query retrieval with HyDE (Hypothetical Document Embedding)
#   2. Intent classification → structured engine OR prose RAG
#   3. Contextual compression → extract relevant passages per chunk
#   4. Grounded generation with strict system prompt (temperature=0)
#   5. Consistent JSON response envelope
# ---------------------------------------------------------------------------

@app.route(route="query", methods=["GET", "POST"])
def query(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("POST /query")

    user_query      = req.params.get("q", "")
    filename_filter = req.params.get("filename", "")
    if not user_query:
        try:
            body            = req.get_json()
            user_query      = body.get("q", "")
            filename_filter = body.get("filename", filename_filter)
        except Exception:
            pass

    if not user_query:
        return func.HttpResponse(
            json.dumps({"error": "'q' is required."}),
            status_code=400, mimetype="application/json")

    # Extract user identity from JWT for document isolation
    uploaded_by = ""
    auth_header = req.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        try:
            import base64 as _b64
            payload_b64 = auth_header.split(".")[1]
            payload_b64 += "=" * (-len(payload_b64) % 4)
            jwt_payload = json.loads(_b64.b64decode(payload_b64))
            uploaded_by = jwt_payload.get("email") or jwt_payload.get("preferred_username") or ""
        except Exception:
            pass

    try:
        t0 = time.time()

        from services.rag_pipeline import run_rag_pipeline
        result = run_rag_pipeline(
            query           = user_query,
            filename_filter = filename_filter,
            uploaded_by     = uploaded_by,
            top_k           = 7,
            use_hyde        = True,
            use_compression = True,
        )

        resp_type    = result.get("type", "text")
        sources      = result.get("sources", [])
        chart_config = result.get("chart_config")
        rows         = result.get("rows", [])
        columns      = result.get("columns", [])

        logging.info("query: type=%s sources=%d elapsed=%.3fs",
                     resp_type, len(sources), time.time() - t0)

        # ── Chart response ────────────────────────────────────────────────
        if resp_type == "chart":
            # Structured engine chart (has chart_config + data rows)
            if chart_config and rows:
                x_key     = chart_config.get("xKey", "")
                axis_info = detect_dual_axis_from_rows(rows, x_key)
                chart_config["series"]   = axis_info["series"]
                chart_config["dualAxis"] = axis_info["dual_axis"]
                if axis_info["dual_axis"]:
                    chart_config["type"] = axis_info.get("chart_type", "composed")
                return func.HttpResponse(
                    _safe_json({
                        "type":         "chart",
                        "answer":       result.get("answer", ""),
                        "data":         rows,
                        "chart_config": chart_config,
                        "script":       result.get("script", ""),
                        "query":        user_query,
                        "sources":      sources,
                    }),
                    status_code=200, mimetype="application/json")

            # RAG-generated chart (has labels + values)
            labels   = result.get("labels", [])
            values   = result.get("values", [])
            series   = result.get("series", {})

            if labels and values:
                return func.HttpResponse(
                    _safe_json({
                        "type":       "chart",
                        "chart_type": result.get("chart_type", "bar"),
                        "labels":     labels,
                        "values":     values,
                        "answer":     result.get("answer", ""),
                        "query":      user_query,
                        "sources":    sources,
                    }),
                    status_code=200, mimetype="application/json")

            if labels and series:
                return func.HttpResponse(
                    _safe_json({
                        "type":       "chart",
                        "chart_type": result.get("chart_type", "bar"),
                        "labels":     labels,
                        "series":     series,
                        "answer":     result.get("answer", ""),
                        "query":      user_query,
                        "sources":    sources,
                    }),
                    status_code=200, mimetype="application/json")

        # ── Table response ────────────────────────────────────────────────
        if resp_type == "table" and rows:
            return func.HttpResponse(
                _safe_json({
                    "type":    "table",
                    "answer":  result.get("answer", ""),
                    "columns": columns,
                    "rows":    rows,
                    "script":  result.get("script", ""),
                    "query":   user_query,
                    "sources": sources,
                }),
                status_code=200, mimetype="application/json")

        # ── Text response (default) ───────────────────────────────────────
        answer = result.get("answer") or "No relevant data found in documents."
        return func.HttpResponse(
            json.dumps({
                "type":    "text",
                "answer":  answer,
                "query":   user_query,
                "sources": sources,
            }),
            status_code=200, mimetype="application/json")

    except Exception as exc:
        logging.exception("Query error.")
        return func.HttpResponse(
            json.dumps({"error": "Internal server error.", "detail": str(exc)}),
            status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# GET /download/:id — generate SAS URL for private blob download
# ---------------------------------------------------------------------------

@app.route(route="download/{id}", methods=["GET"])
def download_document(req: func.HttpRequest) -> func.HttpResponse:
    """Generate a time-limited SAS URL for downloading a private blob."""
    record_id = req.route_params.get("id", "").strip().strip("{}")
    logging.info("[DOWNLOAD] Request | id=%s", record_id)

    if not record_id:
        return func.HttpResponse(
            json.dumps({"error": "Document ID is required."}),
            status_code=400, mimetype="application/json")

    try:
        # Look up the blob_url from Table Storage
        from azure.data.tables import TableServiceClient
        conn_str   = require_env("AZURE_STORAGE_CONNECTION_STRING")
        raw_client = TableServiceClient.from_connection_string(conn_str).get_table_client("documentsmetadata")
        entities   = list(raw_client.query_entities(
            query_filter=f"PartitionKey eq 'documents' and RowKey eq '{record_id}'"
        ))

        if not entities:
            return func.HttpResponse(
                json.dumps({"error": "Document not found."}),
                status_code=404, mimetype="application/json")

        blob_url = entities[0].get("blob_url", "")
        filename = entities[0].get("filename", "file")

        if not blob_url:
            return func.HttpResponse(
                json.dumps({"error": "No blob URL for this document."}),
                status_code=404, mimetype="application/json")

        # Generate SAS URL (valid for 1 hour)
        sas_url = BlobService().generate_sas_url(blob_url, expiry_hours=1)

        return func.HttpResponse(
            json.dumps({"sas_url": sas_url, "filename": filename}),
            status_code=200, mimetype="application/json")

    except Exception as exc:
        logging.exception("[DOWNLOAD] Error for id=%s", record_id)
        return func.HttpResponse(
            json.dumps({"error": "Internal server error.", "detail": str(exc)}),
            status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# DELETE /api/document/{id} — synchronous cascade delete
# ---------------------------------------------------------------------------

@app.route(route="document/{id}", methods=["DELETE"])
def delete_document_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    """
    Synchronous cascade delete.

    Deletes all associated resources (Blob, Search, Table) in-request.
    Queue-based async deletion is disabled — set ENABLE_QUEUE=true in
    local.settings.json to re-enable it and restore the queue worker.
    """
    record_id = req.route_params.get("id", "").strip().strip("{}")
    logging.info("[DELETE API] Request received | id=%s", record_id)

    if not record_id:
        logging.warning("[DELETE API] Missing document ID in request")
        return func.HttpResponse(
            json.dumps({"error": "Document ID is required."}),
            status_code=400,
            mimetype="application/json",
        )

    try:
        logging.info("[DELETE] Processing document | id=%s", record_id)
        result = delete_document(record_id)

        if not result.found:
            logging.warning("[DELETE] Document not found | id=%s", record_id)
            return func.HttpResponse(
                json.dumps({"error": "Document not found", "id": record_id}),
                status_code=404,
                mimetype="application/json",
            )

        if result.success:
            logging.info("[DELETE SUCCESS] id=%s cid=%s", record_id, result.correlation_id[:8])
        else:
            logging.error("[DELETE PARTIAL] id=%s errors=%s", record_id, result.errors)

        return func.HttpResponse(
            json.dumps(result.to_dict()),
            status_code=200,
            mimetype="application/json",
        )

    except Exception as exc:
        logging.error("[DELETE ERROR] id=%s error=%s", record_id, exc)
        logging.exception("DELETE /api/document/%s — unexpected error", record_id)
        return func.HttpResponse(
            json.dumps({"error": "Internal server error.", "detail": str(exc)}),
            status_code=500,
            mimetype="application/json",
        )




# ---------------------------------------------------------------------------
# GET /file?id= — proxy private blob by document ID
# Used by Cards.jsx / ChatInfoSage.jsx to serve private blob images/files
# ---------------------------------------------------------------------------

@app.route(route="file", methods=["GET"])
def serve_file(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("GET /file")
    doc_id = req.params.get("id", "").strip()
    if not doc_id:
        return func.HttpResponse(
            json.dumps({"error": "'id' is required."}),
            status_code=400, mimetype="application/json")
    try:
        # Look up directly by RowKey — faster and works for all statuses
        from azure.data.tables import TableServiceClient
        conn_str   = require_env("AZURE_STORAGE_CONNECTION_STRING")
        raw_client = TableServiceClient.from_connection_string(conn_str).get_table_client("documentsmetadata")
        entities   = list(raw_client.query_entities(
            query_filter=f"PartitionKey eq 'documents' and RowKey eq '{doc_id}'"
        ))

        if not entities:
            return func.HttpResponse(
                json.dumps({"error": "Document not found."}),
                status_code=404, mimetype="application/json")

        entity   = entities[0]
        blob_url = entity.get("blob_url", "")
        filename = entity.get("filename", "file")

        if not blob_url:
            return func.HttpResponse(
                json.dumps({"error": "No blob URL for this document."}),
                status_code=404, mimetype="application/json")

        logging.info("serve_file: downloading blob_url=%s for filename=%s", blob_url, filename)

        from azure.storage.blob import BlobClient, BlobServiceClient as _BSC
        bc = BlobClient.from_blob_url(
            blob_url   = blob_url,
            credential = _BSC.from_connection_string(conn_str).credential,
        )
        data = bc.download_blob().readall()

        # Determine content type from filename extension
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        mime_map = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "gif": "image/gif", "webp": "image/webp", "svg": "image/svg+xml",
            "pdf": "application/pdf", "txt": "text/plain",
            "csv": "text/csv",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "xls": "application/vnd.ms-excel",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "doc": "application/msword",
        }
        content_type = mime_map.get(ext, "application/octet-stream")

        return func.HttpResponse(
            body=data,
            status_code=200,
            mimetype=content_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        logging.exception("serve_file error.")
        return func.HttpResponse(
            json.dumps({"error": "Internal server error.", "detail": str(exc)}),
            status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# POST /cleanup-session — delete all temp blobs for a session
# Called by frontend "Clear Chat" to remove all temp-uploaded files
# ---------------------------------------------------------------------------

@app.route(route="cleanup-session", methods=["POST"])
def cleanup_session(req: func.HttpRequest) -> func.HttpResponse:
    """
    Delete all temporary blobs and Table Storage entities for a given session_id.
    Called when the user clicks "Clear Chat" in the Files Knowledge Bot.
    """
    logging.info("POST /cleanup-session")
    try:
        body       = req.get_json(silent=True) or {}
        session_id = body.get("session_id", "").strip()

        if not session_id:
            return func.HttpResponse(
                json.dumps({"error": "session_id is required."}),
                status_code=400, mimetype="application/json")

        from azure.storage.blob import BlobServiceClient, ContainerClient
        from services.config import require_env as _re

        conn_str   = _re("AZURE_STORAGE_CONNECTION_STRING")
        blob_svc_c = BlobServiceClient.from_connection_string(conn_str)

        deleted_blobs    = 0
        deleted_entities = 0
        errors           = []

        # Delete all blobs under temp/{session_id}/ prefix in both containers
        for container_name in ("documents", "images"):
            try:
                container: ContainerClient = blob_svc_c.get_container_client(container_name)
                prefix = f"temp/{session_id}/"
                blobs  = list(container.list_blobs(name_starts_with=prefix))
                for blob in blobs:
                    try:
                        container.delete_blob(blob.name)
                        deleted_blobs += 1
                        logging.info("Deleted temp blob: %s/%s", container_name, blob.name)
                    except Exception as be:
                        errors.append(f"blob:{container_name}/{blob.name}: {be}")
            except Exception as ce:
                logging.warning("cleanup-session: container '%s' error: %s", container_name, ce)

        # Delete Table Storage entities with matching session_id
        try:
            table_svc = TableService()
            docs      = table_svc.list_documents()
            for doc in docs:
                if doc.get("session_id") == session_id:
                    try:
                        from services.delete_service import delete_document as _del_doc
                        _del_doc(doc["id"])
                        deleted_entities += 1
                        logging.info("Deleted temp entity: %s", doc["id"])
                    except Exception as de:
                        errors.append(f"entity:{doc['id']}: {de}")
        except Exception as te:
            logging.warning("cleanup-session: table cleanup error: %s", te)

        logging.info("cleanup-session: session=%s blobs=%d entities=%d errors=%d",
                     session_id, deleted_blobs, deleted_entities, len(errors))

        return func.HttpResponse(
            json.dumps({
                "message":          f"Session cleanup complete.",
                "deleted_blobs":    deleted_blobs,
                "deleted_entities": deleted_entities,
                "errors":           errors,
            }),
            status_code=200, mimetype="application/json")

    except Exception as exc:
        logging.exception("cleanup-session error.")
        return func.HttpResponse(
            json.dumps({"error": "Internal server error.", "detail": str(exc)}),
            status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# POST /agent/query — Orchestrator: classify → dispatch → save turn → return
# ---------------------------------------------------------------------------

@app.route(route="agent/query", methods=["POST"])
def agent_query(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("POST /agent/query")
    try:
        body = req.get_json()
    except Exception:
        body = {}

    q = (body.get("q") or "").strip()
    session_id = (body.get("session_id") or "").strip()
    image_id = body.get("image_id") or None

    if not q:
        return func.HttpResponse(
            json.dumps({"error": "'q' is required."}),
            status_code=400, mimetype="application/json")
    if not session_id:
        return func.HttpResponse(
            json.dumps({"error": "'session_id' is required."}),
            status_code=400, mimetype="application/json")

    try:
        from services.intent_classifier import classify_intent
        from services.session_service import SessionService
        from services.image_search_service import search_images
        from services.image_understanding_service import analyze_image
        from services.rag_pipeline import run_rag_pipeline
    except Exception as import_exc:
        logging.error("agent_query: import error: %s", import_exc)
        return func.HttpResponse(
            json.dumps({"error": "Internal server error.", "detail": str(import_exc)}),
            status_code=500, mimetype="application/json")

    # 1. Get session context
    context_warning = False
    context = []
    try:
        svc = SessionService()
        context = svc.get_context(session_id, last_n=5)
    except Exception as ctx_exc:
        logging.error("agent_query: session context unavailable: %s", ctx_exc)
        context_warning = True

    # 2. Classify intent
    intent = classify_intent(q, context, image_id=image_id)

    # 3. Dispatch to tool
    try:
        if intent == "image_search":
            from services.image_search_service import (
                _clean_query, _search_searxng, _search_wikimedia_commons,
                _search_wikipedia
            )
            clean_q = _clean_query(q)
            img_source = "wikipedia_fallback"
            results = []

            # Level 1: SearXNG
            searxng_url = os.environ.get("SEARXNG_BASE_URL", "").strip().rstrip("/")
            if searxng_url:
                try:
                    results = _search_searxng(clean_q, searxng_url)
                    if results:
                        img_source = "searxng"
                        logging.info("agent_query: image used SearXNG for %r", q)
                except Exception as exc:
                    logging.warning("agent_query: SearXNG failed (%s)", exc)

            # Level 2: Wikimedia Commons
            if not results:
                try:
                    results = _search_wikimedia_commons(clean_q)
                    if results:
                        img_source = "wikimedia_commons"
                        logging.info("agent_query: image used Wikimedia Commons for %r", q)
                except Exception as exc:
                    logging.warning("agent_query: Wikimedia Commons failed (%s)", exc)

            # Level 3: Wikipedia
            if not results:
                results = _search_wikipedia(clean_q)
                img_source = "wikipedia_fallback"
                logging.info("agent_query: image used Wikipedia fallback for %r", q)

            if not results:
                response = {"type": "text", "data": "No images found for your query.", "source": "image_search", "intent": intent}
            else:
                response = {"type": "image", "data": results, "source": img_source, "intent": intent}

        elif intent == "image_qa":
            if not image_id:
                response = {"type": "text", "data": "Please upload an image first before asking about it.", "source": "upload", "intent": intent}
            else:
                answer = analyze_image(image_id, q)
                response = {"type": "text", "data": answer, "source": "upload", "intent": intent}

        elif intent == "document_qa":
            rag_result = run_rag_pipeline(query=q, top_k=7, use_hyde=True, use_compression=True)
            response = {"type": rag_result.get("type", "text"), "data": rag_result, "source": "knowledge", "intent": intent}

        else:  # general_qa or followup
            # Build context string from session history
            context_str = ""
            if context:
                recent = context[-3:]  # last 3 turns
                parts = []
                for turn in recent:
                    q_text = turn.get("query", "")
                    r_data = turn.get("response_data", "")
                    try:
                        import json as _json
                        r_parsed = _json.loads(r_data) if isinstance(r_data, str) else r_data
                        if isinstance(r_parsed, dict):
                            r_text = r_parsed.get("answer") or r_parsed.get("data") or str(r_parsed)
                        else:
                            r_text = str(r_parsed)
                    except Exception:
                        r_text = str(r_data)
                    if q_text:
                        parts.append(f"User: {q_text}")
                    if r_text and len(str(r_text)) < 500:
                        parts.append(f"Assistant: {r_text[:300]}")
                context_str = "\n".join(parts)

            # Step 1: Try RAG — if uploaded docs have relevant content, use it
            rag_answer = ""
            rag_result = None
            try:
                resolved_q = q
                if intent == "followup" and context:
                    last_turn = context[-1]
                    resolved_q = f"{last_turn.get('query', '')} {q}"
                rag_result = run_rag_pipeline(query=resolved_q, top_k=5, use_hyde=True, use_compression=True)
                rag_answer = rag_result.get("answer", "") or ""
            except Exception as rag_exc:
                logging.warning("agent_query: RAG failed: %s", rag_exc)

            # Detect if RAG found nothing useful
            no_rag_content = (
                not rag_answer.strip()
                or "do not contain" in rag_answer.lower()
                or "no relevant" in rag_answer.lower()
                or "not enough information" in rag_answer.lower()
                or "cannot answer" in rag_answer.lower()
                or "don't have" in rag_answer.lower()
                or "i don't see" in rag_answer.lower()
                or "please upload" in rag_answer.lower()
                or "please specify" in rag_answer.lower()
                or "could you please" in rag_answer.lower()
            )

            if not no_rag_content and rag_result:
                # RAG found relevant content in uploaded docs — use it
                response = {"type": rag_result.get("type", "text"), "data": rag_result, "source": "knowledge", "intent": intent}
            else:
                # RAG found nothing — use OpenAI general knowledge with conversation context
                try:
                    from services.openai_service import _get_client, _deployment
                    client = _get_client()

                    system_prompt = (
                        "You are a helpful, knowledgeable assistant. "
                        "Answer the user's question clearly and concisely. "
                        "If the question refers to something mentioned earlier in the conversation, "
                        "use that context to give a relevant answer. "
                        "Keep answers focused and accurate."
                    )

                    messages = [{"role": "system", "content": system_prompt}]
                    if context_str:
                        messages.append({"role": "user", "content": f"Previous conversation:\n{context_str}"})
                        messages.append({"role": "assistant", "content": "I understand the context."})
                    messages.append({"role": "user", "content": q})

                    oai_resp = client.chat.completions.create(
                        model=_deployment(),
                        messages=messages,
                        temperature=0.7,
                        max_tokens=600,
                    )
                    answer = oai_resp.choices[0].message.content.strip()
                    response = {"type": "text", "data": answer, "source": "knowledge", "intent": intent}

                except Exception as oai_exc:
                    logging.warning("agent_query: OpenAI direct failed: %s", oai_exc)
                    response = {"type": "text", "data": rag_answer or "I could not find an answer. Please try rephrasing.", "source": "knowledge", "intent": intent}

    except KeyError as ke:
        logging.warning("agent_query: not found: %s", ke)
        response = {"type": "text", "data": "The requested resource was not found.", "source": "knowledge", "intent": intent}
    except Exception as tool_exc:
        logging.exception("agent_query: tool execution failed")
        response = {"type": "text", "data": "I encountered an error processing your request. Please try again.", "source": "knowledge", "intent": intent}

    # 4. Attach session metadata
    response["session_id"] = session_id
    if context_warning:
        response["context_warning"] = True

    # 5. Persist turn
    try:
        svc = SessionService()
        svc.save_turn(session_id, q, response)
    except Exception as save_exc:
        logging.error("agent_query: failed to save turn: %s", save_exc)

    return func.HttpResponse(
        _safe_json(response),
        status_code=200, mimetype="application/json")


# ---------------------------------------------------------------------------
# POST /agent/image-search — Image search tool endpoint
# ---------------------------------------------------------------------------

@app.route(route="agent/image-search", methods=["POST"])
def agent_image_search(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("POST /agent/image-search")
    try:
        body = req.get_json()
    except Exception:
        body = {}

    q = (body.get("q") or "").strip()
    session_id = (body.get("session_id") or "").strip()

    if not q:
        return func.HttpResponse(json.dumps({"error": "'q' is required."}), status_code=400, mimetype="application/json")
    if not session_id:
        return func.HttpResponse(json.dumps({"error": "'session_id' is required."}), status_code=400, mimetype="application/json")

    try:
        from services.image_search_service import search_images
        results = search_images(q)
        if not results:
            response = {"type": "text", "data": "No images found for your query.", "source": "image_search", "session_id": session_id, "intent": "image_search"}
        else:
            response = {"type": "image", "data": results, "source": "image_search", "session_id": session_id, "intent": "image_search"}
        return func.HttpResponse(_safe_json(response), status_code=200, mimetype="application/json")
    except Exception as exc:
        logging.exception("agent_image_search error")
        return func.HttpResponse(json.dumps({"error": "Internal server error.", "detail": str(exc)}), status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# POST /agent/analyze-image — Image understanding tool endpoint
# ---------------------------------------------------------------------------

@app.route(route="agent/analyze-image", methods=["POST"])
def agent_analyze_image(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("POST /agent/analyze-image")
    try:
        body = req.get_json()
    except Exception:
        body = {}

    image_id = (body.get("image_id") or "").strip()
    q = (body.get("q") or "").strip()
    session_id = (body.get("session_id") or "").strip()

    if not image_id:
        return func.HttpResponse(json.dumps({"error": "'image_id' is required."}), status_code=400, mimetype="application/json")
    if not q:
        return func.HttpResponse(json.dumps({"error": "'q' is required."}), status_code=400, mimetype="application/json")
    if not session_id:
        return func.HttpResponse(json.dumps({"error": "'session_id' is required."}), status_code=400, mimetype="application/json")

    try:
        from services.image_understanding_service import analyze_image
        answer = analyze_image(image_id, q)
        response = {"type": "text", "data": answer, "source": "upload", "session_id": session_id, "intent": "image_qa"}
        return func.HttpResponse(_safe_json(response), status_code=200, mimetype="application/json")
    except KeyError:
        return func.HttpResponse(json.dumps({"error": "Document not found."}), status_code=404, mimetype="application/json")
    except ValueError as ve:
        return func.HttpResponse(json.dumps({"error": str(ve)}), status_code=400, mimetype="application/json")
    except Exception as exc:
        logging.exception("agent_analyze_image error")
        return func.HttpResponse(json.dumps({"error": "Image analysis failed.", "detail": str(exc)}), status_code=502, mimetype="application/json")


# ---------------------------------------------------------------------------
# POST /agent/ask — Knowledge retrieval tool endpoint
# ---------------------------------------------------------------------------

@app.route(route="agent/ask", methods=["POST"])
def agent_ask(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("POST /agent/ask")
    try:
        body = req.get_json()
    except Exception:
        body = {}

    q = (body.get("q") or "").strip()
    session_id = (body.get("session_id") or "").strip()

    if not q:
        return func.HttpResponse(json.dumps({"error": "'q' is required."}), status_code=400, mimetype="application/json")
    if not session_id:
        return func.HttpResponse(json.dumps({"error": "'session_id' is required."}), status_code=400, mimetype="application/json")

    try:
        from services.rag_pipeline import run_rag_pipeline
        rag_result = run_rag_pipeline(query=q, top_k=7, use_hyde=True, use_compression=True)
        response = {"type": rag_result.get("type", "text"), "data": rag_result, "source": "knowledge", "session_id": session_id, "intent": "general_qa"}
        return func.HttpResponse(_safe_json(response), status_code=200, mimetype="application/json")
    except Exception as exc:
        logging.exception("agent_ask error")
        return func.HttpResponse(json.dumps({"error": "Internal server error.", "detail": str(exc)}), status_code=500, mimetype="application/json")

# ---------------------------------------------------------------------------
# GET /agent/diagnose-search?q= — Diagnose which image source is working
# Shows exactly whether DuckDuckGo or Wikipedia is being used
# ---------------------------------------------------------------------------

@app.route(route="agent/diagnose-search", methods=["GET"])
def agent_diagnose_search(req: func.HttpRequest) -> func.HttpResponse:
    """
    Diagnostic endpoint to test all 3 image search sources.
    Usage: GET /agent/diagnose-search?q=elon+musk
    """
    q = req.params.get("q", "elon musk").strip()
    from services.image_search_service import _clean_query, _search_searxng, _search_wikimedia_commons, _search_wikipedia
    clean_q = _clean_query(q)
    report = {"query": q, "clean_query": clean_q, "searxng": {}, "wikimedia_commons": {}, "wikipedia": {}, "final_source": ""}

    # Test SearXNG
    searxng_url = os.environ.get("SEARXNG_BASE_URL", "").strip()
    if searxng_url:
        try:
            results = _search_searxng(clean_q, searxng_url.rstrip("/"))
            report["searxng"] = {"status": "success" if results else "empty", "count": len(results), "sample": results[0] if results else None}
            if results:
                report["final_source"] = "searxng"
        except Exception as exc:
            report["searxng"] = {"status": "failed", "error": str(exc), "count": 0}
    else:
        report["searxng"] = {"status": "skipped", "reason": "SEARXNG_BASE_URL not set"}

    # Test Wikimedia Commons
    try:
        results = _search_wikimedia_commons(clean_q)
        report["wikimedia_commons"] = {"status": "success" if results else "empty", "count": len(results), "sample": results[0] if results else None}
        if not report["final_source"] and results:
            report["final_source"] = "wikimedia_commons"
    except Exception as exc:
        report["wikimedia_commons"] = {"status": "failed", "error": str(exc), "count": 0}

    # Test Wikipedia
    try:
        results = _search_wikipedia(clean_q)
        report["wikipedia"] = {"status": "success" if results else "empty", "count": len(results), "sample": results[0] if results else None}
        if not report["final_source"] and results:
            report["final_source"] = "wikipedia_fallback"
    except Exception as exc:
        report["wikipedia"] = {"status": "failed", "error": str(exc), "count": 0}

    if not report["final_source"]:
        report["final_source"] = "none — all sources failed"

    return func.HttpResponse(json.dumps(report, indent=2), status_code=200, mimetype="application/json")


# ---------------------------------------------------------------------------
# Chat History Storage — import
# ---------------------------------------------------------------------------

try:
    from services.chat_storage_service import (
        save_message_to_table,
        get_messages_from_table,
        append_message_to_blob,
        get_chat_file_from_blob,
        get_chat_sessions,
        delete_chat_session,
        sync_blob_to_table,
    )
except Exception as _chat_import_exc:
    logging.error("CHAT STORAGE IMPORT ERROR: %s", _chat_import_exc)
    raise

# ---------------------------------------------------------------------------
# POST /api/saveMessage
# ---------------------------------------------------------------------------

@app.route(route="saveMessage", methods=["POST"])
def save_message(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("POST /api/saveMessage")
    try:
        body = req.get_json()
    except Exception:
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON body"}),
            status_code=400, mimetype="application/json"
        )

    user_id    = (body.get("userId")    or "").strip()
    session_id = (body.get("sessionId") or "").strip()
    message    = (body.get("message")   or "").strip()
    role       = (body.get("role")      or "").strip()

    if not all([user_id, session_id, message, role]):
        return func.HttpResponse(
            json.dumps({"error": "userId, sessionId, message, and role are required"}),
            status_code=400, mimetype="application/json"
        )
    if role not in ("user", "assistant"):
        return func.HttpResponse(
            json.dumps({"error": "role must be 'user' or 'assistant'"}),
            status_code=400, mimetype="application/json"
        )

    try:
        entity = save_message_to_table(user_id, session_id, message, role)
        append_message_to_blob(user_id, session_id, message, role, entity["createdAt"])
        return func.HttpResponse(
            json.dumps({"status": "saved", "rowKey": entity["RowKey"], "createdAt": entity["createdAt"]}),
            status_code=201, mimetype="application/json"
        )
    except Exception as exc:
        logging.exception("saveMessage error")
        return func.HttpResponse(
            json.dumps({"error": str(exc)}),
            status_code=500, mimetype="application/json"
        )


# ---------------------------------------------------------------------------
# GET /api/getChatHistory?userId=&sessionId=
# ---------------------------------------------------------------------------

@app.route(route="getChatHistory", methods=["GET"])
def get_chat_history(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("GET /api/getChatHistory")
    user_id    = (req.params.get("userId")    or "").strip()
    session_id = (req.params.get("sessionId") or "").strip()

    if not user_id or not session_id:
        return func.HttpResponse(
            json.dumps({"error": "userId and sessionId query params are required"}),
            status_code=400, mimetype="application/json"
        )

    try:
        messages = get_messages_from_table(user_id, session_id)
        return func.HttpResponse(
            json.dumps({"userId": user_id, "sessionId": session_id, "messages": messages}),
            status_code=200, mimetype="application/json"
        )
    except Exception as exc:
        logging.exception("getChatHistory error")
        return func.HttpResponse(
            json.dumps({"error": str(exc)}),
            status_code=500, mimetype="application/json"
        )


# ---------------------------------------------------------------------------
# GET /api/getChatFile?userId=&sessionId=
# ---------------------------------------------------------------------------

@app.route(route="getChatFile", methods=["GET"])
def get_chat_file(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("GET /api/getChatFile")
    user_id    = (req.params.get("userId")    or "").strip()
    session_id = (req.params.get("sessionId") or "").strip()

    if not user_id or not session_id:
        return func.HttpResponse(
            json.dumps({"error": "userId and sessionId query params are required"}),
            status_code=400, mimetype="application/json"
        )

    try:
        data = get_chat_file_from_blob(user_id, session_id)
        if data is None:
            return func.HttpResponse(
                json.dumps({"error": "Chat file not found"}),
                status_code=404, mimetype="application/json"
            )
        return func.HttpResponse(
            json.dumps(data),
            status_code=200, mimetype="application/json"
        )
    except Exception as exc:
        logging.exception("getChatFile error")
        return func.HttpResponse(
            json.dumps({"error": str(exc)}),
            status_code=500, mimetype="application/json"
        )


# ---------------------------------------------------------------------------
# GET /api/chatSessions?userId=
# ---------------------------------------------------------------------------

@app.route(route="chatSessions", methods=["GET"])
def list_chat_sessions(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("GET /api/chatSessions")
    user_id = (req.params.get("userId") or "").strip()
    if not user_id:
        return func.HttpResponse(
            json.dumps({"error": "userId query param is required"}),
            status_code=400, mimetype="application/json"
        )
    try:
        sessions = get_chat_sessions(user_id)
        return func.HttpResponse(json.dumps(sessions), status_code=200, mimetype="application/json")
    except Exception as exc:
        logging.exception("chatSessions error")
        return func.HttpResponse(json.dumps({"error": str(exc)}), status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# GET /api/chatSession/{sessionId}?userId=
# Returns all messages for a session, sorted by timestamp.
# ---------------------------------------------------------------------------

@app.route(route="chatSession/{sessionId}", methods=["GET"])
def get_chat_session(req: func.HttpRequest) -> func.HttpResponse:
    session_id = req.route_params.get("sessionId", "").strip()
    user_id    = (req.params.get("userId") or "").strip()
    logging.info("GET /api/chatSession/%s", session_id)
    if not user_id or not session_id:
        return func.HttpResponse(
            json.dumps({"error": "userId query param and sessionId path param are required"}),
            status_code=400, mimetype="application/json"
        )
    try:
        raw = get_messages_from_table(user_id, session_id)
        messages = [{"role": m["role"], "content": m["message"], "timestamp": m["createdAt"]} for m in raw]
        return func.HttpResponse(
            json.dumps({"sessionId": session_id, "messages": messages}),
            status_code=200, mimetype="application/json"
        )
    except Exception as exc:
        logging.exception("getChatSession error")
        return func.HttpResponse(json.dumps({"error": str(exc)}), status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# DELETE /api/chatSession/{sessionId}?userId=
# ---------------------------------------------------------------------------

@app.route(route="chatSession/{sessionId}", methods=["DELETE"])
def delete_chat_session_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    session_id = req.route_params.get("sessionId", "").strip()
    user_id    = (req.params.get("userId") or "").strip()
    logging.info("DELETE /api/chatSession/%s", session_id)
    if not user_id or not session_id:
        return func.HttpResponse(
            json.dumps({"error": "userId query param and sessionId path param are required"}),
            status_code=400, mimetype="application/json"
        )
    try:
        delete_chat_session(user_id, session_id)
        return func.HttpResponse(json.dumps({"status": "deleted"}), status_code=200, mimetype="application/json")
    except Exception as exc:
        logging.exception("deleteChat error")
        return func.HttpResponse(json.dumps({"error": str(exc)}), status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# POST /api/shareChat  { userId, sessionId }
# Returns a shareable URL (base URL + sessionId as a read-only link)
# ---------------------------------------------------------------------------

@app.route(route="shareChat", methods=["POST"])
def share_chat(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("POST /api/shareChat")
    try:
        body = req.get_json()
    except Exception:
        return func.HttpResponse(json.dumps({"error": "Invalid JSON body"}), status_code=400, mimetype="application/json")

    user_id    = (body.get("userId")    or "").strip()
    session_id = (body.get("sessionId") or "").strip()
    if not user_id or not session_id:
        return func.HttpResponse(
            json.dumps({"error": "userId and sessionId are required"}),
            status_code=400, mimetype="application/json"
        )

    # Build a shareable link using the frontend base URL env var (falls back to a placeholder)
    frontend_url = os.environ.get("FRONTEND_URL", "https://agreeable-glacier-0b749ee0f.7.azurestaticapps.net").rstrip("/")
    share_url = f"{frontend_url}/chat/{session_id}?userId={user_id}"
    return func.HttpResponse(
        json.dumps({"shareUrl": share_url, "sessionId": session_id}),
        status_code=200, mimetype="application/json"
    )


# ---------------------------------------------------------------------------
# POST /api/syncChat  { userId }
# Syncs Blob Storage → Table Storage for sessions missing from Table.
# Call this once to backfill existing chats so they appear in the sidebar.
# ---------------------------------------------------------------------------

@app.route(route="syncChat", methods=["POST"])
def sync_chat(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("POST /api/syncChat")
    try:
        body = req.get_json()
    except Exception:
        return func.HttpResponse(json.dumps({"error": "Invalid JSON body"}), status_code=400, mimetype="application/json")

    user_id = (body.get("userId") or "").strip()
    if not user_id:
        return func.HttpResponse(json.dumps({"error": "userId is required"}), status_code=400, mimetype="application/json")

    try:
        synced = sync_blob_to_table(user_id)
        return func.HttpResponse(
            json.dumps({"status": "ok", "sessionsSynced": synced}),
            status_code=200, mimetype="application/json"
        )
    except Exception as exc:
        logging.exception("syncChat error")
        return func.HttpResponse(json.dumps({"error": str(exc)}), status_code=500, mimetype="application/json")


# ---------------------------------------------------------------------------
# APIM-friendly aliases (lowercase with underscores)
# ---------------------------------------------------------------------------

# GET /api/get_chat_session?userId=&sessionId=
# Alias for /chatSession/{sessionId}?userId= with query-param-based routing
@app.route(route="get_chat_session", methods=["GET"])
def get_chat_session_alias(req: func.HttpRequest) -> func.HttpResponse:
    session_id = (req.params.get("sessionId") or "").strip()
    user_id    = (req.params.get("userId") or "").strip()
    logging.info("GET /api/get_chat_session (alias)")
    if not user_id or not session_id:
        return func.HttpResponse(
            json.dumps({"error": "userId and sessionId query params are required"}),
            status_code=400, mimetype="application/json"
        )
    try:
        raw = get_messages_from_table(user_id, session_id)
        messages = [{"role": m["role"], "content": m["message"], "timestamp": m["createdAt"]} for m in raw]
        return func.HttpResponse(
            json.dumps({"sessionId": session_id, "messages": messages}),
            status_code=200, mimetype="application/json"
        )
    except Exception as exc:
        logging.exception("get_chat_session_alias error")
        return func.HttpResponse(json.dumps({"error": str(exc)}), status_code=500, mimetype="application/json")


# POST /api/share_chat  { userId, sessionId }
# Alias for /shareChat
@app.route(route="share_chat", methods=["POST"])
def share_chat_alias(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("POST /api/share_chat (alias)")
    try:
        body = req.get_json()
    except Exception:
        return func.HttpResponse(json.dumps({"error": "Invalid JSON body"}), status_code=400, mimetype="application/json")

    user_id    = (body.get("userId")    or "").strip()
    session_id = (body.get("sessionId") or "").strip()
    if not user_id or not session_id:
        return func.HttpResponse(
            json.dumps({"error": "userId and sessionId are required"}),
            status_code=400, mimetype="application/json"
        )

    frontend_url = os.environ.get("FRONTEND_URL", "https://agreeable-glacier-0b749ee0f.7.azurestaticapps.net").rstrip("/")
    share_url = f"{frontend_url}/chat/{session_id}?userId={user_id}"
    return func.HttpResponse(
        json.dumps({"shareUrl": share_url, "sessionId": session_id}),
        status_code=200, mimetype="application/json"
    )
