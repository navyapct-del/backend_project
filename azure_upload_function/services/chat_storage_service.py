"""
Chat history storage service.
Handles both Table Storage (structured messages) and Blob Storage (full chat logs).
"""
import json
import logging
import os
from datetime import datetime, timezone
from threading import Lock
from typing import Optional

from azure.data.tables import TableServiceClient, TableClient
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError

TABLE_NAME = "ChatMessages"
BLOB_CONTAINER = "chat-history"

# Monotonic counter ensures RowKeys are strictly ordered even when two messages
# are saved within the same microsecond (e.g. user + assistant in one request).
_row_key_lock    = Lock()
_row_key_counter = 0


def _next_row_key() -> str:
    global _row_key_counter
    with _row_key_lock:
        now = datetime.now(timezone.utc)
        _row_key_counter += 1
        seq = _row_key_counter
    return now.strftime("%Y%m%dT%H%M%S%f") + f"{seq:06d}Z"


def _get_connection_string() -> str:
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip()
    if not conn:
        raise EnvironmentError("Missing required environment variable: AZURE_STORAGE_CONNECTION_STRING")
    return conn


def _get_table_client() -> TableClient:
    conn = _get_connection_string()
    service = TableServiceClient.from_connection_string(conn)
    try:
        service.create_table(TABLE_NAME)
    except ResourceExistsError:
        pass
    return service.get_table_client(TABLE_NAME)


def _get_blob_container_client():
    conn = _get_connection_string()
    service = BlobServiceClient.from_connection_string(conn)
    container = service.get_container_client(BLOB_CONTAINER)
    try:
        container.create_container()
    except ResourceExistsError:
        pass
    return container


# ---------------------------------------------------------------------------
# Table Storage operations
# ---------------------------------------------------------------------------

def save_message_to_table(user_id: str, session_id: str, message: str, role: str) -> dict:
    """Insert a single chat message into Table Storage."""
    now = datetime.now(timezone.utc)
    # Monotonic RowKey: ISO timestamp + counter suffix — guarantees strict ordering
    # even when two messages are saved within the same microsecond.
    row_key = _next_row_key()

    entity = {
        "PartitionKey": user_id,
        "RowKey": row_key,
        "message": message,
        "role": role,
        "sessionId": session_id,
        "createdAt": now.isoformat(),
    }

    client = _get_table_client()
    client.upsert_entity(entity)  # idempotent
    logging.info("Saved message to table: userId=%s sessionId=%s role=%s", user_id, session_id, role)
    return entity


def get_messages_from_table(user_id: str, session_id: str) -> list[dict]:
    """Fetch all messages for a user+session, ordered by timestamp."""
    client = _get_table_client()
    filter_query = (
        f"PartitionKey eq '{user_id}' and sessionId eq '{session_id}'"
    )
    entities = list(client.query_entities(filter_query))
    # Sort by RowKey (ISO timestamp) ascending
    entities.sort(key=lambda e: e["RowKey"])
    return [
        {
            "userId": e["PartitionKey"],
            "rowKey": e["RowKey"],
            "message": e.get("message", ""),
            "role": e.get("role", ""),
            "sessionId": e.get("sessionId", ""),
            "createdAt": e.get("createdAt", ""),
        }
        for e in entities
    ]


# ---------------------------------------------------------------------------
# Blob Storage operations
# ---------------------------------------------------------------------------

def _blob_path(user_id: str, session_id: str) -> str:
    return f"{user_id}/{session_id}.json"


def append_message_to_blob(user_id: str, session_id: str, message: str, role: str, timestamp: str) -> None:
    """Append a message to the session's JSON blob (create if not exists)."""
    container = _get_blob_container_client()
    blob_name = _blob_path(user_id, session_id)
    blob_client: BlobClient = container.get_blob_client(blob_name)

    new_entry = {"role": role, "content": message, "timestamp": timestamp}

    try:
        existing = blob_client.download_blob().readall()
        data = json.loads(existing)
    except ResourceNotFoundError:
        data = {"userId": user_id, "sessionId": session_id, "messages": []}

    data["messages"].append(new_entry)
    blob_client.upload_blob(json.dumps(data, indent=2), overwrite=True)
    logging.info("Updated blob: %s", blob_name)


def get_chat_file_from_blob(user_id: str, session_id: str) -> Optional[dict]:
    """Fetch the full chat log JSON from Blob Storage. Returns None if not found."""
    container = _get_blob_container_client()
    blob_name = _blob_path(user_id, session_id)
    blob_client: BlobClient = container.get_blob_client(blob_name)
    try:
        content = blob_client.download_blob().readall()
        return json.loads(content)
    except ResourceNotFoundError:
        return None


def get_chat_sessions(user_id: str) -> list[dict]:
    """Return one summary entry per unique sessionId for a user, sorted newest first."""
    client = _get_table_client()
    entities = list(client.query_entities(f"PartitionKey eq '{user_id}'"))

    # First pass: collect all data per session
    sessions: dict[str, dict] = {}
    for e in entities:
        sid = e.get("sessionId", "")
        if not sid:
            continue
        updated = e.get("createdAt", "")
        if sid not in sessions:
            sessions[sid] = {"sessionId": sid, "title": "", "updatedAt": updated, "first_user_msg": ""}
        # Track the latest updatedAt
        if updated > sessions[sid]["updatedAt"]:
            sessions[sid]["updatedAt"] = updated
        # Capture the earliest user message as title candidate (RowKey is ISO timestamp, sort ascending)
        if e.get("role") == "user":
            row_key = e.get("RowKey", "")
            existing_rk = sessions[sid].get("_first_user_rk", "")
            if not existing_rk or row_key < existing_rk:
                sessions[sid]["title"] = (e.get("message") or "")[:60]
                sessions[sid]["_first_user_rk"] = row_key

    # Clean up internal tracking key and apply fallback title
    result = []
    for sid, s in sessions.items():
        s.pop("_first_user_rk", None)
        if not s["title"]:
            s["title"] = sid[:8]
        result.append(s)

    return sorted(result, key=lambda s: s["updatedAt"], reverse=True)


def sync_blob_to_table(user_id: str) -> int:
    """
    Scan Blob Storage for sessions belonging to user_id that have no Table Storage entries.
    For each missing session, insert the messages from the blob into Table Storage.
    Returns the number of sessions synced.
    """
    container = _get_blob_container_client()
    table_client = _get_table_client()

    # List all blobs for this user (prefix = user_id/)
    prefix = f"{user_id}/"
    blobs = list(container.list_blobs(name_starts_with=prefix))

    # Get existing session IDs from Table Storage
    existing_entities = list(table_client.query_entities(f"PartitionKey eq '{user_id}'"))
    existing_sessions = {e.get("sessionId") for e in existing_entities if e.get("sessionId")}

    synced = 0
    for blob in blobs:
        blob_name = blob.name  # e.g. "user@email.com/session-uuid.json"
        # Extract session_id from blob name
        filename = blob_name[len(prefix):]
        if not filename.endswith(".json"):
            continue
        session_id = filename[:-5]  # strip .json

        if session_id in existing_sessions:
            continue  # already in Table Storage

        # Read blob and insert messages into Table Storage
        try:
            blob_client = container.get_blob_client(blob_name)
            content = blob_client.download_blob().readall()
            data = json.loads(content)
            messages = data.get("messages", [])
            for msg in messages:
                role = msg.get("role", "")
                message = msg.get("content", "")
                timestamp = msg.get("timestamp", datetime.now(timezone.utc).isoformat())
                if not role or not message:
                    continue
                # Use timestamp as RowKey base
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    row_key = dt.strftime("%Y%m%dT%H%M%S%f") + "Z"
                except Exception:
                    row_key = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "Z"

                entity = {
                    "PartitionKey": user_id,
                    "RowKey": row_key,
                    "message": message,
                    "role": role,
                    "sessionId": session_id,
                    "createdAt": timestamp,
                }
                table_client.upsert_entity(entity)
            synced += 1
            logging.info("Synced blob session to table: userId=%s sessionId=%s msgs=%d", user_id, session_id, len(messages))
        except Exception as exc:
            logging.warning("Failed to sync blob %s: %s", blob_name, exc)

    return synced


def delete_chat_session(user_id: str, session_id: str) -> None:
    """Delete all Table rows and the Blob for a session."""
    # Table rows
    client = _get_table_client()
    entities = list(client.query_entities(
        f"PartitionKey eq '{user_id}' and sessionId eq '{session_id}'"
    ))
    for e in entities:
        try:
            client.delete_entity(partition_key=e["PartitionKey"], row_key=e["RowKey"])
        except ResourceNotFoundError:
            pass

    # Blob
    try:
        container = _get_blob_container_client()
        container.get_blob_client(_blob_path(user_id, session_id)).delete_blob()
    except ResourceNotFoundError:
        pass

    logging.info("Deleted session: userId=%s sessionId=%s", user_id, session_id)
