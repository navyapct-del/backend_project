"""
Session service for the Agentic Bot.

Manages conversational session state in Azure Table Storage.
Table name: agentsessions
PartitionKey: session_id
RowKey: zero-padded 10-digit turn index (e.g. "0000000000", "0000000001")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from azure.data.tables import TableServiceClient
from services.config import require_env

TABLE_NAME = "agentsessions"

# Maximum field sizes (Azure Table Storage property limit is 64 KB per property)
_MAX_QUERY_BYTES = 8_000   # 8 KB
_MAX_RESPONSE_BYTES = 32_000  # 32 KB


class SessionService:
    """Manages conversational session turns in Azure Table Storage."""

    def __init__(self) -> None:
        conn_str = require_env("AZURE_STORAGE_CONNECTION_STRING")
        self._svc = TableServiceClient.from_connection_string(
            conn_str,
            connection_timeout=5,
            read_timeout=8,
        )
        try:
            self._svc.create_table_if_not_exists(TABLE_NAME)
            logging.info("SessionService: table '%s' ready.", TABLE_NAME)
        except Exception as exc:
            logging.warning("SessionService: table init warning (non-fatal): %s", exc)

        self._client = self._svc.get_table_client(TABLE_NAME)

    # ------------------------------------------------------------------
    # save_turn
    # ------------------------------------------------------------------

    def save_turn(self, session_id: str, query: str, response: dict) -> None:
        """
        Persist a single conversation turn.

        Args:
            session_id: Unique session identifier (PartitionKey).
            query:      User query text (truncated to 8 KB if needed).
            response:   Structured response dict with optional keys:
                        type, data, intent.
        """
        # Determine next turn index
        try:
            existing = list(
                self._client.query_entities(
                    query_filter=f"PartitionKey eq '{session_id}'",
                    select=["RowKey"],
                )
            )
            turn_index = len(existing)
        except Exception as exc:
            logging.warning(
                "save_turn: could not count existing turns for session=%s: %s",
                session_id, exc,
            )
            turn_index = 0

        row_key = str(turn_index).zfill(10)  # e.g. "0000000000"

        # Truncate query to 8 KB
        query_safe = query.encode("utf-8")[:_MAX_QUERY_BYTES].decode("utf-8", errors="ignore")

        # Serialize response_data, truncate to 32 KB
        response_data_raw = json.dumps(response.get("data", ""), ensure_ascii=False)
        response_data_safe = (
            response_data_raw.encode("utf-8")[:_MAX_RESPONSE_BYTES]
            .decode("utf-8", errors="ignore")
        )

        now = datetime.now(timezone.utc)
        entity = {
            "PartitionKey":   session_id,
            "RowKey":         row_key,
            "query":          query_safe,
            "response_type":  response.get("type", "text"),
            "response_data":  response_data_safe,
            "intent":         response.get("intent", "general_qa"),
            "timestamp":      now.isoformat(),
            "expires_at":     (now + timedelta(hours=24)).isoformat(),
        }

        try:
            self._client.create_entity(entity=entity)
            logging.info(
                "save_turn: session=%s turn=%s intent=%s",
                session_id, row_key, entity["intent"],
            )
        except Exception:
            logging.exception(
                "save_turn: failed to write entity for session=%s turn=%s",
                session_id, row_key,
            )
            raise

    # ------------------------------------------------------------------
    # get_context
    # ------------------------------------------------------------------

    def get_context(self, session_id: str, last_n: int = 5) -> list[dict]:
        """
        Retrieve the last *last_n* non-expired turns for a session.

        Args:
            session_id: Unique session identifier.
            last_n:     Maximum number of turns to return (default 5).

        Returns:
            List of dicts with keys: query, response_type, response_data,
            intent, timestamp.  Ordered oldest → newest.  Empty list if
            the session has no turns or does not exist.
        """
        try:
            entities = list(
                self._client.query_entities(
                    query_filter=f"PartitionKey eq '{session_id}'",
                )
            )
        except Exception as exc:
            logging.warning(
                "get_context: query failed for session=%s: %s", session_id, exc
            )
            return []

        if not entities:
            return []

        now_iso = datetime.now(timezone.utc).isoformat()

        # Filter out expired turns
        valid = [
            e for e in entities
            if e.get("expires_at", "") >= now_iso
        ]

        # Sort chronologically (RowKey is zero-padded → lexicographic == chronological)
        valid.sort(key=lambda e: e.get("RowKey", ""))

        # Return last last_n
        recent = valid[-last_n:] if last_n > 0 else []

        return [
            {
                "query":         e.get("query", ""),
                "response_type": e.get("response_type", "text"),
                "response_data": e.get("response_data", ""),
                "intent":        e.get("intent", "general_qa"),
                "timestamp":     e.get("timestamp", ""),
            }
            for e in recent
        ]
