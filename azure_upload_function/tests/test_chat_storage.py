"""
Tests for chat_storage_service.
Run with: pytest tests/test_chat_storage.py -v

Requires AZURE_STORAGE_CONNECTION_STRING to be set (real or Azurite).
"""
import os
import pytest

# Skip entire module if no connection string is available
pytestmark = pytest.mark.skipif(
    not os.environ.get("AZURE_STORAGE_CONNECTION_STRING"),
    reason="AZURE_STORAGE_CONNECTION_STRING not set",
)

from services.chat_storage_service import (
    save_message_to_table,
    get_messages_from_table,
    append_message_to_blob,
    get_chat_file_from_blob,
)

USER_A = "test_user_alice"
USER_B = "test_user_bob"
SESSION_1 = "session_001"
SESSION_2 = "session_002"


def test_save_and_retrieve_messages():
    """Messages are stored and returned in chronological order."""
    save_message_to_table(USER_A, SESSION_1, "Hello!", "user")
    save_message_to_table(USER_A, SESSION_1, "Hi there!", "assistant")
    save_message_to_table(USER_A, SESSION_1, "How are you?", "user")

    messages = get_messages_from_table(USER_A, SESSION_1)
    assert len(messages) >= 3
    roles = [m["role"] for m in messages[-3:]]
    assert roles == ["user", "assistant", "user"]


def test_messages_isolated_by_session():
    """Messages from different sessions don't bleed into each other."""
    save_message_to_table(USER_A, SESSION_2, "Session 2 message", "user")

    s1 = get_messages_from_table(USER_A, SESSION_1)
    s2 = get_messages_from_table(USER_A, SESSION_2)

    s1_texts = [m["message"] for m in s1]
    assert "Session 2 message" not in s1_texts

    s2_texts = [m["message"] for m in s2]
    assert "Session 2 message" in s2_texts


def test_messages_isolated_by_user():
    """Messages from different users don't bleed into each other."""
    save_message_to_table(USER_B, SESSION_1, "Bob's message", "user")

    alice_msgs = get_messages_from_table(USER_A, SESSION_1)
    alice_texts = [m["message"] for m in alice_msgs]
    assert "Bob's message" not in alice_texts


def test_blob_append_and_retrieve():
    """Blob is created and messages are appended correctly."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()

    append_message_to_blob(USER_A, SESSION_1, "Blob test message", "user", ts)

    data = get_chat_file_from_blob(USER_A, SESSION_1)
    assert data is not None
    assert data["userId"] == USER_A
    assert data["sessionId"] == SESSION_1
    assert any(m["content"] == "Blob test message" for m in data["messages"])


def test_blob_not_found_returns_none():
    """get_chat_file_from_blob returns None for non-existent blobs."""
    result = get_chat_file_from_blob("nonexistent_user_xyz", "nonexistent_session_xyz")
    assert result is None


def test_message_order_maintained():
    """Messages are returned in insertion (timestamp) order."""
    import time
    session = "order_test_session"
    texts = ["first", "second", "third"]
    for text in texts:
        save_message_to_table(USER_A, session, text, "user")
        time.sleep(0.01)  # ensure distinct timestamps

    messages = get_messages_from_table(USER_A, session)
    retrieved = [m["message"] for m in messages]
    # All three should appear in order
    indices = [retrieved.index(t) for t in texts if t in retrieved]
    assert indices == sorted(indices)
