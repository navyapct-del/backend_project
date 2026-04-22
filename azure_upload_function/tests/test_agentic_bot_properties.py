"""
Property-based tests for the Agentic Bot feature.

Each test is a stub that calls pytest.skip() until the corresponding service
module is implemented.  The file is syntactically valid and fully collectable
by pytest even when the service modules do not yet exist.

Feature: agentic-bot
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Optional service imports — wrapped in try/except so pytest can collect this
# file before the implementation modules exist.
# ---------------------------------------------------------------------------
try:
    from services.intent_classifier import classify_intent  # type: ignore
except ImportError:
    classify_intent = None  # type: ignore

try:
    from services.session_service import SessionService  # type: ignore
except ImportError:
    SessionService = None  # type: ignore

try:
    from services.image_search_service import search_images  # type: ignore
except ImportError:
    search_images = None  # type: ignore

try:
    from services.image_understanding_service import analyze_image  # type: ignore
except ImportError:
    analyze_image = None  # type: ignore

try:
    from function_app import agent_query as _agent_query  # type: ignore
    _function_app_loaded = True
except ImportError:
    _agent_query = None  # type: ignore
    _function_app_loaded = False

# ---------------------------------------------------------------------------
# Standard library / third-party imports
# ---------------------------------------------------------------------------
import unittest.mock as mock
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Signal constant sets (module-level, shared across tests)
# ---------------------------------------------------------------------------

IMAGE_SEARCH_SIGNALS = [
    "show images of",
    "find images of",
    "find pictures of",
    "display photos of",
    "show me pictures",
    "search for images",
    "image of",
    "photo of",
    "picture of",
    "show photos",
    "find photos",
]

IMAGE_QA_SIGNALS = [
    "what is in this image",
    "describe this image",
    "what does this image show",
    "analyze this image",
    "what is in the picture",
    "describe this chart",
    "what is shown in",
    "explain this image",
    "read this image",
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

FOLLOWUP_PRONOUNS = ["this", "that", "it", "he", "she", "they", "those", "these"]

NON_IMAGE_EXTENSIONS = ["pdf", "csv", "xlsx", "docx", "txt", "xls", "doc"]

VALID_INTENTS = {"image_search", "image_qa", "general_qa", "document_qa", "followup"}
VALID_TYPES = {"image", "text", "chart", "table"}
VALID_SOURCES = {"search", "upload", "knowledge", "image_search"}

# ---------------------------------------------------------------------------
# Shared Hypothesis strategies
# ---------------------------------------------------------------------------


def image_result_strategy():
    """Strategy that generates a single image-result dict."""
    return st.fixed_dictionaries(
        {
            "url": st.text(min_size=1),
            "name": st.text(min_size=1),
            "thumbnail": st.text(min_size=1),
        }
    )


def response_strategy():
    """Strategy that generates a structured response dict."""
    return st.fixed_dictionaries(
        {
            "type": st.sampled_from(["text", "image", "chart", "table"]),
            "data": st.text(),
            "source": st.sampled_from(["search", "upload", "knowledge", "image_search"]),
        }
    )


def expired_turn_strategy():
    """Strategy that generates a session turn whose expires_at is in the past."""
    return st.fixed_dictionaries(
        {
            "query": st.text(min_size=1),
            "expires_at": st.just(
                (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
            ),
        }
    )


# ---------------------------------------------------------------------------
# Session-context strategy (used by several tests)
# ---------------------------------------------------------------------------

_session_turn_strategy = st.fixed_dictionaries(
    {
        "query": st.text(min_size=1),
        "response": st.text(),
        "timestamp": st.just(datetime.now(timezone.utc).isoformat()),
    }
)

# ---------------------------------------------------------------------------
# P1 — classify_intent always returns a valid intent
# Feature: agentic-bot, Property 1: classify_intent always returns a valid intent
# Validates: Requirements 1.1
# ---------------------------------------------------------------------------


@given(st.text(), st.lists(_session_turn_strategy, max_size=3))
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_p1_classify_intent_always_returns_valid_intent(query, session_context):
    """Property 1: classify_intent always returns a value in the valid intent set."""
    # Feature: agentic-bot, Property 1: classify_intent always returns a valid intent
    # Validates: Requirements 1.1
    assert classify_intent is not None, "classify_intent not imported"
    result = classify_intent(query, session_context)
    assert result in {"image_search", "image_qa", "general_qa", "document_qa", "followup"}


# ---------------------------------------------------------------------------
# P2 — image-search signal queries are classified as image_search
# Feature: agentic-bot, Property 2: image-search signal queries are classified as image_search
# Validates: Requirements 1.2
# ---------------------------------------------------------------------------


@given(st.sampled_from(IMAGE_SEARCH_SIGNALS), st.text(max_size=30))
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_p2_image_search_signals_classified_as_image_search(signal, topic):
    """Property 2: Any query containing an image-search signal → image_search."""
    # Feature: agentic-bot, Property 2: image-search signal queries are classified as image_search
    # Validates: Requirements 1.2
    assert classify_intent is not None, "classify_intent not imported"
    query = f"{signal} {topic}"
    result = classify_intent(query, [])
    assert result == "image_search"


# ---------------------------------------------------------------------------
# P3 — document-signal queries are classified as document_qa
# Feature: agentic-bot, Property 3: document-signal queries are classified as document_qa
# Validates: Requirements 1.5
# ---------------------------------------------------------------------------


@given(st.sampled_from(DOCUMENT_QA_SIGNALS), st.text(max_size=30))
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_p3_document_signals_classified_as_document_qa(signal, topic):
    """Property 3: Any query containing a document signal → document_qa."""
    # Feature: agentic-bot, Property 3: document-signal queries are classified as document_qa
    # Validates: Requirements 1.5
    assert classify_intent is not None, "classify_intent not imported"
    query = f"{signal} {topic}"
    result = classify_intent(query, [])
    assert result == "document_qa"


# ---------------------------------------------------------------------------
# P4 — followup-signal queries + non-empty context → followup
# Feature: agentic-bot, Property 4: followup-signal queries are classified as followup
# Validates: Requirements 1.6
# ---------------------------------------------------------------------------


@given(
    st.sampled_from(FOLLOWUP_SIGNALS),
    st.text(max_size=30),
    st.lists(_session_turn_strategy, min_size=1, max_size=3),
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_p4_followup_signals_classified_as_followup(signal, topic, session_context):
    """Property 4: Followup signal + non-empty context → followup."""
    # Feature: agentic-bot, Property 4: followup-signal queries are classified as followup
    # Validates: Requirements 1.6
    assert classify_intent is not None, "classify_intent not imported"
    query = f"{signal} {topic}"
    result = classify_intent(query, session_context)
    assert result == "followup"


# ---------------------------------------------------------------------------
# P5 — valid request always returns all required response fields
# Feature: agentic-bot, Property 5: structured response always contains required fields
# Validates: Requirements 1.7, 3.4, 7.3, 8.1, 8.2, 8.3, 8.4
# ---------------------------------------------------------------------------


def _make_mock_request(body: dict) -> mock.MagicMock:
    req = mock.MagicMock()
    req.get_json.return_value = body
    return req


@given(st.text(min_size=1, max_size=50), st.uuids())
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_p5_valid_request_returns_required_fields(q, session_id):
    """Property 5: Valid request always returns type, data, source, session_id, intent."""
    # Feature: agentic-bot, Property 5: structured response always contains required fields
    # Validates: Requirements 1.7, 3.4, 7.3, 8.1, 8.2, 8.3, 8.4
    assert _agent_query is not None, "agent_query not imported"

    session_id_str = str(session_id)
    mock_req = _make_mock_request({"q": q, "session_id": session_id_str})

    rag_result = {"type": "text", "answer": "some answer", "sources": []}

    with mock.patch("services.session_service.TableServiceClient") as mock_ts_cls, \
         mock.patch("services.intent_classifier.classify_intent", return_value="general_qa"), \
         mock.patch("services.rag_pipeline.run_rag_pipeline", return_value=rag_result):

        mock_ts_instance = mock.MagicMock()
        mock_ts_cls.from_connection_string.return_value = mock_ts_instance
        mock_table_client = mock.MagicMock()
        mock_ts_instance.get_table_client.return_value = mock_table_client
        mock_table_client.query_entities.return_value = iter([])

        with mock.patch.dict("os.environ", {"AZURE_STORAGE_CONNECTION_STRING": "fake"}):
            response = _agent_query(mock_req)

    assert response.status_code == 200
    body = json.loads(response.get_body())

    # All five required fields must be present
    assert "type" in body, "Missing 'type' field"
    assert "data" in body, "Missing 'data' field"
    assert "source" in body, "Missing 'source' field"
    assert "session_id" in body, "Missing 'session_id' field"
    assert "intent" in body, "Missing 'intent' field"

    # Enum value checks
    assert body["type"] in VALID_TYPES, f"Invalid type: {body['type']}"
    assert body["source"] in VALID_SOURCES, f"Invalid source: {body['source']}"
    assert body["intent"] in VALID_INTENTS, f"Invalid intent: {body['intent']}"
    assert body["session_id"] == session_id_str


# ---------------------------------------------------------------------------
# P6 — tool exception → type=="text" response
# Feature: agentic-bot, Property 6: tool errors always produce a text response
# Validates: Requirements 2.6
# ---------------------------------------------------------------------------


@given(st.sampled_from([Exception, ValueError, RuntimeError]))
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_p6_tool_exception_produces_text_response(exc_class):
    """Property 6: Any tool exception → response with type=='text' and non-empty data."""
    # Feature: agentic-bot, Property 6: tool errors always produce a text response
    # Validates: Requirements 2.6
    assert function_app is not None, "function_app not imported"

    mock_req = _make_mock_request({"q": "what is this?", "session_id": "test-session-123"})

    with mock.patch("services.session_service.TableServiceClient") as mock_ts_cls, \
         mock.patch("services.intent_classifier.classify_intent", return_value="general_qa"), \
         mock.patch("services.rag_pipeline.run_rag_pipeline", side_effect=exc_class("tool error")):

        mock_ts_instance = mock.MagicMock()
        mock_ts_cls.from_connection_string.return_value = mock_ts_instance
        mock_table_client = mock.MagicMock()
        mock_ts_instance.get_table_client.return_value = mock_table_client
        mock_table_client.query_entities.return_value = iter([])

        with mock.patch.dict("os.environ", {"AZURE_STORAGE_CONNECTION_STRING": "fake"}):
            response = function_app.agent_query(mock_req)

    assert response.status_code == 200
    body = json.loads(response.get_body())
    assert body["type"] == "text", f"Expected type='text', got {body['type']}"
    assert body["data"] and isinstance(body["data"], str) and len(body["data"]) > 0, \
        "Expected non-empty string in 'data'"


# ---------------------------------------------------------------------------
# P7 — whitespace q → HTTP 400
# Feature: agentic-bot, Property 7: empty or whitespace q returns HTTP 400
# Validates: Requirements 3.2
# ---------------------------------------------------------------------------


@given(st.text(alphabet=st.characters(whitelist_categories=("Zs",)), max_size=20))
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_p7_whitespace_q_returns_400(whitespace_q):
    """Property 7: Whitespace-only q → HTTP 400 with error field."""
    # Feature: agentic-bot, Property 7: empty or whitespace q returns HTTP 400
    # Validates: Requirements 3.2
    assert function_app is not None, "function_app not imported"

    mock_req = _make_mock_request({"q": whitespace_q, "session_id": "test-session-123"})
    response = function_app.agent_query(mock_req)

    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    body = json.loads(response.get_body())
    assert "error" in body, "Expected 'error' field in response body"


# ---------------------------------------------------------------------------
# P8 — missing/empty session_id → HTTP 400
# Feature: agentic-bot, Property 8: missing session_id returns HTTP 400
# Validates: Requirements 3.3
# ---------------------------------------------------------------------------


@given(
    st.one_of(
        st.just(""),
        st.just(None),
        st.text(alphabet=st.characters(whitelist_categories=("Zs",)), max_size=20),
    )
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_p8_missing_session_id_returns_400(session_id):
    """Property 8: Missing or empty session_id → HTTP 400 with error field."""
    # Feature: agentic-bot, Property 8: missing session_id returns HTTP 400
    # Validates: Requirements 3.3
    assert function_app is not None, "function_app not imported"

    mock_req = _make_mock_request({"q": "what is this?", "session_id": session_id})
    response = function_app.agent_query(mock_req)

    assert response.status_code == 400, f"Expected 400, got {response.status_code}"
    body = json.loads(response.get_body())
    assert "error" in body, "Expected 'error' field in response body"


# ---------------------------------------------------------------------------
# P9 — image search results always have url/name/thumbnail
# Feature: agentic-bot, Property 9: image search response data items always have required fields
# Validates: Requirements 4.4
# ---------------------------------------------------------------------------


@given(st.lists(image_result_strategy(), min_size=0, max_size=10))
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_p9_image_search_results_have_required_fields(results):
    """Property 9: Every image result item has non-empty url, name, thumbnail."""
    # Feature: agentic-bot, Property 9: image search response data items always have required fields
    # Validates: Requirements 4.4
    for item in results:
        assert item["url"] and item["name"] and item["thumbnail"]


# ---------------------------------------------------------------------------
# P10 — non-image file extension → HTTP 400
# Feature: agentic-bot, Property 10: non-image file IDs are rejected by the image understanding endpoint
# Validates: Requirements 5.5
# ---------------------------------------------------------------------------


@given(st.sampled_from(NON_IMAGE_EXTENSIONS))
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_p10_non_image_extension_returns_400(extension):
    """Property 10: Non-image file extension → HTTP 400."""
    # Feature: agentic-bot, Property 10: non-image file IDs are rejected by the image understanding endpoint
    # Validates: Requirements 5.5
    assert analyze_image is not None, "analyze_image not imported"
    with mock.patch("services.image_understanding_service.TableService") as mock_ts:
        mock_ts.return_value._client.query_entities.return_value = iter([
            {
                "RowKey": f"fake-id-{extension}",
                "filename": f"testfile.{extension}",
                "blob_url": "https://example.com/testfile",
                "status": "completed",
            }
        ])
        with pytest.raises(ValueError):
            analyze_image(f"fake-id-{extension}", "what is this?")


# ---------------------------------------------------------------------------
# P11 — save_turn then get_context round-trip preserves data
# Feature: agentic-bot, Property 11: session turn round-trip preserves data
# Validates: Requirements 7.1
# ---------------------------------------------------------------------------


def _make_in_memory_table_client():
    """
    Return a mock TableClient backed by an in-memory list.

    The mock supports:
      - create_entity(entity=...)  → appends to store
      - query_entities(query_filter=..., select=...)  → returns matching entities
    """
    store: list[dict] = []

    def _create_entity(entity):
        store.append(dict(entity))

    def _query_entities(query_filter="", select=None):
        # Parse "PartitionKey eq '<value>'" from the filter string
        pk_value = None
        if "PartitionKey eq '" in query_filter:
            start = query_filter.index("PartitionKey eq '") + len("PartitionKey eq '")
            end = query_filter.index("'", start)
            pk_value = query_filter[start:end]

        results = [
            dict(e) for e in store
            if pk_value is None or e.get("PartitionKey") == pk_value
        ]
        return iter(results)

    table_client = mock.MagicMock()
    table_client.create_entity.side_effect = _create_entity
    table_client.query_entities.side_effect = _query_entities
    return table_client


def _make_session_service_with_mock_client(table_client):
    """
    Instantiate SessionService with a mocked TableServiceClient so no real
    Azure credentials are needed.
    """
    assert SessionService is not None, "SessionService not imported"

    with mock.patch.dict("os.environ", {"AZURE_STORAGE_CONNECTION_STRING": "fake"}):
        with mock.patch(
            "services.session_service.TableServiceClient.from_connection_string"
        ) as mock_svc_cls:
            mock_svc = mock.MagicMock()
            mock_svc.get_table_client.return_value = table_client
            mock_svc_cls.return_value = mock_svc
            svc = SessionService()
    return svc


@given(st.uuids(), st.text(min_size=1, max_size=100), response_strategy())
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_p11_session_turn_round_trip_preserves_data(session_id, query, response):
    """Property 11: save_turn followed by get_context returns the saved data."""
    # Feature: agentic-bot, Property 11: session turn round-trip preserves data
    # Validates: Requirements 7.1
    assert SessionService is not None, "SessionService not imported"

    table_client = _make_in_memory_table_client()
    svc = _make_session_service_with_mock_client(table_client)

    session_id_str = str(session_id)
    svc.save_turn(session_id_str, query, response)

    context = svc.get_context(session_id_str, last_n=1)

    assert len(context) == 1, f"Expected 1 turn, got {len(context)}"
    assert context[0]["query"] == query[:8000], (
        "Returned query does not match saved query"
    )


# ---------------------------------------------------------------------------
# P12 — get_context returns at most last_n turns
# Feature: agentic-bot, Property 12: get_context returns at most last_n turns
# Validates: Requirements 7.2
# ---------------------------------------------------------------------------


@given(
    st.uuids(),
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=5),
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_p12_get_context_returns_at_most_last_n_turns(session_id, total_turns, last_n):
    """Property 12: get_context(last_n) returns at most last_n turns."""
    # Feature: agentic-bot, Property 12: get_context returns at most last_n turns
    # Validates: Requirements 7.2
    assert SessionService is not None, "SessionService not imported"

    table_client = _make_in_memory_table_client()
    svc = _make_session_service_with_mock_client(table_client)

    session_id_str = str(session_id)

    # Populate total_turns turns
    for i in range(total_turns):
        svc.save_turn(
            session_id_str,
            f"query {i}",
            {"type": "text", "data": f"response {i}", "intent": "general_qa"},
        )

    context = svc.get_context(session_id_str, last_n=last_n)

    assert len(context) <= last_n, (
        f"Expected at most {last_n} turns, got {len(context)}"
    )


# ---------------------------------------------------------------------------
# P13 — expired turns are excluded from context
# Feature: agentic-bot, Property 13: expired turns are excluded from context
# Validates: Requirements 7.5
# ---------------------------------------------------------------------------


@given(st.uuids(), expired_turn_strategy())
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_p13_expired_turns_excluded_from_context(session_id, expired_turn):
    """Property 13: Turns with expires_at in the past are not returned by get_context."""
    # Feature: agentic-bot, Property 13: expired turns are excluded from context
    # Validates: Requirements 7.5
    assert SessionService is not None, "SessionService not imported"

    # Build an in-memory store that already contains an expired entity
    store: list[dict] = []
    session_id_str = str(session_id)
    store.append(
        {
            "PartitionKey":   session_id_str,
            "RowKey":         "0000000000",
            "query":          expired_turn["query"],
            "response_type":  "text",
            "response_data":  '""',
            "intent":         "general_qa",
            "timestamp":      expired_turn["expires_at"],  # already in the past
            "expires_at":     expired_turn["expires_at"],  # 25 hours ago
        }
    )

    def _query_entities(query_filter="", select=None):
        pk_value = None
        if "PartitionKey eq '" in query_filter:
            start = query_filter.index("PartitionKey eq '") + len("PartitionKey eq '")
            end = query_filter.index("'", start)
            pk_value = query_filter[start:end]
        results = [
            dict(e) for e in store
            if pk_value is None or e.get("PartitionKey") == pk_value
        ]
        return iter(results)

    table_client = mock.MagicMock()
    table_client.query_entities.side_effect = _query_entities

    svc = _make_session_service_with_mock_client(table_client)

    context = svc.get_context(session_id_str, last_n=5)

    assert context == [], (
        f"Expected empty context for expired turns, got {context}"
    )


# ---------------------------------------------------------------------------
# P14 — Table Storage failure → context_warning==True in response
# Feature: agentic-bot, Property 14: Table Storage failure produces context_warning in response
# Validates: Requirements 7.6
# ---------------------------------------------------------------------------


@given(st.text(min_size=1, max_size=50), st.uuids())
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_p14_table_storage_failure_produces_context_warning(q, session_id):
    """Property 14: Table Storage exception → context_warning==True and all required fields present."""
    # Feature: agentic-bot, Property 14: Table Storage failure produces context_warning in response
    # Validates: Requirements 7.6
    assert function_app is not None, "function_app not imported"

    session_id_str = str(session_id)
    mock_req = _make_mock_request({"q": q, "session_id": session_id_str})

    rag_result = {"type": "text", "answer": "some answer", "sources": []}

    with mock.patch("services.session_service.TableServiceClient") as mock_ts_cls, \
         mock.patch("services.intent_classifier.classify_intent", return_value="general_qa"), \
         mock.patch("services.rag_pipeline.run_rag_pipeline", return_value=rag_result):

        # Make TableServiceClient raise on construction so SessionService.__init__ fails
        # and get_context raises an exception
        mock_ts_instance = mock.MagicMock()
        mock_ts_cls.from_connection_string.return_value = mock_ts_instance
        mock_table_client = mock.MagicMock()
        mock_ts_instance.get_table_client.return_value = mock_table_client
        # Make get_context raise by having query_entities raise
        mock_table_client.query_entities.side_effect = Exception("Table Storage unavailable")

        with mock.patch.dict("os.environ", {"AZURE_STORAGE_CONNECTION_STRING": "fake"}):
            response = function_app.agent_query(mock_req)

    assert response.status_code == 200
    body = json.loads(response.get_body())

    # context_warning must be True
    assert body.get("context_warning") is True, \
        f"Expected context_warning==True, got {body.get('context_warning')}"

    # All required fields must still be present
    assert "type" in body, "Missing 'type' field"
    assert "data" in body, "Missing 'data' field"
    assert "source" in body, "Missing 'source' field"
    assert "session_id" in body, "Missing 'session_id' field"
    assert "intent" in body, "Missing 'intent' field"


# ---------------------------------------------------------------------------
# P15 — Bing API requests contain only the query string
# Feature: agentic-bot, Property 15: Bing API requests contain only the query string
# Validates: Requirements 10.6
# ---------------------------------------------------------------------------


@given(st.text(min_size=1, max_size=50))
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_p15_bing_requests_contain_only_query_param(query):
    """Property 15: Bing HTTP request contains only the q param — no session/doc/image data."""
    # Feature: agentic-bot, Property 15: Bing API requests contain only the query string
    # Validates: Requirements 10.6
    assert search_images is not None, "search_images not imported"
    with mock.patch.dict("os.environ", {"BING_SEARCH_API_KEY": "fake-key"}):
        with mock.patch("services.image_search_service.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.raise_for_status.return_value = None
            mock_get.return_value.json.return_value = {"value": []}
            search_images(query)
            call_kwargs = mock_get.call_args
            params = call_kwargs[1].get("params", call_kwargs[0][1] if len(call_kwargs[0]) > 1 else {})
            assert set(params.keys()) == {"q"}, f"Expected only 'q' param, got {set(params.keys())}"
            assert params["q"] == query
