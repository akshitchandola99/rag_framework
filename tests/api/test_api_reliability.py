"""Reliability: what the API does when the LLM server is down."""

import pytest
import requests

from api_helpers import REQUEST_TIMEOUT_SECONDS
from api_helpers import chat
from api_helpers import search
from api_helpers import start_server
from api_helpers import stop_server
from api_helpers import unique_text
from api_helpers import upload_and_wait

pytestmark = [pytest.mark.integration, pytest.mark.api]

# Nothing listens on port 9, so every call to this "Ollama" fails at once.
UNREACHABLE_OLLAMA_URL = "http://127.0.0.1:9"


# Start a second server whose Ollama address is wrong, with one document in it.
@pytest.fixture(scope="module")
def api_url_without_llm(tmp_path_factory):
    storage_folder = tmp_path_factory.mktemp("api_storage_no_llm")
    process, base_url = start_server(storage_folder, {"RAGQA_OLLAMA_URL": UNREACHABLE_OLLAMA_URL})
    upload_and_wait(base_url, "leave.md", unique_text("Annual leave is 27 days per year."))
    yield base_url
    stop_server(process)


# Liveness stays OK: the process itself is fine.
def test_health_is_still_ok(api_url_without_llm):
    response = requests.get(f"{api_url_without_llm}/health", timeout=REQUEST_TIMEOUT_SECONDS)

    assert response.status_code == 200


# Readiness says 503 and names the broken dependency.
def test_ready_is_503_and_names_the_llm(api_url_without_llm):
    response = requests.get(f"{api_url_without_llm}/ready", timeout=REQUEST_TIMEOUT_SECONDS)
    body = response.json()

    assert response.status_code == 503
    assert body["ready"] is False
    assert body["checks"]["llm_server"]["ok"] is False
    assert body["checks"]["embedding_model"]["ok"] is True


# Chat gives a clear 503 error instead of crashing or hanging.
def test_chat_gives_503_with_a_clear_message(api_url_without_llm):
    response = chat(api_url_without_llm, "How much annual leave do I get?")

    assert response.status_code == 503
    assert "not reachable" in response.json()["error"]


# Search does not need the LLM, so it keeps working (graceful degradation).
def test_search_still_works_without_the_llm(api_url_without_llm):
    hits = search(api_url_without_llm, "How much annual leave do I get?")["hits"]

    assert len(hits) == 1
    assert "27 days" in hits[0]["text"]
