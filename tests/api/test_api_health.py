"""Phase 0 through the API: is the service alive, and is everything it needs ready?"""

import pytest
import requests

from api_helpers import REQUEST_TIMEOUT_SECONDS

pytestmark = [pytest.mark.integration, pytest.mark.api]


# The service process is up.
def test_health_says_ok(api_url):
    response = requests.get(f"{api_url}/health", timeout=REQUEST_TIMEOUT_SECONDS)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# Every dependency (embedding model, vector store, Ollama + model) is ready.
def test_ready_reports_every_dependency_ok(api_url):
    response = requests.get(f"{api_url}/ready", timeout=REQUEST_TIMEOUT_SECONDS)
    body = response.json()

    assert response.status_code == 200, body
    assert body["ready"] is True
    for check_name in ["embedding_model", "vector_store", "llm_server"]:
        assert body["checks"][check_name]["ok"] is True, body["checks"][check_name]


# The service tells us exactly which models and settings it runs with.
def test_ready_reports_the_settings(api_url):
    settings = requests.get(f"{api_url}/ready", timeout=REQUEST_TIMEOUT_SECONDS).json()["settings"]

    assert settings["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert settings["embedding_dimension"] == 384
    assert settings["generator_model"] == "llama3.2:3b"
    assert settings["chunk_overlap_chars"] < settings["chunk_size_chars"]
