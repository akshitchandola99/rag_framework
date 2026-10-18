"""Phase 8: guard rails for unit tests, which must stay fast and offline."""

import pytest
import requests


# Fail any unit test that tries a real HTTP call (for example to Ollama) instead of faking it.
@pytest.fixture(autouse=True)
def block_real_http(monkeypatch):

    # Stands in for every real request; a test that fakes requests.post itself replaces this.
    def refuse_request(self, method, url, *args, **kwargs):
        raise RuntimeError(f"Unit tests must not make real HTTP calls ({method} {url}). Fake it, or move the test to tests/integration.")

    monkeypatch.setattr(requests.Session, "request", refuse_request)
