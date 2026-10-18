"""Phase 4 through the API: search results, limits, bad input and access rules."""

import pytest
import requests

from api_helpers import REQUEST_TIMEOUT_SECONDS
from api_helpers import search
from api_helpers import unique_text
from api_helpers import upload_and_wait

pytestmark = [pytest.mark.integration, pytest.mark.api]

RECEIPTS_QUESTION = "How many days do I have to submit travel receipts?"


# Call POST /search and return the raw HTTP response (for error cases).
def post_search(api_url: str, request_body: dict) -> requests.Response:
    return requests.post(f"{api_url}/search", json=request_body, timeout=REQUEST_TIMEOUT_SECONDS)


# A known question finds its evidence in the top 3, ranked 1, 2, 3 with the best score first.
def test_known_question_finds_the_evidence(api_url, sample_documents):
    hits = search(api_url, RECEIPTS_QUESTION, top_k=3)["hits"]

    assert len(hits) == 3
    evidence_found = False
    for hit in hits:
        if "within 30 days" in " ".join(hit["text"].split()):
            evidence_found = True
    assert evidence_found
    ranks = []
    for hit in hits:
        ranks.append(hit["rank"])
    assert ranks == [1, 2, 3]
    assert hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"]


# top_k decides how many hits come back.
def test_top_k_limits_the_number_of_hits(api_url, sample_documents):
    hits = search(api_url, RECEIPTS_QUESTION, top_k=2)["hits"]

    assert len(hits) == 2


# The same question twice gives the same chunks and the same scores.
def test_search_is_repeatable(api_url, sample_documents):
    first_hits = search(api_url, RECEIPTS_QUESTION)["hits"]
    second_hits = search(api_url, RECEIPTS_QUESTION)["hits"]

    for first_hit, second_hit in zip(first_hits, second_hits):
        assert first_hit["chunk_id"] == second_hit["chunk_id"]
        assert abs(first_hit["score"] - second_hit["score"]) < 0.000001


# top_k of 0 or above the maximum is refused with 422.
@pytest.mark.parametrize("top_k", [0, 21])
def test_bad_top_k_is_rejected(api_url, top_k):
    response = post_search(api_url, {"query": RECEIPTS_QUESTION, "top_k": top_k})

    assert response.status_code == 422


# An empty or blank query is refused with 422.
@pytest.mark.parametrize("query", ["", "   "])
def test_blank_query_is_rejected(api_url, query):
    response = post_search(api_url, {"query": query})

    assert response.status_code == 422


# A user never gets chunks from another user's private document; the owner does.
def test_user_cannot_see_another_users_documents(api_url, sample_documents):
    secret = unique_text("Project Kestrel has a secret budget of 88,000 euros.")
    alice_document_id = upload_and_wait(api_url, "kestrel.md", secret, owner="alice")

    for hit in search(api_url, "Project Kestrel secret budget", top_k=5, user_id="bob")["hits"]:
        assert hit["document_id"] != alice_document_id

    alice_found_it = False
    for hit in search(api_url, "Project Kestrel secret budget", top_k=5, user_id="alice")["hits"]:
        if hit["document_id"] == alice_document_id:
            alice_found_it = True
    assert alice_found_it
