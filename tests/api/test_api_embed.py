"""Phase 3 through the API: the embedding service as a black box."""

import math

import pytest
import requests

from api_helpers import REQUEST_TIMEOUT_SECONDS

pytestmark = [pytest.mark.integration, pytest.mark.api]


# Call POST /embed and return the raw HTTP response.
def embed(api_url: str, texts: list[str], is_query: bool = False) -> requests.Response:
    request_body = {"texts": texts, "is_query": is_query}
    return requests.post(f"{api_url}/embed", json=request_body, timeout=REQUEST_TIMEOUT_SECONDS)


# Dot product of two equal-length lists (equals cosine similarity for length-one vectors).
def dot(first: list[float], second: list[float]) -> float:
    total = 0.0
    for first_value, second_value in zip(first, second):
        total += first_value * second_value
    return total


# The service reports the model we expect and one 384-number vector per text.
def test_embedding_service_reports_model_and_size(api_url):
    reply = embed(api_url, ["Annual leave is 27 days.", "Laptops are replaced every 3 years."]).json()

    assert reply["model"] == "BAAI/bge-small-en-v1.5"
    assert reply["dimension"] == 384
    assert len(reply["vectors"]) == 2
    for vector in reply["vectors"]:
        assert len(vector) == 384


# Every vector has length one, and the same text always gives the same vector.
def test_vectors_are_normalized_and_repeatable(api_url):
    first = embed(api_url, ["Annual leave is 27 days."]).json()["vectors"][0]
    second = embed(api_url, ["Annual leave is 27 days."]).json()["vectors"][0]

    assert abs(math.sqrt(dot(first, first)) - 1.0) < 0.0001
    for first_value, second_value in zip(first, second):
        assert abs(first_value - second_value) < 0.000001


# A related passage scores higher than an unrelated one for the same question.
def test_related_text_scores_higher_than_unrelated(api_url):
    question = embed(api_url, ["How many vacation days do I get?"], is_query=True).json()["vectors"][0]
    passages = embed(
        api_url,
        ["Annual leave: 27 days per year, plus public holidays.", "Laptops are replaced every 3 years."],
    ).json()["vectors"]

    assert dot(question, passages[0]) > dot(question, passages[1])


# The query flag really reaches the model (the search prefix changes the vector).
def test_query_flag_changes_the_vector(api_url):
    as_passage = embed(api_url, ["Can I work from home?"], is_query=False).json()["vectors"][0]
    as_query = embed(api_url, ["Can I work from home?"], is_query=True).json()["vectors"][0]

    assert dot(as_passage, as_query) < 0.9999


# An empty list and a list over the limit are both refused with 422.
@pytest.mark.parametrize("text_count", [0, 65])
def test_bad_batch_size_is_rejected(api_url, text_count):
    texts = []
    for index in range(text_count):
        texts.append(f"text {index}")

    response = embed(api_url, texts)

    assert response.status_code == 422
