import numpy as np
import pytest

from ragqa import embeddings
from ragqa.similarity import cosine_scores


# Each text becomes one row of 384 numbers.
@pytest.mark.integration
def test_one_vector_of_384_numbers_per_text(embedding_model):
    vectors = embeddings.embed_texts(embedding_model, ["first text", "second text", "third text"])

    assert vectors.shape == (3, 384)


# Every vector is normalized to length 1.
@pytest.mark.integration
def test_every_vector_has_length_one(embedding_model):
    vectors = embeddings.embed_texts(embedding_model, ["Annual leave is 27 days.", "Laptops last 3 years."])

    for vector in vectors:
        assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)


# The related passage outranks the unrelated one.
@pytest.mark.integration
def test_related_passage_scores_higher_than_unrelated(embedding_model):
    question_vector = embeddings.embed_query(embedding_model, "How many vacation days do I get?")
    passage_vectors = embeddings.embed_texts(
        embedding_model,
        [
            "Annual leave: 27 days per year, plus public holidays.",
            "Laptops are replaced every 3 years.",
        ],
    )

    scores = cosine_scores(question_vector, passage_vectors)

    related_score = scores[0]
    unrelated_score = scores[1]
    assert related_score > unrelated_score


# Embedding the same text twice gives the same vector.
@pytest.mark.integration
def test_same_text_gives_the_same_vector(embedding_model):
    first_vector = embeddings.embed_texts(embedding_model, ["The daily meal allowance is 45 euros."])[0]
    second_vector = embeddings.embed_texts(embedding_model, ["The daily meal allowance is 45 euros."])[0]

    assert np.allclose(first_vector, second_vector, atol=1e-6)


# The query prefix really changes the question's vector.
@pytest.mark.integration
def test_query_prefix_changes_the_vector(embedding_model):
    with_prefix = embeddings.embed_query(embedding_model, "Can I work from home?")
    without_prefix = embeddings.embed_texts(embedding_model, ["Can I work from home?"])[0]

    assert not np.allclose(with_prefix, without_prefix)
