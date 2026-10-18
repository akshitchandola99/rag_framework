import numpy as np
import pytest

from ragqa.similarity import cosine_scores


# The maths matches the 2-D example from section 3.1.
def test_cosine_scores_match_the_worked_example():
    vector_a = np.array([0.6, 0.8])
    other_vectors = np.array([[0.6, 0.8], [0.8, 0.6], [0.8, -0.6]])

    scores = cosine_scores(vector_a, other_vectors)

    assert scores == pytest.approx([1.0, 0.96, 0.0])
