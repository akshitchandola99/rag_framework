import numpy as np
import pytest

from ragqa.chunking import Chunk
from ragqa.retrieval import search_chunks


# Build a test chunk in memory with a given ID and text.
def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, source="policy.pdf", page=2, chunk_index=0, start_char=0, end_char=len(text), text=text)


CHUNKS = [make_chunk("a", "text a"), make_chunk("b", "text b"), make_chunk("c", "text c")]
VECTORS = np.array([[0.8, -0.6], [0.6, 0.8], [0.8, 0.6]])
QUESTION = np.array([0.6, 0.8])


# The best-scoring chunk comes first, then the rest in score order.
def test_hits_come_back_best_first():
    hits = search_chunks(QUESTION, CHUNKS, VECTORS, top_k=3)

    hit_ids = []
    for hit in hits:
        hit_ids.append(hit.chunk_id)
    assert hit_ids == ["b", "c", "a"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(0.96)


# Only top_k hits are returned, ranked from 1.
def test_top_k_limits_the_number_of_hits():
    hits = search_chunks(QUESTION, CHUNKS, VECTORS, top_k=2)

    assert len(hits) == 2
    assert hits[0].rank == 1
    assert hits[1].rank == 2


# Asking for more hits than chunks returns every chunk, not an error.
def test_top_k_larger_than_chunk_count_returns_all():
    hits = search_chunks(QUESTION, CHUNKS, VECTORS, top_k=10)

    assert len(hits) == 3


# Each hit carries the chunk's source, page and text.
def test_hits_copy_chunk_metadata():
    hits = search_chunks(QUESTION, CHUNKS, VECTORS, top_k=1)

    assert hits[0].source == "policy.pdf"
    assert hits[0].page == 2
    assert hits[0].text == "text b"


# A top_k of 0 is rejected.
def test_top_k_zero_raises():
    with pytest.raises(ValueError, match="top_k must be above 0"):
        search_chunks(QUESTION, CHUNKS, VECTORS, top_k=0)
