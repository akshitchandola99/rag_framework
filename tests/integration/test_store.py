import numpy as np
import pytest

from ragqa import store
from ragqa.chunking import Chunk


# Build a test chunk in memory with a given ID and text.
def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, source="policy.pdf", page=2, chunk_index=0, start_char=0, end_char=len(text), text=text)


CHUNKS = [make_chunk("a", "text a"), make_chunk("b", "text b"), make_chunk("c", "text c")]
VECTORS = np.array([[0.8, -0.6], [0.6, 0.8], [0.8, 0.6]])
QUESTION = np.array([0.6, 0.8])


# The same chunk always gets the same ID; a changed text gets a new one.
@pytest.mark.integration
def test_content_id_is_stable_and_follows_the_text():
    first_id = store.make_content_id(make_chunk("a", "Annual leave is 27 days."))
    second_id = store.make_content_id(make_chunk("a", "Annual leave is 27 days."))
    changed_id = store.make_content_id(make_chunk("a", "Annual leave is 28 days."))

    assert first_id == second_id
    assert first_id != changed_id
    assert len(first_id) == 16


# Adding the same chunks twice keeps the count the same.
@pytest.mark.integration
def test_adding_twice_does_not_duplicate(tmp_path):
    collection = store.open_collection(tmp_path)

    store.add_chunks(collection, CHUNKS, VECTORS)
    store.add_chunks(collection, CHUNKS, VECTORS)

    assert collection.count() == 3


# Chroma returns the closest chunks first, with cosine similarity as the score.
@pytest.mark.integration
def test_store_search_returns_best_first(tmp_path):
    collection = store.open_collection(tmp_path)
    store.add_chunks(collection, CHUNKS, VECTORS)

    hits = store.search_store(collection, QUESTION, top_k=3)

    hit_ids = []
    for hit in hits:
        hit_ids.append(hit.chunk_id)
    assert hit_ids == ["b", "c", "a"]
    assert hits[1].score == pytest.approx(0.96, abs=1e-4)


# Data written to disk is still there when the folder is opened again.
@pytest.mark.integration
def test_stored_chunks_survive_reopening(tmp_path):
    first_collection = store.open_collection(tmp_path)
    store.add_chunks(first_collection, CHUNKS, VECTORS)

    reopened_collection = store.open_collection(tmp_path)

    assert reopened_collection.count() == 3
