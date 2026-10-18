import pytest

from ragqa import embeddings
from ragqa import store
from ragqa.chunking import chunk_pages
from ragqa.loaders import DEFAULT_DATA_FOLDER
from ragqa.loaders import load_folder
from ragqa.retrieval import search_chunks

QUESTIONS = [
    "How many vacation days do I get?",
    "How much can I spend on a hotel in Paris?",
    "Who do I call when IT is closed?",
    "What happens if I lose my laptop?",
    "Can I fly business class?",
]


# numpy and Chroma return the same top 3 chunks, in the same order, for real questions.
@pytest.mark.integration
def test_numpy_and_chroma_agree_on_real_questions(tmp_path, embedding_model):
    pages, skipped_files = load_folder(DEFAULT_DATA_FOLDER)
    chunks = chunk_pages(pages, 500, 100)
    chunk_texts = []
    for chunk in chunks:
        chunk_texts.append(chunk.text)
    chunk_vectors = embeddings.embed_texts(embedding_model, chunk_texts)
    collection = store.open_collection(tmp_path)
    store.add_chunks(collection, chunks, chunk_vectors)

    for question in QUESTIONS:
        query_vector = embeddings.embed_query(embedding_model, question)
        numpy_hits = search_chunks(query_vector, chunks, chunk_vectors, top_k=3)
        chroma_hits = store.search_store(collection, query_vector, top_k=3)
        for index in range(3):
            assert numpy_hits[index].chunk_id == chroma_hits[index].chunk_id, question
            assert numpy_hits[index].score == pytest.approx(chroma_hits[index].score, abs=1e-4)
