"""Phase 3: turn chunks and questions into normalized vectors with a local embedding model."""

import time

import numpy as np
from sentence_transformers import SentenceTransformer

from ragqa.chunking import DEFAULT_CHUNK_SIZE
from ragqa.chunking import DEFAULT_OVERLAP
from ragqa.chunking import chunk_pages
from ragqa.loaders import DEFAULT_DATA_FOLDER
from ragqa.loaders import load_folder
from ragqa.similarity import cosine_scores

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DEVICE = "cpu"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 32

SIMILARITY_EXAMPLES = [
    ("How many vacation days do I get?", "Annual leave: 27 days per year, plus public holidays."),
    ("How many vacation days do I get?", "Laptops are replaced every 3 years."),
    ("Can I work from home?", "Employees may work remotely up to 3 days per week."),
    ("Can I work from home?", "The daily meal allowance is 45 euros."),
]


# Load the bge-small embedding model on the CPU.
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)


# Turn texts into normalized vectors, one row per text.
def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors


# Embed a question, with the search prefix in front.
def embed_query(model: SentenceTransformer, question: str) -> np.ndarray:
    vectors = embed_texts(model, [QUERY_PREFIX + question])
    return vectors[0]


# Embed all chunks and print shapes, lengths and similarity checks.
def main() -> None:
    start_time = time.perf_counter()
    model = load_embedding_model()
    load_seconds = time.perf_counter() - start_time
    print(f"Model: {EMBEDDING_MODEL_NAME} on {model.device}, loaded in {load_seconds:.1f} s")

    pages, skipped_files = load_folder(DEFAULT_DATA_FOLDER)
    chunks = chunk_pages(pages, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP)
    chunk_texts = []
    for chunk in chunks:
        chunk_texts.append(chunk.text)

    start_time = time.perf_counter()
    chunk_vectors = embed_texts(model, chunk_texts)
    embed_seconds = time.perf_counter() - start_time
    print(f"Embedded {len(chunks)} chunks in {embed_seconds:.2f} s")
    print(f"Array shape: {chunk_vectors.shape}, number type: {chunk_vectors.dtype}")

    first_vector = chunk_vectors[0]
    print(f"First 5 numbers of {chunks[0].chunk_id}: {np.round(first_vector[:5], 4)}")

    print("Vector lengths (should all be 1.0):")
    for index, chunk in enumerate(chunks):
        length = float(np.linalg.norm(chunk_vectors[index]))
        print(f"  {chunk.chunk_id}: {length:.4f}")

    print("Similarity checks (question vs passage):")
    for question, passage in SIMILARITY_EXAMPLES:
        question_vector = embed_query(model, question)
        passage_vectors = embed_texts(model, [passage])
        score = float(cosine_scores(question_vector, passage_vectors)[0])
        print(f"  {score:.3f}  {question!r} vs {passage!r}")

    for reason in skipped_files:
        print(f"SKIPPED {reason}")


if __name__ == "__main__":
    main()
