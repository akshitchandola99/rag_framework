"""Phase 3 exercise: rank every chunk for one question, with and without the query prefix."""

import sys

import numpy as np

from ragqa.chunking import DEFAULT_CHUNK_SIZE
from ragqa.chunking import DEFAULT_OVERLAP
from ragqa.chunking import chunk_pages
from ragqa.embeddings import embed_query
from ragqa.embeddings import embed_texts
from ragqa.embeddings import load_embedding_model
from ragqa.loaders import DEFAULT_DATA_FOLDER
from ragqa.loaders import load_folder
from ragqa.similarity import cosine_scores


# Print chunk IDs from the best score to the worst.
def print_ranking(title: str, scores: np.ndarray, chunk_ids: list[str]) -> None:
    print(title)
    positions_best_first = np.argsort(-scores)
    for rank, position in enumerate(positions_best_first, start=1):
        print(f"  {rank}. {scores[position]:.3f}  {chunk_ids[position]}")


# Rank all chunks for one question, with and without the prefix.
def main() -> None:
    if len(sys.argv) != 2:
        print('Usage: uv run python -m ragqa.try_ranking "your question"')
        sys.exit(1)
    question = sys.argv[1]

    model = load_embedding_model()
    pages, skipped_files = load_folder(DEFAULT_DATA_FOLDER)
    chunks = chunk_pages(pages, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP)

    chunk_texts = []
    chunk_ids = []
    for chunk in chunks:
        chunk_texts.append(chunk.text)
        chunk_ids.append(chunk.chunk_id)
    chunk_vectors = embed_texts(model, chunk_texts)

    with_prefix = embed_query(model, question)
    without_prefix = embed_texts(model, [question])[0]

    print(f"Question: {question}")
    print_ranking("With query prefix:", cosine_scores(with_prefix, chunk_vectors), chunk_ids)
    print_ranking("Without prefix:", cosine_scores(without_prefix, chunk_vectors), chunk_ids)


if __name__ == "__main__":
    main()
