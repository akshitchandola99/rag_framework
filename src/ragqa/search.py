"""Phase 4: build the Chroma index, then answer one question with numpy and with Chroma."""

import sys

from ragqa.chunking import DEFAULT_CHUNK_SIZE
from ragqa.chunking import DEFAULT_OVERLAP
from ragqa.chunking import chunk_pages
from ragqa.embeddings import embed_query
from ragqa.embeddings import embed_texts
from ragqa.embeddings import load_embedding_model
from ragqa.loaders import DEFAULT_DATA_FOLDER
from ragqa.loaders import load_folder
from ragqa.retrieval import DEFAULT_TOP_K
from ragqa.retrieval import SearchHit
from ragqa.retrieval import search_chunks
from ragqa.store import DEFAULT_STORE_FOLDER
from ragqa.store import add_chunks
from ragqa.store import open_collection
from ragqa.store import search_store

PREVIEW_LENGTH = 70


# Print search hits as a small numbered list with a one-line preview.
def print_hits(title: str, hits: list[SearchHit]) -> None:
    print(title)
    for hit in hits:
        preview = " ".join(hit.text.split())[:PREVIEW_LENGTH]
        print(f"  {hit.rank}. {hit.score:.4f}  {hit.chunk_id}  {preview!r}")


# Index the documents in Chroma and compare numpy and Chroma results for one question.
def main() -> None:
    if len(sys.argv) != 2:
        print('Usage: uv run python -m ragqa.search "your question"')
        sys.exit(1)
    question = sys.argv[1]

    model = load_embedding_model()
    pages, skipped_files = load_folder(DEFAULT_DATA_FOLDER)
    chunks = chunk_pages(pages, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP)
    chunk_texts = []
    for chunk in chunks:
        chunk_texts.append(chunk.text)
    chunk_vectors = embed_texts(model, chunk_texts)

    collection = open_collection(DEFAULT_STORE_FOLDER)
    count_before = collection.count()
    add_chunks(collection, chunks, chunk_vectors)
    count_after = collection.count()
    print(f"Chroma folder: {DEFAULT_STORE_FOLDER}")
    print(f"Stored chunks: {count_before} before, {count_after} after adding {len(chunks)}")

    query_vector = embed_query(model, question)
    numpy_hits = search_chunks(query_vector, chunks, chunk_vectors, DEFAULT_TOP_K)
    chroma_hits = search_store(collection, query_vector, DEFAULT_TOP_K)

    print(f"Question: {question}")
    print_hits(f"numpy top {DEFAULT_TOP_K}:", numpy_hits)
    print_hits(f"Chroma top {DEFAULT_TOP_K}:", chroma_hits)

    same_order = True
    for index in range(len(numpy_hits)):
        if numpy_hits[index].chunk_id != chroma_hits[index].chunk_id:
            same_order = False
    print(f"Same chunks in the same order: {same_order}")

    for reason in skipped_files:
        print(f"SKIPPED {reason}")


if __name__ == "__main__":
    main()
