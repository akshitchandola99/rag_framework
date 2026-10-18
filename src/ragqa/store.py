"""Phase 4: keep chunks and their vectors in Chroma on disk, and search them there."""

import hashlib
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

from ragqa.chunking import Chunk
from ragqa.retrieval import SearchHit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE_FOLDER = PROJECT_ROOT / "storage" / "chroma"
COLLECTION_NAME = "chunks"


# Build an ID from the chunk's content, so the same chunk always gets the same ID.
def make_content_id(chunk: Chunk) -> str:
    content = f"{chunk.source}|{chunk.page}|{chunk.start_char}|{chunk.text}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


# Open the chunk collection in a folder on disk, creating it the first time.
def open_collection(store_folder: Path) -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=str(store_folder),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )
    return collection


# Save chunks with their vectors; chunks already stored are replaced, never duplicated.
def add_chunks(collection: chromadb.Collection, chunks: list[Chunk], chunk_vectors: np.ndarray) -> None:
    ids = []
    documents = []
    metadatas = []
    for chunk in chunks:
        ids.append(make_content_id(chunk))
        documents.append(chunk.text)
        metadatas.append({"chunk_id": chunk.chunk_id, "source": chunk.source, "page": chunk.page})
    collection.upsert(ids=ids, embeddings=chunk_vectors.tolist(), documents=documents, metadatas=metadatas)


# Ask Chroma for the top_k stored chunks closest to the question, best first.
def search_store(collection: chromadb.Collection, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
    result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    found_metadatas = result["metadatas"][0]
    found_documents = result["documents"][0]
    found_distances = result["distances"][0]

    hits = []
    for index in range(len(found_metadatas)):
        metadata = found_metadatas[index]
        hit = SearchHit(
            rank=index + 1,
            chunk_id=metadata["chunk_id"],
            source=metadata["source"],
            page=metadata["page"],
            score=1.0 - found_distances[index],
            text=found_documents[index],
        )
        hits.append(hit)
    return hits
