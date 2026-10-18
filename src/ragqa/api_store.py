"""Storage for the API: a JSON registry of documents, plus their chunks in Chroma."""

import hashlib
import json
import threading
from pathlib import Path

import chromadb
import numpy as np

from ragqa.chunking import Chunk
from ragqa.retrieval import SearchHit

PUBLIC_OWNER = "public"
REGISTRY_FILE_NAME = "documents.json"


# Holds the document registry in memory and writes it to disk on every change.
class DocumentRegistry:
    # Load the registry file, or start empty the first time.
    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.lock = threading.Lock()
        self.documents = {}
        if registry_path.exists():
            self.documents = json.loads(registry_path.read_text(encoding="utf-8"))

    # Write the whole registry to disk.
    def save(self) -> None:
        text = json.dumps(self.documents, indent=2)
        self.registry_path.write_text(text, encoding="utf-8")

    # Add or replace one document record.
    def put(self, document: dict) -> None:
        with self.lock:
            self.documents[document["document_id"]] = document
            self.save()

    # Change some fields of one document record.
    def update(self, document_id: str, changes: dict) -> None:
        with self.lock:
            self.documents[document_id].update(changes)
            self.save()

    # Remove one document record.
    def remove(self, document_id: str) -> None:
        with self.lock:
            del self.documents[document_id]
            self.save()

    # Return one document record, or None if it does not exist.
    def get(self, document_id: str) -> dict | None:
        return self.documents.get(document_id)

    # Return every document record.
    def all(self) -> list[dict]:
        return list(self.documents.values())


# Build a document ID from the owner and the file bytes, so a re-upload is spotted.
def make_document_id(owner: str, file_bytes: bytes) -> str:
    hasher = hashlib.sha256()
    hasher.update(owner.encode("utf-8"))
    hasher.update(b"|")
    hasher.update(file_bytes)
    return hasher.hexdigest()[:16]


# Save one document's chunks and vectors, tagged with the document ID and owner.
def add_document_chunks(
    collection: chromadb.Collection,
    document_id: str,
    owner: str,
    chunks: list[Chunk],
    chunk_vectors: np.ndarray,
    token_counts: list[int],
) -> None:
    ids = []
    documents = []
    metadatas = []
    for index, chunk in enumerate(chunks):
        ids.append(f"{document_id}:{chunk.chunk_id}")
        documents.append(chunk.text)
        metadatas.append(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": document_id,
                "owner": owner,
                "source": chunk.source,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "token_count": token_counts[index],
            }
        )
    collection.upsert(ids=ids, embeddings=chunk_vectors.tolist(), documents=documents, metadatas=metadatas)


# Sort key for chunks: by page, then by position on the page.
def chunk_order(chunk: dict) -> tuple[int, int]:
    return (chunk["page"], chunk["start"])


# Return all stored chunks of one document, in reading order.
def get_document_chunks(collection: chromadb.Collection, document_id: str) -> list[dict]:
    result = collection.get(where={"document_id": document_id}, include=["documents", "metadatas"])
    chunks = []
    for index in range(len(result["ids"])):
        metadata = result["metadatas"][index]
        chunks.append(
            {
                "chunk_id": metadata["chunk_id"],
                "page": metadata["page"],
                "chunk_index": metadata["chunk_index"],
                "start": metadata["start_char"],
                "end": metadata["end_char"],
                "token_count": metadata["token_count"],
                "text": result["documents"][index],
            }
        )
    chunks.sort(key=chunk_order)
    return chunks


# Delete every stored chunk of one document.
def delete_document_chunks(collection: chromadb.Collection, document_id: str) -> None:
    collection.delete(where={"document_id": document_id})


# Search only the chunks this user may read (their own and public ones), best first.
def search_for_user(
    collection: chromadb.Collection,
    query_vector: np.ndarray,
    top_k: int,
    user_id: str,
) -> list[tuple[SearchHit, str]]:
    if collection.count() == 0:
        return []

    allowed_owners = [PUBLIC_OWNER]
    if user_id != PUBLIC_OWNER:
        allowed_owners.append(user_id)

    result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=top_k,
        where={"owner": {"$in": allowed_owners}},
        include=["documents", "metadatas", "distances"],
    )
    found_metadatas = result["metadatas"][0]
    found_documents = result["documents"][0]
    found_distances = result["distances"][0]

    hits_with_document = []
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
        hits_with_document.append((hit, metadata["document_id"]))
    return hits_with_document
