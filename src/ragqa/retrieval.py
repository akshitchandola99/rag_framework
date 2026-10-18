"""Phase 4: find the chunks most similar to a question with plain numpy (brute force)."""

from dataclasses import dataclass

import numpy as np

from ragqa.chunking import Chunk
from ragqa.similarity import cosine_scores

DEFAULT_TOP_K = 3


# One search result: which chunk, how well it matched, and where it came from.
@dataclass
class SearchHit:
    rank: int
    chunk_id: str
    source: str
    page: int
    score: float
    text: str


# Score every chunk against the question and return the top_k best, best first.
def search_chunks(
    query_vector: np.ndarray,
    chunks: list[Chunk],
    chunk_vectors: np.ndarray,
    top_k: int,
) -> list[SearchHit]:
    if top_k <= 0:
        raise ValueError(f"top_k must be above 0, got {top_k}")

    scores = cosine_scores(query_vector, chunk_vectors)
    positions_best_first = np.argsort(-scores, kind="stable")

    hits = []
    for rank, position in enumerate(positions_best_first[:top_k], start=1):
        chunk = chunks[position]
        hit = SearchHit(
            rank=rank,
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            page=chunk.page,
            score=float(scores[position]),
            text=chunk.text,
        )
        hits.append(hit)
    return hits
