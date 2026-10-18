"""Phase 3: similarity math for normalized vectors. Only numpy, so it imports fast."""

import numpy as np


# Score one vector against many (dot product of normalized vectors).
def cosine_scores(query_vector: np.ndarray, chunk_vectors: np.ndarray) -> np.ndarray:
    # Both sides are normalized, so the dot product is the cosine similarity.
    return chunk_vectors @ query_vector
