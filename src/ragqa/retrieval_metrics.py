"""Phase 5: score retrieval against the golden set with hit@k and MRR."""

from ragqa.golden import Evidence
from ragqa.golden import normalize_whitespace
from ragqa.retrieval import SearchHit


# True when the hit is on the evidence's page and contains its quote.
def hit_contains_evidence(hit: SearchHit, evidence: Evidence) -> bool:
    if hit.source != evidence.source or hit.page != evidence.page:
        return False
    return normalize_whitespace(evidence.quote) in normalize_whitespace(hit.text)


# Rank of the first hit that contains any expected quote, or None if no hit does.
def first_relevant_rank(hits: list[SearchHit], expected_evidence: list[Evidence]) -> int | None:
    for hit in hits:
        for evidence in expected_evidence:
            if hit_contains_evidence(hit, evidence):
                return hit.rank
    return None


# Share of questions whose first relevant hit is at rank k or better.
def hit_at_k(first_ranks: list[int | None], k: int) -> float:
    if len(first_ranks) == 0:
        raise ValueError("no questions to score")
    found = 0
    for rank in first_ranks:
        if rank is not None and rank <= k:
            found = found + 1
    return found / len(first_ranks)


# Average of 1/rank over all questions; a question with no relevant hit counts as 0.
def mean_reciprocal_rank(first_ranks: list[int | None]) -> float:
    if len(first_ranks) == 0:
        raise ValueError("no questions to score")
    total = 0.0
    for rank in first_ranks:
        if rank is not None:
            total = total + 1 / rank
    return total / len(first_ranks)
