import pytest

from ragqa.golden import Evidence
from ragqa.retrieval import SearchHit
from ragqa.retrieval_metrics import first_relevant_rank
from ragqa.retrieval_metrics import hit_at_k
from ragqa.retrieval_metrics import hit_contains_evidence
from ragqa.retrieval_metrics import mean_reciprocal_rank


# Build a search hit in memory with a given rank, page and text.
def make_hit(rank: int, page: int, text: str) -> SearchHit:
    return SearchHit(rank=rank, chunk_id=f"policy.pdf:p{page}:c0", source="policy.pdf", page=page, score=0.5, text=text)


MEAL_EVIDENCE = Evidence(source="policy.pdf", page=4, quote="The daily meal allowance is 45 euros")


# A hit on the right page that contains the quote counts.
def test_hit_with_the_quote_counts():
    hit = make_hit(1, 4, "4. Daily allowance. The daily meal allowance is 45 euros. Alcohol is never reimbursed.")

    assert hit_contains_evidence(hit, MEAL_EVIDENCE) is True


# A line break inside the quote (as in PDF text) still counts.
def test_line_breaks_in_the_chunk_do_not_matter():
    hit = make_hit(1, 4, "The daily meal\nallowance is 45 euros.")

    assert hit_contains_evidence(hit, MEAL_EVIDENCE) is True


# The same words on a different page do not count.
def test_quote_on_the_wrong_page_does_not_count():
    hit = make_hit(1, 2, "The daily meal allowance is 45 euros.")

    assert hit_contains_evidence(hit, MEAL_EVIDENCE) is False


# The rank of the first relevant hit is returned, skipping irrelevant ones.
def test_first_relevant_rank_skips_irrelevant_hits():
    hits = [
        make_hit(1, 2, "Hotels cost 140 euros."),
        make_hit(2, 4, "The daily meal allowance is 45 euros."),
        make_hit(3, 4, "The daily meal allowance is 45 euros, again."),
    ]

    assert first_relevant_rank(hits, [MEAL_EVIDENCE]) == 2


# No relevant hit at all gives None.
def test_first_relevant_rank_is_none_when_nothing_matches():
    hits = [make_hit(1, 2, "Hotels cost 140 euros.")]

    assert first_relevant_rank(hits, [MEAL_EVIDENCE]) is None


# hit@k and MRR match the worked example from section 5.1.
def test_metrics_match_the_worked_example():
    first_ranks = [1, None, 3, 2]

    assert hit_at_k(first_ranks, 1) == pytest.approx(0.25)
    assert hit_at_k(first_ranks, 3) == pytest.approx(0.75)
    assert mean_reciprocal_rank(first_ranks) == pytest.approx(0.4583, abs=1e-4)


# Scoring an empty list of questions is refused instead of dividing by zero.
def test_metrics_refuse_an_empty_question_list():
    with pytest.raises(ValueError, match="no questions to score"):
        hit_at_k([], 3)
    with pytest.raises(ValueError, match="no questions to score"):
        mean_reciprocal_rank([])
