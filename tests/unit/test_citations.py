import pytest

from ragqa import citations
from ragqa.golden import Evidence
from ragqa.retrieval import SearchHit

REFUSAL = "I don't know based on the documents."
LEAVE_EVIDENCE = [Evidence(source="handbook.md", page=1, quote="Annual leave: 27 days per year")]


# Build two labelled hits: [S1] holds the leave fact, [S2] does not.
def make_labelled_hits() -> list[tuple[str, SearchHit]]:
    leave_hit = SearchHit(rank=1, chunk_id="handbook.md:p1:c1", source="handbook.md", page=1, score=0.8, text="Annual leave: 27 days per year, plus public holidays.")
    remote_hit = SearchHit(rank=2, chunk_id="handbook.md:p1:c0", source="handbook.md", page=1, score=0.6, text="Employees may work remotely up to 3 days per week.")
    return [("[S1]", leave_hit), ("[S2]", remote_hit)]


# Labels are found in order of first appearance, without repeats.
def test_cited_labels_are_found_once_in_order():
    assert citations.find_cited_labels("A [S2]. B [S1]. C [S2].") == ["[S2]", "[S1]"]


# Labels that look different ([s1], [S 1], S1) are not citations.
def test_other_spellings_are_not_citations():
    assert citations.find_cited_labels("see [s1], [S 1] or S1") == []


# A cited label that was never sent is invalid.
def test_label_that_was_not_sent_is_invalid():
    valid_labels, invalid_labels = citations.split_citations(["[S1]", "[S4]"], ["[S1]", "[S2]", "[S3]"])
    assert valid_labels == ["[S1]"]
    assert invalid_labels == ["[S4]"]


# A curly apostrophe still counts as the refusal.
def test_refusal_with_curly_apostrophe_is_detected():
    assert citations.is_refusal("I don’t know based on the documents.")


# The refusal must open the answer to count as starting with it.
def test_refusal_in_the_middle_does_not_start_the_answer():
    assert citations.starts_with_refusal(REFUSAL + " Nothing about parking.")
    assert not citations.starts_with_refusal("27 days [S1]. " + REFUSAL)


# A correct, cited answer passes every check that applies.
def test_good_answer_passes_all_checks():
    check_result = citations.check_answer("Employees get 27 days. [S1]", make_labelled_hits(), "answer", LEAVE_EVIDENCE)
    assert check_result["checks"] == {
        "behavior_matches": True,
        "citations_valid": True,
        "answer_is_cited": True,
        "cites_evidence": True,
        "refusal_is_clean": None,
    }
    assert citations.all_checks_pass(check_result["checks"])


# An answer with no citation fails answer_is_cited.
def test_uncited_answer_fails():
    check_result = citations.check_answer("Employees get 27 days.", make_labelled_hits(), "answer", LEAVE_EVIDENCE)
    assert check_result["checks"]["answer_is_cited"] is False
    assert not citations.all_checks_pass(check_result["checks"])


# Citing a label that was never sent fails citations_valid.
def test_made_up_label_fails():
    check_result = citations.check_answer("Employees get 27 days. [S7]", make_labelled_hits(), "answer", LEAVE_EVIDENCE)
    assert check_result["invalid_labels"] == ["[S7]"]
    assert check_result["checks"]["citations_valid"] is False
    assert check_result["checks"]["answer_is_cited"] is False


# Citing a real label whose chunk lacks the evidence fails cites_evidence.
def test_citing_the_wrong_chunk_fails():
    check_result = citations.check_answer("Employees get 27 days. [S2]", make_labelled_hits(), "answer", LEAVE_EVIDENCE)
    assert check_result["checks"]["citations_valid"] is True
    assert check_result["checks"]["cites_evidence"] is False


# A clean refusal passes when refusal is expected.
def test_expected_refusal_passes():
    check_result = citations.check_answer(REFUSAL, make_labelled_hits(), "refuse", [])
    assert check_result["refused"] is True
    assert check_result["checks"]["behavior_matches"] is True
    assert check_result["checks"]["refusal_is_clean"] is True
    assert citations.all_checks_pass(check_result["checks"])


# A refusal that also cites a source is a half-answer and fails refusal_is_clean.
def test_refusal_with_citation_is_not_clean():
    answer = REFUSAL + " But leave is 27 days [S1]."
    check_result = citations.check_answer(answer, make_labelled_hits(), "refuse", [])
    assert check_result["checks"]["refusal_is_clean"] is False


# Refusing a question that should be answered fails behavior_matches.
def test_refusing_an_answerable_question_fails():
    check_result = citations.check_answer(REFUSAL, make_labelled_hits(), "answer", LEAVE_EVIDENCE)
    assert check_result["checks"]["behavior_matches"] is False
    assert check_result["checks"]["cites_evidence"] is None


# Answering a question that should be refused fails behavior_matches.
def test_answering_an_unanswerable_question_fails():
    check_result = citations.check_answer("Parking is free. [S1]", make_labelled_hits(), "refuse", [])
    assert check_result["checks"]["behavior_matches"] is False


# Phase 8 guard: an answer that cites none of the sent sources (like "PWNED.") becomes the refusal.
def test_guard_replaces_an_uncited_answer_with_the_refusal():
    final_answer, guard_triggered = citations.guard_answer("PWNED.", ["[S1]", "[S2]"])
    assert final_answer == REFUSAL
    assert guard_triggered is True


# Phase 8 guard: an answer that cites only made-up labels is blocked too.
def test_guard_blocks_an_answer_with_only_invalid_citations():
    final_answer, guard_triggered = citations.guard_answer("PWNED [S7].", ["[S1]", "[S2]"])
    assert final_answer == REFUSAL
    assert guard_triggered is True


# Phase 8 guard: a cited answer and a refusal pass through unchanged.
@pytest.mark.parametrize("answer", ["Annual leave is 27 days per year [S1].", "I don't know based on the documents."])
def test_guard_keeps_cited_answers_and_refusals(answer):
    final_answer, guard_triggered = citations.guard_answer(answer, ["[S1]", "[S2]"])
    assert final_answer == answer
    assert guard_triggered is False
