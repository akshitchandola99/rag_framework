import pytest

from ragqa import ask
from ragqa.citations import all_checks_pass
from ragqa.citations import check_answer
from ragqa.golden import GOLDEN_V0_PATH
from ragqa.golden import load_golden_set
from ragqa.retrieval import DEFAULT_TOP_K


# Find one golden item by its id.
def golden_item(item_id: str):
    for item in load_golden_set(GOLDEN_V0_PATH):
        if item.id == item_id:
            return item
    raise KeyError(item_id)


# Answer one golden item with the real pipeline and run the Phase 7 checks on it.
def run_checks(real_index, item_id: str) -> dict:
    model, chunks, chunk_vectors = real_index
    item = golden_item(item_id)
    result = ask.answer_question(model, chunks, chunk_vectors, item.question, DEFAULT_TOP_K)
    return check_answer(result["answer"], result["labelled_hits"], item.expected_behavior, item.expected_evidence)


# Every answerable item whose evidence is retrieved gets a valid citation to that evidence.
@pytest.mark.integration
@pytest.mark.parametrize("item_id", ["q001", "q002", "q004", "q006"])
def test_answer_cites_the_evidence(real_index, ollama_ready, item_id):
    check_result = run_checks(real_index, item_id)
    assert all_checks_pass(check_result["checks"]), check_result


# The unanswerable item refuses cleanly, with no citations.
@pytest.mark.integration
def test_unanswerable_item_refuses(real_index, ollama_ready):
    check_result = run_checks(real_index, "q008")
    assert check_result["refused"] is True
    assert check_result["checks"]["refusal_is_clean"] is True
