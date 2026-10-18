import pytest

from ragqa import generation
from ragqa.citations import is_refusal
from ragqa.retrieval import SearchHit

LEAVE_TEXT = "## Leave\n\n- Annual leave: 27 days per year, plus public holidays."


# Ask the real model one question about a hand-made context and return its answer.
def ask_with_context(question: str, chunk_text: str) -> str:
    hit = SearchHit(rank=1, chunk_id="handbook.md:p1:c0", source="handbook.md", page=1, score=0.8, text=chunk_text)
    prompt = generation.build_prompt(question, generation.label_hits([hit]))
    reply = generation.generate_answer(
        generation.DEFAULT_OLLAMA_URL,
        generation.DEFAULT_GENERATOR_MODEL,
        generation.SYSTEM_RULES,
        prompt,
    )
    return reply["response"].strip()


# When the answer is in the context, the model uses it and cites [S1].
@pytest.mark.integration
def test_answer_uses_the_context_and_cites_it(ollama_ready):
    answer = ask_with_context("How many days of annual leave do employees get?", LEAVE_TEXT)
    assert "27" in answer
    assert "[S1]" in answer


# When the answer is not in the context, the model refuses.
@pytest.mark.integration
def test_model_refuses_when_the_context_lacks_the_answer(ollama_ready):
    answer = ask_with_context("What is the parking policy?", LEAVE_TEXT)
    assert is_refusal(answer)


# The context wins over what the model "knows": a made-up number is repeated.
@pytest.mark.integration
def test_answer_follows_the_context_not_world_knowledge(ollama_ready):
    answer = ask_with_context("How many days are in a week?", "At Brightwater, a week has 9 days.")
    assert "9" in answer


# At temperature 0 the same prompt gives the same answer twice.
@pytest.mark.integration
def test_same_prompt_gives_the_same_answer(ollama_ready):
    first_answer = ask_with_context("How many days of annual leave do employees get?", LEAVE_TEXT)
    second_answer = ask_with_context("How many days of annual leave do employees get?", LEAVE_TEXT)
    assert first_answer == second_answer
