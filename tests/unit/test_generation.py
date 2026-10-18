import pytest
import requests

from ragqa import generation
from ragqa.retrieval import SearchHit


# Build a search hit in memory with a given rank, source, page and text.
def make_hit(rank: int, source: str, page: int, text: str) -> SearchHit:
    return SearchHit(rank=rank, chunk_id=f"{source}:p{page}:c{rank}", source=source, page=page, score=0.5, text=text)


# A stand-in for requests' reply object, so no real HTTP call is made.
class FakeResponse:
    # Keep the status code and body to hand back later.
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self.body = body

    # Raise like requests does when the status code is an error.
    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    # Return the JSON body we were given.
    def json(self) -> dict:
        return self.body


# Hits get [S1], [S2], [S3] in the order they were ranked.
def test_hits_are_labelled_in_rank_order():
    hits = [
        make_hit(1, "a.md", 1, "first"),
        make_hit(2, "b.md", 1, "second"),
        make_hit(3, "c.md", 2, "third"),
    ]
    labelled_hits = generation.label_hits(hits)
    labels = []
    for label, hit in labelled_hits:
        labels.append(label)
    assert labels == ["[S1]", "[S2]", "[S3]"]
    assert labelled_hits[0][1].text == "first"


# Every chunk is inside <context>, after its label, source and page.
def test_prompt_puts_labelled_chunks_inside_context():
    hits = [make_hit(1, "handbook.md", 1, "Annual leave: 27 days per year.")]
    prompt = generation.build_prompt("How much leave?", generation.label_hits(hits))
    assert prompt.startswith("<context>")
    assert "[S1] (source: handbook.md, page 1)\nAnnual leave: 27 days per year." in prompt
    assert prompt.index("Annual leave") < prompt.index("</context>")


# The question comes last, after the closing </context> tag.
def test_prompt_ends_with_the_question():
    hits = [make_hit(1, "faq.txt", 1, "Passwords expire every 180 days.")]
    prompt = generation.build_prompt("How often do passwords expire?", generation.label_hits(hits))
    assert prompt.endswith("Question: How often do passwords expire?")
    assert prompt.index("</context>") < prompt.index("Question:")


# With no hits the prompt still has an empty context block, not a crash.
def test_prompt_with_no_hits_has_an_empty_context():
    prompt = generation.build_prompt("What is the parking policy?", [])
    assert prompt == "<context>\n</context>\n\nQuestion: What is the parking policy?"


# The rules tell the model the exact refusal sentence.
def test_system_rules_contain_the_refusal_sentence():
    assert generation.REFUSAL_TEXT in generation.SYSTEM_RULES
    assert "<context>" in generation.SYSTEM_RULES


# The request to Ollama is non-streaming, temperature 0, with our context window.
def test_generate_answer_sends_deterministic_settings(monkeypatch):
    sent = {}

    # Record what would have been sent and return a canned reply.
    def fake_post(url, json, timeout):
        sent["url"] = url
        sent["body"] = json
        sent["timeout"] = timeout
        return FakeResponse(200, {"response": "27 days [S1]", "prompt_eval_count": 50, "eval_count": 5})

    monkeypatch.setattr(generation.requests, "post", fake_post)
    reply = generation.generate_answer("http://ollama.test", "llama3.2:3b", "rules", "prompt text")

    assert reply["response"] == "27 days [S1]"
    assert sent["url"] == "http://ollama.test/api/generate"
    assert sent["body"]["system"] == "rules"
    assert sent["body"]["prompt"] == "prompt text"
    assert sent["body"]["stream"] is False
    assert sent["body"]["options"]["temperature"] == 0
    assert sent["body"]["options"]["num_ctx"] == generation.CONTEXT_WINDOW_TOKENS
    assert sent["timeout"] == generation.GENERATION_TIMEOUT_SECONDS


# An error status from Ollama (e.g. 404 model not found) is raised, not hidden.
def test_generate_answer_raises_on_http_error(monkeypatch):

    # Pretend Ollama answered 404.
    def fake_post(url, json, timeout):
        return FakeResponse(404, {"error": "model 'llama3.2:3b' not found"})

    monkeypatch.setattr(generation.requests, "post", fake_post)
    with pytest.raises(requests.HTTPError):
        generation.generate_answer("http://ollama.test", "llama3.2:3b", "rules", "prompt text")
