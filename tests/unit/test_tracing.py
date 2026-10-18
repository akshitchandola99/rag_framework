import json
from pathlib import Path

from ragqa import tracing
from ragqa.errors import GenerationError
from ragqa.generation import CONTEXT_WINDOW_TOKENS
from ragqa.retrieval import SearchHit


# Build a fake answer_question() result, as if one question went through every stage.
def make_result(prompt_tokens: int, done_reason: str) -> dict:
    hit = SearchHit(rank=1, chunk_id="faq.txt:p1:c0", source="faq.txt", page=1, score=0.81234, text="Passwords expire every 180 days.")
    return {
        "question": "How often do passwords expire?",
        "labelled_hits": [("[S1]", hit)],
        "prompt": "<context>\n[S1] (source: faq.txt, page 1)\nPasswords expire every 180 days.\n</context>\n\nQuestion: How often do passwords expire?",
        "answer": "Every 180 days. [S1]",
        "raw_output": "Every 180 days. [S1]\n",
        "done_reason": done_reason,
        "retrieval_seconds": 0.0123,
        "generation_seconds": 1.234,
        "prompt_tokens": prompt_tokens,
        "output_tokens": 8,
    }


# A normal call has no warnings.
def test_normal_call_has_no_warnings():
    assert tracing.find_warnings(400, "stop") == []


# A prompt at 90% of the window or more is flagged.
def test_prompt_near_the_window_is_flagged():
    warnings = tracing.find_warnings(CONTEXT_WINDOW_TOKENS - 10, "stop")
    assert len(warnings) == 1
    assert "may have been cut off" in warnings[0]


# An answer that stopped on a length limit is flagged.
def test_cut_short_answer_is_flagged():
    warnings = tracing.find_warnings(400, "length")
    assert warnings == ["done_reason is 'length', not 'stop': the answer may be cut short"]


# Failed answerable questions with no evidence in the context point at retrieval.
def test_missing_evidence_points_at_retrieval():
    assert tracing.diagnose_stage(False, "answer", None) == "retrieval"


# Failed questions with the evidence in the context point at generation.
def test_present_evidence_points_at_generation():
    assert tracing.diagnose_stage(False, "answer", 1) == "generation"
    assert tracing.diagnose_stage(False, "refuse", None) == "generation"


# Passed questions have no failing stage.
def test_passed_question_has_no_stage():
    assert tracing.diagnose_stage(True, "answer", 1) is None


# The trace keeps chunk IDs, scores, the exact prompt, raw output and per-stage times.
def test_trace_has_every_stage():
    check_result = {"refused": False, "cited_labels": ["[S1]"], "invalid_labels": [], "checks": {"behavior_matches": True}}
    trace = tracing.build_trace("q006", make_result(91, "stop"), 1, check_result, True, None)

    assert trace["status"] == "ok"
    assert list(trace["stages"].keys()) == ["retrieval", "prompt", "generation", "checks"]
    assert trace["stages"]["retrieval"]["hits"][0] == {"label": "[S1]", "chunk_id": "faq.txt:p1:c0", "source": "faq.txt", "page": 1, "score": 0.8123}
    assert trace["stages"]["retrieval"]["seconds"] == 0.012
    assert trace["stages"]["prompt"]["text"].endswith("Question: How often do passwords expire?")
    assert trace["stages"]["generation"]["raw_output"] == "Every 180 days. [S1]\n"
    assert trace["stages"]["generation"]["seconds"] == 1.23
    assert trace["warnings"] == []


# An error trace names the stage, the error type and its message.
def test_error_trace_names_the_stage():
    error = GenerationError("Ollama is not reachable")
    trace = tracing.build_error_trace("q001", "How much leave?", "generation", error)
    assert trace["status"] == "error"
    assert trace["passed"] is False
    assert trace["error"] == {"stage": "generation", "type": "GenerationError", "message": "Ollama is not reachable"}


# The trace file has the settings on line 1 and one JSON trace per line after it.
def test_traces_are_written_as_json_lines(tmp_path: Path):
    trace_path = tmp_path / "traces" / "run.jsonl"
    tracing.write_traces(trace_path, {"top_k": 3}, [{"id": "q001"}, {"id": "q002"}])

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0]) == {"settings": {"top_k": 3}}
    assert json.loads(lines[2])["id"] == "q002"
