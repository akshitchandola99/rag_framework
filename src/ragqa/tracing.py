"""Phase 7: one trace record per question, with what happened at every stage."""

import json
from pathlib import Path

from ragqa.generation import CONTEXT_WINDOW_TOKENS
from ragqa.generation import DEFAULT_GENERATOR_MODEL

PROMPT_WARNING_SHARE = 0.9


# List anything suspicious about one LLM call: a prompt near the window, or an answer cut short.
def find_warnings(prompt_tokens: int, done_reason: str | None) -> list[str]:
    warnings = []
    if prompt_tokens >= PROMPT_WARNING_SHARE * CONTEXT_WINDOW_TOKENS:
        warnings.append(f"prompt uses {prompt_tokens} of {CONTEXT_WINDOW_TOKENS} tokens: it may have been cut off")
    if done_reason != "stop":
        warnings.append(f"done_reason is {done_reason!r}, not 'stop': the answer may be cut short")
    return warnings


# Name the stage to look at first when a question fails, or None when it passed.
def diagnose_stage(passed: bool, expected_behavior: str, evidence_rank: int | None) -> str | None:
    if passed:
        return None
    if expected_behavior == "answer" and evidence_rank is None:
        return "retrieval"
    return "generation"


# Build the trace for a question that went through every stage.
def build_trace(
    item_id: str,
    result: dict,
    evidence_rank: int | None,
    check_result: dict,
    passed: bool,
    failing_stage: str | None,
) -> dict:
    retrieved = []
    for label, hit in result["labelled_hits"]:
        retrieved.append(
            {
                "label": label,
                "chunk_id": hit.chunk_id,
                "source": hit.source,
                "page": hit.page,
                "score": round(hit.score, 4),
            }
        )

    return {
        "id": item_id,
        "question": result["question"],
        "status": "ok",
        "passed": passed,
        "failing_stage": failing_stage,
        "answer": result["answer"],
        "stages": {
            "retrieval": {
                "seconds": round(result["retrieval_seconds"], 3),
                "hits": retrieved,
                "evidence_rank": evidence_rank,
            },
            "prompt": {
                "characters": len(result["prompt"]),
                "text": result["prompt"],
            },
            "generation": {
                "seconds": round(result["generation_seconds"], 2),
                "model": DEFAULT_GENERATOR_MODEL,
                "prompt_tokens": result["prompt_tokens"],
                "output_tokens": result["output_tokens"],
                "context_window_tokens": CONTEXT_WINDOW_TOKENS,
                "done_reason": result["done_reason"],
                "raw_output": result["raw_output"],
                "guard_triggered": result.get("guard_triggered", False),
            },
            "checks": {
                "refused": check_result["refused"],
                "cited_labels": check_result["cited_labels"],
                "invalid_labels": check_result["invalid_labels"],
                "results": check_result["checks"],
            },
        },
        "warnings": find_warnings(result["prompt_tokens"], result["done_reason"]),
        "error": None,
    }


# Build the trace for a question that stopped with an error at one stage.
def build_error_trace(item_id: str, question: str, stage: str, error: Exception) -> dict:
    return {
        "id": item_id,
        "question": question,
        "status": "error",
        "passed": False,
        "failing_stage": stage,
        "answer": None,
        "stages": {},
        "warnings": [],
        "error": {"stage": stage, "type": type(error).__name__, "message": str(error)},
    }


# Save the run settings on line 1, then one trace per line (JSON Lines).
def write_traces(path: Path, settings: dict, traces: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as trace_file:
        trace_file.write(json.dumps({"settings": settings}) + "\n")
        for trace in traces:
            trace_file.write(json.dumps(trace) + "\n")
