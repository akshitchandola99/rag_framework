"""Phase 7: answer every golden question, check citations and refusals, and save a per-stage trace."""

import sys
from datetime import datetime
from pathlib import Path

from ragqa.ask import answer_question
from ragqa.ask import build_index
from ragqa.chunking import DEFAULT_CHUNK_SIZE
from ragqa.chunking import DEFAULT_OVERLAP
from ragqa.citations import all_checks_pass
from ragqa.citations import check_answer
from ragqa.embeddings import EMBEDDING_MODEL_NAME
from ragqa.embeddings import load_embedding_model
from ragqa.errors import GenerationError
from ragqa.generation import CONTEXT_WINDOW_TOKENS
from ragqa.generation import DEFAULT_GENERATOR_MODEL
from ragqa.generation import SYSTEM_RULES
from ragqa.golden import GOLDEN_V0_PATH
from ragqa.golden import load_golden_set
from ragqa.loaders import DEFAULT_DATA_FOLDER
from ragqa.loaders import load_folder
from ragqa.retrieval import DEFAULT_TOP_K
from ragqa.retrieval_metrics import first_relevant_rank
from ragqa.tracing import build_error_trace
from ragqa.tracing import build_trace
from ragqa.tracing import diagnose_stage
from ragqa.tracing import write_traces

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACES_FOLDER = PROJECT_ROOT / "results" / "traces"


# Turn one check result into a short word for the printed table.
def check_word(result: bool | None) -> str:
    if result is None:
        return "-"
    if result:
        return "ok"
    return "FAIL"


# Answer and check every golden question, print one line each, save the traces, exit 1 on any failure.
def main() -> None:
    golden_items = load_golden_set(GOLDEN_V0_PATH)
    model = load_embedding_model()
    chunks, chunk_vectors = build_index(model)
    # Load once more only to record which files were skipped; build_index just prints them.
    pages, skipped_files = load_folder(DEFAULT_DATA_FOLDER)

    print("id    expect  behavior citations cited evidence refusal  stage       answer")
    traces = []
    for item in golden_items:
        try:
            result = answer_question(model, chunks, chunk_vectors, item.question, DEFAULT_TOP_K)
        except GenerationError as error:
            trace = build_error_trace(item.id, item.question, "generation", error)
            traces.append(trace)
            print(f"{item.id}  ERROR at generation: {error}")
            continue

        hits = []
        for label, hit in result["labelled_hits"]:
            hits.append(hit)
        evidence_rank = first_relevant_rank(hits, item.expected_evidence)

        check_result = check_answer(
            result["answer"],
            result["labelled_hits"],
            item.expected_behavior,
            item.expected_evidence,
        )
        checks = check_result["checks"]
        passed = all_checks_pass(checks)
        failing_stage = diagnose_stage(passed, item.expected_behavior, evidence_rank)
        trace = build_trace(item.id, result, evidence_rank, check_result, passed, failing_stage)
        traces.append(trace)

        short_answer = " ".join(result["answer"].split())[:45]
        print(
            f"{item.id}  {item.expected_behavior:<7} "
            f"{check_word(checks['behavior_matches']):<8} "
            f"{check_word(checks['citations_valid']):<9} "
            f"{check_word(checks['answer_is_cited']):<5} "
            f"{check_word(checks['cites_evidence']):<8} "
            f"{check_word(checks['refusal_is_clean']):<8} "
            f"{failing_stage or '-':<11} "
            f"{short_answer!r}"
        )
        for warning in trace["warnings"]:
            print(f"      WARNING {warning}")

    passed_count = 0
    error_count = 0
    for trace in traces:
        if trace["passed"]:
            passed_count = passed_count + 1
        if trace["status"] == "error":
            error_count = error_count + 1

    run_settings = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "golden_file": GOLDEN_V0_PATH.name,
        "generator_model": DEFAULT_GENERATOR_MODEL,
        "temperature": 0,
        "num_ctx": CONTEXT_WINDOW_TOKENS,
        "top_k": DEFAULT_TOP_K,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "overlap": DEFAULT_OVERLAP,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "skipped_files": skipped_files,
        "system": SYSTEM_RULES,
    }
    trace_path = TRACES_FOLDER / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_traces.jsonl"
    write_traces(trace_path, run_settings, traces)

    print()
    print(f"Passed {passed_count} of {len(traces)} questions ({error_count} errors)")
    print(f"Saved: {trace_path}")
    if passed_count < len(traces):
        sys.exit(1)


if __name__ == "__main__":
    main()
