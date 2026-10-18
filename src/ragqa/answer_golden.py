"""Phase 6: answer every golden question and save the answers for manual review."""

import json
from datetime import datetime
from pathlib import Path

from ragqa.ask import answer_question
from ragqa.ask import build_index
from ragqa.chunking import DEFAULT_CHUNK_SIZE
from ragqa.chunking import DEFAULT_OVERLAP
from ragqa.embeddings import EMBEDDING_MODEL_NAME
from ragqa.embeddings import load_embedding_model
from ragqa.generation import CONTEXT_WINDOW_TOKENS
from ragqa.generation import DEFAULT_GENERATOR_MODEL
from ragqa.generation import SYSTEM_RULES
from ragqa.golden import GOLDEN_V0_PATH
from ragqa.golden import load_golden_set
from ragqa.retrieval import DEFAULT_TOP_K
from ragqa.retrieval_metrics import first_relevant_rank

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_FOLDER = PROJECT_ROOT / "results" / "generation"


# Answer each golden question, print a short summary line, and save every answer.
def main() -> None:
    golden_items = load_golden_set(GOLDEN_V0_PATH)
    model = load_embedding_model()
    chunks, chunk_vectors = build_index(model)

    records = []
    for item in golden_items:
        result = answer_question(model, chunks, chunk_vectors, item.question, DEFAULT_TOP_K)

        hits = []
        for label, hit in result["labelled_hits"]:
            hits.append(hit)
        if item.expected_behavior == "answer":
            evidence_rank = first_relevant_rank(hits, item.expected_evidence)
            if evidence_rank is None:
                evidence_shown = f"not in top {DEFAULT_TOP_K}"
            else:
                evidence_shown = f"at [S{evidence_rank}]"
        else:
            evidence_rank = None
            evidence_shown = "none expected"

        print(f"{item.id}  ({item.expected_behavior}, evidence {evidence_shown}, {result['generation_seconds']:.1f} s)")
        print(f"  Q: {item.question}")
        print(f"  A: {result['answer']}")
        print(f"  Reference: {item.reference_answer or '(should refuse)'}")
        print()

        sources = []
        for label, hit in result["labelled_hits"]:
            sources.append({"label": label, "chunk_id": hit.chunk_id, "score": round(hit.score, 4)})
        records.append(
            {
                "id": item.id,
                "category": item.category,
                "expected_behavior": item.expected_behavior,
                "question": item.question,
                "reference_answer": item.reference_answer,
                "answer": result["answer"],
                "evidence_rank_in_context": evidence_rank,
                "sources": sources,
                "prompt": result["prompt"],
                "prompt_tokens": result["prompt_tokens"],
                "output_tokens": result["output_tokens"],
                "generation_seconds": round(result["generation_seconds"], 2),
                "review": "",
            }
        )

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
        "system": SYSTEM_RULES,
    }
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_FOLDER / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_answers.jsonl"
    with open(result_path, "w", encoding="utf-8") as result_file:
        result_file.write(json.dumps({"settings": run_settings}) + "\n")
        for record in records:
            result_file.write(json.dumps(record) + "\n")

    total_seconds = 0.0
    for record in records:
        total_seconds = total_seconds + record["generation_seconds"]
    print(f"Answered {len(records)} questions in {total_seconds:.1f} s of generation")
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
