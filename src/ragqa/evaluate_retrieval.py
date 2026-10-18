"""Phase 5: measure retrieval on the golden set and save the numbers as a baseline."""

import json
import sys
from datetime import datetime
from pathlib import Path

from ragqa.chunking import DEFAULT_CHUNK_SIZE
from ragqa.chunking import DEFAULT_OVERLAP
from ragqa.chunking import chunk_pages
from ragqa.embeddings import EMBEDDING_MODEL_NAME
from ragqa.embeddings import embed_query
from ragqa.embeddings import embed_texts
from ragqa.embeddings import load_embedding_model
from ragqa.golden import GOLDEN_V0_PATH
from ragqa.golden import find_missing_quotes
from ragqa.golden import load_golden_set
from ragqa.loaders import DEFAULT_DATA_FOLDER
from ragqa.loaders import load_folder
from ragqa.retrieval import search_chunks
from ragqa.retrieval_metrics import first_relevant_rank
from ragqa.retrieval_metrics import hit_at_k
from ragqa.retrieval_metrics import mean_reciprocal_rank

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_FOLDER = PROJECT_ROOT / "results" / "retrieval"
K_VALUES = [1, 3, 5]


# Run every golden question through retrieval, print the scores and save them.
def main() -> None:
    golden_items = load_golden_set(GOLDEN_V0_PATH)
    pages, skipped_files = load_folder(DEFAULT_DATA_FOLDER)

    problems = find_missing_quotes(golden_items, pages)
    if len(problems) > 0:
        for problem in problems:
            print(f"GOLDEN SET ERROR {problem}")
        sys.exit(1)

    model = load_embedding_model()
    chunks = chunk_pages(pages, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP)
    chunk_texts = []
    for chunk in chunks:
        chunk_texts.append(chunk.text)
    chunk_vectors = embed_texts(model, chunk_texts)

    first_ranks = []
    per_question = []
    print(f"{'id':5} {'category':13} {'rank':>4}  top hit (score)")
    for item in golden_items:
        query_vector = embed_query(model, item.question)
        hits = search_chunks(query_vector, chunks, chunk_vectors, top_k=len(chunks))
        top_hit = hits[0]
        if item.expected_behavior == "answer":
            rank = first_relevant_rank(hits, item.expected_evidence)
            first_ranks.append(rank)
            if rank is None:
                shown_rank = "-"
            else:
                shown_rank = str(rank)
        else:
            rank = None
            shown_rank = "n/a"
        print(f"{item.id:5} {item.category:13} {shown_rank:>4}  {top_hit.chunk_id} ({top_hit.score:.3f})")
        per_question.append(
            {
                "id": item.id,
                "category": item.category,
                "expected_behavior": item.expected_behavior,
                "first_relevant_rank": rank,
                "top_chunk_id": top_hit.chunk_id,
                "top_score": round(top_hit.score, 4),
            }
        )

    metrics = {}
    for k in K_VALUES:
        metrics[f"hit@{k}"] = round(hit_at_k(first_ranks, k), 4)
    metrics["mrr"] = round(mean_reciprocal_rank(first_ranks), 4)

    print()
    print(f"Scored {len(first_ranks)} answerable questions (unanswerable ones are listed but not scored)")
    for metric_name, value in metrics.items():
        print(f"  {metric_name:6} {value:.3f}")

    run = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "golden_file": GOLDEN_V0_PATH.name,
        "settings": {
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "overlap": DEFAULT_OVERLAP,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "search": "numpy brute force",
        },
        "metrics": metrics,
        "per_question": per_question,
    }
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_FOLDER / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_baseline.json"
    with open(result_path, "w", encoding="utf-8") as result_file:
        json.dump(run, result_file, indent=2)
    print(f"Saved: {result_path}")

    for reason in skipped_files:
        print(f"SKIPPED {reason}")


if __name__ == "__main__":
    main()
