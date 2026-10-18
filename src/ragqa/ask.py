"""Phases 6-8: answer one question from the documents with a grounded prompt."""

import sys
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from ragqa.chunking import DEFAULT_CHUNK_SIZE
from ragqa.chunking import DEFAULT_OVERLAP
from ragqa.chunking import Chunk
from ragqa.chunking import chunk_pages
from ragqa.citations import guard_answer
from ragqa.embeddings import embed_query
from ragqa.embeddings import embed_texts
from ragqa.embeddings import load_embedding_model
from ragqa.errors import GenerationError
from ragqa.generation import CONTEXT_WINDOW_TOKENS
from ragqa.generation import DEFAULT_GENERATOR_MODEL
from ragqa.generation import build_prompt
from ragqa.generation import call_generator
from ragqa.generation import label_hits
from ragqa.loaders import DEFAULT_DATA_FOLDER
from ragqa.loaders import load_folder
from ragqa.retrieval import DEFAULT_TOP_K
from ragqa.retrieval import search_chunks

PREVIEW_LENGTH = 70


# Load, chunk and embed every document once, so many questions can reuse it.
def build_index(model: SentenceTransformer) -> tuple[list[Chunk], np.ndarray]:
    pages, skipped_files = load_folder(DEFAULT_DATA_FOLDER)
    for reason in skipped_files:
        print(f"SKIPPED {reason}")
    chunks = chunk_pages(pages, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP)
    chunk_texts = []
    for chunk in chunks:
        chunk_texts.append(chunk.text)
    chunk_vectors = embed_texts(model, chunk_texts)
    return chunks, chunk_vectors


# Retrieve the top chunks for a question, then ask Ollama for a grounded answer.
def answer_question(
    model: SentenceTransformer,
    chunks: list[Chunk],
    chunk_vectors: np.ndarray,
    question: str,
    top_k: int,
) -> dict:
    retrieval_start = time.perf_counter()
    query_vector = embed_query(model, question)
    hits = search_chunks(query_vector, chunks, chunk_vectors, top_k)
    retrieval_seconds = time.perf_counter() - retrieval_start

    labelled_hits = label_hits(hits)
    prompt = build_prompt(question, labelled_hits)

    start_time = time.perf_counter()
    llm_reply = call_generator(prompt)
    generation_seconds = time.perf_counter() - start_time

    # Phase 8: block answers that cite none of our sources (a sign of prompt injection or made-up text).
    sent_labels = []
    for label, hit in labelled_hits:
        sent_labels.append(label)
    final_answer, guard_triggered = guard_answer(llm_reply["response"].strip(), sent_labels)

    return {
        "question": question,
        "labelled_hits": labelled_hits,
        "prompt": prompt,
        "answer": final_answer,
        "raw_output": llm_reply["response"],
        "guard_triggered": guard_triggered,
        "done_reason": llm_reply.get("done_reason"),
        "retrieval_seconds": retrieval_seconds,
        "generation_seconds": generation_seconds,
        "prompt_tokens": llm_reply["prompt_eval_count"],
        "output_tokens": llm_reply["eval_count"],
    }


# Answer the question given on the command line and show what the model saw.
def main() -> None:
    if len(sys.argv) != 2:
        print('Usage: uv run python -m ragqa.ask "your question"')
        sys.exit(1)
    question = sys.argv[1]

    model = load_embedding_model()
    chunks, chunk_vectors = build_index(model)
    try:
        result = answer_question(model, chunks, chunk_vectors, question, DEFAULT_TOP_K)
    except GenerationError as error:
        print(f"ERROR: {error}")
        sys.exit(2)

    print(f"Question: {question}")
    print(f"Sources sent to {DEFAULT_GENERATOR_MODEL}:")
    for label, hit in result["labelled_hits"]:
        preview = " ".join(hit.text.split())[:PREVIEW_LENGTH]
        print(f"  {label} {hit.score:.3f}  {hit.chunk_id}  {preview!r}")
    print()
    print("Answer:")
    print(result["answer"])
    print()
    print(f"Generation: {result['generation_seconds']:.1f} s")
    print(f"Tokens: {result['prompt_tokens']} in the prompt (limit {CONTEXT_WINDOW_TOKENS}), {result['output_tokens']} in the answer")


if __name__ == "__main__":
    main()
