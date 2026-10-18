"""A small RAG service with HTTP endpoints for every stage, built on our pipeline code.

Start it with: uv run python -m ragqa.api
Then open http://127.0.0.1:8000/docs to see and try every endpoint.
"""

import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np
import requests
import uvicorn
from fastapi import BackgroundTasks
from fastapi import Depends
from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import Request
from fastapi import UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from pydantic import BaseModel
from pydantic import Field

from ragqa.api_store import PUBLIC_OWNER
from ragqa.api_store import REGISTRY_FILE_NAME
from ragqa.api_store import DocumentRegistry
from ragqa.api_store import add_document_chunks
from ragqa.api_store import delete_document_chunks
from ragqa.api_store import get_document_chunks
from ragqa.api_store import make_document_id
from ragqa.api_store import search_for_user
from ragqa.chunking import DEFAULT_CHUNK_SIZE
from ragqa.chunking import DEFAULT_OVERLAP
from ragqa.chunking import chunk_pages
from ragqa.citations import find_cited_labels
from ragqa.citations import guard_answer
from ragqa.citations import is_refusal
from ragqa.citations import split_citations
from ragqa.embeddings import EMBEDDING_MODEL_NAME
from ragqa.embeddings import QUERY_PREFIX
from ragqa.embeddings import embed_query
from ragqa.embeddings import embed_texts
from ragqa.embeddings import load_embedding_model
from ragqa.generation import CONTEXT_WINDOW_TOKENS
from ragqa.generation import DEFAULT_GENERATOR_MODEL
from ragqa.generation import DEFAULT_OLLAMA_URL
from ragqa.generation import REFUSAL_TEXT
from ragqa.generation import SYSTEM_RULES
from ragqa.generation import build_prompt
from ragqa.generation import generate_answer
from ragqa.generation import label_hits
from ragqa.loaders import load_document
from ragqa.retrieval import DEFAULT_TOP_K
from ragqa.store import open_collection

API_VERSION = "0.1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE_FOLDER = PROJECT_ROOT / "storage" / "api"

# Settings can be changed with environment variables (tests use this to get a clean, separate server).
STORAGE_FOLDER = Path(os.environ.get("RAGQA_STORAGE_DIR", str(DEFAULT_STORAGE_FOLDER))).resolve()
OLLAMA_URL = os.environ.get("RAGQA_OLLAMA_URL", DEFAULT_OLLAMA_URL)
GENERATOR_MODEL = os.environ.get("RAGQA_GENERATOR_MODEL", DEFAULT_GENERATOR_MODEL)
API_HOST = "127.0.0.1"
API_PORT = int(os.environ.get("RAGQA_PORT", "8000"))

SUPPORTED_EXTENSIONS = [".pdf", ".md", ".txt"]
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_TOP_K = 20
MAX_EMBED_TEXTS = 64
MAX_QUESTION_CHARS = 1000
READY_CHECK_TIMEOUT_SECONDS = 5

# Filled in at startup: the embedding model, the Chroma collection and the document registry.
state = {}


# Body of POST /embed.
class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_EMBED_TEXTS)
    is_query: bool = False


# Body of POST /search.
class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    user_id: str = PUBLIC_OWNER


# Body of POST /chat.
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    user_id: str = PUBLIC_OWNER
    include_sources: bool = True
    include_trace: bool = False


# Load the model, open the vector store and the registry once, when the server starts.
@asynccontextmanager
async def lifespan(app: FastAPI):
    STORAGE_FOLDER.mkdir(parents=True, exist_ok=True)
    (STORAGE_FOLDER / "uploads").mkdir(exist_ok=True)

    model = load_embedding_model()
    state["model"] = model
    state["embedding_dimension"] = int(embed_texts(model, ["dimension check"]).shape[1])
    state["collection"] = open_collection(STORAGE_FOLDER / "chroma")
    registry = DocumentRegistry(STORAGE_FOLDER / REGISTRY_FILE_NAME)
    state["registry"] = registry

    # A job cut off by a restart can never finish, so say so instead of "processing" forever.
    for document in registry.all():
        if document["status"] == "processing":
            registry.update(document["document_id"], {"status": "failed", "error": "interrupted by a server restart"})
    yield
    state.clear()


app = FastAPI(title="RAG Quality Lab API", version=API_VERSION, lifespan=lifespan)


# Build a JSON error reply with one clear message.
def error_response(status_code: int, message: str, extra: dict | None = None) -> JSONResponse:
    content = {"error": message}
    if extra is not None:
        content.update(extra)
    return JSONResponse(status_code=status_code, content=content)


# Turn FastAPI's input-validation errors into our error format (still status 422).
@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    details = jsonable_encoder(error.errors())
    return error_response(422, "invalid request", {"details": details})


# Milliseconds since a perf_counter start time, rounded for reading.
def elapsed_ms(start_time: float) -> float:
    return round((time.perf_counter() - start_time) * 1000, 1)


# Count how many model tokens each text uses (including the model's start and end tokens).
def count_tokens(texts: list[str]) -> list[int]:
    tokenizer = state["model"].tokenizer
    token_counts = []
    for text in texts:
        token_counts.append(len(tokenizer(text)["input_ids"]))
    return token_counts


# Liveness: the server process is up and answering.
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Readiness: every dependency works, plus the settings this server runs with.
@app.get("/ready")
def ready() -> JSONResponse:
    checks = {}
    checks["embedding_model"] = {"ok": "model" in state, "detail": EMBEDDING_MODEL_NAME}

    try:
        chunk_count = state["collection"].count()
        checks["vector_store"] = {"ok": True, "detail": f"{chunk_count} chunks stored"}
    except Exception as error:
        checks["vector_store"] = {"ok": False, "detail": str(error)}

    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=READY_CHECK_TIMEOUT_SECONDS)
        response.raise_for_status()
        model_names = []
        for model in response.json()["models"]:
            model_names.append(model["name"])
        if GENERATOR_MODEL in model_names:
            checks["llm_server"] = {"ok": True, "detail": f"{GENERATOR_MODEL} is installed"}
        else:
            checks["llm_server"] = {"ok": False, "detail": f"{GENERATOR_MODEL} is not installed in Ollama"}
    except requests.RequestException as error:
        checks["llm_server"] = {"ok": False, "detail": f"Ollama not reachable at {OLLAMA_URL}: {error}"}

    all_ok = True
    for check in checks.values():
        if not check["ok"]:
            all_ok = False

    settings = {
        "api_version": API_VERSION,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": state["embedding_dimension"],
        "embedding_max_tokens": state["model"].max_seq_length,
        "generator_model": GENERATOR_MODEL,
        "ollama_url": OLLAMA_URL,
        "context_window_tokens": CONTEXT_WINDOW_TOKENS,
        "chunk_size_chars": DEFAULT_CHUNK_SIZE,
        "chunk_overlap_chars": DEFAULT_OVERLAP,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_top_k": MAX_TOP_K,
    }
    body = {"ready": all_ok, "checks": checks, "settings": settings}
    if all_ok:
        return JSONResponse(status_code=200, content=body)
    return JSONResponse(status_code=503, content=body)


# Background job: chunk, embed and store one uploaded document, then mark it ready or failed.
def process_document(document_id: str, owner: str, pages: list) -> None:
    registry = state["registry"]
    try:
        chunks = chunk_pages(pages, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP)
        chunk_texts = []
        for chunk in chunks:
            chunk_texts.append(chunk.text)
        chunk_vectors = embed_texts(state["model"], chunk_texts)
        token_counts = count_tokens(chunk_texts)
        add_document_chunks(state["collection"], document_id, owner, chunks, chunk_vectors, token_counts)
        registry.update(document_id, {"status": "ready", "chunk_count": len(chunks), "error": None})
    except Exception as error:
        # A background job has nobody to raise to, so the error goes into the job status.
        registry.update(document_id, {"status": "failed", "error": str(error)})


# Check a file's type and size; return an error reply, or None when the file is fine.
def check_file(safe_name: str, file_bytes: bytes) -> JSONResponse | None:
    extension = Path(safe_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        return error_response(
            415,
            f"{safe_name}: unsupported file type '{extension}'",
            {"supported": SUPPORTED_EXTENSIONS},
        )
    if len(file_bytes) == 0:
        return error_response(422, f"{safe_name}: file is empty")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        return error_response(413, f"{safe_name}: file is larger than {MAX_UPLOAD_BYTES} bytes")
    return None


# Turn loaded pages into JSON: page number, character count and the full extracted text.
def pages_to_json(pages: list) -> list[dict]:
    pages_json = []
    for page in pages:
        pages_json.append({"page": page.page, "char_count": len(page.text), "text": page.text})
    return pages_json


# Read the owner field exactly as the client sent it: None if it was not sent at all.
async def read_raw_owner(request: Request) -> str | None:
    # FastAPI swaps an empty form field ("") for the default ("public") before our code sees it,
    # so a blank owner would silently make a private upload public. The raw form keeps the "".
    form = await request.form()
    return form.get("owner")


# Upload a document: check it, load its pages now, then chunk and embed it in the background.
@app.post("/documents", status_code=201)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    owner: str = Form(PUBLIC_OWNER),
    raw_owner: str | None = Depends(read_raw_owner),
):
    # Keep only the file name, so a name like "../../secret.md" can't write outside our folder.
    safe_name = Path(file.filename or "").name
    if safe_name == "":
        return error_response(422, "the uploaded file has no name")
    # Not sent at all is fine (the document is public); sent but blank is a mistake.
    if raw_owner is not None and raw_owner.strip() == "":
        return error_response(422, "owner must not be empty")

    file_bytes = file.file.read()
    file_problem = check_file(safe_name, file_bytes)
    if file_problem is not None:
        return file_problem

    extension = Path(safe_name).suffix.lower()
    registry = state["registry"]
    document_id = make_document_id(owner, file_bytes)
    if registry.get(document_id) is not None:
        return error_response(409, f"{safe_name}: this file was already uploaded by {owner}", {"id": document_id})

    upload_folder = STORAGE_FOLDER / "uploads" / document_id
    upload_folder.mkdir(parents=True, exist_ok=True)
    file_path = upload_folder / safe_name
    file_path.write_bytes(file_bytes)

    try:
        pages = load_document(file_path)
    except ValueError as error:
        shutil.rmtree(upload_folder)
        # Some loader errors quote the full server path; never show that to the caller.
        message = str(error).replace(str(file_path), safe_name)
        return error_response(422, message)

    page_summaries = []
    for page in pages:
        page_summaries.append({"page": page.page, "char_count": len(page.text)})

    job_id = f"job-{document_id}"
    document = {
        "document_id": document_id,
        "job_id": job_id,
        "source": safe_name,
        "file_type": extension.lstrip("."),
        "owner": owner,
        "size_bytes": len(file_bytes),
        "page_count": len(pages),
        "pages": page_summaries,
        "status": "processing",
        "chunk_count": 0,
        "error": None,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    registry.put(document)
    background_tasks.add_task(process_document, document_id, owner, pages)
    return {"id": document_id, "job_id": job_id, "status": "processing", "source": safe_name}


# Extract a file's text page by page and return it, without storing anything (a dry run of Phase 1).
@app.post("/extract")
def extract_text(file: UploadFile = File(...)):
    safe_name = Path(file.filename or "").name
    if safe_name == "":
        return error_response(422, "the uploaded file has no name")
    file_bytes = file.file.read()
    file_problem = check_file(safe_name, file_bytes)
    if file_problem is not None:
        return file_problem

    # The loader needs a real file, so write it to a temporary folder that is deleted right after.
    with tempfile.TemporaryDirectory() as temporary_folder:
        file_path = Path(temporary_folder) / safe_name
        file_path.write_bytes(file_bytes)
        try:
            pages = load_document(file_path)
        except ValueError as error:
            message = str(error).replace(str(file_path), safe_name)
            return error_response(422, message)

    return {
        "source": safe_name,
        "file_type": Path(safe_name).suffix.lower().lstrip("."),
        "page_count": len(pages),
        "pages": pages_to_json(pages),
    }


# One stored document's text, page by page, extracted again from its uploaded file.
@app.get("/documents/{document_id}/pages")
def get_pages(document_id: str):
    document = state["registry"].get(document_id)
    if document is None:
        return error_response(404, f"no document with id {document_id}")
    file_path = STORAGE_FOLDER / "uploads" / document_id / document["source"]
    pages = load_document(file_path)
    return {
        "document_id": document_id,
        "source": document["source"],
        "page_count": len(pages),
        "pages": pages_to_json(pages),
    }


# List documents, optionally only one owner's.
@app.get("/documents")
def list_documents(owner: str | None = None) -> dict:
    documents = []
    for document in state["registry"].all():
        if owner is None or document["owner"] == owner:
            documents.append(document)
    return {"documents": documents}


# One document's metadata.
@app.get("/documents/{document_id}")
def get_document(document_id: str):
    document = state["registry"].get(document_id)
    if document is None:
        return error_response(404, f"no document with id {document_id}")
    return document


# One document's stored chunks, in reading order.
@app.get("/documents/{document_id}/chunks")
def get_chunks(document_id: str):
    document = state["registry"].get(document_id)
    if document is None:
        return error_response(404, f"no document with id {document_id}")
    if document["status"] != "ready":
        return error_response(409, f"document is {document['status']}, not ready", {"status": document["status"]})
    chunks = get_document_chunks(state["collection"], document_id)
    return {"document_id": document_id, "chunk_count": len(chunks), "chunks": chunks}


# Delete a document, its chunks and its uploaded file.
@app.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str):
    registry = state["registry"]
    if registry.get(document_id) is None:
        return error_response(404, f"no document with id {document_id}")
    delete_document_chunks(state["collection"], document_id)
    upload_folder = STORAGE_FOLDER / "uploads" / document_id
    if upload_folder.exists():
        shutil.rmtree(upload_folder)
    registry.remove(document_id)
    return Response(status_code=204)


# The status of one upload job.
@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    document_id = job_id.removeprefix("job-")
    document = state["registry"].get(document_id)
    if document is None or document["job_id"] != job_id:
        return error_response(404, f"no job with id {job_id}")
    return {
        "job_id": job_id,
        "document_id": document_id,
        "status": document["status"],
        "error": document["error"],
    }


# Turn texts into vectors with the same model the index uses.
@app.post("/embed")
def embed(request: EmbedRequest) -> dict:
    texts = []
    for text in request.texts:
        if request.is_query:
            texts.append(QUERY_PREFIX + text)
        else:
            texts.append(text)
    vectors = embed_texts(state["model"], texts)
    vector_lists = []
    for vector in vectors:
        vector_lists.append(vector.tolist())
    return {
        "model": EMBEDDING_MODEL_NAME,
        "dimension": int(vectors.shape[1]),
        "is_query": request.is_query,
        "vectors": vector_lists,
    }


# Turn (hit, document_id) pairs into JSON-ready dictionaries.
def hits_to_json(hits_with_document: list) -> list[dict]:
    hits_json = []
    for hit, document_id in hits_with_document:
        hits_json.append(
            {
                "rank": hit.rank,
                "chunk_id": hit.chunk_id,
                "document_id": document_id,
                "source": hit.source,
                "page": hit.page,
                "score": round(hit.score, 6),
                "text": hit.text,
            }
        )
    return hits_json


# Find the top_k chunks this user may read that best match the query.
@app.post("/search")
def search(request: SearchRequest):
    if request.query.strip() == "":
        return error_response(422, "query must not be only spaces")
    start_time = time.perf_counter()
    query_vector = embed_query(state["model"], request.query)
    hits_with_document = search_for_user(state["collection"], query_vector, request.top_k, request.user_id)
    return {
        "query": request.query,
        "top_k": request.top_k,
        "user_id": request.user_id,
        "hits": hits_to_json(hits_with_document),
        "latency_ms": elapsed_ms(start_time),
    }


# Answer a question from the documents, with citations, sources and an optional trace.
@app.post("/chat")
def chat(request: ChatRequest):
    if request.question.strip() == "":
        return error_response(422, "question must not be only spaces")
    total_start = time.perf_counter()

    retrieval_start = time.perf_counter()
    query_vector: np.ndarray = embed_query(state["model"], request.question)
    hits_with_document = search_for_user(state["collection"], query_vector, request.top_k, request.user_id)
    retrieval_ms = elapsed_ms(retrieval_start)

    hits = []
    for hit, document_id in hits_with_document:
        hits.append(hit)
    labelled_hits = label_hits(hits)

    sources = []
    sent_labels = []
    for index, (label, hit) in enumerate(labelled_hits):
        sent_labels.append(label)
        sources.append(
            {
                "label": label,
                "chunk_id": hit.chunk_id,
                "document_id": hits_with_document[index][1],
                "source": hit.source,
                "page": hit.page,
                "score": round(hit.score, 6),
                "text": hit.text,
            }
        )

    prompt = build_prompt(request.question, labelled_hits)
    llm_reply = {}
    generation_ms = 0.0
    if len(labelled_hits) == 0:
        # Nothing to ground an answer on, so refuse without calling the LLM.
        answer = REFUSAL_TEXT
    else:
        generation_start = time.perf_counter()
        try:
            llm_reply = generate_answer(OLLAMA_URL, GENERATOR_MODEL, SYSTEM_RULES, prompt)
        except requests.ConnectionError:
            return error_response(503, f"LLM server not reachable at {OLLAMA_URL}")
        except requests.Timeout:
            return error_response(504, "LLM server did not answer in time")
        except requests.HTTPError as error:
            return error_response(502, f"LLM server returned an error: {error}")
        generation_ms = elapsed_ms(generation_start)
        answer = llm_reply["response"].strip()

    # Phase 8: block answers that cite none of our sources (a sign of prompt injection or made-up text).
    # The raw model output stays in the trace, so we can still see what was blocked.
    answer, guard_triggered = guard_answer(answer, sent_labels)

    cited_labels = find_cited_labels(answer)
    valid_labels, invalid_labels = split_citations(cited_labels, sent_labels)

    body = {
        "question": request.question,
        "answer": answer,
        "refused": is_refusal(answer),
        "citations": cited_labels,
        "invalid_citations": invalid_labels,
        "guard_triggered": guard_triggered,
        "sources": [],
        "latency_ms": elapsed_ms(total_start),
        "trace": None,
    }
    if request.include_sources:
        body["sources"] = sources
    if request.include_trace:
        body["trace"] = {
            "llm_called": len(labelled_hits) > 0,
            "generator_model": GENERATOR_MODEL,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "system": SYSTEM_RULES,
            "prompt": prompt,
            "raw_output": llm_reply.get("response"),
            "guard_triggered": guard_triggered,
            "prompt_tokens": llm_reply.get("prompt_eval_count"),
            "output_tokens": llm_reply.get("eval_count"),
            "context_window_tokens": CONTEXT_WINDOW_TOKENS,
        }
    return body


# Start the API server on this computer only.
def main() -> None:
    uvicorn.run(app, host=API_HOST, port=API_PORT)


if __name__ == "__main__":
    main()
