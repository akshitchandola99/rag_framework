# RAG Quality Lab — Phase 4: Retrieval

Oct 15, 2026 · @Akshit

## Phase tracker

Tick a phase only when its verification checks pass.

- [ ] Phase 0 — Environment, Git, Ollama, hardware baseline ([separate doc](https://claude.ai/code/artifact/b32269ff-ce8f-4186-b984-8cf68f206b1b))
- [ ] Phase 1 — Load documents with metadata ([separate doc](https://claude.ai/code/artifact/7df3b55e-6623-464f-8990-7106ec3af43f))
- [ ] Phase 2 — Chunking ([separate doc](https://claude.ai/code/artifact/4b5e335a-21aa-435a-98c4-3d6105208c7f))
- [ ] Phase 3 — Local embeddings ([separate doc](https://claude.ai/code/artifact/a37f9484-2b02-439c-bcbd-42d9145c7a5a))
- [ ] Phase 4 — Retrieval (numpy brute force, then Chroma) (this doc)
- [ ] Phase 5 — Golden v0 + hit@k / MRR ([separate doc](https://claude.ai/code/artifact/af02e615-699e-4072-b059-e6fd541a7eac))
- [ ] Phase 6 — Grounded generation with Ollama ([separate doc](https://claude.ai/code/artifact/dc112016-b8cb-489d-b01d-377efa3083a1))
- [ ] Phase 7 — Citations, refusal, errors, tracing ([separate doc](https://claude.ai/code/artifact/1e0d77c4-cdca-4be6-af3c-6ac45a522e8d))
- [ ] Phase 8 — Consolidated pytest suite + robustness fixtures ([separate doc](https://claude.ai/code/artifact/8dd46bfa-af84-4bf8-a1bb-8537f5671950))
- [ ] Phase 9 — Golden v1, human-verified
- [ ] Phase 10 — RAGAS with a local judge
- [ ] Phase 11 — DeepEval with Ollama + G-Eval
- [ ] Phase 12 — Regression runner, thresholds, reports
- [ ] Phase 13 — Evidence-driven improvement (+ optional UI)
- [ ] Phase 14 — Documentation and portfolio polish

**Goal of this phase:** given a question, find the **top k** chunks whose vectors are closest to it. First with a few lines of numpy, so you see exactly what search is. Then with **Chroma**, a vector database that saves chunks and vectors on disk. Finally, prove both give the same answer, and that adding the same documents twice never creates duplicates. Ollama can stay off. Previous phase: [Phase 3](https://claude.ai/code/artifact/a37f9484-2b02-439c-bcbd-42d9145c7a5a).

## 4.1 Concepts

In Phase 3's exercise you already ranked chunks by score. Phase 4 turns that into two proper tools: a numpy search for learning and testing, and a stored index that later phases query without re-embedding every document each time.

**Words you'll meet in this phase**

| Term | Plain meaning | In our project |
| --- | --- | --- |
| Retrieval | Finding the stored pieces of text that best match a question | The R in RAG |
| Top k | The k best results, best first | `DEFAULT_TOP_K = 3` |
| Brute force search | Score the question against **every** chunk, then sort. Simple and exact | `search_chunks()` in numpy |
| Vector database | A program that stores vectors with their text and metadata, and finds the closest ones quickly | Chroma |
| Chroma | A free, open-source vector database (Apache 2.0) that runs inside our Python program, with no server to start | Data saved in `storage/chroma/` |
| Collection | Chroma's name for one table of items: IDs, vectors, texts and metadata | One collection called `chunks` |
| Cosine distance | 1 minus cosine similarity. Chroma sorts by *distance* (smaller is closer); we turn it back into a similarity score | distance 0.04 → score 0.96 |
| Index (HNSW) | A shortcut structure that lets a database find close vectors without checking every one. HNSW ("Hierarchical Navigable Small World") is the one Chroma uses | Exact for our 7 chunks; matters with thousands |
| Content-hash ID | An ID made by running a chunk's content through a hash function, so the same content always gets the same ID | 16 characters, e.g. from `sha256` |
| Hash function | A recipe that turns any text into a short fixed-length code; the same input always gives the same code, and any change gives a different one | `hashlib.sha256` from Python's standard library |
| Upsert | "Update or insert": add an item, or replace it if its ID already exists | How we avoid duplicates |
| Idempotent | Doing something twice has the same effect as doing it once | Re-ingesting gives the same count |
| Ingest | Load, chunk, embed and store documents | What `search.py` does before searching |

**Why brute force first?** It's about 10 lines of numpy, so nothing is hidden. With 7 chunks it's instant and exact. Because it's exact, it's the **reference**: if Chroma ever returns something different, Chroma (or how we use it) is wrong. The plan's check for this phase is exactly that: both give the same top k.

**Why a vector database at all, then?** Three reasons, and none of them is speed at our size:

1. **It remembers.** Vectors are saved on disk, so later phases (generation, evaluation) don't re-embed every document on every run.
2. **It keeps text and metadata with the vector**, so a search result already carries its source and page for citations.
3. **It scales.** Brute force checks every chunk; with millions of chunks that gets slow. HNSW finds near neighbours without checking them all.

**Why content-hash IDs?** Chroma needs an ID for every item. If we used a counter (1, 2, 3…), adding the same documents again would create items 8–14: duplicates, and every search would return each chunk twice. With an ID built from the content, the same chunk always gets the same ID, and `upsert` simply replaces it. The plan's second check, "re-ingesting doesn't duplicate", depends on this.

**How it connects to the architecture.** Phase 4 completes the indexing flow and starts the query flow:

1. **Load → Chunk → Embed** (Phases 1–3): files → chunks → vectors. Reused unchanged.
2. **Store** (Phase 4): chunks + vectors + metadata go into Chroma under content-hash IDs.
3. **Search** (Phase 4): the question is embedded with the Phase 3 query prefix, and the top 3 chunks come back with scores.
4. **Next:** Phase 5 measures *how often* the right chunk is in the top k (hit@k), and Phase 6 sends the top chunks to the LLM.

**Design choices for this phase**

| Question | Our choice | Why |
| --- | --- | --- |
| Who embeds the text, us or Chroma? | Us (Phase 3's model). Chroma gets ready-made vectors with `embedding_function=None` | Chroma's default would silently download a *different* model (all-MiniLM). One model, under our control, keeps results comparable |
| Distance | Cosine | Matches the dot-product scores from Phase 3 exactly |
| ID recipe | `sha256(source \| page \| start_char \| text)`, first 16 characters | Same chunk → same ID. Includes position, so two identical sentences on one page don't merge |
| Default top k | 3 | Enough to include the right chunk even when it isn't #1 (remember the 0.043 gap for "vacation days" in Phase 3), small enough for an 8 GB LLM prompt later |
| Where data lives | `PROJECT_ROOT / "storage" / "chroma"` | An absolute path, as in Phase 1. `storage/` is already in `.gitignore` (Phase 0), so the database never goes into Git |
| Telemetry | Off: `anonymized_telemetry=False` | No data leaves your machine |

## 4.2 Setup steps

Run every command from your project folder (`/Users/nowgg/Automation/rag_framework`). Phases 1–3 must work first: `uv run python -m ragqa.embeddings` should print `Array shape: (7, 384)`.

### Step 1 — Install Chroma

```bash
uv add chromadb
```

- **What it does:** downloads Chroma and its helpers, adds `chromadb` to `pyproject.toml`, and pins exact versions in `uv.lock`.
- **Why:** Chroma is the vector database this phase stores chunks in. No `--dev`, because the app itself needs it.
- **What to expect:** on my test run it installed `chromadb==1.5.9` plus **36 packages in 22 seconds**, and `.venv` grew from 855 MB to **1.1 GB**.

Some of those 36 packages look surprising for a small local database:

| Package | Why Chroma brings it | Do we use it? |
| --- | --- | --- |
| `onnxruntime` | Runs Chroma's *default* embedding model | No: we pass `embedding_function=None` and give our own vectors |
| `uvicorn`, `uvloop`, websockets | Chroma can also run as a server | No: we use it embedded inside Python |
| `kubernetes`, `opentelemetry-*` | Cloud deployment and monitoring features | No |
| `pydantic` | Checks Chroma's settings | Yes, indirectly |

Unused packages only cost disk space; they aren't loaded unless used. Chroma is licensed **Apache 2.0** (free, permissive).

After this step `pyproject.toml` lists:

```toml
dependencies = [
    "chromadb>=1.5.9",
    "numpy>=2.5.4",
    "pymupdf>=1.28.2",
    "requests>=2.34.2",
    "sentence-transformers>=6.1.0",
]
```

### Step 2 — Check that `storage/` stays out of Git

```bash
grep storage .gitignore
```

- **What it does:** searches `.gitignore` for the word `storage`. `grep` prints every line that contains the word.
- **Why:** Chroma writes its database to `storage/chroma/`. A database is generated data, so it must never be committed. Phase 0, Step 7 already added `storage/` to `.gitignore`.
- **Expected output:** `storage/`. If nothing prints, add a line `storage/` to `.gitignore`.

Nothing else to set up. Chroma creates `storage/chroma/` by itself the first time the code in section 4.3 runs: on my run it held one file, `chroma.sqlite3`, plus one folder for the vector index, **496 KB** in total for our 7 chunks.

## 4.3 Code: numpy search, Chroma store, and a comparison script

Three small files, split by what they import (the same lesson as Phase 3: keep slow imports away from fast tests):

| File | Job | Imports | Import time on your Mac |
| --- | --- | --- | --- |
| `retrieval.py` | Brute-force search with numpy, and the `SearchHit` record | numpy only | Instant |
| `store.py` | Save chunks in Chroma and search them there | `chromadb` | About 5 s |
| `search.py` | Ingest the documents, then answer one question both ways | Everything, including PyTorch | About 15 s |

### `retrieval.py`: brute-force search

Create `src/ragqa/retrieval.py`:

```python
"""Phase 4: find the chunks most similar to a question with plain numpy (brute force)."""

from dataclasses import dataclass

import numpy as np

from ragqa.chunking import Chunk
from ragqa.similarity import cosine_scores

DEFAULT_TOP_K = 3


# One search result: which chunk, how well it matched, and where it came from.
@dataclass
class SearchHit:
    rank: int
    chunk_id: str
    source: str
    page: int
    score: float
    text: str


# Score every chunk against the question and return the top_k best, best first.
def search_chunks(
    query_vector: np.ndarray,
    chunks: list[Chunk],
    chunk_vectors: np.ndarray,
    top_k: int,
) -> list[SearchHit]:
    if top_k <= 0:
        raise ValueError(f"top_k must be above 0, got {top_k}")

    scores = cosine_scores(query_vector, chunk_vectors)
    positions_best_first = np.argsort(-scores, kind="stable")

    hits = []
    for rank, position in enumerate(positions_best_first[:top_k], start=1):
        chunk = chunks[position]
        hit = SearchHit(
            rank=rank,
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            page=chunk.page,
            score=float(scores[position]),
            text=chunk.text,
        )
        hits.append(hit)
    return hits
```

**What the code does, part by part**

- **The imports** reuse Phase 2's `Chunk` and Phase 3's `cosine_scores`. No PyTorch, no Chroma, so this file loads instantly.
- **`DEFAULT_TOP_K = 3`**: how many results we return by default (design table, 4.1).
- **`class SearchHit`**: one result. `rank` is its place (1 = best), `score` is the cosine similarity, and `source`, `page` and `text` travel with it so later phases can cite it. **Both** search methods return this same record, which is what makes them easy to compare.
- **`search_chunks(query_vector, chunks, chunk_vectors, top_k)`**: input is the question vector, the chunks, their vectors (same order as the chunks: row 0 belongs to chunk 0) and k. Output: a list of up to k `SearchHit`s, best first.
  - The `if top_k <= 0` check rejects a search that could never return anything.
  - `cosine_scores(...)` gives one score per chunk (Phase 3).
  - `np.argsort(-scores, kind="stable")` returns chunk *positions* from best to worst (the minus sign flips "smallest first" into "largest first", as in the Phase 3 exercise). `kind="stable"` means that if two chunks score exactly the same, the earlier one stays first, so results never flip randomly between runs.
  - `positions_best_first[:top_k]` keeps only the first k positions. If k is bigger than the number of chunks, Python simply returns all of them; no error.
  - The loop builds one `SearchHit` per position. `float(...)` turns numpy's number type into a plain Python number.

### `store.py`: Chroma

Create `src/ragqa/store.py`:

```python
"""Phase 4: keep chunks and their vectors in Chroma on disk, and search them there."""

import hashlib
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

from ragqa.chunking import Chunk
from ragqa.retrieval import SearchHit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE_FOLDER = PROJECT_ROOT / "storage" / "chroma"
COLLECTION_NAME = "chunks"


# Build an ID from the chunk's content, so the same chunk always gets the same ID.
def make_content_id(chunk: Chunk) -> str:
    content = f"{chunk.source}|{chunk.page}|{chunk.start_char}|{chunk.text}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


# Open the chunk collection in a folder on disk, creating it the first time.
def open_collection(store_folder: Path) -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=str(store_folder),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )
    return collection


# Save chunks with their vectors; chunks already stored are replaced, never duplicated.
def add_chunks(collection: chromadb.Collection, chunks: list[Chunk], chunk_vectors: np.ndarray) -> None:
    ids = []
    documents = []
    metadatas = []
    for chunk in chunks:
        ids.append(make_content_id(chunk))
        documents.append(chunk.text)
        metadatas.append({"chunk_id": chunk.chunk_id, "source": chunk.source, "page": chunk.page})
    collection.upsert(ids=ids, embeddings=chunk_vectors.tolist(), documents=documents, metadatas=metadatas)


# Ask Chroma for the top_k stored chunks closest to the question, best first.
def search_store(collection: chromadb.Collection, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
    result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    found_metadatas = result["metadatas"][0]
    found_documents = result["documents"][0]
    found_distances = result["distances"][0]

    hits = []
    for index in range(len(found_metadatas)):
        metadata = found_metadatas[index]
        hit = SearchHit(
            rank=index + 1,
            chunk_id=metadata["chunk_id"],
            source=metadata["source"],
            page=metadata["page"],
            score=1.0 - found_distances[index],
            text=found_documents[index],
        )
        hits.append(hit)
    return hits
```

**What the code does, part by part**

- **The imports.** `hashlib` (Python's standard library) makes the hash; `chromadb` and its `Settings` are the database; `SearchHit` is reused from `retrieval.py`.
- **The CAPITAL settings.** `DEFAULT_STORE_FOLDER` is an absolute path built from `PROJECT_ROOT`, exactly like Phase 1's data folder. `COLLECTION_NAME` names our one collection.
- **`make_content_id(chunk)`**: input a chunk, output a 16-character ID.
  - The f-string glues source, page, start position and text into one string, with `|` between the parts so `"a" + "bc"` can't look like `"ab" + "c"`.
  - `.encode("utf-8")` turns text into bytes, which is what a hash function eats.
  - `hashlib.sha256(...).hexdigest()` gives a 64-character code made of `0-9` and `a-f`. `[:16]` keeps the first 16, still unique enough for millions of chunks.
- **`open_collection(store_folder)`**: input a folder, output the collection, ready to use.
  - `chromadb.PersistentClient(path=...)` opens (or creates) a database saved in that folder. "Persistent" means it survives after the program ends. Chroma wants the path as text, hence `str(...)`.
  - `Settings(anonymized_telemetry=False)` switches off usage reporting.
  - `get_or_create_collection(...)` opens the `chunks` collection, or creates it the first time. `embedding_function=None` tells Chroma "never embed text yourself, I give you vectors". `configuration={"hnsw": {"space": "cosine"}}` makes it measure cosine distance.
- **`add_chunks(collection, chunks, chunk_vectors)`**: no output; it writes to disk.
  - The loop builds three lists in the same order: IDs, texts, and metadata dictionaries. We keep our readable `chunk_id` (like `brightwater_handbook.md:p1:c1`) as metadata, so results are easy to read and match against numpy.
  - `chunk_vectors.tolist()` turns the numpy array into plain Python lists, which is what Chroma accepts.
  - `collection.upsert(...)` adds every item, or replaces it if the ID already exists. This one word is what makes re-ingesting safe.
- **`search_store(collection, query_vector, top_k)`**: same inputs and output as `search_chunks`, but Chroma does the work.
  - `collection.query(...)` takes a *list* of question vectors (it can search for several at once), so we wrap ours in `[ ]`. `include=` lists what we want back.
  - The result holds one inner list per question. We asked one question, so `[0]` picks its results.
  - `score=1.0 - found_distances[index]` turns cosine distance back into cosine similarity, so both methods report the same kind of number.

### `search.py`: ingest, then search both ways

Create `src/ragqa/search.py`:

```python
"""Phase 4: build the Chroma index, then answer one question with numpy and with Chroma."""

import sys

from ragqa.chunking import DEFAULT_CHUNK_SIZE
from ragqa.chunking import DEFAULT_OVERLAP
from ragqa.chunking import chunk_pages
from ragqa.embeddings import embed_query
from ragqa.embeddings import embed_texts
from ragqa.embeddings import load_embedding_model
from ragqa.loaders import DEFAULT_DATA_FOLDER
from ragqa.loaders import load_folder
from ragqa.retrieval import DEFAULT_TOP_K
from ragqa.retrieval import SearchHit
from ragqa.retrieval import search_chunks
from ragqa.store import DEFAULT_STORE_FOLDER
from ragqa.store import add_chunks
from ragqa.store import open_collection
from ragqa.store import search_store

PREVIEW_LENGTH = 70


# Print search hits as a small numbered list with a one-line preview.
def print_hits(title: str, hits: list[SearchHit]) -> None:
    print(title)
    for hit in hits:
        preview = " ".join(hit.text.split())[:PREVIEW_LENGTH]
        print(f"  {hit.rank}. {hit.score:.4f}  {hit.chunk_id}  {preview!r}")


# Index the documents in Chroma and compare numpy and Chroma results for one question.
def main() -> None:
    if len(sys.argv) != 2:
        print('Usage: uv run python -m ragqa.search "your question"')
        sys.exit(1)
    question = sys.argv[1]

    model = load_embedding_model()
    pages, skipped_files = load_folder(DEFAULT_DATA_FOLDER)
    chunks = chunk_pages(pages, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP)
    chunk_texts = []
    for chunk in chunks:
        chunk_texts.append(chunk.text)
    chunk_vectors = embed_texts(model, chunk_texts)

    collection = open_collection(DEFAULT_STORE_FOLDER)
    count_before = collection.count()
    add_chunks(collection, chunks, chunk_vectors)
    count_after = collection.count()
    print(f"Chroma folder: {DEFAULT_STORE_FOLDER}")
    print(f"Stored chunks: {count_before} before, {count_after} after adding {len(chunks)}")

    query_vector = embed_query(model, question)
    numpy_hits = search_chunks(query_vector, chunks, chunk_vectors, DEFAULT_TOP_K)
    chroma_hits = search_store(collection, query_vector, DEFAULT_TOP_K)

    print(f"Question: {question}")
    print_hits(f"numpy top {DEFAULT_TOP_K}:", numpy_hits)
    print_hits(f"Chroma top {DEFAULT_TOP_K}:", chroma_hits)

    same_order = True
    for index in range(len(numpy_hits)):
        if numpy_hits[index].chunk_id != chroma_hits[index].chunk_id:
            same_order = False
    print(f"Same chunks in the same order: {same_order}")

    for reason in skipped_files:
        print(f"SKIPPED {reason}")


if __name__ == "__main__":
    main()
```

**What the code does, part by part**

- **The imports** pull in every phase so far: loader (1), chunker (2), embedder (3), and both search methods (4). Nothing is copied.
- **`print_hits(title, hits)`** prints one line per hit: rank, score with 4 decimals (`:.4f`), chunk ID, and the first 70 characters squashed onto one line (the same trick as Phase 2's `flatten`).
- **`main()`**, step by step:
  1. **Read the question** from the command line (`sys.argv`, as in the Phase 3 exercise). Anything other than exactly one quoted question prints the usage line and stops.
  2. **Ingest:** load → chunk → embed, exactly as Phase 3.
  3. **Store:** open the collection, count what's already there, add the chunks, count again. On the first run you'll see `0 before, 7 after`; on every later run `7 before, 7 after`. That second line is the plan's "no duplicates" check, visible.
  4. **Search both ways** with the same question vector, then print both lists.
  5. **Compare:** walk both lists side by side; if any position holds a different chunk, `same_order` becomes `False`. The plan's "both give the same top-k" check, visible.

## 4.4 Tests

Ten tests in three files. Together they cover both plan checks for this phase: **numpy and Chroma give the same top k**, and **re-ingesting doesn't duplicate**.

| File | Tests | Kind | Why that kind | Time on my run |
| --- | --- | --- | --- | --- |
| `tests/unit/test_retrieval.py` | 5 | Unit | Only numpy and tiny hand-made vectors | Part of the 0.97 s unit run |
| `tests/integration/test_store.py` | 4 | Integration | Uses a real Chroma database on disk (5 s import). This is the plan's L1 layer: "temp Chroma" | A few seconds |
| `tests/integration/test_search_agreement.py` | 1 | Integration | Needs the real embedding model | About 15 s |

The unit tests and the Chroma tests reuse the 2-D vectors from Phase 3's worked example, in a new order, so you can check the expected answer by hand: chunk `b` = `[0.6, 0.8]` (score 1.00), `c` = `[0.8, 0.6]` (0.96), `a` = `[0.8, -0.6]` (0.00).

The Chroma tests use pytest's `tmp_path` fixture (Phase 1) as the database folder, so every test starts with an empty database and your real `storage/` is never touched.

Create `tests/unit/test_retrieval.py`:

```python
import numpy as np
import pytest

from ragqa.chunking import Chunk
from ragqa.retrieval import search_chunks


# Build a test chunk in memory with a given ID and text.
def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, source="policy.pdf", page=2, chunk_index=0, start_char=0, end_char=len(text), text=text)


CHUNKS = [make_chunk("a", "text a"), make_chunk("b", "text b"), make_chunk("c", "text c")]
VECTORS = np.array([[0.8, -0.6], [0.6, 0.8], [0.8, 0.6]])
QUESTION = np.array([0.6, 0.8])


# The best-scoring chunk comes first, then the rest in score order.
def test_hits_come_back_best_first():
    hits = search_chunks(QUESTION, CHUNKS, VECTORS, top_k=3)

    hit_ids = []
    for hit in hits:
        hit_ids.append(hit.chunk_id)
    assert hit_ids == ["b", "c", "a"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(0.96)


# Only top_k hits are returned, ranked from 1.
def test_top_k_limits_the_number_of_hits():
    hits = search_chunks(QUESTION, CHUNKS, VECTORS, top_k=2)

    assert len(hits) == 2
    assert hits[0].rank == 1
    assert hits[1].rank == 2


# Asking for more hits than chunks returns every chunk, not an error.
def test_top_k_larger_than_chunk_count_returns_all():
    hits = search_chunks(QUESTION, CHUNKS, VECTORS, top_k=10)

    assert len(hits) == 3


# Each hit carries the chunk's source, page and text.
def test_hits_copy_chunk_metadata():
    hits = search_chunks(QUESTION, CHUNKS, VECTORS, top_k=1)

    assert hits[0].source == "policy.pdf"
    assert hits[0].page == 2
    assert hits[0].text == "text b"


# A top_k of 0 is rejected.
def test_top_k_zero_raises():
    with pytest.raises(ValueError, match="top_k must be above 0"):
        search_chunks(QUESTION, CHUNKS, VECTORS, top_k=0)
```

Create `tests/integration/test_store.py`:

```python
import numpy as np
import pytest

from ragqa import store
from ragqa.chunking import Chunk


# Build a test chunk in memory with a given ID and text.
def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, source="policy.pdf", page=2, chunk_index=0, start_char=0, end_char=len(text), text=text)


CHUNKS = [make_chunk("a", "text a"), make_chunk("b", "text b"), make_chunk("c", "text c")]
VECTORS = np.array([[0.8, -0.6], [0.6, 0.8], [0.8, 0.6]])
QUESTION = np.array([0.6, 0.8])


# The same chunk always gets the same ID; a changed text gets a new one.
@pytest.mark.integration
def test_content_id_is_stable_and_follows_the_text():
    first_id = store.make_content_id(make_chunk("a", "Annual leave is 27 days."))
    second_id = store.make_content_id(make_chunk("a", "Annual leave is 27 days."))
    changed_id = store.make_content_id(make_chunk("a", "Annual leave is 28 days."))

    assert first_id == second_id
    assert first_id != changed_id
    assert len(first_id) == 16


# Adding the same chunks twice keeps the count the same.
@pytest.mark.integration
def test_adding_twice_does_not_duplicate(tmp_path):
    collection = store.open_collection(tmp_path)

    store.add_chunks(collection, CHUNKS, VECTORS)
    store.add_chunks(collection, CHUNKS, VECTORS)

    assert collection.count() == 3


# Chroma returns the closest chunks first, with cosine similarity as the score.
@pytest.mark.integration
def test_store_search_returns_best_first(tmp_path):
    collection = store.open_collection(tmp_path)
    store.add_chunks(collection, CHUNKS, VECTORS)

    hits = store.search_store(collection, QUESTION, top_k=3)

    hit_ids = []
    for hit in hits:
        hit_ids.append(hit.chunk_id)
    assert hit_ids == ["b", "c", "a"]
    assert hits[1].score == pytest.approx(0.96, abs=1e-4)


# Data written to disk is still there when the folder is opened again.
@pytest.mark.integration
def test_stored_chunks_survive_reopening(tmp_path):
    first_collection = store.open_collection(tmp_path)
    store.add_chunks(first_collection, CHUNKS, VECTORS)

    reopened_collection = store.open_collection(tmp_path)

    assert reopened_collection.count() == 3
```

Create `tests/integration/test_search_agreement.py`:

```python
import pytest

from ragqa import embeddings
from ragqa import store
from ragqa.chunking import chunk_pages
from ragqa.loaders import DEFAULT_DATA_FOLDER
from ragqa.loaders import load_folder
from ragqa.retrieval import search_chunks

QUESTIONS = [
    "How many vacation days do I get?",
    "How much can I spend on a hotel in Paris?",
    "Who do I call when IT is closed?",
    "What happens if I lose my laptop?",
    "Can I fly business class?",
]


# numpy and Chroma return the same top 3 chunks, in the same order, for real questions.
@pytest.mark.integration
def test_numpy_and_chroma_agree_on_real_questions(tmp_path):
    model = embeddings.load_embedding_model()
    pages, skipped_files = load_folder(DEFAULT_DATA_FOLDER)
    chunks = chunk_pages(pages, 500, 100)
    chunk_texts = []
    for chunk in chunks:
        chunk_texts.append(chunk.text)
    chunk_vectors = embeddings.embed_texts(model, chunk_texts)
    collection = store.open_collection(tmp_path)
    store.add_chunks(collection, chunks, chunk_vectors)

    for question in QUESTIONS:
        query_vector = embeddings.embed_query(model, question)
        numpy_hits = search_chunks(query_vector, chunks, chunk_vectors, top_k=3)
        chroma_hits = store.search_store(collection, query_vector, top_k=3)
        for index in range(3):
            assert numpy_hits[index].chunk_id == chroma_hits[index].chunk_id, question
            assert numpy_hits[index].score == pytest.approx(chroma_hits[index].score, abs=1e-4)
```

**What the tests check, and why each matters**

| Test | What it proves | What would break later without it |
| --- | --- | --- |
| `hits_come_back_best_first` | Order is `b, c, a` with scores 1.00 and 0.96, matching the hand calculation | Every retrieval result and every hit@k number in Phase 5 |
| `top_k_limits_the_number_of_hits` | Exactly k hits, ranked 1, 2… | Prompts in Phase 6 could get too many chunks |
| `top_k_larger_than_chunk_count_returns_all` | Asking for 10 of 3 returns 3, no crash | Small test corpora would crash the search |
| `hits_copy_chunk_metadata` | Source, page and text travel with each hit | Citations (Phase 7) would have nothing to point to |
| `top_k_zero_raises` | `top_k=0` is refused | A silent empty result that looks like "nothing relevant" |
| `content_id_is_stable_and_follows_the_text` | Same content → same ID; one changed character → new ID | Re-ingesting would duplicate; edited chunks would never update |
| `adding_twice_does_not_duplicate` | 3 chunks added twice → count stays 3 | The plan's "no duplicates" check |
| `store_search_returns_best_first` | Chroma's order and score (0.96, within 0.0001) match the numpy maths | Chroma's distance could be misread as a score |
| `stored_chunks_survive_reopening` | Data is still there after reopening the folder | Every later run would have to re-ingest |
| `numpy_and_chroma_agree_on_real_questions` | For 5 real questions, both methods return the same top 3, scores within 0.0001 | The plan's "same top-k" check, on real data |

A few lines explained:

- `CHUNKS`, `VECTORS` and `QUESTION` are written in CAPITALS at the top of the file because every test shares them. They're test data, read-only.
- `assert ... == chroma_hits[index].chunk_id, question`: the text after the comma is shown only if the assert fails, so you see *which* question disagreed.
- `abs=1e-4` allows a difference of 0.0001. Chroma does its own maths internally, so its scores can differ from numpy's by a tiny amount. On my run the largest difference across all 5 questions was 0.0000008.

### QA role for this phase

**How important is testing here: high.** Most RAG failures start at retrieval: if the evidence chunk is not in the top k, the LLM cannot answer correctly, however good it is. Phase 4 tests make the retriever *correct and repeatable*. Phase 5 then measures how *good* it is.

A useful QA idea here: the numpy brute-force search is a **test oracle**, a simple trusted reference whose answers you compare the real system (Chroma) against. If both agree, Chroma is doing its job.

**What to test**

| What can go wrong | How to check it | Manual or automated | Covered by |
| --- | --- | --- | --- |
| Results come back in the wrong order | Best score first; matches the hand calculation | Automated | `hits_come_back_best_first` |
| Too many or too few results | Exactly k hits, ranked 1, 2… | Automated | `top_k_limits_the_number_of_hits` |
| Small corpus crashes the search | Asking for 10 of 3 returns 3 | Automated | `top_k_larger_than_chunk_count_returns_all` |
| Hits lose their source and page | Metadata travels with every hit | Automated | `hits_copy_chunk_metadata` |
| `top_k=0` looks like "nothing relevant" | It is refused with an error | Automated | `top_k_zero_raises` |
| Re-ingesting duplicates chunks | Same content gives the same ID; count stays the same | Automated | `content_id_is_stable_and_follows_the_text`, `adding_twice_does_not_duplicate` |
| Chroma's distance is misread as a score | Chroma's order and score match numpy | Automated (integration) | `store_search_returns_best_first` |
| Data is lost between runs | Chunks are still there after reopening | Automated (integration) | `stored_chunks_survive_reopening` |
| Chroma and the oracle disagree on real data | Same top 3 for 5 real questions, scores within 0.0001 | Automated (integration) | `numpy_and_chroma_agree_on_real_questions` |
| Results are correct but not *relevant* | Read the top 3 for real questions | Manual | Run it (4.5), Exercise Part A |
| An edited document leaves old chunks behind | Edit a file, re-ingest, check the results | Manual, then automated | Exercise Parts B and C |

**Manual testing: how**

1. Run the search (4.5) for 5 questions whose answers you know.
2. For each, write down the rank of the right chunk and its score, and the score of the best wrong chunk. A small gap between them means the result is fragile.
3. Ask the same question in other words. Does the right chunk keep its rank?
4. Run the ingest twice and confirm the stored chunk count does not change.
5. Edit one sentence in a document and re-ingest (Exercise Part B). Check whether the old version still appears in the results.

**Automated testing: how**

- `tests/unit/test_retrieval.py`: fast unit tests with tiny hand-made vectors, so the right answer is known exactly.
- `tests/integration/test_store.py`: tests Chroma in a temporary folder (`tmp_path`), so your real `storage/` is never touched.
- `tests/integration/test_search_agreement.py`: the oracle test on real questions.
- Run them with the commands in 4.5: unit tests on every change, integration tests before each commit.

**Not covered yet:** these tests prove the search is correct, not that it finds the *right evidence*. Relevance is measured with a golden set and hit@k / MRR in Phase 5.

**How a real system tests this phase, and how we test it now**

In a real system, retrieval is a *search API*: `POST /search` takes a question and returns the top k hits with their text, source, page and score. This is the door QA uses most, because retrieval is where most RAG failures start.

| What is checked | In a real system | In our project now |
| --- | --- | --- |
| Right chunk is found | Top k hits for a golden set of questions, scored with hit@k and MRR | Call `search_chunks`; scored in Phase 5 |
| The index is exact enough | Vector databases use *approximate* search: fast, but it can miss a result. QA compares it with exact brute-force search | The numpy brute-force search is the exact reference (an *oracle*). Same idea |
| Filters | Search only one source, date range or department | No filters |
| Access rules | User A never gets chunks from documents only user B may read | One user |
| Fresh data | A new document is searchable within a set time. A deleted document disappears. An edited document replaces the old text | Content-hash IDs and re-ingest tests |
| Speed | p95 time with millions of chunks | 35 chunks |
| Odd input | Empty text, very long text, other languages, special characters give a clear reply | `top_k_zero_raises` |

**Example: the same checks through a search API** (a sketch with made-up endpoints, not run):

```python
import requests

API_URL = "http://localhost:8000"


# Search the knowledge base as one user and return the hits.
def search(question: str, user_id: str, top_k: int = 3) -> list[dict]:
    request_body = {"query": question, "top_k": top_k, "user_id": user_id}
    response = requests.post(f"{API_URL}/search", json=request_body, timeout=30)
    response.raise_for_status()
    return response.json()["hits"]


# A known question finds its evidence in the top 3, best score first.
def test_known_question_finds_the_evidence():
    hits = search("How many days do I have to submit travel receipts?", "alice")

    assert len(hits) == 3
    evidence_found = False
    for hit in hits:
        if "within 30 days" in hit["text"]:
            evidence_found = True
    assert evidence_found
    assert hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"]


# A user must never get chunks from documents they may not read.
def test_user_cannot_see_another_users_documents():
    hits = search("What is the salary table?", "bob")

    for hit in hits:
        assert hit["source"] != "hr_salaries_confidential.pdf"
```

- **Test 1** is hit@3 for one question. A real team runs the same check for every question in the golden set (Phase 5) and reports one number. It also checks the order of the scores, as our `hits_come_back_best_first` test does.
- **Test 2** is a *security* check. Our project can't have it, because there are no users. In a real company it is one of the most important tests: a wrong answer is bad, but leaking a document is much worse.
- Both tests only use what the API returns. QA does not touch the vector database.

**The gap, in one line:** our numpy-vs-Chroma test is already what real teams do (compare the fast index with an exact reference). What a real system adds is users, filters, fresh-data rules and speed at large size.

## 4.5 Run it

Run these from your project folder. Ollama doesn't need to be running.

```bash
uv run python -m ragqa.search "How many vacation days do I get?"
```

- **What it does:** ingests `data/raw/` into Chroma (load, chunk, embed, store), then answers the question with numpy and with Chroma and compares the two top-3 lists.
- **The parts:** `uv run` uses the project's environment; `python -m ragqa.search` runs `search.py`; the quoted text becomes `sys.argv[1]`, the question. The quotes keep it as one item.
- **Why:** this shows both plan checks at once: the same top k, and no duplicates (run it twice).

```bash
uv run pytest tests/unit -v
```

- **What it does:** the quick check, now including the 5 new numpy search tests. It never loads Chroma or PyTorch.

```bash
uv run pytest tests/integration/test_store.py tests/integration/test_search_agreement.py -v
```

- **What it does:** runs the 5 new integration tests: 4 with Chroma, 1 with the real model and real documents.
- **The parts:** two file paths, so only those two files run.

**Expected output, first run** (real output from my test run, starting with no `storage/` folder):

```text
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Loading weights: 100%|██████████| 199/199 [00:00<00:00, 5300.45it/s]
Chroma folder: /Users/nowgg/Automation/rag_framework/storage/chroma
Stored chunks: 0 before, 7 after adding 7
Question: How many vacation days do I get?
numpy top 3:
  1. 0.6553  brightwater_handbook.md:p1:c1  'rking days per calendar year and needs approval from People Operations'
  2. 0.6118  brightwater_travel_policy.pdf:p2:c0  '2. Booking travel Trains are the default for trips under 4 hours. Flig'
  3. 0.5977  brightwater_travel_policy.pdf:p4:c0  '4. Daily allowance and claims The daily meal allowance is 45 euros. Al'
Chroma top 3:
  1. 0.6553  brightwater_handbook.md:p1:c1  'rking days per calendar year and needs approval from People Operations'
  2. 0.6118  brightwater_travel_policy.pdf:p2:c0  '2. Booking travel Trains are the default for trips under 4 hours. Flig'
  3. 0.5977  brightwater_travel_policy.pdf:p4:c0  '4. Daily allowance and claims The daily meal allowance is 45 euros. Al'
Same chunks in the same order: True
```

(My test copy lives in a different folder; on your Mac the `Chroma folder:` line shows your project path, as above.)

**Expected output, second run** (only the lines that change):

```text
Stored chunks: 7 before, 7 after adding 7
Same chunks in the same order: True
```

What to notice:

| What you see | What it means |
| --- | --- |
| `0 before, 7 after` on run 1 | Chroma created the database and stored all 7 chunks |
| `7 before, 7 after` on run 2 | **No duplicates.** The same 7 content-hash IDs were upserted, so they replaced themselves |
| The two top-3 lists are identical, to 4 decimals | numpy (exact, our reference) and Chroma agree. Across 5 questions the biggest difference I measured was 0.0000008 |
| #1 is handbook `c1` (0.6553), #2 a travel page (0.6118) | The right chunk wins, by a small margin: the same 0.043 gap you met in Phase 3's exercise. With `top_k = 3`, a slightly worse ranking would still include the right chunk; Phase 5 measures exactly that |
| The preview of `c1` starts mid-word (`rking days...`) | The Phase 2 chunk boundary; the leave rules come later in this chunk |
| The whole command took about 23 s | Almost all of it is importing PyTorch (13 s) and Chroma (5 s), plus the online model check. Searching itself takes milliseconds |

**Expected output of the new tests** (real):

```text
tests/unit/test_retrieval.py::test_hits_come_back_best_first PASSED      [ 10%]
tests/unit/test_retrieval.py::test_top_k_limits_the_number_of_hits PASSED [ 20%]
tests/unit/test_retrieval.py::test_top_k_larger_than_chunk_count_returns_all PASSED [ 30%]
tests/unit/test_retrieval.py::test_hits_copy_chunk_metadata PASSED       [ 40%]
tests/unit/test_retrieval.py::test_top_k_zero_raises PASSED              [ 50%]
tests/integration/test_store.py::test_content_id_is_stable_and_follows_the_text PASSED [ 60%]
tests/integration/test_store.py::test_adding_twice_does_not_duplicate PASSED [ 70%]
tests/integration/test_store.py::test_store_search_returns_best_first PASSED [ 80%]
tests/integration/test_store.py::test_stored_chunks_survive_reopening PASSED [ 90%]
tests/integration/test_search_agreement.py::test_numpy_and_chroma_agree_on_real_questions PASSED [100%]
============================= 10 passed in 19.59s ==============================
```

On my test copy, `uv run pytest tests/unit -q` gave `24 passed in 0.97s`, and the full `uv run pytest -q` gave `35 passed in 16.02s`. Your totals depend on which earlier exercise tests you've written; the full run also includes Phase 0's Ollama test, so start Ollama first.

## 4.6 Common errors and debugging

The messages below are real outputs from my test runs (the last line of each traceback).

| What you see | What it means | How to fix it |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'chromadb'` | Chroma isn't installed in the project's environment | Run `uv add chromadb` (Step 1); always start commands with `uv run` |
| `InvalidArgumentError: Collection expecting embedding with dimension of 384, got 768` *(numbers depend on your case; I got `dimension of 2, got 3` in a small test)* | Vectors of a different length than the ones already stored: usually you switched to another embedding model but kept the old database | Delete `storage/chroma` and ingest again. One collection = one embedding model |
| `Same chunks in the same order: False`, and the count grew (e.g. `7 before, 8 after adding 7`) | **Stale chunks.** A document was edited, so the changed chunk got a new ID, but the old version is still stored. On my run the outdated "27 days" chunk became Chroma's #1 result | Delete `storage/chroma` and ingest again (exercise, Part B). For now, rebuilding is the fix; removing a document's old chunks automatically can come later, if evaluation shows it's needed |
| Scores look odd: negative, above 1, or very different from numpy, but no error | The collection was created earlier with Chroma's default `l2` distance. Asking for cosine later is **silently ignored** for an existing collection (I checked: it stays `l2`) | Check the distance with the debugging command below; if it isn't `cosine`, delete `storage/chroma` and ingest again |
| `ValueError: You must provide an embedding function to compute embeddings` | Code called `collection.query(query_texts=[...])`, asking Chroma to embed text, but we told it not to (`embedding_function=None`) | Embed the question yourself with `embed_query` and pass `query_embeddings=` (as `search_store` does) |
| `TypeError: Number of requested results 0, cannot be negative, or zero.` | `search_store` was called with `top_k=0` | Use a top k of 1 or more |
| Fewer hits than `top_k`, no error | You asked for more results than there are stored chunks (I asked for 10 of 2 and got 2) | Normal; both search methods behave this way |
| `Usage: uv run python -m ragqa.search "your question"` | No question, or a question without quotes (each word becomes a separate item) | Put the whole question in double quotes |
| Every run takes 15–25 s | Importing PyTorch (13 s) and Chroma (5 s), plus the online model check | Normal. Add `HF_HUB_OFFLINE=1` in front to skip the online check (Phase 3, Step 4) |

**Debugging techniques for this phase**

```bash
uv run python -c "from ragqa.store import DEFAULT_STORE_FOLDER, open_collection; print(open_collection(DEFAULT_STORE_FOLDER).count())"
```

- **What it does:** prints how many chunks are stored in your Chroma database. Real output on a clean database: `7`.
- **Why:** the fastest check for duplicates or stale chunks. More than your chunk count (7) means something old is still in there.

```bash
uv run python -c "from ragqa.store import DEFAULT_STORE_FOLDER, open_collection; stored = open_collection(DEFAULT_STORE_FOLDER).get(include=['metadatas']); print(sorted(m['chunk_id'] for m in stored['metadatas']))"
```

- **What it does:** lists the readable chunk ID of every stored item, sorted.
- **Why:** after an edit, a stale chunk shows up as the **same chunk ID twice** (e.g. `brightwater_handbook.md:p1:c1` two times). Real output on a clean database: the 7 IDs from Phase 2, each once.
- **The parts:** `.get(include=['metadatas'])` reads stored items without searching; `sorted(... for m in ...)` is a one-line loop that collects and sorts the IDs. Fine for a quick check; in project code we write the loop out.

```bash
uv run python -c "from ragqa.store import DEFAULT_STORE_FOLDER, open_collection; print(open_collection(DEFAULT_STORE_FOLDER).configuration['hnsw']['space'])"
```

- **What it does:** prints which distance the stored collection really uses. It must print `cosine` (it did on my run).
- **Why:** catches the silent `l2` trap from the table above.

```bash
rm -r storage/chroma
```

- **What it does:** deletes the Chroma database folder. The next `ragqa.search` run rebuilds it from `data/raw/` in a few seconds.
- **The parts:** `rm` removes files; `-r` ("recursive") includes the folder and everything inside it. Run it from your project folder, and double-check the path: `rm -r` doesn't use the Trash.
- **Why:** the database is generated data, never the only copy of anything. Rebuilding is the simple, safe fix for duplicates, stale chunks and wrong settings.

## 4.7 Exercise: search, break the index, write a test

**Part A — Search like a tester.** Run `ragqa.search` with each question below and fill in the table. My real #1 results are given so you can compare:

```bash
uv run python -m ragqa.search "How long do I have to submit receipts?"
```

- **What it does:** the same search as in 4.5, with a different question. Change the text in quotes for each row.

| Question | Is the answer in the documents? | My #1 (score) | My #2 (score) | Your notes |
| --- | --- | --- | --- | --- |
| How long do I have to submit receipts? | Yes: policy page 4 | policy `p4:c0` (0.636) | handbook `c1` (0.583) |  |
| Can I work from another country? | Yes: the whole sentence is in handbook `c0`; its second half is repeated in `c1` | handbook `c1` (0.541) | policy `p1:c0` (0.528) |  |
| What is the parking policy? | **No** | policy `p1:c0` (0.573) | handbook `c1` (0.531) |  |

Answer these in a few words each:

1. The parking question has no answer anywhere, yet search still returned a #1 with a score of 0.573. A *real* match (receipts) scored 0.636. Could a simple rule like "ignore scores below 0.6" tell them apart reliably? What should a RAG system do when the best chunk doesn't contain the answer? (This is the plan's "answer absent from corpus" scenario; Phase 7 adds the refusal.)
2. "Can I work from another country?" is answered by one sentence that sits **whole** in handbook `c0` (characters 348–471). Because of the 100-character overlap, its second half ("...working days per calendar year and needs approval from People Operations") is repeated at the start of `c1`. Search ranked `c1`, which holds only half the sentence, above `c0`, which isn't even in my top 2. Why might half a sentence score higher than the whole one? Which chunk is "the right one"? Why does that make the plan label evidence as a **quote**, not as a chunk ID?
3. Search never says "I found nothing". It always returns k chunks. Why is that a risk for the LLM in Phase 6?

**Part B — Create stale chunks on purpose, then fix them.**

1. In `data/raw/brightwater_handbook.md`, change `Annual leave: 27 days` to `Annual leave: 28 days` and save.
2. Run `uv run python -m ragqa.search "How many vacation days do I get?"`. On my run the count went `7 before, 8 after adding 7`, Chroma's #1 was the **old** "27 days" chunk (0.6553), and the last line said `Same chunks in the same order: False`.
3. Use the "list stored chunk IDs" command from 4.6. Which ID appears twice?
4. Change `28` back to `27`, save, and run the search again. On my run the count stayed at **8**: undoing the edit doesn't remove anything.
5. Delete the database and rebuild:

```bash
rm -r storage/chroma
```

- **What it does:** deletes the Chroma folder (see 4.6). The next search rebuilds it.

6. Run the search once more. Expected: `0 before, 7 after adding 7` and `Same chunks in the same order: True`.

Question: content-hash IDs prevent duplicates of *unchanged* chunks. Why can't they prevent stale copies of *changed* chunks? (Hint: what does the ID of the edited chunk depend on?)

**Part C — Write one test yourself.** Add `test_edited_chunk_leaves_the_old_version_behind` to `tests/integration/test_store.py`, with `@pytest.mark.integration` and the `tmp_path` fixture. It turns Part B into an automatic check:

- Build two `Chunk`s with the same `chunk_id`, source and page, but texts `"Annual leave is 27 days."` and `"Annual leave is 28 days."`.
- Open a collection in `tmp_path`, add the first chunk with vector `np.array([[0.6, 0.8]])`, then the second with `np.array([[0.8, 0.6]])`.
- Assert the count is **2**, not 1.

The test documents a known limitation instead of hiding it. If you ever add automatic clean-up, this test will start failing, and you'll update it on purpose. It passed on my test copy.

## 4.8 Done criteria

- [ ] `uv add chromadb` is done and `chromadb` appears in `pyproject.toml`
- [ ] `uv run python -m ragqa.search "How many vacation days do I get?"` prints identical numpy and Chroma top-3 lists and `Same chunks in the same order: True`
- [ ] Running it a second time prints `Stored chunks: 7 before, 7 after adding 7` (no duplicates)
- [ ] `uv run pytest tests/unit -v` passes in about a second, including the 5 retrieval tests
- [ ] `uv run pytest tests/integration/test_store.py tests/integration/test_search_agreement.py -v` passes all 5 tests (6 with your own)
- [ ] Part B done: you created stale chunks, found the duplicated chunk ID, and fixed it by rebuilding
- [ ] Your table and answers from Part A are written down
- [ ] You can explain why numpy brute force is the reference that Chroma is checked against
- [ ] Your work is committed: `git add .` then `git commit -m "Phase 4: retrieval with numpy and Chroma"`, then `git push`. Check that `storage/` is **not** in the commit (`git status` must not list it)
- [ ] Phase 4 ticked in the phase tracker

When these pass, share your Part A table and answers, and we'll start Phase 5: a small golden set and the first retrieval score, hit@k.
