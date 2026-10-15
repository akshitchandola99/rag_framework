# RAG Quality Lab — Phase 3: Local embeddings

Oct 14, 2026 · @Akshit

**Goal of this phase:** turn every chunk from Phase 2 into an **embedding**, a list of 384 numbers that captures what the text *means*, using a small model that runs on your Mac. Then prove the numbers behave: similar sentences must score higher than unrelated ones. No LLM is needed, so Ollama can stay off. Previous phase: [Phase 2](https://claude.ai/code/artifact/4b5e335a-21aa-435a-98c4-3d6105208c7f). Tracker: [Phase 0 doc](https://claude.ai/code/artifact/b32269ff-ce8f-4186-b984-8cf68f206b1b).

## 3.1 Concepts

Retrieval needs a way to ask "which chunk is about the same thing as this question?" Matching exact words fails often: a question about "holidays" should find a chunk about "annual leave". Embeddings solve this by turning text into numbers so that texts with similar meaning get similar numbers.

**Words you'll meet in this phase**

| Term | Plain meaning | In our project |
| --- | --- | --- |
| Embedding | A list of numbers that represents a text's meaning | One per chunk, one per question |
| Vector | Another word for a list of numbers, seen as an arrow pointing somewhere in space | Each embedding is a vector |
| Dimension | How many numbers the vector has | 384 for our model |
| Embedding model | A neural network trained so that texts with similar meaning get vectors pointing in similar directions | `BAAI/bge-small-en-v1.5` |
| Cosine similarity | A score for how closely two vectors point the same way: 1 = same direction, 0 = unrelated, negative = opposite | How we compare a question with chunks |
| Normalization | Rescaling a vector so its length is exactly 1, keeping its direction | Makes cosine similarity a simple multiply-and-add |
| Dot product | Multiply two vectors number by number, then add everything up | Equals cosine similarity once both vectors are normalized |
| NumPy array | A fast table of numbers from the `numpy` library; `shape` tells its size | Our 7 chunk vectors form one array of shape `(7, 384)` |
| Batch | Several texts sent to the model at once, which is faster than one at a time | All chunks are embedded in one call |
| Hugging Face Hub | A free website that hosts open models; libraries download from it once, then reuse the copy on disk | Where `bge-small-en-v1.5` comes from (no account needed) |

**Cosine similarity in two dimensions.** Real vectors have 384 numbers, but the idea is the same with 2. All three vectors below are already normalized (length 1):

| Vector | Numbers | Dot product with A | Meaning |
| --- | --- | --- | --- |
| A | `[0.6, 0.8]` | 0.6×0.6 + 0.8×0.8 = **1.00** | Same direction as itself |
| B | `[0.8, 0.6]` | 0.6×0.8 + 0.8×0.6 = **0.96** | Almost the same direction: similar meaning |
| C | `[0.8, -0.6]` | 0.6×0.8 + 0.8×(−0.6) = **0.00** | At a right angle: unrelated |

That multiply-and-add is all retrieval does in Phase 4: score the question against every chunk, keep the highest.

**Scores are relative, not absolute.** With this model, even unrelated English sentences score well above 0: on your Mac, a question and an unrelated policy line scored 0.46, while the matching line scored 0.66. Never read a single score as "good" or "bad". What matters is the **order**: the right chunk must score higher than the wrong ones. Section 3.5 shows the real numbers from your Mac.

**How it connects to the architecture.** Phase 3 is step 3 of the indexing flow, and it's also the first step of every query:

1. **Load** (Phase 1) and **Chunk** (Phase 2): files → `Chunk` records. `embeddings.py` reuses `load_folder()` and `chunk_pages()`.
2. **Embed** (Phase 3): each chunk's text → a 384-number vector. A question is embedded the same way at query time.
3. **Store and search** (Phase 4): vectors go into numpy, then Chroma, and the question vector is compared with every chunk vector.

**Design choices for this phase**

| Question | Our choice | Why |
| --- | --- | --- |
| Which library? | `sentence-transformers` (Apache 2.0 license) | The plan's choice: it shows normalization, batching and vector shapes clearly. It installs PyTorch, which is large on disk |
| Which model? | `BAAI/bge-small-en-v1.5` (MIT license) | About 130 MB, 384 dimensions, strong for its size, runs well on an M1 with 8 GB |
| Normalize vectors? | Yes, always | Cosine similarity becomes a plain dot product, and every score lands between −1 and 1 |
| Treat questions differently from chunks? | Yes: questions get a short instruction in front | The model's authors recommend the prefix ` Represent this sentence for searching relevant passages:  ` for short search queries. Chunks get no prefix |
| Load the model how often? | Once per program run, then reuse it | Loading takes seconds; embedding a sentence takes milliseconds |
| Model files | Downloaded once into the Hugging Face cache (`~/.cache/huggingface`), reused offline afterwards | Zero cost and no account. After the first download, no internet is needed |
| Run on the CPU or the M1's GPU ("mps")? | CPU, set explicitly | Measured on your Mac: same scores to 4 decimals, but CPU embedded 41 texts in 0.10 s against 0.45 s on the GPU (small batches don't pay off the GPU's start-up cost). It also leaves the GPU free for Ollama in Phase 6 |

**The query prefix, in plain words.** bge models were trained so that a short question with this instruction in front lands close to the passages that *answer* it, not just passages that *sound like* it. Phase 4 and Phase 5 measure whether it helps on our data; the exercise in 3.7 gives you a first look.

## 3.2 Setup steps

Run every command from your project folder (`/Users/nowgg/Automation/rag_framework`). Phases 1 and 2 must work first: `uv run python -m ragqa.chunking` should end with `5 pages -> 7 chunks`.

### Step 1 — Install sentence-transformers

```bash
uv add sentence-transformers
```

- **What it does:** downloads `sentence-transformers` and everything it needs, adds it to `dependencies` in `pyproject.toml`, and pins exact versions in `uv.lock`.
- **Why:** this library loads the embedding model and turns text into vectors. No `--dev`, because the app itself needs it.
- **What to expect:** a big download. On my test run it added **21 packages in 56 seconds**, and the project's `.venv` grew to **856 MB**. Most of that is **PyTorch** (`torch`), the maths engine that runs neural networks.

The main packages it installed, with versions from my run and their licenses:

| Package | Version | What it does | License |
| --- | --- | --- | --- |
| `sentence-transformers` | 6.1.0 | Loads embedding models, turns text into vectors | Apache 2.0 |
| `transformers` | 5.19.0 | Hugging Face's library for reading model files | Apache 2.0 |
| `torch` | 2.14.1 | PyTorch, runs the neural network | BSD-3-Clause |
| `numpy` | 2.5.4 | Arrays of numbers; vectors come back as numpy arrays | BSD-3-Clause |
| `scikit-learn`, `scipy` | 1.9.1, 1.18.1 | Maths helpers that sentence-transformers depends on | BSD-3-Clause |

All free and open source, with permissive licenses (you can use them in any project).

### Step 2 — Declare numpy as a direct dependency

```bash
uv add numpy
```

- **What it does:** adds `numpy` to `pyproject.toml` by name. It's already installed (Step 1 pulled it in), so nothing new downloads.
- **Why:** our code writes `import numpy` itself. A package you import directly should be listed directly. If a future version of sentence-transformers stopped depending on numpy, our code would otherwise break without warning.

After Steps 1 and 2, the `dependencies` list in `pyproject.toml` reads:

```toml
dependencies = [
    "numpy>=2.5.4",
    "pymupdf>=1.28.2",
    "requests>=2.34.2",
    "sentence-transformers>=6.1.0",
]
```

### Step 3 — Let the first run download the model

There's no separate download command. The first time the code in section 3.3 asks for `BAAI/bge-small-en-v1.5`, the library downloads it from the Hugging Face Hub and saves it in a cache folder. Every later run reads it from disk.

What I measured on the first and later runs:

| Measurement | First run (downloads) | Later runs |
| --- | --- | --- |
| Model load time printed by the script | 38.6 s | 7.4 s online (it asks the Hub whether the model changed), 0.2–0.3 s offline |
| Whole command, start to finish | 107 s | 17–24 s (13 s of that is importing PyTorch) |
| Internet needed | Yes | No |

Where the model lives afterwards:

```bash
du -sh ~/.cache/huggingface/hub
```

- **What it does:** shows how much disk the Hugging Face cache uses. `du` means "disk usage"; `-s` gives one total and `-h` prints it in human units (K, M, G). `~` is your home folder.
- **Why:** so you know where the 137 MB went (that was my result), and can delete the folder to force a fresh download.

The first run also prints this line:

```text
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

Ignore it. A token needs a Hugging Face account, and this project uses no accounts. Downloading a public model works fine without one.

### Step 4 (optional) — Prove it works offline

```bash
HF_HUB_OFFLINE=1 uv run python -m ragqa.embeddings
```

- **What it does:** runs the script with the environment variable `HF_HUB_OFFLINE` set to `1`, which tells the Hugging Face library "never use the internet, only the cache".
- **The parts:** `NAME=value command` sets a variable for that one command only. Nothing is changed permanently.
- **Why:** proof that, after the first download, the project runs fully locally. On my run it gave exactly the same numbers as the online run, and the `HF_TOKEN` warning disappeared.

**Memory on 8 GB.** The whole script peaked at about **550–660 MB** of RAM (measured with `/usr/bin/time -l`). That fits comfortably next to Ollama's \~2.5 GB model in later phases, which matches the plan's budget of 0.5–1 GB for Python + PyTorch + bge-small.

## 3.3 Code: embeddings and similarity

This phase has two files:

| File | Job | Imports PyTorch? |
| --- | --- | --- |
| `similarity.py` | The maths: score one vector against many | No, only numpy. Imports in a blink |
| `embeddings.py` | Load the model, turn texts into vectors, print checks | Yes, so importing it takes about 13 s |

Why split them? `import sentence_transformers` took **13–14 seconds** on every run on your Mac, because PyTorch is large. Anything that imports `embeddings.py` pays that cost, including tests. Keeping the maths in its own small file means the fast unit test (and Phase 4's search code) never waits for PyTorch. I found this by measuring: with the maths inside `embeddings.py`, the fast test run jumped from under 1 s to 14 s.

Create `src/ragqa/similarity.py`:

```python
"""Phase 3: similarity math for normalized vectors. Only numpy, so it imports fast."""

import numpy as np


# Score one vector against many (dot product of normalized vectors).
def cosine_scores(query_vector: np.ndarray, chunk_vectors: np.ndarray) -> np.ndarray:
    # Both sides are normalized, so the dot product is the cosine similarity.
    return chunk_vectors @ query_vector
```

Create `src/ragqa/embeddings.py`:

```python
"""Phase 3: turn chunks and questions into normalized vectors with a local embedding model."""

import time

import numpy as np
from sentence_transformers import SentenceTransformer

from ragqa.chunking import DEFAULT_CHUNK_SIZE
from ragqa.chunking import DEFAULT_OVERLAP
from ragqa.chunking import chunk_pages
from ragqa.loaders import DEFAULT_DATA_FOLDER
from ragqa.loaders import load_folder
from ragqa.similarity import cosine_scores

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DEVICE = "cpu"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 32

SIMILARITY_EXAMPLES = [
    ("How many vacation days do I get?", "Annual leave: 27 days per year, plus public holidays."),
    ("How many vacation days do I get?", "Laptops are replaced every 3 years."),
    ("Can I work from home?", "Employees may work remotely up to 3 days per week."),
    ("Can I work from home?", "The daily meal allowance is 45 euros."),
]


# Load the bge-small embedding model on the CPU.
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)


# Turn texts into normalized vectors, one row per text.
def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors


# Embed a question, with the search prefix in front.
def embed_query(model: SentenceTransformer, question: str) -> np.ndarray:
    vectors = embed_texts(model, [QUERY_PREFIX + question])
    return vectors[0]


# Embed all chunks and print shapes, lengths and similarity checks.
def main() -> None:
    start_time = time.perf_counter()
    model = load_embedding_model()
    load_seconds = time.perf_counter() - start_time
    print(f"Model: {EMBEDDING_MODEL_NAME} on {model.device}, loaded in {load_seconds:.1f} s")

    pages, skipped_files = load_folder(DEFAULT_DATA_FOLDER)
    chunks = chunk_pages(pages, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP)
    chunk_texts = []
    for chunk in chunks:
        chunk_texts.append(chunk.text)

    start_time = time.perf_counter()
    chunk_vectors = embed_texts(model, chunk_texts)
    embed_seconds = time.perf_counter() - start_time
    print(f"Embedded {len(chunks)} chunks in {embed_seconds:.2f} s")
    print(f"Array shape: {chunk_vectors.shape}, number type: {chunk_vectors.dtype}")

    first_vector = chunk_vectors[0]
    print(f"First 5 numbers of {chunks[0].chunk_id}: {np.round(first_vector[:5], 4)}")

    print("Vector lengths (should all be 1.0):")
    for index, chunk in enumerate(chunks):
        length = float(np.linalg.norm(chunk_vectors[index]))
        print(f"  {chunk.chunk_id}: {length:.4f}")

    print("Similarity checks (question vs passage):")
    for question, passage in SIMILARITY_EXAMPLES:
        question_vector = embed_query(model, question)
        passage_vectors = embed_texts(model, [passage])
        score = float(cosine_scores(question_vector, passage_vectors)[0])
        print(f"  {score:.3f}  {question!r} vs {passage!r}")

    for reason in skipped_files:
        print(f"SKIPPED {reason}")


if __name__ == "__main__":
    main()
```

**What the code does, part by part**

- **`similarity.py` → `cosine_scores(query_vector, chunk_vectors)`.** Input: one question vector (384 numbers) and a 2-D array of chunk vectors (one row per chunk, e.g. shape `(7, 384)`). Output: one score per chunk, shape `(7,)`.
  - `@` is numpy's **matrix multiplication** sign. `chunk_vectors @ query_vector` computes the dot product of the question with *every* row at once: 7 multiply-and-add sums in one line, the same sum as the 2-D table in section 3.1.
  - It only equals cosine similarity because every vector was normalized. That's the job of `normalize_embeddings=True` below.
- **The imports in `embeddings.py`.** `time` measures how long steps take. `import numpy as np` is the universal short name for numpy. `SentenceTransformer` is the class that loads and runs the model. The five `from ragqa...` lines **reuse Phases 1 and 2** (loader, chunker, their default settings) and the new `cosine_scores`.
- **The CAPITAL settings.**
  - `EMBEDDING_MODEL_NAME`: the model's name on the Hugging Face Hub: organisation (`BAAI`) / model.
  - `EMBEDDING_DEVICE = "cpu"`: where the maths runs. Faster than the GPU for our small batches (design table, 3.1).
  - `QUERY_PREFIX`: the instruction bge models expect in front of a search question. Note the space at the end.
  - `BATCH_SIZE = 32`: how many texts the model processes together. Our 7 chunks fit in one batch.
  - `SIMILARITY_EXAMPLES`: four (question, passage) pairs: for each question, one related passage from our documents, then one unrelated.
- **`load_embedding_model()`.** Creates the model object. The first call downloads the files (Step 3); later calls read the cache. It returns the model so the caller can reuse it; loading it again for every sentence would waste seconds each time.
- **`embed_texts(model, texts)`.** Input: the model and a list of strings. Output: a numpy array with one row of 384 numbers per string. `model.encode(...)` does the real work:
  - `batch_size=BATCH_SIZE` processes up to 32 texts per step.
  - `normalize_embeddings=True` rescales every vector to length 1, so dot product = cosine similarity.
  - `convert_to_numpy=True` returns a numpy array (instead of a PyTorch tensor), which is what `cosine_scores` and Phase 4 expect.
  - `show_progress_bar=False` keeps the output clean.
- **`embed_query(model, question)`.** Adds `QUERY_PREFIX` to the question, embeds it with the same function, and returns row `[0]`: a single vector of shape `(384,)`. Chunks never get the prefix; only questions do.
- **`main()`**, step by step:
  1. **Load and time the model.** `time.perf_counter()` reads a precise clock; the difference between two readings is the time in seconds. `model.device` reports where it runs (`cpu`).
  2. **Load and chunk** the documents with the Phase 1 and 2 functions, then collect each chunk's text into `chunk_texts`.
  3. **Embed all chunks in one call** and time it.
  4. **Print the array's shape and number type.** `(7, 384)` means 7 rows (chunks) by 384 columns (numbers). `float32` means each number is a 32-bit decimal, which takes 4 bytes, so one vector is 384 × 4 = 1,536 bytes.
  5. **Print the first 5 numbers** of the first vector, rounded with `np.round(..., 4)`, so you can see what an embedding actually looks like.
  6. **Check normalization.** `np.linalg.norm(vector)` computes a vector's length (square root of the sum of its squared numbers). Every line should read `1.0000`. `float(...)` turns numpy's number type into a plain Python number for printing.
  7. **Similarity checks.** For each pair: embed the question (with prefix) and the passage (without), score them with `cosine_scores`, and print the score. `[0]` takes the single score out of the 1-item result. In each question's pair of lines, the related passage should score higher.

## 3.4 Tests

Six tests cover the plan's Phase 3 checks: **vector shape, normalization, and "similar scores higher than unrelated"**.

| Kind | File | Needs the model? | Time |
| --- | --- | --- | --- |
| 1 unit test | `tests/unit/test_similarity.py` | No: tiny hand-made vectors | Under 1 s |
| 5 integration tests | `tests/integration/test_embeddings.py` | Yes: real model from the cache | About 18 s, mostly the PyTorch import |

The integration tests get the `@pytest.mark.integration` label: they need a downloaded model, just like the Ollama test from Phase 0. They don't need Ollama running.

One new pytest idea appears here:

- **A module-scoped fixture.** `@pytest.fixture(scope="module")` above a function called `model` means "build this once for the whole file, then hand the same object to every test that asks for `model`". Without `scope="module"`, pytest would reload the model for each of the 5 tests.

Create `tests/unit/test_similarity.py`:

```python
import numpy as np
import pytest

from ragqa.similarity import cosine_scores


# The maths matches the 2-D example from section 3.1.
def test_cosine_scores_match_the_worked_example():
    vector_a = np.array([0.6, 0.8])
    other_vectors = np.array([[0.6, 0.8], [0.8, 0.6], [0.8, -0.6]])

    scores = cosine_scores(vector_a, other_vectors)

    assert scores == pytest.approx([1.0, 0.96, 0.0])
```

Create `tests/integration/test_embeddings.py`:

```python
import numpy as np
import pytest

from ragqa import embeddings
from ragqa.similarity import cosine_scores


# Load the model once and share it with every test in this file.
@pytest.fixture(scope="module")
def model():
    return embeddings.load_embedding_model()


# Each text becomes one row of 384 numbers.
@pytest.mark.integration
def test_one_vector_of_384_numbers_per_text(model):
    vectors = embeddings.embed_texts(model, ["first text", "second text", "third text"])

    assert vectors.shape == (3, 384)


# Every vector is normalized to length 1.
@pytest.mark.integration
def test_every_vector_has_length_one(model):
    vectors = embeddings.embed_texts(model, ["Annual leave is 27 days.", "Laptops last 3 years."])

    for vector in vectors:
        assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)


# The related passage outranks the unrelated one.
@pytest.mark.integration
def test_related_passage_scores_higher_than_unrelated(model):
    question_vector = embeddings.embed_query(model, "How many vacation days do I get?")
    passage_vectors = embeddings.embed_texts(
        model,
        [
            "Annual leave: 27 days per year, plus public holidays.",
            "Laptops are replaced every 3 years.",
        ],
    )

    scores = cosine_scores(question_vector, passage_vectors)

    related_score = scores[0]
    unrelated_score = scores[1]
    assert related_score > unrelated_score


# Embedding the same text twice gives the same vector.
@pytest.mark.integration
def test_same_text_gives_the_same_vector(model):
    first_vector = embeddings.embed_texts(model, ["The daily meal allowance is 45 euros."])[0]
    second_vector = embeddings.embed_texts(model, ["The daily meal allowance is 45 euros."])[0]

    assert np.allclose(first_vector, second_vector, atol=1e-6)


# The query prefix really changes the question's vector.
@pytest.mark.integration
def test_query_prefix_changes_the_vector(model):
    with_prefix = embeddings.embed_query(model, "Can I work from home?")
    without_prefix = embeddings.embed_texts(model, ["Can I work from home?"])[0]

    assert not np.allclose(with_prefix, without_prefix)
```

**What the tests check, and why each matters**

| Test | What it proves | What would break later without it |
| --- | --- | --- |
| `cosine_scores_match_the_worked_example` | The maths gives 1.00, 0.96, 0.00, exactly the table in 3.1 | Every retrieval score in Phase 4 would be wrong |
| `one_vector_of_384_numbers_per_text` | 3 texts in → array of shape `(3, 384)` out | Phase 4 would mix up which vector belongs to which chunk |
| `every_vector_has_length_one` | Normalization is on | Dot product would no longer equal cosine similarity; long chunks would get unfairly high scores |
| `related_passage_scores_higher_than_unrelated` | The model ranks the leave sentence above the laptop sentence | The whole idea of semantic search would fail |
| `same_text_gives_the_same_vector` | Embedding is repeatable (to 0.000001) | Evaluation runs (Phase 12) couldn't be compared |
| `query_prefix_changes_the_vector` | The prefix really reaches the model | A typo in `embed_query` could silently drop it |

A few lines explained:

- `pytest.approx([1.0, 0.96, 0.0])` compares decimals with a tiny tolerance. Computers store 0.96 as 0.95999999..., so an exact `==` on decimals can fail for no real reason.
- `pytest.approx(1.0, abs=1e-5)` allows a difference of up to 0.00001. `1e-5` is scientific notation for 0.00001.
- `np.allclose(a, b, atol=1e-6)` checks that every one of the 384 numbers matches within 0.000001.
- The related-vs-unrelated test checks **order**, never a fixed score like `> 0.6`. Scores shift slightly between model versions and machines; the order is what retrieval depends on.

**Why the fast test command changes from this phase on.** Until now, `uv run pytest -m "not integration"` was the quick check. It still gives the right results, but on my run it took **15.96 s** instead of under 1 s. The reason: pytest first *imports every test file* to find the tests, and only then skips the labelled ones. Importing `test_embeddings.py` imports PyTorch, and that alone takes 13 s. Running only the unit-test folder avoids the import entirely: **0.93 s** on my run. So from Phase 3 on:

| You want | Command | My time |
| --- | --- | --- |
| Quick check while coding | `uv run pytest tests/unit -v` | 0.93 s (19 tests) |
| Everything, before a commit | `uv run pytest -v` | 15.56 s (24 tests) |

## 3.5 Run it

Run these from your project folder. Ollama doesn't need to be running.

```bash
uv run python -m ragqa.embeddings
```

- **What it does:** loads the model, loads and chunks `data/raw/`, embeds all 7 chunks, then prints the array shape, a sample of numbers, every vector's length, and four similarity scores.
- **The parts:** `uv run` uses the project's environment; `python -m ragqa.embeddings` runs `embeddings.py` inside the `ragqa` package.
- **Why:** this is the plan's Phase 3 check: shape and normalization verified, and similar sentences scoring higher than unrelated ones.

```bash
uv run pytest tests/integration/test_embeddings.py tests/unit/test_similarity.py -v
```

- **What it does:** runs only the 6 new Phase 3 tests.
- **The parts:** each file path limits the run to that file; `-v` prints each test's name.

```bash
uv run pytest -v
```

- **What it does:** runs every test from Phases 0–3, including integration tests.
- **Why:** your full check before a commit. Phase 0's Ollama integration test is included, so start Ollama first, or expect that one test to fail.

**Expected output of the script** (real output from my run, online mode):

```text
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Loading weights: 100%|██████████| 199/199 [00:00<00:00, 5232.17it/s]
Model: BAAI/bge-small-en-v1.5 on cpu, loaded in 7.4 s
Embedded 7 chunks in 0.43 s
Array shape: (7, 384), number type: float32
First 5 numbers of brightwater_faq.txt:p1:c0: [-0.0424 -0.0383  0.0481 -0.035  -0.0357]
Vector lengths (should all be 1.0):
  brightwater_faq.txt:p1:c0: 1.0000
  brightwater_handbook.md:p1:c0: 1.0000
  brightwater_handbook.md:p1:c1: 1.0000
  brightwater_handbook.md:p1:c2: 1.0000
  brightwater_travel_policy.pdf:p1:c0: 1.0000
  brightwater_travel_policy.pdf:p2:c0: 1.0000
  brightwater_travel_policy.pdf:p4:c0: 1.0000
Similarity checks (question vs passage):
  0.658  'How many vacation days do I get?' vs 'Annual leave: 27 days per year, plus public holidays.'
  0.462  'How many vacation days do I get?' vs 'Laptops are replaced every 3 years.'
  0.646  'Can I work from home?' vs 'Employees may work remotely up to 3 days per week.'
  0.459  'Can I work from home?' vs 'The daily meal allowance is 45 euros.'
```

What to notice:

| What you see | What it means |
| --- | --- |
| `Warning: ... HF_TOKEN` | Harmless; no account needed (Step 3). Gone in offline mode |
| `Loading weights: 199/199` | The model has 199 blocks of learned numbers ("weights"), read from the cache |
| `on cpu, loaded in 7.4 s` | Most of that is the online update check. With `HF_HUB_OFFLINE=1` it was 0.2 s |
| `Embedded 7 chunks in 0.43 s` | Embedding itself is fast: about 60 ms per chunk, including warm-up |
| `Array shape: (7, 384)` | 7 chunks, 384 numbers each, matching the Phase 2 chunk count |
| `float32` | Each number takes 4 bytes, so one vector is 1.5 KB and all 7 together about 10 KB |
| First 5 numbers like `-0.0424` | Single numbers mean nothing on their own; meaning lives in the direction of all 384 together |
| Every length `1.0000` | Normalization works, so dot product = cosine similarity |
| `0.658` vs `0.462`, `0.646` vs `0.459` | The related passage wins by about 0.19 both times. Note the unrelated pairs still score around 0.46, not 0: **only the order matters** |
| "vacation days" matched "Annual leave" | No shared keyword ("vacation" vs "leave"), yet the highest score. That's what embeddings add over word search |

Your numbers should match these exactly, because embedding on the CPU is deterministic (same input, same output). I checked this with **your** files too: they have no newline at the very end (noticed in Phase 2), so the FAQ and the last handbook chunk are one character shorter, yet every vector came out identical to mine to 5 decimals. The model ignores a trailing line break.

**Expected output of the Phase 3 tests** (real):

```text
tests/integration/test_embeddings.py::test_one_vector_of_384_numbers_per_text PASSED [ 16%]
tests/integration/test_embeddings.py::test_every_vector_has_length_one PASSED [ 33%]
tests/integration/test_embeddings.py::test_related_passage_scores_higher_than_unrelated PASSED [ 50%]
tests/integration/test_embeddings.py::test_same_text_gives_the_same_vector PASSED [ 66%]
tests/integration/test_embeddings.py::test_query_prefix_changes_the_vector PASSED [ 83%]
tests/unit/test_similarity.py::test_cosine_scores_match_the_worked_example PASSED [100%]
============================== 6 passed in 21.25s ==============================
```

## 3.6 Common errors and debugging

The messages in the first column are real outputs from my test runs (the last lines of each traceback), except where marked.

| What you see | What it means | How to fix it |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'sentence_transformers'` | The package isn't in the project's environment | Run `uv add sentence-transformers` (Step 1); always start commands with `uv run` |
| `OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.` | Offline mode (or no internet) and the model was never downloaded | Run once with internet and without `HF_HUB_OFFLINE=1`, so the model reaches the cache |
| `huggingface_hub.errors.RepositoryNotFoundError: 401 Client Error.` followed by `Invalid username or password.` | The model name is misspelled (I tried `BAAI/bge-smal-en-v1.5`). The Hub answers "not found or private", and the message wrongly hints at a login problem | Compare `EMBEDDING_MODEL_NAME` letter by letter with `BAAI/bge-small-en-v1.5`. No login is needed |
| `Warning: You are sending unauthenticated requests to the HF Hub...` | Not an error | Ignore it, or use `HF_HUB_OFFLINE=1` once the model is cached |
| Every `uv run` takes 15–25 s even for a tiny job | Importing PyTorch takes about 13 s on an M1, every run | Normal. For quick tests use `uv run pytest tests/unit`, which never imports PyTorch |
| `uv run pytest -m "not integration"` suddenly takes 16 s | pytest imports every test file before skipping labelled tests, and `test_embeddings.py` imports PyTorch | Use `uv run pytest tests/unit -v` for the quick check (section 3.4) |
| A unit test is slow after you add a new one | Your new unit test imports `ragqa.embeddings` (directly or through another module) | Keep model-free maths in `similarity.py`; put anything needing the model in `tests/integration/` |
| Shape `(384,)` where you expected `(1, 384)` | You passed a string, not a list: `embed_texts(model, "text")`. The library then returns one flat vector (real result from my run) | Always pass a list: `embed_texts(model, ["text"])`, then take `[0]` if you want one vector |
| `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, ... (size 7 is different from 384)` | You passed two arrays of chunk vectors to `cosine_scores`, instead of one question vector and many chunk vectors | The first argument must be a single vector of shape `(384,)`, e.g. from `embed_query` |
| Vector lengths not `1.0000` | `normalize_embeddings=True` is missing or misspelled in `embed_texts` | Fix the argument; `test_every_vector_has_length_one` catches this |
| Scores differ from 3.5 in the 3rd or 4th decimal | You run on `mps` (GPU) instead of `cpu`, or a different model version | Check `EMBEDDING_DEVICE = "cpu"`; small differences don't matter, the order does |
| Mac slows down, memory pressure turns yellow | Too many big apps open together with Python and PyTorch (\~0.6 GB) | Close Chrome and similar apps; check Activity Monitor as in Phase 0 |

**Debugging techniques for this phase**

```bash
uv run python -c "import time; t = time.perf_counter(); import sentence_transformers; print(round(time.perf_counter() - t, 1), 'seconds to import')"
```

- **What it does:** measures only how long `import sentence_transformers` takes.
- **Why:** when something feels slow, measure each step separately before guessing. On my runs this printed 13–14 seconds, which is how I found the slow-import problem in section 3.3.
- **The parts:** `time.perf_counter()` reads a precise clock; reading it before and after the import gives the time in between. `round(..., 1)` keeps one decimal.

```bash
HF_HUB_OFFLINE=1 uv run python -c "from ragqa.embeddings import load_embedding_model, embed_texts, embed_query; from ragqa.similarity import cosine_scores; m = load_embedding_model(); q = embed_query(m, 'How long do I have to submit receipts?'); p = embed_texts(m, ['Submit receipts in Compass within 30 days of returning.', 'Passwords expire every 180 days.']); print(cosine_scores(q, p))"
```

- **What it does:** scores one question of your choice against passages of your choice, offline.
- **Why:** the fastest way to check "does the model think these two are related?" when a retrieval result in Phase 4 surprises you. Change the question and passages freely.
- **The parts:** it imports the Phase 3 functions, loads the model once (`m`), embeds the question with the prefix (`q`) and two passages without (`p`), then prints both scores. My run printed `[0.8082475  0.57955325]`: the receipts passage wins clearly.

```bash
ls ~/.cache/huggingface/hub
```

- **What it does:** lists the models in your local Hugging Face cache.
- **Why:** confirms the model was downloaded. You should see `models--BAAI--bge-small-en-v1.5`. To force a fresh download, delete that folder (in Finder, or with `rm -r` on the full path).

## 3.7 Exercise: a first taste of search

**Part A — Rank all chunks for a question.** This small script scores one question against all 7 chunks, twice: with the query prefix and without it. It's a preview of Phase 4, built only from Phase 1–3 functions.

Create `src/ragqa/try_ranking.py`:

```python
"""Phase 3 exercise: rank every chunk for one question, with and without the query prefix."""

import sys

import numpy as np

from ragqa.chunking import DEFAULT_CHUNK_SIZE
from ragqa.chunking import DEFAULT_OVERLAP
from ragqa.chunking import chunk_pages
from ragqa.embeddings import embed_query
from ragqa.embeddings import embed_texts
from ragqa.embeddings import load_embedding_model
from ragqa.loaders import DEFAULT_DATA_FOLDER
from ragqa.loaders import load_folder
from ragqa.similarity import cosine_scores


# Print chunk IDs from the best score to the worst.
def print_ranking(title: str, scores: np.ndarray, chunk_ids: list[str]) -> None:
    print(title)
    positions_best_first = np.argsort(-scores)
    for rank, position in enumerate(positions_best_first, start=1):
        print(f"  {rank}. {scores[position]:.3f}  {chunk_ids[position]}")


# Rank all chunks for one question, with and without the prefix.
def main() -> None:
    if len(sys.argv) != 2:
        print('Usage: uv run python -m ragqa.try_ranking "your question"')
        sys.exit(1)
    question = sys.argv[1]

    model = load_embedding_model()
    pages, skipped_files = load_folder(DEFAULT_DATA_FOLDER)
    chunks = chunk_pages(pages, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP)

    chunk_texts = []
    chunk_ids = []
    for chunk in chunks:
        chunk_texts.append(chunk.text)
        chunk_ids.append(chunk.chunk_id)
    chunk_vectors = embed_texts(model, chunk_texts)

    with_prefix = embed_query(model, question)
    without_prefix = embed_texts(model, [question])[0]

    print(f"Question: {question}")
    print_ranking("With query prefix:", cosine_scores(with_prefix, chunk_vectors), chunk_ids)
    print_ranking("Without prefix:", cosine_scores(without_prefix, chunk_vectors), chunk_ids)


if __name__ == "__main__":
    main()
```

Two new lines explained:

- `np.argsort(-scores)` returns the *positions* of the scores in sorted order. `argsort` sorts from smallest to largest, so negating the scores (`-scores`) puts the highest first. With scores `[0.5, 0.9, 0.7]` it returns `[1, 2, 0]`.
- `enumerate(..., start=1)` counts from 1 instead of 0, so ranks read 1, 2, 3.

Run it with a question in double quotes:

```bash
uv run python -m ragqa.try_ranking "How many vacation days do I get?"
```

- **What it does:** prints all 7 chunks, best first, with and without the prefix.
- **The parts:** the text in quotes becomes `sys.argv[1]`. The quotes keep the whole question as one word for the program.

Real output from my run:

```text
Question: How many vacation days do I get?
With query prefix:
  1. 0.655  brightwater_handbook.md:p1:c1
  2. 0.612  brightwater_travel_policy.pdf:p2:c0
  3. 0.598  brightwater_travel_policy.pdf:p4:c0
  4. 0.560  brightwater_handbook.md:p1:c2
  5. 0.558  brightwater_faq.txt:p1:c0
  6. 0.548  brightwater_travel_policy.pdf:p1:c0
  7. 0.546  brightwater_handbook.md:p1:c0
Without prefix:
  1. 0.653  brightwater_handbook.md:p1:c1
  2. 0.619  brightwater_travel_policy.pdf:p2:c0
  3. 0.595  brightwater_travel_policy.pdf:p4:c0
  4. 0.564  brightwater_handbook.md:p1:c2
  5. 0.560  brightwater_faq.txt:p1:c0
  6. 0.552  brightwater_handbook.md:p1:c0
  7. 0.547  brightwater_travel_policy.pdf:p1:c0
```

Now run it with these four questions too, and fill in the table. My top result and score (with prefix) are given so you can check yours:

| Question | Correct chunk | My #1 (with prefix) | Gap to #2 | Your notes |
| --- | --- | --- | --- | --- |
| How many vacation days do I get? | handbook `c1` | handbook `c1`, 0.655 | 0.043 |  |
| How much can I spend on a hotel in Paris? | policy `p2:c0` | policy `p2:c0`, 0.723 | 0.157 |  |
| Who do I call when IT is closed? | FAQ `c0` | FAQ `c0`, 0.662 | 0.119 |  |
| What happens if I lose my laptop? | handbook `c2` | handbook `c2`, 0.639 | 0.150 |  |
| Can I fly business class? | policy `p2:c0` | policy `p2:c0`, 0.605 | 0.071 |  |

Answer these in a few words each:

1. All five #1 results were correct on my run. But for "vacation days", the gap to #2 was only 0.043, and #2 was a *travel* page. Why might a page about hotels and flights look similar to a question about vacation? What does a small gap mean for retrieval with top k = 1 versus top k = 3?
2. Handbook `c0` ends with `Annual leave: 2` (the cut from Phase 2), yet it ranked **last** for the vacation question. Why doesn't a half sentence about leave help it?
3. With and without the prefix, the #1 result never changed, and scores moved by at most 0.03. Does that prove the prefix is useless? (Hint: 5 questions and 7 chunks is a tiny test. Phase 5 measures it properly with hit@k.)
4. Make up a question whose answer is **not** in any document, like "What is the parking policy?". What is the #1 score? Would you trust the #1 chunk? This is the "answer absent from corpus" scenario from the plan's test table.

**Part B — Write one test yourself.** Add `test_batching_does_not_change_vectors` to `tests/integration/test_embeddings.py` (with `@pytest.mark.integration` and the `model` fixture). It proves that embedding several texts in one batch gives the same vectors as embedding them one at a time.

- Embed a list of three short texts in one call with `embeddings.embed_texts`.
- Loop over the texts with `enumerate`; embed each one alone and compare it with the matching row of the batch result.
- Use `np.allclose(..., atol=1e-5)`. On my run the largest difference was 0.00000018, so tiny differences are normal and must not fail the test.

## 3.8 Done criteria

- [ ] `uv add sentence-transformers` and `uv add numpy` are done, and both appear in `pyproject.toml`
- [ ] `uv run python -m ragqa.embeddings` prints `Array shape: (7, 384)`, seven lengths of `1.0000`, and higher scores for both related pairs
- [ ] `HF_HUB_OFFLINE=1 uv run python -m ragqa.embeddings` works with no internet
- [ ] `uv run pytest tests/unit -v` passes in about a second
- [ ] `uv run pytest -v` passes, including the 5 embedding integration tests (6 with your own)
- [ ] Your ranking table from Part A is filled in, with answers to the four questions
- [ ] You can explain why normalized vectors make cosine similarity a plain dot product
- [ ] Your work is committed: `git add .` then `git commit -m "Phase 3: local embeddings"`, then `git push`
- [ ] Phase 3 ticked in the tracker of the [Phase 0 doc](https://claude.ai/code/artifact/b32269ff-ce8f-4186-b984-8cf68f206b1b)

When these pass, share your ranking table and answers, and we'll start Phase 4: retrieval, first with a few lines of numpy, then with Chroma.
