# RAG Quality Lab — Phase 5: Golden set v0 and retrieval metrics

Oct 15, 2026 · @Akshit

## Phase tracker

Tick a phase only when its verification checks pass.

- [ ] Phase 0 — Environment, Git, Ollama, hardware baseline ([separate doc](https://claude.ai/code/artifact/b32269ff-ce8f-4186-b984-8cf68f206b1b))
- [ ] Phase 1 — Load documents with metadata ([separate doc](https://claude.ai/code/artifact/7df3b55e-6623-464f-8990-7106ec3af43f))
- [ ] Phase 2 — Chunking ([separate doc](https://claude.ai/code/artifact/4b5e335a-21aa-435a-98c4-3d6105208c7f))
- [ ] Phase 3 — Local embeddings ([separate doc](https://claude.ai/code/artifact/a37f9484-2b02-439c-bcbd-42d9145c7a5a))
- [ ] Phase 4 — Retrieval (numpy brute force, then Chroma) ([separate doc](https://claude.ai/code/artifact/5c0d6903-3508-4447-89e0-79e7f586b656))
- [ ] Phase 5 — Golden v0 + hit@k / MRR (this doc)
- [ ] Phase 6 — Grounded generation with Ollama ([separate doc](https://claude.ai/code/artifact/dc112016-b8cb-489d-b01d-377efa3083a1))
- [ ] Phase 7 — Citations, refusal, errors, tracing ([separate doc](https://claude.ai/code/artifact/1e0d77c4-cdca-4be6-af3c-6ac45a522e8d))
- [ ] Phase 8 — Consolidated pytest suite + robustness fixtures ([separate doc](https://claude.ai/code/artifact/8dd46bfa-af84-4bf8-a1bb-8537f5671950))
- [ ] Phase 9 — Golden v1, human-verified
- [ ] Phase 10 — RAGAS with a local judge
- [ ] Phase 11 — DeepEval with Ollama + G-Eval
- [ ] Phase 12 — Regression runner, thresholds, reports
- [ ] Phase 13 — Evidence-driven improvement (+ optional UI)
- [ ] Phase 14 — Documentation and portfolio polish

**Goal of this phase:** write a small **golden set** of 8 test questions, each with the exact quote that answers it, and use it to measure retrieval with two numbers: **hit@k** and **MRR**. Save the result as the project's first **baseline**, the number every later change is compared against. No LLM is needed, so Ollama can stay off. Previous phase: [Phase 4](https://claude.ai/code/artifact/5c0d6903-3508-4447-89e0-79e7f586b656).

## 5.1 Concepts

So far you've judged search by eye: "the right chunk is #1, looks good". That doesn't scale and can't be compared between versions. This phase turns "looks good" into a number you can track, which is the core skill of AI quality engineering.

**Words you'll meet in this phase**

| Term | Plain meaning | In our project |
| --- | --- | --- |
| Golden set | A fixed list of test questions with known correct answers and evidence, written and checked by a person | `eval/datasets/golden_v0.jsonl`, 8 items |
| Evidence quote | The exact words in a document that answer the question, plus their source and page | `{"source": ..., "page": ..., "quote": ...}` |
| JSON Lines (`.jsonl`) | A text file with one JSON object per line. Easy to read, diff in Git, and add to line by line | The golden set's file format |
| Relevant chunk | A retrieved chunk that is on the evidence's page **and** contains the evidence quote | Checked by `hit_contains_evidence()` |
| First relevant rank | The rank of the first relevant chunk in the results (1 = top) | e.g. q003 has rank 6 |
| hit@k | The share of questions with a relevant chunk in the top k. Read "hit at k" | hit@3 = 0.857 means 6 of 7 questions |
| MRR | Mean Reciprocal Rank: the average of 1 / first relevant rank. Rank 1 → 1.0, rank 2 → 0.5, rank 3 → 0.33, not found → 0 | Rewards finding the right chunk *early* |
| Baseline | The first measured score, saved so later changes can be compared against it | `results/retrieval/<date>_baseline.json` |
| Answerable / unanswerable | Whether the documents contain the answer at all | q008 ("parking policy") is unanswerable |

**A worked example.** Four questions, and the rank where each one's relevant chunk appeared:

| Question | First relevant rank | In top 1? | In top 3? | 1 / rank |
| --- | --- | --- | --- | --- |
| A | 1 | Yes | Yes | 1.000 |
| B | not found | No | No | 0 |
| C | 3 | No | Yes | 0.333 |
| D | 2 | No | Yes | 0.500 |
| **Result** |  | **hit@1 = 1/4 = 0.25** | **hit@3 = 3/4 = 0.75** | **MRR = 1.833 / 4 = 0.458** |

hit@k answers "how often is the evidence somewhere the LLM will see it?" (with top k = 3, the LLM in Phase 6 sees 3 chunks). MRR answers "how high up is it?". You need both: two systems can have the same hit@3 while one always ranks the evidence #1 and the other always #3.

**Why evidence is a quote, not a chunk ID.** Chunk IDs change whenever chunking changes (Phase 2's exercise produced 6 to 15 chunks from the same files). A quote, a source and a page don't. So a hit is defined as "a retrieved chunk contains the quote", and the same golden set keeps working when you change chunk size in Phase 13. This is the plan's "chunking-agnostic evidence matching".

**Why we compare text after squashing whitespace.** PDF text keeps the PDF's line breaks, even in the middle of a sentence. On your Mac, "Submit receipts in Compass within 30 days of returning" is **not** found exactly in the page-4 text, because there is a line break after "in". Both sides are compared with every run of spaces and line breaks squashed into one space (`normalize_whitespace`), and then it matches.

**What the golden set does not score (yet).** The unanswerable question has no evidence, so it can't have a "rank". It's listed in the output with its top score, but hit@k and MRR use the 7 answerable questions only. Refusing to answer is Phase 7's job.

**How it connects to the architecture.** This is layer L2 of the plan's testing architecture, "retrieval eval": deterministic numbers that need only the embedder, no LLM.

1. **Phases 1–4** built the indexing and search path.
2. **Phase 5** scores that path against hand-written questions and saves a baseline.
3. **Phase 9** grows the golden set to 15–20 human-verified items; **Phase 12** turns the baseline into thresholds that flag regressions; **Phase 13** uses these numbers to decide whether a change (chunk size, another embedding model) is actually better.

**Design choices for this phase**

| Question | Our choice | Why |
| --- | --- | --- |
| How many items? | 8: 6 single-fact, 1 multi-chunk, 1 unanswerable | The plan's "golden v0 (8 items)". Small enough to write and check by hand today |
| Which search do we score? | numpy brute force (Phase 4) | Exact and fast, and the plan's L2 needs "embedder only". Phase 4 proved Chroma returns the same ranking |
| How deep do we rank? | Every chunk (7) | So every relevant chunk gets a real rank, even rank 6, and MRR is exact |
| Multi-chunk questions | Count a hit when **any** of the expected quotes is found | Simple first version. Checking that *all* quotes are found is recall@k, which can come with the larger golden set |
| Where results go | `PROJECT_ROOT / "results" / "retrieval"`, one JSON file per run | `results/` is git-ignored (Phase 0), so you'll copy the baseline numbers into a small Markdown note you *do* commit (exercise) |

## 5.2 Setup steps

**Nothing new to install.** This phase uses only Python's built-in `json` and `datetime` modules plus code from Phases 1–4. Run every command from your project folder.

### Step 1 — Read the 8 questions before you write them down

Each item was written the way the plan's ground-truth method says: read the document, write the question, copy the evidence **word for word**, then write the reference answer using only that evidence.

| ID | Question | Category | Evidence (source, page) | Quote |
| --- | --- | --- | --- | --- |
| q001 | How many days of annual leave do employees get? | single\_fact | handbook, 1 | Annual leave: 27 days per year |
| q002 | How many days per week can I work remotely? | single\_fact | handbook, 1 | Employees may work remotely up to 3 days per week |
| q003 | How many days a year can I work from another country? | single\_fact | handbook, 1 | Remote work from another country is limited to 20 working days per calendar year |
| q004 | What is the nightly hotel limit in Paris? | single\_fact | travel policy, 2 | 190 euros in London, Paris and Zurich |
| q005 | How much is the daily meal allowance? | single\_fact | travel policy, 4 | The daily meal allowance is 45 euros |
| q006 | How often do passwords expire? | single\_fact | FAQ, 1 | Passwords expire every 180 days |
| q007 | Which portal do I use to get travel approved and to claim expenses? | multi\_chunk | travel policy, 1 **and** 4 | Travel must be approved in the Compass portal by your team lead / Submit receipts in Compass within 30 days of returning |
| q008 | What is the parking policy? | unanswerable | — | — |

Why these? Every answer is a specific fact (a number, a name), so right and wrong are obvious. q003 is deliberately tricky (Phase 4 showed search struggles with it). q007 needs two pages. q008 checks behaviour when there is nothing to find.

### Step 2 — Create the golden set file

Create `eval/datasets/golden_v0.jsonl` (the `eval/datasets/` folder exists since Phase 0). Each line is one complete JSON object; don't break a line in two:

```json
{"id": "q001", "question": "How many days of annual leave do employees get?", "category": "single_fact", "expected_behavior": "answer", "reference_answer": "27 days per year, plus public holidays.", "expected_evidence": [{"source": "brightwater_handbook.md", "page": 1, "quote": "Annual leave: 27 days per year"}], "notes": "", "verified_by": "Akshit", "verified_on": "2026-10-15"}
{"id": "q002", "question": "How many days per week can I work remotely?", "category": "single_fact", "expected_behavior": "answer", "reference_answer": "Up to 3 days per week.", "expected_evidence": [{"source": "brightwater_handbook.md", "page": 1, "quote": "Employees may work remotely up to 3 days per week"}], "notes": "", "verified_by": "Akshit", "verified_on": "2026-10-15"}
{"id": "q003", "question": "How many days a year can I work from another country?", "category": "single_fact", "expected_behavior": "answer", "reference_answer": "Up to 20 working days per calendar year, with approval from People Operations.", "expected_evidence": [{"source": "brightwater_handbook.md", "page": 1, "quote": "Remote work from another country is limited to 20 working days per calendar year"}], "notes": "The full sentence is only in handbook c0; c1 repeats its second half.", "verified_by": "Akshit", "verified_on": "2026-10-15"}
{"id": "q004", "question": "What is the nightly hotel limit in Paris?", "category": "single_fact", "expected_behavior": "answer", "reference_answer": "190 euros per night.", "expected_evidence": [{"source": "brightwater_travel_policy.pdf", "page": 2, "quote": "190 euros in London, Paris and Zurich"}], "notes": "", "verified_by": "Akshit", "verified_on": "2026-10-15"}
{"id": "q005", "question": "How much is the daily meal allowance?", "category": "single_fact", "expected_behavior": "answer", "reference_answer": "45 euros per day.", "expected_evidence": [{"source": "brightwater_travel_policy.pdf", "page": 4, "quote": "The daily meal allowance is 45 euros"}], "notes": "", "verified_by": "Akshit", "verified_on": "2026-10-15"}
{"id": "q006", "question": "How often do passwords expire?", "category": "single_fact", "expected_behavior": "answer", "reference_answer": "Every 180 days.", "expected_evidence": [{"source": "brightwater_faq.txt", "page": 1, "quote": "Passwords expire every 180 days"}], "notes": "", "verified_by": "Akshit", "verified_on": "2026-10-15"}
{"id": "q007", "question": "Which portal do I use to get travel approved and to claim expenses?", "category": "multi_chunk", "expected_behavior": "answer", "reference_answer": "Compass: travel is approved there by your team lead, and receipts are submitted there within 30 days of returning.", "expected_evidence": [{"source": "brightwater_travel_policy.pdf", "page": 1, "quote": "Travel must be approved in the Compass portal by your team lead"}, {"source": "brightwater_travel_policy.pdf", "page": 4, "quote": "Submit receipts in Compass within 30 days of returning"}], "notes": "The page 4 quote has a line break inside it in the PDF text.", "verified_by": "Akshit", "verified_on": "2026-10-15"}
{"id": "q008", "question": "What is the parking policy?", "category": "unanswerable", "expected_behavior": "refuse", "reference_answer": "", "expected_evidence": [], "notes": "Nothing about parking in any document.", "verified_by": "Akshit", "verified_on": "2026-10-15"}
```

**The same 8 items, formatted for reading.** Use this to read and review the items, but **don't paste it into the file**. A `.jsonl` file needs each item on one line, so this formatted version would fail to load with `golden_v0.jsonl line 1: not valid JSON`. Copy the block above instead.

```json
[
  {
    "id": "q001",
    "question": "How many days of annual leave do employees get?",
    "category": "single_fact",
    "expected_behavior": "answer",
    "reference_answer": "27 days per year, plus public holidays.",
    "expected_evidence": [
      {
        "source": "brightwater_handbook.md",
        "page": 1,
        "quote": "Annual leave: 27 days per year"
      }
    ],
    "notes": "",
    "verified_by": "Akshit",
    "verified_on": "2026-10-15"
  },
  {
    "id": "q002",
    "question": "How many days per week can I work remotely?",
    "category": "single_fact",
    "expected_behavior": "answer",
    "reference_answer": "Up to 3 days per week.",
    "expected_evidence": [
      {
        "source": "brightwater_handbook.md",
        "page": 1,
        "quote": "Employees may work remotely up to 3 days per week"
      }
    ],
    "notes": "",
    "verified_by": "Akshit",
    "verified_on": "2026-10-15"
  },
  {
    "id": "q003",
    "question": "How many days a year can I work from another country?",
    "category": "single_fact",
    "expected_behavior": "answer",
    "reference_answer": "Up to 20 working days per calendar year, with approval from People Operations.",
    "expected_evidence": [
      {
        "source": "brightwater_handbook.md",
        "page": 1,
        "quote": "Remote work from another country is limited to 20 working days per calendar year"
      }
    ],
    "notes": "The full sentence is only in handbook c0; c1 repeats its second half.",
    "verified_by": "Akshit",
    "verified_on": "2026-10-15"
  },
  {
    "id": "q004",
    "question": "What is the nightly hotel limit in Paris?",
    "category": "single_fact",
    "expected_behavior": "answer",
    "reference_answer": "190 euros per night.",
    "expected_evidence": [
      {
        "source": "brightwater_travel_policy.pdf",
        "page": 2,
        "quote": "190 euros in London, Paris and Zurich"
      }
    ],
    "notes": "",
    "verified_by": "Akshit",
    "verified_on": "2026-10-15"
  },
  {
    "id": "q005",
    "question": "How much is the daily meal allowance?",
    "category": "single_fact",
    "expected_behavior": "answer",
    "reference_answer": "45 euros per day.",
    "expected_evidence": [
      {
        "source": "brightwater_travel_policy.pdf",
        "page": 4,
        "quote": "The daily meal allowance is 45 euros"
      }
    ],
    "notes": "",
    "verified_by": "Akshit",
    "verified_on": "2026-10-15"
  },
  {
    "id": "q006",
    "question": "How often do passwords expire?",
    "category": "single_fact",
    "expected_behavior": "answer",
    "reference_answer": "Every 180 days.",
    "expected_evidence": [
      {
        "source": "brightwater_faq.txt",
        "page": 1,
        "quote": "Passwords expire every 180 days"
      }
    ],
    "notes": "",
    "verified_by": "Akshit",
    "verified_on": "2026-10-15"
  },
  {
    "id": "q007",
    "question": "Which portal do I use to get travel approved and to claim expenses?",
    "category": "multi_chunk",
    "expected_behavior": "answer",
    "reference_answer": "Compass: travel is approved there by your team lead, and receipts are submitted there within 30 days of returning.",
    "expected_evidence": [
      {
        "source": "brightwater_travel_policy.pdf",
        "page": 1,
        "quote": "Travel must be approved in the Compass portal by your team lead"
      },
      {
        "source": "brightwater_travel_policy.pdf",
        "page": 4,
        "quote": "Submit receipts in Compass within 30 days of returning"
      }
    ],
    "notes": "The page 4 quote has a line break inside it in the PDF text.",
    "verified_by": "Akshit",
    "verified_on": "2026-10-15"
  },
  {
    "id": "q008",
    "question": "What is the parking policy?",
    "category": "unanswerable",
    "expected_behavior": "refuse",
    "reference_answer": "",
    "expected_evidence": [],
    "notes": "Nothing about parking in any document.",
    "verified_by": "Akshit",
    "verified_on": "2026-10-15"
  }
]
```

**The fields, one by one** (the plan's dataset design):

| Field | Meaning | Example |
| --- | --- | --- |
| `id` | A short unique name, so results can be compared item by item across runs | `"q003"` |
| `question` | What a user would type | `"How often do passwords expire?"` |
| `category` | What kind of test this is | `single_fact`, `multi_chunk`, `unanswerable` |
| `expected_behavior` | What the system *should* do | `"answer"` or `"refuse"` |
| `reference_answer` | The correct answer, written only from the evidence. Used from Phase 10 on | `"Every 180 days."` |
| `expected_evidence` | A list of `{source, page, quote}`. Empty for unanswerable items | see q007 for two |
| `notes` | Anything a future reader should know | why q003 is tricky |
| `verified_by` / `verified_on` | Who checked the item against the documents, and when | your name, today's date |

Put your own name in `verified_by` once you've checked each quote yourself (the exercise asks you to). Unlike `results/`, this file **is** committed to Git: it's a hand-made test asset, like a test case in a test-management tool.

## 5.3 Code: golden set loader, metrics, and the evaluation script

| File | Job | Imports PyTorch? |
| --- | --- | --- |
| `golden.py` | Read `golden_v0.jsonl` into records; check every quote exists in the documents | No |
| `retrieval_metrics.py` | Decide whether a hit is relevant; compute hit@k and MRR | No |
| `evaluate_retrieval.py` | Run every golden question through search, print scores, save the baseline | Yes (needs the embedder) |

The first two stay fast, so their tests run in the quick `tests/unit` check (the lesson from Phase 3).

### `golden.py`

Create `src/ragqa/golden.py`:

```python
"""Phase 5: load the golden set and check that its evidence quotes really exist."""

import json
from dataclasses import dataclass
from pathlib import Path

from ragqa.loaders import Page

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_V0_PATH = PROJECT_ROOT / "eval" / "datasets" / "golden_v0.jsonl"

REQUIRED_FIELDS = [
    "id",
    "question",
    "category",
    "expected_behavior",
    "reference_answer",
    "expected_evidence",
    "notes",
    "verified_by",
    "verified_on",
]


# One piece of expected evidence: an exact quote and where it lives.
@dataclass
class Evidence:
    source: str
    page: int
    quote: str


# One hand-written test question with its expected answer and evidence.
@dataclass
class GoldenItem:
    id: str
    question: str
    category: str
    expected_behavior: str
    reference_answer: str
    expected_evidence: list[Evidence]
    notes: str
    verified_by: str
    verified_on: str


# Squash all runs of spaces and line breaks into single spaces.
def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


# Read a JSON Lines file into golden items, one item per line.
def load_golden_set(path: Path) -> list[GoldenItem]:
    items = []
    with open(path, encoding="utf-8") as golden_file:
        for line_number, line in enumerate(golden_file, start=1):
            if line.strip() == "":
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name} line {line_number}: not valid JSON ({error})") from error
            for field_name in REQUIRED_FIELDS:
                if field_name not in record:
                    raise ValueError(f"{path.name} line {line_number}: missing field '{field_name}'")
            evidence_list = []
            for evidence_record in record["expected_evidence"]:
                evidence_list.append(
                    Evidence(
                        source=evidence_record["source"],
                        page=evidence_record["page"],
                        quote=evidence_record["quote"],
                    )
                )
            record["expected_evidence"] = evidence_list
            items.append(GoldenItem(**record))
    return items


# Return a problem message for every quote that can't be found on its page.
def find_missing_quotes(items: list[GoldenItem], pages: list[Page]) -> list[str]:
    problems = []
    for item in items:
        for evidence in item.expected_evidence:
            found = False
            for page in pages:
                same_place = page.source == evidence.source and page.page == evidence.page
                if same_place and normalize_whitespace(evidence.quote) in normalize_whitespace(page.text):
                    found = True
                    break
            if not found:
                problems.append(f"{item.id}: quote not found on {evidence.source} page {evidence.page}: {evidence.quote!r}")
    return problems
```

**What the code does, part by part**

- **`GOLDEN_V0_PATH`**: the golden file's absolute path, built from `PROJECT_ROOT` (the project rule).
- **`REQUIRED_FIELDS`**: the 9 field names every line must have. Checking them turns a typo in the file into a clear error instead of a confusing crash later.
- **`Evidence` and `GoldenItem`**: dataclass records (as in Phase 1), one field per JSON key. `expected_evidence: list[Evidence]` means "a list of `Evidence` records".
- **`normalize_whitespace(text)`**: `text.split()` breaks at every space, tab and line break; `" ".join(...)` glues the words back with single spaces. `"Submit receipts in\nCompass"` becomes `"Submit receipts in Compass"`.
- **`load_golden_set(path)`**: input a `.jsonl` path, output a list of `GoldenItem`s.
  - `with open(path, encoding="utf-8") as golden_file:` opens the file and closes it automatically afterwards. Looping over a file gives one line at a time; `enumerate(..., start=1)` numbers them like an editor does.
  - Empty lines are skipped, so a blank line at the end of the file is harmless.
  - `json.loads(line)` turns one line of JSON text into a Python dictionary. If the line isn't valid JSON, Python raises `json.JSONDecodeError`, whose message gives a position *inside that one line*, not a file line. The `try`/`except` turns it into a `ValueError` that names the file line, like `golden_v0.jsonl line 1: not valid JSON`.
  - The field check raises a `ValueError` (Phase 1's "`ValueError` in short") that names the file, the line and the missing field.
  - The inner loop turns each evidence dictionary into an `Evidence` record.
  - `GoldenItem(**record)` is "dictionary unpacking": `**` passes every key as a named argument, so `{"id": "q001", ...}` becomes `GoldenItem(id="q001", ...)`. It only works because the JSON keys match the dataclass fields exactly.
- **`find_missing_quotes(items, pages)`**: for every quote, look for a page with the same source and page number whose (normalized) text contains the (normalized) quote. Output: a list of problem messages; empty means all good. This is a light version of the plan's "quote-existence script" (Phase 9), and it runs before every evaluation.

### `retrieval_metrics.py`

Create `src/ragqa/retrieval_metrics.py`:

```python
"""Phase 5: score retrieval against the golden set with hit@k and MRR."""

from ragqa.golden import Evidence
from ragqa.golden import normalize_whitespace
from ragqa.retrieval import SearchHit


# True when the hit is on the evidence's page and contains its quote.
def hit_contains_evidence(hit: SearchHit, evidence: Evidence) -> bool:
    if hit.source != evidence.source or hit.page != evidence.page:
        return False
    return normalize_whitespace(evidence.quote) in normalize_whitespace(hit.text)


# Rank of the first hit that contains any expected quote, or None if no hit does.
def first_relevant_rank(hits: list[SearchHit], expected_evidence: list[Evidence]) -> int | None:
    for hit in hits:
        for evidence in expected_evidence:
            if hit_contains_evidence(hit, evidence):
                return hit.rank
    return None


# Share of questions whose first relevant hit is at rank k or better.
def hit_at_k(first_ranks: list[int | None], k: int) -> float:
    if len(first_ranks) == 0:
        raise ValueError("no questions to score")
    found = 0
    for rank in first_ranks:
        if rank is not None and rank <= k:
            found = found + 1
    return found / len(first_ranks)


# Average of 1/rank over all questions; a question with no relevant hit counts as 0.
def mean_reciprocal_rank(first_ranks: list[int | None]) -> float:
    if len(first_ranks) == 0:
        raise ValueError("no questions to score")
    total = 0.0
    for rank in first_ranks:
        if rank is not None:
            total = total + 1 / rank
    return total / len(first_ranks)
```

**What the code does, part by part**

- **`hit_contains_evidence(hit, evidence)`**: the definition of "relevant". Same source **and** same page first (the same sentence on another page doesn't count, which matters for citations later), then the normalized quote must appear inside the normalized chunk text. `in` on two strings means "is the first a piece of the second?".
- **`first_relevant_rank(hits, expected_evidence)`**: walks the hits from best to worst and returns the rank of the first relevant one. For multi-chunk questions any expected quote counts. If none matches, it returns `None`, Python's "no value". `int | None` in the type hint means "either a whole number or None".
- **`hit_at_k(first_ranks, k)`**: counts the questions whose rank exists and is `<= k`, then divides by the number of questions. Output between 0.0 and 1.0. An empty list is refused with `ValueError("no questions to score")`; without that check, a golden set with no answerable items crashed with `ZeroDivisionError: division by zero` on my run.
- **`mean_reciprocal_rank(first_ranks)`**: adds `1 / rank` for every found rank (a missing one adds nothing, i.e. 0) and divides by the number of questions, with the same empty-list check. These last two functions are exactly the worked example in 5.1.

### `evaluate_retrieval.py`

Create `src/ragqa/evaluate_retrieval.py`:

```python
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
```

**What the code does, part by part**

- **The imports** reuse every earlier phase: loader, chunker and their defaults, the embedder and its model name, Phase 4's numpy search, and the two new modules.
- **`RESULTS_FOLDER`** and **`K_VALUES`**: where runs are saved (absolute path), and which k values to report.
- **`main()`**, step by step:
  1. **Load and check the golden set.** If any quote can't be found, print every problem and stop with exit code 1. Scoring against a broken golden set would give meaningless numbers, so failing early is the safe choice.
  2. **Build the index** in memory: chunk and embed, as in Phase 3.
  3. **For each question:** embed it with the query prefix, then rank **all** chunks (`top_k=len(chunks)`), so even a relevant chunk at rank 6 gets a real rank.
     - Answerable items: find the first relevant rank and remember it for the metrics. The `if rank is None:` block then chooses what to print: `-` when nothing matched, otherwise the rank turned into text with `str(rank)`.
     - Unanswerable items: no rank (`n/a`); we only record the top hit and its score, as data for Phase 7.
     - The `print` uses fixed widths (`:5`, `:13`, `:>4` = right-aligned in 4 characters) so the columns line up.
  4. **Compute the metrics** with the functions from `retrieval_metrics.py`. `round(..., 4)` keeps 4 decimals. `f"hit@{k}"` builds the names `hit@1`, `hit@3`, `hit@5`.
  5. **Save the run** as one JSON file: when it ran, which golden file, the **settings** that produced it, the metrics, and one row per question. The settings are what make a number meaningful later: "0.857 with 500/100 and bge-small" can be compared; "0.857" alone can't.
     - `mkdir(parents=True, exist_ok=True)` creates `results/retrieval/` if needed and does nothing if it exists.
     - `strftime('%Y%m%d-%H%M%S')` formats the time like `20261015-140148`, so file names sort in time order and never overwrite each other.
     - `json.dump(run, result_file, indent=2)` writes the dictionary as pretty-printed JSON.

## 5.4 Tests

Twelve unit tests. None of them needs the embedding model, so they all run in the quick check (well under a second). They test the **measuring tools themselves**: if hit@k is computed wrong, every number from now on is wrong, so the metrics get tests before anything is measured with them.

| File | Tests | What they protect |
| --- | --- | --- |
| `tests/unit/test_retrieval_metrics.py` | 7 | The meaning of "relevant", hit@k and MRR |
| `tests/unit/test_golden.py` | 5 | The golden file format, and that every quote really exists |

Create `tests/unit/test_retrieval_metrics.py`:

```python
import pytest

from ragqa.golden import Evidence
from ragqa.retrieval import SearchHit
from ragqa.retrieval_metrics import first_relevant_rank
from ragqa.retrieval_metrics import hit_at_k
from ragqa.retrieval_metrics import hit_contains_evidence
from ragqa.retrieval_metrics import mean_reciprocal_rank


# Build a search hit in memory with a given rank, page and text.
def make_hit(rank: int, page: int, text: str) -> SearchHit:
    return SearchHit(rank=rank, chunk_id=f"policy.pdf:p{page}:c0", source="policy.pdf", page=page, score=0.5, text=text)


MEAL_EVIDENCE = Evidence(source="policy.pdf", page=4, quote="The daily meal allowance is 45 euros")


# A hit on the right page that contains the quote counts.
def test_hit_with_the_quote_counts():
    hit = make_hit(1, 4, "4. Daily allowance. The daily meal allowance is 45 euros. Alcohol is never reimbursed.")

    assert hit_contains_evidence(hit, MEAL_EVIDENCE) is True


# A line break inside the quote (as in PDF text) still counts.
def test_line_breaks_in_the_chunk_do_not_matter():
    hit = make_hit(1, 4, "The daily meal\nallowance is 45 euros.")

    assert hit_contains_evidence(hit, MEAL_EVIDENCE) is True


# The same words on a different page do not count.
def test_quote_on_the_wrong_page_does_not_count():
    hit = make_hit(1, 2, "The daily meal allowance is 45 euros.")

    assert hit_contains_evidence(hit, MEAL_EVIDENCE) is False


# The rank of the first relevant hit is returned, skipping irrelevant ones.
def test_first_relevant_rank_skips_irrelevant_hits():
    hits = [
        make_hit(1, 2, "Hotels cost 140 euros."),
        make_hit(2, 4, "The daily meal allowance is 45 euros."),
        make_hit(3, 4, "The daily meal allowance is 45 euros, again."),
    ]

    assert first_relevant_rank(hits, [MEAL_EVIDENCE]) == 2


# No relevant hit at all gives None.
def test_first_relevant_rank_is_none_when_nothing_matches():
    hits = [make_hit(1, 2, "Hotels cost 140 euros.")]

    assert first_relevant_rank(hits, [MEAL_EVIDENCE]) is None


# hit@k and MRR match the worked example from section 5.1.
def test_metrics_match_the_worked_example():
    first_ranks = [1, None, 3, 2]

    assert hit_at_k(first_ranks, 1) == pytest.approx(0.25)
    assert hit_at_k(first_ranks, 3) == pytest.approx(0.75)
    assert mean_reciprocal_rank(first_ranks) == pytest.approx(0.4583, abs=1e-4)


# Scoring an empty list of questions is refused instead of dividing by zero.
def test_metrics_refuse_an_empty_question_list():
    with pytest.raises(ValueError, match="no questions to score"):
        hit_at_k([], 3)
    with pytest.raises(ValueError, match="no questions to score"):
        mean_reciprocal_rank([])
```

Create `tests/unit/test_golden.py`:

```python
import pytest

from ragqa import golden
from ragqa.loaders import DEFAULT_DATA_FOLDER
from ragqa.loaders import load_folder

GOOD_LINE = '{"id": "t1", "question": "Q?", "category": "single_fact", "expected_behavior": "answer", "reference_answer": "A.", "expected_evidence": [{"source": "a.md", "page": 1, "quote": "hello"}], "notes": "", "verified_by": "me", "verified_on": "2026-10-15"}'


# A valid JSON Lines file loads into GoldenItem and Evidence records.
def test_golden_file_loads_into_records(tmp_path):
    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text(GOOD_LINE + "\n", encoding="utf-8")

    items = golden.load_golden_set(golden_path)

    assert len(items) == 1
    assert items[0].id == "t1"
    assert items[0].expected_evidence[0].quote == "hello"


# A line missing a required field is rejected with its line number.
def test_missing_field_raises(tmp_path):
    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text('{"id": "t1", "question": "Q?"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1: missing field 'category'"):
        golden.load_golden_set(golden_path)


# The real golden_v0 file has 8 items with unique IDs.
def test_golden_v0_has_eight_unique_items():
    items = golden.load_golden_set(golden.GOLDEN_V0_PATH)

    item_ids = set()
    for item in items:
        item_ids.add(item.id)
    assert len(items) == 8
    assert len(item_ids) == 8


# Every evidence quote in golden_v0 exists on its page in the real documents.
def test_every_golden_v0_quote_exists_in_the_documents():
    items = golden.load_golden_set(golden.GOLDEN_V0_PATH)
    pages, skipped_files = load_folder(DEFAULT_DATA_FOLDER)

    problems = golden.find_missing_quotes(items, pages)

    assert problems == []


# A line that isn't valid JSON is rejected with its line number.
def test_broken_json_line_raises(tmp_path):
    golden_path = tmp_path / "golden.jsonl"
    golden_path.write_text(GOOD_LINE + "\n" + GOOD_LINE[:-1] + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2: not valid JSON"):
        golden.load_golden_set(golden_path)
```

**What the tests check, and why each matters**

| Test | What it proves | What would go wrong without it |
| --- | --- | --- |
| `hit_with_the_quote_counts` | A chunk on the right page containing the quote is relevant | Nothing would ever count as a hit |
| `line_breaks_in_the_chunk_do_not_matter` | `"meal\nallowance"` still matches `"meal allowance"` | PDF evidence (like q007's) would always look "not found": a false failure |
| `quote_on_the_wrong_page_does_not_count` | Same words, wrong page → not relevant | A chunk could be counted for evidence it can't cite |
| `first_relevant_rank_skips_irrelevant_hits` | Rank 2 is found after an irrelevant rank 1 | Wrong ranks, so wrong MRR |
| `first_relevant_rank_is_none_when_nothing_matches` | "Not found" is `None`, not 0 or an error | A missing answer could look like rank 0, the best rank of all |
| `metrics_match_the_worked_example` | 0.25, 0.75 and 0.4583, exactly the table in 5.1 | The doc and the code could disagree on what the numbers mean |
| `golden_file_loads_into_records` | A valid line becomes a `GoldenItem` with `Evidence` inside | The evaluation couldn't read its questions |
| `missing_field_raises` | A broken line fails with `line 1: missing field 'category'` | A typo would crash later with an unclear `TypeError` |
| `golden_v0_has_eight_unique_items` | 8 items, 8 different IDs | Two items with the same ID would make per-question comparisons ambiguous |
| `every_golden_v0_quote_exists_in_the_documents` | All 8 evidence quotes are found on their pages | A mistyped quote would silently become a "retrieval failure" |
| `metrics_refuse_an_empty_question_list` | Scoring zero questions raises ValueError("no questions to score") | A ZeroDivisionError crash (what happened on my run before the check existed) |
| `broken_json_line_raises` | A broken second line fails with "line 2: not valid JSON" | Python's own JSON error, which points inside one line and not at the file line |

The last test is the important habit: **test your test data**. If someone edits a document or a quote, this test fails *before* any metric is computed, and tells you exactly which item is broken.

A few lines explained:

- `GOOD_LINE` is wrapped in single quotes (`'...'`) because the JSON inside uses double quotes.
- `assert ... is True` / `is False` checks for the exact values `True` and `False`, not just "something truthy".
- `assert problems == []` passes only when the list of problems is empty.

### QA role for this phase

**How important is testing here: very high. This is the core QA phase.** The golden set is your test oracle for retrieval, hit@k and MRR turn "looks good" into numbers, and the baseline is what every later change is compared against. Here the tester also tests the *test itself*: a wrong quote or a metric bug gives false failures or false passes, and every later decision would rest on them.

**What to test**

| What can go wrong | How to check it | Manual or automated | Covered by |
| --- | --- | --- | --- |
| The relevance rule is wrong | A chunk on the right page containing the quote counts | Automated | `hit_with_the_quote_counts` |
| PDF line breaks cause false misses | `"meal\nallowance"` matches `"meal allowance"` | Automated | `line_breaks_in_the_chunk_do_not_matter` |
| A chunk counts for evidence it can't cite | Same words on the wrong page do not count | Automated | `quote_on_the_wrong_page_does_not_count` |
| Ranks are wrong, so MRR is wrong | Rank 2 is found after an irrelevant rank 1; "not found" is `None` | Automated | `first_relevant_rank_...` (2 tests) |
| Code and docs disagree on the maths | 0.25, 0.75, 0.4583 for the worked example | Automated | `metrics_match_the_worked_example` |
| Scoring zero questions crashes | A clear `ValueError` | Automated | `metrics_refuse_an_empty_question_list` |
| A broken golden file fails with an unclear error | Missing field or bad JSON names the line | Automated | `missing_field_raises`, `broken_json_line_raises` |
| Duplicate question IDs | 8 items, 8 different IDs | Automated | `golden_v0_has_eight_unique_items` |
| A mistyped quote looks like a retrieval failure | Every quote exists on its page | Automated | `every_golden_v0_quote_exists_in_the_documents` |
| Questions are unrealistic, ambiguous or all one type | Review each item against the categories | Manual | Step 1 table |
| A reference answer is wrong | Re-check it against the quote, ideally a day later | Manual | Exercise Part B |
| The unanswerable item is actually answerable | Search the documents for its topic | Manual | Exercise Part B |

**Manual testing: how**

1. Write each item while reading the document. Copy the evidence quote word for word, then write the reference answer using only that quote.
2. Compute hit@3 and MRR by hand for 4 questions, then check that the script gives the same numbers.
3. For every miss, walk the stages: is the quote in the loaded text (Phase 1), whole in one chunk (Phase 2), in the top k (Phase 4)? The "Investigating q003" steps in 5.5 show how.
4. Verify every item yourself (Exercise Part B), add your name in `verified_by`, and log every change in `notes`.
5. Save the baseline and write down what it means in plain words (Exercise Part A).

**Automated testing: how**

- `tests/unit/test_retrieval_metrics.py`: tests the relevance rule, ranks and metric maths with small hand-made inputs.
- `tests/unit/test_golden.py`: tests the golden file itself (loads, valid fields, unique IDs, every quote exists). This is *data validation*: automated checks on test data, the same way you would validate a test-data spreadsheet at work.
- The evaluation script is not a pass/fail test. It produces numbers you compare between runs; turning that comparison into automatic thresholds comes in Phase 12.
- Run the tests with the commands in 5.5, on every change to the golden set or the metric code.

**Not covered yet:** 8 items is a tiny sample: one question moves hit@k by about 0.14. Phase 9 grows the set to 15 to 20 human-verified items, and Phase 12 adds automatic regression thresholds.

**How a real system tests this phase, and how we test it now**

In a real system, the golden set and its metrics are a *quality gate*: a check in the build pipeline (CI) that runs the golden set against the real search API on every change, and blocks the change if quality drops. This is the heart of AI testing, and our Phase 5 is a small version of it.

| What is checked | In a real system | In our project now |
| --- | --- | --- |
| Golden set | 100 to 500 questions, versioned in Git or an evaluation tool. It grows from real user questions, with personal data removed | 8 questions you wrote by hand in `golden_v0.jsonl` |
| Quality of the labels | Two or more people label the items. Their agreement is measured (for example with Cohen's kappa, a score for how often two people agree beyond chance). Disagreements are discussed | One person: you. Re-check a day later |
| Metrics | hit@k, MRR, recall@k, plus a score per category and a confidence range | hit@k and MRR, overall |
| How it runs | Automatically in CI, by calling the real API | You run the script by hand (5.5) |
| Pass or fail | A threshold: below the baseline minus a tolerance blocks the change | You compare the numbers by eye |
| History | Results are saved per run, shown on a dashboard, with per-question changes | `results/retrieval/<date>_baseline.json` |
| Small-sample noise | Hundreds of items and repeated runs, so one question moves the score by a fraction of a point | 8 items: one question moves hit@k by about 14 points |

**Example: a quality gate through a search API** (a sketch with made-up endpoints, not run; the item fields follow the plan's golden-set format):

```python
import json
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_FILE = PROJECT_ROOT / "eval" / "datasets" / "golden_v1.jsonl"
API_URL = "http://localhost:8000"
BASELINE_HIT_AT_3 = 0.857
TOLERANCE = 0.05


# Read the golden set: one JSON object per line.
def load_golden_items() -> list[dict]:
    items = []
    with open(GOLDEN_FILE, encoding="utf-8") as golden_file:
        for line in golden_file:
            items.append(json.loads(line))
    return items


# Check whether any hit contains the evidence quote, ignoring line breaks.
def hits_contain_quote(hits: list[dict], quote: str) -> bool:
    for hit in hits:
        if quote in " ".join(hit["text"].split()):
            return True
    return False


# The hit@3 score must not fall below the saved baseline.
def test_hit_at_3_has_not_dropped():
    answerable_items = []
    for item in load_golden_items():
        if item["expected_behavior"] == "answer":
            answerable_items.append(item)

    found = 0
    missed_ids = []
    for item in answerable_items:
        request_body = {"query": item["question"], "top_k": 3}
        response = requests.post(f"{API_URL}/search", json=request_body, timeout=30)
        response.raise_for_status()
        quote = item["expected_evidence"][0]["quote"]
        if hits_contain_quote(response.json()["hits"], quote):
            found += 1
        else:
            missed_ids.append(item["id"])

    hit_at_3 = found / len(answerable_items)
    print("hit@3:", round(hit_at_3, 3), "missed:", missed_ids)
    assert hit_at_3 >= BASELINE_HIT_AT_3 - TOLERANCE
```

- The test sends every answerable golden question to the search API and counts how many get their evidence quote back in the top 3. This is the same hit@3 idea as `hit_with_the_quote_counts`, run through the API.
- Line breaks are squashed before matching, as in Phase 5, so PDF line breaks don't cause false misses.
- It prints the missed question IDs, so a failing run tells the tester *which* questions broke, not only that the score dropped.
- With 7 answerable items, one lost question drops hit@3 by 0.143, so the 0.05 tolerance catches every loss. The tolerance only matters with larger sets.
- Phase 12 builds the real version of this: thresholds in a file, run comparison and a report.

**The gap, in one line:** our numbers are right, but a person runs and reads them. A real system runs them on every change and stops a bad change by itself.

## 5.5 Run it

Run these from your project folder. Ollama doesn't need to be running.

```bash
uv run python -m ragqa.evaluate_retrieval
```

- **What it does:** checks the golden set, embeds the documents, ranks every chunk for each of the 8 questions, prints one line per question and the metrics, and saves the run to `results/retrieval/`.
- **The parts:** `uv run` uses the project's environment; `python -m ragqa.evaluate_retrieval` runs `evaluate_retrieval.py`.
- **Why:** this is the plan's Phase 5 check, "first retrieval baseline number saved".

```bash
uv run pytest tests/unit -v
```

- **What it does:** the quick check, now including the 10 new tests.

**Expected output** (real output from my test run, offline mode):

```text
id    category      rank  top hit (score)
q001  single_fact      1  brightwater_handbook.md:p1:c1 (0.788)
q002  single_fact      1  brightwater_handbook.md:p1:c0 (0.679)
q003  single_fact      6  brightwater_handbook.md:p1:c1 (0.707)
q004  single_fact      1  brightwater_travel_policy.pdf:p2:c0 (0.786)
q005  single_fact      1  brightwater_travel_policy.pdf:p4:c0 (0.789)
q006  single_fact      1  brightwater_faq.txt:p1:c0 (0.702)
q007  multi_chunk      1  brightwater_travel_policy.pdf:p1:c0 (0.649)
q008  unanswerable   n/a  brightwater_travel_policy.pdf:p1:c0 (0.573)

Scored 7 answerable questions (unanswerable ones are listed but not scored)
  hit@1  0.857
  hit@3  0.857
  hit@5  0.857
  mrr    0.881
Saved: /Users/nowgg/Automation/rag_framework/results/retrieval/20261015-140437_baseline.json
```

(Online you'll also see the `HF_TOKEN` warning and the `Loading weights` bar from Phase 3. Your file name has your own date and time.)

**Your first baseline**

| Metric | Value | How it was computed |
| --- | --- | --- |
| hit@1 | **0.857** | 6 of the 7 answerable questions had a relevant chunk at rank 1 |
| hit@3 | **0.857** | The same 6: the one miss (q003) is not in the top 3 either |
| hit@5 | **0.857** | Still 6: q003's relevant chunk is at rank 6 |
| MRR | **0.881** | (6 × 1/1 + 1/6) ÷ 7 = 6.167 ÷ 7 |

**Reading the per-question lines**

| What you see | What it means |
| --- | --- |
| `rank 1` for q001, q002, q004–q007 | The chunk holding the evidence quote came first |
| q007 `rank 1` | The multi-chunk question counts as a hit because **one** of its two quotes (page 1) is in the top chunk. Whether the page-4 quote is also retrieved isn't measured yet |
| **q003 `rank 6`** | The chunk with the whole sentence (handbook `c0`) came 6th of 7. Your first real retrieval failure, examined below |
| q008 `n/a`, top score 0.573 | Not scored. The "best" chunk for a parking question is the travel policy's first page, with a score close to real matches |

**Investigating q003** (the plan's "finding the failing component" walk, steps 1–3):

1. **Ingestion:** is the quote anywhere in the stored text? **Yes**: the golden set check passed.
2. **Chunking:** is the quote split across chunks? **No**: the whole sentence (characters 348–471) is inside handbook `c0`.
3. **Retrieval:** is `c0` in the top k? **No**: here is the full ranking (`uv run python -m ragqa.try_ranking "How many days a year can I work from another country?"`):

| Rank | Score | Chunk | Contains the full quote? |
| --- | --- | --- | --- |
| 1 | 0.707 | handbook `c1` | No: only the sentence's second half ("...working days per calendar year and needs approval...") |
| 2 | 0.623 | travel policy `p2:c0` | No |
| 3 | 0.614 | handbook `c2` | No |
| 4 | 0.591 | travel policy `p4:c0` | No |
| 5 | 0.583 | travel policy `p1:c0` | No |
| 6 | 0.580 | **handbook `c0`** | **Yes** |

So the failure is in **retrieval**, not in loading or chunking. A likely cause: `c0` is 500 characters covering four topics (the title, working hours, remote work, the start of the leave rules), so its single vector is a blur of all of them. `c1` begins with almost nothing but the matching words ("rking days per calendar year and needs approval", cut mid-word by the Phase 2 boundary), so it looks more similar, even though it can't answer the question. This is a real, measurable trade-off of chunk size, and exactly the kind of evidence Phase 13 needs before changing chunking. Notice what the investigation needed: the golden quote, the chunk boundaries from Phase 2, and the full ranking from Phase 4. Each earlier phase was a tool for this moment.

**The saved result file** (`results/retrieval/<date>_baseline.json`, shortened):

```json
{
  "created_at": "2026-10-15T14:01:48",
  "golden_file": "golden_v0.jsonl",
  "settings": {
    "chunk_size": 500,
    "overlap": 100,
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "search": "numpy brute force"
  },
  "metrics": {
    "hit@1": 0.8571,
    "hit@3": 0.8571,
    "hit@5": 0.8571,
    "mrr": 0.881
  },
  "per_question": [
    {
      "id": "q003",
      "category": "single_fact",
      "expected_behavior": "answer",
      "first_relevant_rank": 6,
      "top_chunk_id": "brightwater_handbook.md:p1:c1",
      "top_score": 0.7069
    }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `created_at` | When the run happened (ISO format: date, `T`, time) |
| `golden_file` | Which version of the golden set was used |
| `settings` | Everything that could change the numbers. Two runs are comparable only if you know what differed |
| `metrics` | The four numbers, rounded to 4 decimals |
| `per_question` | One row per item (the real file has all 8). This is what lets Phase 12 report "q003 went from rank 6 to rank 1", not just "MRR went up" |

**Expected output of the new tests** (real):

```text
============================== 12 passed in 0.90s ==============================
```

On my test copy the whole quick check, `uv run pytest tests/unit -q`, gave `36 passed in 0.90s`.

## 5.6 Common errors and debugging

I broke the golden file on purpose in six ways and ran the evaluation each time; the messages below are the real last lines. Two of them led to the fixes now in the code (the JSON line number and the empty-list check).

| What you see | What it means | How to fix it |
| --- | --- | --- |
| `GOLDEN SET ERROR q006: quote not found on brightwater_faq.txt page 1: 'Passwords expire every 90 days'` (then exit code 1) | The quote has a typo, or the document was edited, so the words don't exist on that page | Copy the quote again, character for character, from the document |
| `GOLDEN SET ERROR q005: quote not found on brightwater_travel_policy.pdf page 2: ...` | The quote is real but the **page** (or source name) is wrong | Check the page with Phase 1's loader output; PDF pages count from 1 and blank page 3 has no text |
| `ValueError: golden_v0.jsonl line 1: not valid JSON (Expecting ',' delimiter: line 2 column 1 (char 381))` | Line 1 of the file isn't valid JSON: here its closing `}` was missing. The part in brackets is Python's own message; its "line 2" counts *inside that one line*, so trust the file line at the start | Fix the named line. Common causes: missing `}` or `"`, a trailing comma, or straight quotes typed as curly quotes (`“”`) |
| `ValueError: golden_v0.jsonl line 1: not valid JSON (Expecting property name enclosed in double quotes ...)` | One item was split over two lines. In JSON Lines, each item must stay on **one** line | Join the two lines back into one |
| `ValueError: golden_v0.jsonl line 1: missing field 'category'` | A required field is missing or misspelled (`catgory`) | Add the field; the 9 names are in `REQUIRED_FIELDS` |
| `ValueError: no questions to score` | The golden set has no item with `expected_behavior` `"answer"` (I tested a file with only q008) | Add answerable items, or check `expected_behavior` is spelled `answer` |
| `FileNotFoundError: [Errno 2] No such file or directory: '.../eval/datasets/golden_v0.jsonl'` | The golden file is missing or has another name | Create it as in 5.2, Step 2, with exactly that name |
| Your metrics differ from 0.857 / 0.881 | Something that produces the numbers is different: documents, chunk settings, golden items, or model version | Compare the `settings` in your result file with the doc's; then compare your `per_question` ranks with the table in 5.5 |
| `results/` doesn't appear in `git status` | Correct: `results/` is git-ignored since Phase 0 | Copy the baseline into a committed note (exercise, Part A) |

**Debugging techniques for this phase**

```bash
uv run python -c "from ragqa.golden import GOLDEN_V0_PATH, find_missing_quotes, load_golden_set; from ragqa.loaders import DEFAULT_DATA_FOLDER, load_folder; print(find_missing_quotes(load_golden_set(GOLDEN_V0_PATH), load_folder(DEFAULT_DATA_FOLDER)[0]))"
```

- **What it does:** checks only the golden set against the documents, without loading the model. `[]` means every quote was found; otherwise you get the list of problems.
- **Why:** fast (1 second on my run, against about 16 s for the full evaluation). Run it after every edit to the golden file.
- **The parts:** `load_folder(...)[0]` takes the first of the two values `load_folder` returns, the pages.

```bash
uv run python -m ragqa.try_ranking "How many days a year can I work from another country?"
```

- **What it does:** Phase 3's exercise script, reused: the full ranking of all 7 chunks for one question.
- **Why:** when a question has a bad rank, this shows *where* its chunk landed and what beat it, the first step of every investigation (as for q003 in 5.5).

```bash
uv run python -c "import json; from ragqa.golden import PROJECT_ROOT; [print(path.name, json.load(open(path))['metrics']) for path in sorted((PROJECT_ROOT / 'results' / 'retrieval').glob('*.json'))]"
```

- **What it does:** prints the metrics of every saved run, oldest first. Real output from my test copy (three runs, same settings):

```text
20261015-140148_baseline.json {'hit@1': 0.8571, 'hit@3': 0.8571, 'hit@5': 0.8571, 'mrr': 0.881}
20261015-140437_baseline.json {'hit@1': 0.8571, 'hit@3': 0.8571, 'hit@5': 0.8571, 'mrr': 0.881}
20261015-140934_baseline.json {'hit@1': 0.8571, 'hit@3': 0.8571, 'hit@5': 0.8571, 'mrr': 0.881}
```

- **Why:** identical numbers across runs prove the evaluation is **deterministic**: same inputs, same score. That's what makes a baseline trustworthy. If two runs with the same settings ever differ, investigate before comparing anything.
- **The parts:** `.glob('*.json')` finds every JSON file in the folder; `sorted(...)` puts them in time order (the file names start with the date); the `[... for path in ...]` one-liner loop is fine in the Terminal, though project code writes loops out in full.

## 5.7 Exercise: own your golden set and your baseline

**Part A — Record the baseline where Git keeps it.** `results/` is git-ignored, so copy your numbers into a small note that *is* committed. Create `docs/experiments/retrieval_baseline.md` (the folder exists since Phase 0):

```markdown
## Retrieval baseline (Phase 5)

| Setting | Value |
| --- | --- |
| Date | |
| Golden set | golden_v0.jsonl (7 answerable + 1 unanswerable) |
| Chunk size / overlap | 500 / 100 |
| Embedding model | BAAI/bge-small-en-v1.5 |
| Search | numpy brute force |

| Metric | Value |
| --- | --- |
| hit@1 | |
| hit@3 | |
| hit@5 | |
| MRR | |

Known failure: q003 (rank ...). Cause: ...
```

Fill it from your own run and write two sentences on q003 in your own words. This becomes the first entry of the plan's `docs/experiments/` evidence, and the first line of the portfolio's results table.

**Part B — Verify the golden set yourself.** The plan's ground-truth method says a golden set is only as good as the person who checked it. For each of the 8 items:

1. Open the document named in `source` and find the quote on that page.
2. Check the reference answer says nothing the quote doesn't support.
3. Put your name in `verified_by` and today's date in `verified_on`.

Then run the quick check from 5.6 (it should print `[]`). Did you find anything you'd write differently? Write it in the item's `notes`.

**Part C — Grow the set by two items and watch the numbers move.** Add these two lines to the end of `golden_v0.jsonl`:

```json
{"id": "q009", "question": "What is my yearly learning budget?", "category": "single_fact", "expected_behavior": "answer", "reference_answer": "1,200 euros per year; it does not roll over.", "expected_evidence": [{"source": "brightwater_handbook.md", "page": 1, "quote": "a learning budget of 1,200 euros per year"}], "notes": "", "verified_by": "Akshit", "verified_on": "2026-10-15"}
{"id": "q010", "question": "Does Brightwater pay for a gym membership?", "category": "unanswerable", "expected_behavior": "refuse", "reference_answer": "", "expected_evidence": [], "notes": "No gym benefit in any document.", "verified_by": "Akshit", "verified_on": "2026-10-15"}
```

Run `uv run python -m ragqa.evaluate_retrieval` again. My real result:

|  | Before (8 items) | After (10 items) |
| --- | --- | --- |
| Answerable questions scored | 7 | 8 |
| hit@1 / hit@3 / hit@5 | 0.857 | 0.875 |
| MRR | 0.881 | 0.896 |
| q009 rank | — | 1 (handbook `c2`, 0.739) |
| q010 top score | — | 0.605 (handbook `c0`) |

Answer these in a few words each:

1. The search code didn't change, yet hit@1 went up from 0.857 to 0.875. Why? What does this tell you about comparing two numbers measured on **different** golden sets?
2. With only 7–8 answerable questions, one question moves hit@k by about 12–14 percentage points. Is 0.857 vs 0.875 a real improvement? (The plan's risk table: "with 20 items, one sample is 5 percentage points".)
3. The unanswerable gym question's top score (0.605) is *higher* than the parking question's (0.573), and close to real matches. What does that say about using a fixed score threshold to refuse in Phase 7?

Afterwards, decide whether to keep q009 and q010 in v0 (if you do, the test `test_golden_v0_has_eight_unique_items` must change to 10, on purpose) or remove them again and save them for golden v1 in Phase 9.

**Part D — Write one test yourself.** Add `test_multi_chunk_question_uses_the_earliest_matching_hit` to `tests/unit/test_retrieval_metrics.py`. It pins down how multi-chunk questions are scored:

- Make two `Evidence` records: one for page 1 ("approved in the Compass portal"), one for page 4 ("Submit receipts in Compass").
- Make three `SearchHit`s in rank order: rank 1 on page 2 with unrelated text, rank 2 on page 4 containing the page-4 quote, rank 3 on page 1 containing the page-1 quote.
- Assert that `first_relevant_rank(hits, [page_one, page_four])` is `2`: the earliest hit holding **any** of the quotes wins, even though the page-1 quote is listed first.

It passed on my test copy.

## 5.8 Done criteria

- [ ] `eval/datasets/golden_v0.jsonl` exists with 8 items, and the quick check in 5.6 prints `[]`
- [ ] `uv run python -m ragqa.evaluate_retrieval` prints the per-question table and the four metrics, and saves a file in `results/retrieval/`
- [ ] Your baseline matches mine (hit@1/3/5 0.857, MRR 0.881), or you can explain the difference from your `settings` and `per_question` rows
- [ ] `uv run pytest tests/unit -v` passes, including the 12 new tests (13 with your own)
- [ ] `docs/experiments/retrieval_baseline.md` holds your numbers and your q003 explanation (Part A)
- [ ] Every golden item carries your name in `verified_by` (Part B)
- [ ] Your answers to the three Part C questions are written down
- [ ] Your work is committed: `git add .` then `git commit -m "Phase 5: golden set v0 and retrieval baseline"`, then `git push`. The golden file and the baseline note go in; `results/` stays out
- [ ] Phase 5 ticked in the phase tracker

When these pass, share your baseline note and Part C answers, and we'll start Phase 6: sending the retrieved chunks to the LLM with a grounded prompt.
