# RAG Quality Lab — Phase 7: Citations, refusal, errors, tracing

Oct 18, 2026 · @Akshit

## Phase tracker

Tick a phase only when its verification checks pass.

- [ ] Phase 0 — Environment, Git, Ollama, hardware baseline ([separate doc](https://claude.ai/code/artifact/b32269ff-ce8f-4186-b984-8cf68f206b1b))
- [ ] Phase 1 — Load documents with metadata ([separate doc](https://claude.ai/code/artifact/7df3b55e-6623-464f-8990-7106ec3af43f))
- [ ] Phase 2 — Chunking ([separate doc](https://claude.ai/code/artifact/4b5e335a-21aa-435a-98c4-3d6105208c7f))
- [ ] Phase 3 — Local embeddings ([separate doc](https://claude.ai/code/artifact/a37f9484-2b02-439c-bcbd-42d9145c7a5a))
- [ ] Phase 4 — Retrieval (numpy brute force, then Chroma) ([separate doc](https://claude.ai/code/artifact/5c0d6903-3508-4447-89e0-79e7f586b656))
- [ ] Phase 5 — Golden v0 + hit@k / MRR ([separate doc](https://claude.ai/code/artifact/af02e615-699e-4072-b059-e6fd541a7eac))
- [ ] Phase 6 — Grounded generation with Ollama ([separate doc](https://claude.ai/code/artifact/dc112016-b8cb-489d-b01d-377efa3083a1))
- [ ] Phase 7 — Citations, refusal, errors, tracing (this doc)
- [ ] Phase 8 — Consolidated pytest suite + robustness fixtures ([separate doc](https://claude.ai/code/artifact/8dd46bfa-af84-4bf8-a1bb-8537f5671950))
- [ ] Phase 9 — Golden v1, human-verified
- [ ] Phase 10 — RAGAS with a local judge
- [ ] Phase 11 — DeepEval with Ollama + G-Eval
- [ ] Phase 12 — Regression runner, thresholds, reports
- [ ] Phase 13 — Evidence-driven improvement (+ optional UI)
- [ ] Phase 14 — Documentation and portfolio polish

**Goal of this phase:** turn the manual review from Phase 6 into **automatic checks**. Code now checks every golden answer for valid citations, a citation to the right evidence, and a clean refusal when one is expected. Bad files and a stopped Ollama give **clear, typed errors** instead of crashes. Every question leaves a **per-stage trace** (retrieval, prompt, generation, checks) in `results/traces/`. On my Mac, 7 of 8 golden questions passed; the one fail (q003) was labelled `retrieval` automatically, the same verdict you reached by hand in Phase 6. Previous phase: [Phase 6](https://claude.ai/code/artifact/dc112016-b8cb-489d-b01d-377efa3083a1).

## 7.1 Concepts

In Phase 6 you read 8 answers and judged them by eye. That doesn't scale: Phase 9 will have 20 questions, Phase 12 will rerun them after every change. This phase writes your review rules down as code, so a computer applies them the same way every time. It also makes failures **easy to explain**: when something breaks, the trace says where.

**Words you'll meet in this phase**

| Term | Plain meaning | In our project |
| --- | --- | --- |
| Deterministic check | A check written in plain code. Same input, same verdict, every time. No LLM judge needed | The 5 checks in `check_answer()` |
| Citation validity | Every `[S#]` in the answer is a label we really sent. `[S7]` when we sent 3 chunks is invalid: the model made it up | `citations_valid` |
| Citation accuracy | The cited chunk really contains the expected evidence quote from the golden set | `cites_evidence` |
| Refusal sentinel | A fixed sentence that means "I can't answer". A *sentinel* is a special value code can look for | `I don't know based on the documents.` |
| Clean refusal | The answer **starts** with the sentinel and cites nothing. A refusal that also cites `[S1]` is a half-answer | `refusal_is_clean` |
| Exception | Python's way of saying "something went wrong": the normal flow stops until some code *catches* it with `try`/`except` | `DocumentLoadError`, `GenerationError` |
| Typed error (custom exception) | Our own exception class, so code can catch *exactly* the failures it expects and let real bugs crash | `errors.py` |
| Trace | A record of what happened to one request at every stage: inputs, outputs, times | One JSON line per question in `results/traces/` |
| Latency | How long a step took | `seconds` per stage in the trace |
| `done_reason` | Ollama's reason for stopping: `"stop"` = finished by itself, `"length"` = hit a length limit | Saved in the trace; anything but `stop` is a warning |
| Exit code | A number a program returns when it ends: 0 = success, anything else = failure. Scripts and CI read it | `check_golden` exits with 1 if any question fails |

**The three jobs of this phase**

| Job | Question it answers | Before (Phase 6) | Now (Phase 7) |
| --- | --- | --- | --- |
| Citations and refusal | Is this answer well-formed and grounded? | You read each answer | `check_answer()` gives 5 verdicts per answer |
| Errors | What happens when a file or Ollama is broken? | Bad files skipped (Phase 1); a stopped Ollama crashed with a long traceback | Typed errors, one clear line, the run continues |
| Tracing | Where did a failure happen? | `evidence at [S#]` printed, prompt saved | One trace per question with every stage, its time, warnings and a `failing_stage` guess |

**The five checks**

Each check returns `True` (pass), `False` (fail) or `None` (doesn't apply to this answer). A question passes when no check is `False`.

| Check | Applies when | Passes when | Catches |
| --- | --- | --- | --- |
| `behavior_matches` | Always | It refused when the golden item says `refuse`, and answered when it says `answer` | Guessing on unanswerable questions; refusing answerable ones |
| `citations_valid` | Always | No cited label is outside the labels we sent | Made-up sources like `[S7]` |
| `answer_is_cited` | It answered | At least one valid citation | Uncited answers (can't be traced to a chunk) |
| `cites_evidence` | It answered, and the item has evidence quotes | A cited chunk contains an expected quote (Phase 5's `hit_contains_evidence()`) | Citing the wrong chunk |
| `refusal_is_clean` | It refused | The answer starts with the sentinel and has no citations | Half-refusals: "I don't know… but it's 27 days \[S1\]" |

The plan's citation-accuracy rule has two parts: "cited `[S#]` ⊆ retrieved labels" (that's `citations_valid`) and "the cited chunk contains the evidence quote" (that's `cites_evidence`). ⊆ means "is a subset of": every cited label is among the sent ones.

**Decision from Phase 6: how strict is the refusal check?** Phase 6 left open whether q008's answer, `I don't know based on the documents. There is no mention of a parking policy…`, counts. We accept it: it **starts** with the sentinel and cites nothing, so code can still detect it and no fact leaked. We reject a refusal that cites a source, because that means the model refused *and* answered. Exact-match ("nothing but the sentence") would fail a harmless extra sentence and teach you to ignore the check.

**Errors: expected failures vs bugs.** Some failures are part of normal life: a user drops an empty file into the folder, or Ollama isn't started. Others are bugs in our code. They need different handling:

| Kind | Example | What should happen |
| --- | --- | --- |
| Expected, per file | Empty `.md`, corrupted `.pdf`, `.docx` | Skip that file, record the reason, carry on (`DocumentLoadError`) |
| Expected, per question | Ollama stopped, too slow, model missing | Record an error trace for that question, carry on, fail the run at the end (`GenerationError`) |
| Bug | A `KeyError` inside our loader | **Crash loudly.** Hiding it would make results silently wrong |

That's why Phase 1's `except ValueError` becomes `except DocumentLoadError`. `ValueError` is a very general Python error; catching it could hide a real bug. Our own class is specific. `DocumentLoadError` still *is* a `ValueError` (it inherits from it), so the API Lab code that catches `ValueError` keeps working.

**Tracing: why per stage.** The plan's design choice: "each sample keeps retrieved chunk IDs and scores, the exact prompt, raw output and per-stage latency, so a failure can be pinned on one component". A single score ("7 of 8") tells you *that* something failed. The trace tells you *where*: was the evidence retrieved, what did the prompt look like, did the model stop on its own?

**How the trace finds the failing stage** (the plan's "Finding the failing component", steps 3–4, in code):

1. All checks passed → no failing stage.
2. The question should be answered, but no sent chunk contains the evidence → `retrieval`.
3. Anything else → `generation` (the evidence was there, or the model should have refused).

It's a first guess, not proof. Ingestion and chunking problems also show up as `retrieval` here; Phase 5's `find_missing_quotes()` tells them apart.

**How it connects to the architecture.** Phase 7 adds the plan's **post-processing** stage (citation parsing and checks) after generation, and wraps every stage in a trace:

1. **Indexing** (Phases 1–3): loaders now raise `DocumentLoadError`; skipped files go into the run settings.
2. **Query flow** (Phases 4–6): `answer_question()` now times retrieval too and keeps `done_reason` and the raw output.
3. **Post-processing** (new): `check_answer()` turns the answer into 5 verdicts.
4. **Evaluation**: `check_golden` is the first pass/fail runner. Phase 12's regression runner grows out of it; Phases 10–11's judges will read the same saved answers.

**Design choices for this phase**

| Question | Our choice | Why |
| --- | --- | --- |
| New module for citations? | No: extend `citations.py` from the API Lab | It already finds labels and refusals; the API uses it |
| Where does the Ollama error handling go? | `call_generator()` in `generation.py` | Next to the call it wraps. Putting it in `ask.py` made the unit tests 5 s slower, because importing `ask.py` loads PyTorch |
| Change `generate_answer()`? | No | The API Lab maps each `requests` error to its own HTTP status (503, 504, 502) and needs the original errors |
| Run continues after an Ollama error? | Yes, one error trace per question | One bad question shouldn't hide the results of the others. The exit code still says the run failed |
| Trace format | JSON Lines, settings on line 1, like Phase 6's answers file | Same tools read both; one line per question is easy to `grep` |

## 7.2 Setup steps

**Nothing new to install.** Every check in this phase is plain Python (`re`, `json`) plus code from earlier phases. Run every command from your project folder.

### Step 1 — Check that Phase 6 works

```bash
uv run pytest tests/unit -q
```

- **What it does:** runs the quick unit tests (no Ollama needed).
- **Why:** Phase 7 changes `loaders.py`, `generation.py`, `ask.py` and `citations.py`. A green run now means any red later comes from this phase.
- **The parts:** `-q` (quiet) prints one dot per test and a summary line.

### Step 2 — Create the bad-file fixtures

A *fixture* here is a small test file that exists only to be tested against. The plan's folder layout already has `data/adversarial/` (it holds only `.gitkeep` so far). Add three files that the loader must skip:

```bash
printf '   \n\n' > data/adversarial/empty.md
```

- **What it does:** creates a Markdown file with only spaces and line breaks.
- **The parts:** `printf` prints text exactly as given (`\n` = line break); `>` writes it into the file.

```bash
printf 'this is not really a pdf' > data/adversarial/corrupted.pdf
```

- **What it does:** creates a file with a `.pdf` name but plain-text content, like a broken download.

```bash
printf 'pretend word file' > data/adversarial/notes.docx
```

- **What it does:** creates a file with an extension we don't support.
- **Why all three:** they are the plan's robustness cases "empty / corrupted / unsupported files". Phase 8 adds injection fixtures to the same folder.

### Step 3 — Ollama for the integration steps

The golden check and integration tests call the model. If the Ollama app isn't running, start the server in its own terminal tab:

```bash
ollama serve
```

- **What it does:** starts Ollama on `http://localhost:11434`.
- **Why:** `check_golden` and `tests/integration/` send requests there. You'll also stop it on purpose in 7.6 to see the error handling.

## 7.3 Code: errors, checks, traces and the golden check

| File | Job | New in this phase? |
| --- | --- | --- |
| `errors.py` | Two error classes: `DocumentLoadError`, `GenerationError` | Yes |
| `loaders.py` | Raise `DocumentLoadError`; skip only that error | Changed (7 lines) |
| `generation.py` | New `call_generator()`: turns network/HTTP errors into `GenerationError` | Changed (one function added) |
| `ask.py` | Times retrieval too; keeps `done_reason` and the raw output; clean error message | Changed |
| `citations.py` | The 5 checks: `check_answer()` and helpers | Changed (functions added) |
| `tracing.py` | Builds and saves one trace per question; warnings; failing-stage guess | Yes |
| `check_golden.py` | Runs the whole golden set through answer → checks → trace | Yes |

The import chain: `check_golden.py` → `ask.py` → `generation.py` → `errors.py`; and `check_golden.py` → `citations.py`, `tracing.py`. Phase 6's `answer_golden.py` still works unchanged; it just ignores the new fields.

### `errors.py`

Create `src/ragqa/errors.py`:

```python
"""Phase 7: our own error types, so callers can tell expected failures from bugs."""


# A document we can't use (empty, corrupted, unsupported). The run skips it and goes on.
class DocumentLoadError(ValueError):
    pass


# The LLM call failed (Ollama down, too slow, or an error status). The question can't be answered.
class GenerationError(RuntimeError):
    pass
```

**What the code does, part by part**

- `class DocumentLoadError(ValueError)`: a new exception type. The name in brackets is its *parent class*: our error is a special kind of `ValueError`, so `except ValueError` still catches it (the API Lab relies on this), but `except DocumentLoadError` catches **only** it.
- `pass` means "nothing more to add": the class needs no code, only a name. The name is the whole point: it says *what kind* of failure happened.
- `GenerationError(RuntimeError)`: `RuntimeError` is Python's general "something went wrong while running" error. The model call is a runtime failure, not a bad value.

### `loaders.py` (changed)

In `src/ragqa/loaders.py`, add the import and replace every `raise ValueError(` with `raise DocumentLoadError(`. Your password-protected PDF check from Phase 1 stays; only its error type changes. The changed lines:

```python
import pymupdf

from ragqa.errors import DocumentLoadError
```

```python
        raise DocumentLoadError(f"{file_path.name}: not valid UTF-8 text ({error})") from error
        raise DocumentLoadError(f"{file_path.name}: file has no text")
        raise DocumentLoadError(f"{file_path.name}: not a readable PDF ({error})") from error
        raise DocumentLoadError(f"{file_path.name}: PDF is password-protected")
        raise DocumentLoadError(f"{file_path.name}: PDF has no extractable text (empty or scanned)")
    raise DocumentLoadError(f"{file_path.name}: unsupported file type '{extension}'")
```

And in `load_folder()`:

```python
        try:
            pages = load_document(file_path)
        except DocumentLoadError as error:
            # Only our own "bad document" error is skipped. Any other error is a bug and stops the run.
            skipped_files.append(str(error))
            continue
        all_pages.extend(pages)
```

**What the code does, part by part**

- Every "this file can't be used" case now raises the same specific type, with the file name first in the message.
- `except DocumentLoadError` skips exactly those cases. A different error (say a `KeyError` from a typo in our code) is **not** caught, so the run stops with a traceback. That's what you want for a bug: a loud failure, not a missing document nobody notices.
- `from error` keeps the original error attached (Python prints it as "The above exception was the direct cause…"), so you can still see PyMuPDF's own message when debugging.

### `generation.py` (changed)

Add one import at the top and one function at the end of `src/ragqa/generation.py`. `generate_answer()` stays exactly as in Phase 6.

```python
import requests

from ragqa.errors import GenerationError
from ragqa.retrieval import SearchHit
```

```python
# Call Ollama with our settings, turning network and HTTP errors into one GenerationError with a clear message.
def call_generator(prompt: str) -> dict:
    try:
        return generate_answer(DEFAULT_OLLAMA_URL, DEFAULT_GENERATOR_MODEL, SYSTEM_RULES, prompt)
    except requests.ConnectionError as error:
        raise GenerationError(f"Ollama is not reachable at {DEFAULT_OLLAMA_URL}. Start it with: ollama serve") from error
    except requests.Timeout as error:
        raise GenerationError(f"Ollama did not answer within {GENERATION_TIMEOUT_SECONDS} s") from error
    except requests.HTTPError as error:
        raise GenerationError(f"Ollama returned an error: {error}") from error
```

**What the code does, part by part**

- Input: the prompt. Output: Ollama's reply dictionary, exactly like `generate_answer()`, using our default URL, model and rules.
- The three `except` blocks catch the three ways the call can fail and re-raise them as **one** type with a message a person can act on:

| `requests` error | When | Our message |
| --- | --- | --- |
| `ConnectionError` | Nothing listens on port 11434 (Ollama stopped) | `Ollama is not reachable at http://localhost:11434. Start it with: ollama serve` |
| `Timeout` | No answer within 180 s (memory pressure) | `Ollama did not answer within 180 s` |
| `HTTPError` | Ollama answered with an error status, e.g. 404 model not found (from `raise_for_status()`) | `Ollama returned an error: 404 Client Error…` |

- Callers now need only one `except GenerationError`. The API Lab doesn't use this function; it still calls `generate_answer()` and maps each `requests` error to its own HTTP status.

### `ask.py` (changed)

In `src/ragqa/ask.py`, the imports from `generation` become:

```python
from ragqa.errors import GenerationError
from ragqa.generation import CONTEXT_WINDOW_TOKENS
from ragqa.generation import DEFAULT_GENERATOR_MODEL
from ragqa.generation import build_prompt
from ragqa.generation import call_generator
from ragqa.generation import label_hits
```

(`DEFAULT_OLLAMA_URL`, `SYSTEM_RULES` and `generate_answer` are no longer imported here; `call_generator()` uses them inside `generation.py`.) Replace `answer_question()` with:

```python
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

    return {
        "question": question,
        "labelled_hits": labelled_hits,
        "prompt": prompt,
        "answer": llm_reply["response"].strip(),
        "raw_output": llm_reply["response"],
        "done_reason": llm_reply.get("done_reason"),
        "retrieval_seconds": retrieval_seconds,
        "generation_seconds": generation_seconds,
        "prompt_tokens": llm_reply["prompt_eval_count"],
        "output_tokens": llm_reply["eval_count"],
    }
```

In `main()`, wrap the call:

```python
    try:
        result = answer_question(model, chunks, chunk_vectors, question, DEFAULT_TOP_K)
    except GenerationError as error:
        print(f"ERROR: {error}")
        sys.exit(2)
```

**What the code does, part by part**

- **Retrieval is timed now** (`retrieval_seconds`: embedding the question + searching). With both times, the trace shows which stage is slow.
- `call_generator(prompt)` replaces the direct `generate_answer(...)` call, so a stopped Ollama becomes a `GenerationError`.
- Two new fields: `raw_output` (the model's text before `.strip()`, exactly as received; useful when whitespace or a stray line matters) and `done_reason` (`.get()` returns `None` instead of crashing if a field is missing).
- In `main()`: on a `GenerationError`, print one clear line and exit with code 2 ("failed"), instead of a 40-line traceback.

### `citations.py` (changed)

Replace `src/ragqa/citations.py` with this version. The first two functions are unchanged from the API Lab; `is_refusal()` behaves the same but shares a helper now.

```python
"""Check an answer's [S1]-style citations and spot refusals."""

import re

from ragqa.generation import REFUSAL_TEXT
from ragqa.golden import Evidence
from ragqa.retrieval import SearchHit
from ragqa.retrieval_metrics import hit_contains_evidence

CITATION_PATTERN = re.compile(r"\[S\d+\]")


# Find every [S1]-style label in an answer, first appearance first, no repeats.
def find_cited_labels(answer_text: str) -> list[str]:
    cited_labels = []
    for label in CITATION_PATTERN.findall(answer_text):
        if label not in cited_labels:
            cited_labels.append(label)
    return cited_labels


# Split cited labels into ones we really sent and ones we did not.
def split_citations(cited_labels: list[str], sent_labels: list[str]) -> tuple[list[str], list[str]]:
    valid_labels = []
    invalid_labels = []
    for label in cited_labels:
        if label in sent_labels:
            valid_labels.append(label)
        else:
            invalid_labels.append(label)
    return valid_labels, invalid_labels


# Lower-case the text and turn curly apostrophes into straight ones.
def normalize_for_refusal(text: str) -> str:
    # Models sometimes write a curly apostrophe (’) instead of a straight one (').
    return text.lower().replace("’", "'").strip()


# True when the answer is the refusal sentence (case and punctuation ignored).
def is_refusal(answer_text: str) -> bool:
    refusal_core = REFUSAL_TEXT.lower().rstrip(".")
    return refusal_core in normalize_for_refusal(answer_text)


# True when the answer opens with the refusal sentence, not just mentions it later.
def starts_with_refusal(answer_text: str) -> bool:
    refusal_core = REFUSAL_TEXT.lower().rstrip(".")
    return normalize_for_refusal(answer_text).startswith(refusal_core)


# True when at least one cited chunk contains one of the expected evidence quotes.
def cites_expected_evidence(
    valid_labels: list[str],
    labelled_hits: list[tuple[str, SearchHit]],
    expected_evidence: list[Evidence],
) -> bool:
    for label, hit in labelled_hits:
        if label not in valid_labels:
            continue
        for evidence in expected_evidence:
            if hit_contains_evidence(hit, evidence):
                return True
    return False


# Run the Phase 7 checks on one answer. Each check is True (pass), False (fail) or None (does not apply).
def check_answer(
    answer_text: str,
    labelled_hits: list[tuple[str, SearchHit]],
    expected_behavior: str,
    expected_evidence: list[Evidence],
) -> dict:
    sent_labels = []
    for label, hit in labelled_hits:
        sent_labels.append(label)
    cited_labels = find_cited_labels(answer_text)
    valid_labels, invalid_labels = split_citations(cited_labels, sent_labels)
    refused = is_refusal(answer_text)

    checks = {}
    if expected_behavior == "refuse":
        checks["behavior_matches"] = refused
    else:
        checks["behavior_matches"] = not refused
    checks["citations_valid"] = len(invalid_labels) == 0

    if refused:
        checks["answer_is_cited"] = None
        checks["cites_evidence"] = None
        checks["refusal_is_clean"] = starts_with_refusal(answer_text) and len(cited_labels) == 0
    else:
        checks["answer_is_cited"] = len(valid_labels) > 0
        if len(expected_evidence) == 0:
            checks["cites_evidence"] = None
        else:
            checks["cites_evidence"] = cites_expected_evidence(valid_labels, labelled_hits, expected_evidence)
        checks["refusal_is_clean"] = None

    return {
        "refused": refused,
        "cited_labels": cited_labels,
        "invalid_labels": invalid_labels,
        "checks": checks,
    }


# True when no check failed (checks that do not apply are ignored).
def all_checks_pass(checks: dict) -> bool:
    for check_name, result in checks.items():
        if result is False:
            return False
    return True
```

**What the code does, part by part**

- `CITATION_PATTERN`: a *regular expression* (a text pattern). `\[` and `\]` are literal square brackets, `S` is the letter S, `\d+` is one or more digits. So it matches `[S1]` and `[S12]`, but not `[s1]` or `[S 1]`. `re.compile()` prepares the pattern once so it can be reused.
- `find_cited_labels()` / `split_citations()`: from the API Lab. Input: the answer, and the labels we sent. Output: the cited labels split into valid (sent) and invalid (made up).
- `normalize_for_refusal()`: lower-cases, swaps the curly apostrophe for a straight one, trims the ends. Both refusal functions use it, so they always agree.
- `is_refusal()` asks "does the sentence appear **anywhere**?" (used for the `refused` flag, and by the API). `starts_with_refusal()` asks "does the answer **open** with it?" (used for `refusal_is_clean`). `.rstrip(".")` drops the final full stop, so `I don't know based on the documents` matches with or without it.
- `cites_expected_evidence()`: for each **validly cited** chunk, it reuses Phase 5's `hit_contains_evidence()` (same source, same page, quote inside the chunk text). `continue` skips chunks that weren't cited. Returns `True` at the first match.
- `check_answer()`: the heart of the phase.
  1. Collect the sent labels, find the cited ones, split them.
  2. `behavior_matches`: compares what happened (`refused`) with what the golden item expects.
  3. `citations_valid`: no invalid labels.
  4. Then two branches: a refusal gets `refusal_is_clean`, and the answer checks become `None`; an answer gets `answer_is_cited` and `cites_evidence` (only when the item has evidence quotes; q008 has none).
  5. Output: a dictionary with `refused`, the cited and invalid labels, and the 5 verdicts. Everything a trace needs.
- `all_checks_pass()`: `result is False` (not `== False`, not `not result`) so that `None` ("doesn't apply") never counts as a fail.

### `tracing.py`

Create `src/ragqa/tracing.py`:

```python
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
```

**What the code does, part by part**

- `find_warnings()`: two early-warning signs from Phase 6, now automatic.
  - Prompt at 90% (3,687 tokens) or more of the 4,096 window: Ollama may have cut it **silently**, like the `num_ctx` 256 demo in Phase 6.
  - `done_reason` other than `"stop"`: the model didn't finish on its own (`"length"` = it hit a token limit mid-sentence).
  - Warnings don't fail a question; they tell you to look.
- `diagnose_stage()`: the three rules from 7.1 as code. `str | None` means "returns text or `None`".
- `build_trace()`: inputs are everything known about one question; output is one dictionary with four `stages`, in pipeline order:

| Stage | What it keeps | Why |
| --- | --- | --- |
| `retrieval` | Time; each hit's label, chunk ID, source, page, score; where the evidence ranked | Was the evidence found, and how confidently? |
| `prompt` | Length in characters; the exact text | Reproduce the exact request; spot a broken layout |
| `generation` | Time, model, token counts, window, `done_reason`, raw output | Slow? Cut off? What did the model really write? |
| `checks` | `refused`, cited and invalid labels, the 5 verdicts | Why it passed or failed |

- `round(..., 4)` / `round(..., 3)` / `round(..., 2)`: keep the file readable; scores need 4 decimals to compare, times don't.
- `build_error_trace()`: same top-level shape, but `status: "error"`, empty `stages`, and an `error` with the stage, the error's class name (`type(error).__name__`, e.g. `"GenerationError"`) and its message. Same shape means one script can read both kinds.
- `write_traces()`: like Phase 6's answers file. `mkdir(parents=True, exist_ok=True)` creates `results/traces/` if needed and doesn't complain if it exists.

### `check_golden.py`

Create `src/ragqa/check_golden.py`:

```python
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
```

**What the code does, part by part**

- **Setup:** load the golden set and build the index once (Phase 6's `build_index()`). Then `load_folder()` once more, only to keep the list of skipped files for the settings line. Loading 3 small files twice costs milliseconds; it keeps `build_index()` unchanged for Phase 6's code.
- **The loop**, once per golden item:
  1. **Answer** with `answer_question()`. If Ollama fails, `except GenerationError` records an error trace and `continue` moves on to the next question.
  2. **Evidence rank:** Phase 5's `first_relevant_rank()`: which sent chunk holds the quote, or `None`.
  3. **Check** with `check_answer()`, then decide `passed` and the `failing_stage`.
  4. **Trace** it, and print one table row. `{text:<8}` pads text to 8 characters on the left so the columns line up; `failing_stage or '-'` prints `-` when the stage is `None`.
  5. Print any warnings under the row.
- **After the loop:** count passes and errors, save the settings (now including `skipped_files`) and the traces, print a summary.
- `sys.exit(1)` when anything failed. In a terminal you see the same output either way; a script or CI job reads the exit code to decide "red or green". This is the first step toward Phase 12's regression gate.

## 7.4 What the code does, part by part: one question, end to end

Each file was explained under its code in 7.3. Here is how the parts work **together**, following q003 ("How many days a year can I work from another country?") through one real run on my Mac.

| Step | Code | What happened to q003 | Result kept in the trace |
| --- | --- | --- | --- |
| 1. Index | `build_index()` + `load_folder()` | 3 files loaded, none skipped | `settings.skipped_files: []` |
| 2. Retrieve | `answer_question()`: `embed_query()` + `search_chunks()` | Top 3: handbook `c1` (0.707), travel policy p2 `c0` (0.623), handbook `c2` (0.614) | `retrieval.hits`, `retrieval.seconds: 0.033` |
| 3. Evidence rank | `first_relevant_rank()` | The full quote is only in handbook `c0`, which wasn't sent | `retrieval.evidence_rank: null` |
| 4. Prompt | `label_hits()` + `build_prompt()` | 1,313 characters | `prompt.text`, `prompt.characters` |
| 5. Generate | `call_generator()` | `I don't know based on the documents.` in 2.51 s, 450 prompt tokens, `done_reason: "stop"` | `generation.*`; no warnings |
| 6. Check | `check_answer()` | Refused; expected `answer` → `behavior_matches: false`. No citations, so `citations_valid: true`; `refusal_is_clean: true` | `checks.results` |
| 7. Verdict | `all_checks_pass()` + `diagnose_stage()` | One `False` → failed. Expected `answer` and evidence rank `null` → `retrieval` | `passed: false`, `failing_stage: "retrieval"` |
| 8. Save | `write_traces()` | Line 4 of `results/traces/<date>_traces.jsonl` | The whole record |

Read step 6 carefully: the **model** behaved well (it refused instead of guessing, cleanly). The **system** failed: the user didn't get an answer the documents contain. That's why `failing_stage` matters more than the pass count: it points you at Phase 5's retrieval problem, not at the prompt.

**The same question with Ollama stopped:** step 5 raises `GenerationError`; `check_golden` catches it, writes an error trace (`status: "error"`, `failing_stage: "generation"`, the message), skips steps 6–7 for this question and continues with the next one.

## 7.5 Tests

32 new tests in four files. Almost everything in this phase is **code we control** (parsing labels, comparing verdicts, building dictionaries, catching errors), so most tests are fast unit tests. Only "does the real model pass the checks?" needs Ollama.

| File | Tests | Needs Ollama? | What they protect |
| --- | --- | --- | --- |
| `tests/unit/test_citations.py` | 13 | No | Label parsing, refusal detection, each of the 5 checks, pass/fail logic |
| `tests/unit/test_tracing.py` | 9 | No | Warnings, failing-stage rules, trace shape, the saved file |
| `tests/unit/test_errors.py` | 5 | No (fakes) | Bad files skipped with typed errors; bugs not swallowed; Ollama errors become `GenerationError` |
| `tests/integration/test_check_golden.py` | 5 | Yes, `@pytest.mark.integration` | The real pipeline passes the checks for 4 answerable items and refuses cleanly for q008 |

**A lesson learned while writing these tests.** My first version of `test_errors.py` imported `ask.py` to test `call_generator()`. The unit suite went from 0.4 s to 5.3 s, because `ask.py` imports `sentence_transformers`, which loads PyTorch. That's why `call_generator()` lives in `generation.py` (light imports only). Rule: unit tests should import only light modules; watch the total time (`--durations=5` shows the slowest tests).

Create `tests/unit/test_citations.py`:

```python
from ragqa import citations
from ragqa.golden import Evidence
from ragqa.retrieval import SearchHit

REFUSAL = "I don't know based on the documents."
LEAVE_EVIDENCE = [Evidence(source="handbook.md", page=1, quote="Annual leave: 27 days per year")]


# Build two labelled hits: [S1] holds the leave fact, [S2] does not.
def make_labelled_hits() -> list[tuple[str, SearchHit]]:
    leave_hit = SearchHit(rank=1, chunk_id="handbook.md:p1:c1", source="handbook.md", page=1, score=0.8, text="Annual leave: 27 days per year, plus public holidays.")
    remote_hit = SearchHit(rank=2, chunk_id="handbook.md:p1:c0", source="handbook.md", page=1, score=0.6, text="Employees may work remotely up to 3 days per week.")
    return [("[S1]", leave_hit), ("[S2]", remote_hit)]


# Labels are found in order of first appearance, without repeats.
def test_cited_labels_are_found_once_in_order():
    assert citations.find_cited_labels("A [S2]. B [S1]. C [S2].") == ["[S2]", "[S1]"]


# Labels that look different ([s1], [S 1], S1) are not citations.
def test_other_spellings_are_not_citations():
    assert citations.find_cited_labels("see [s1], [S 1] or S1") == []


# A cited label that was never sent is invalid.
def test_label_that_was_not_sent_is_invalid():
    valid_labels, invalid_labels = citations.split_citations(["[S1]", "[S4]"], ["[S1]", "[S2]", "[S3]"])
    assert valid_labels == ["[S1]"]
    assert invalid_labels == ["[S4]"]


# A curly apostrophe still counts as the refusal.
def test_refusal_with_curly_apostrophe_is_detected():
    assert citations.is_refusal("I don’t know based on the documents.")


# The refusal must open the answer to count as starting with it.
def test_refusal_in_the_middle_does_not_start_the_answer():
    assert citations.starts_with_refusal(REFUSAL + " Nothing about parking.")
    assert not citations.starts_with_refusal("27 days [S1]. " + REFUSAL)


# A correct, cited answer passes every check that applies.
def test_good_answer_passes_all_checks():
    check_result = citations.check_answer("Employees get 27 days. [S1]", make_labelled_hits(), "answer", LEAVE_EVIDENCE)
    assert check_result["checks"] == {
        "behavior_matches": True,
        "citations_valid": True,
        "answer_is_cited": True,
        "cites_evidence": True,
        "refusal_is_clean": None,
    }
    assert citations.all_checks_pass(check_result["checks"])


# An answer with no citation fails answer_is_cited.
def test_uncited_answer_fails():
    check_result = citations.check_answer("Employees get 27 days.", make_labelled_hits(), "answer", LEAVE_EVIDENCE)
    assert check_result["checks"]["answer_is_cited"] is False
    assert not citations.all_checks_pass(check_result["checks"])


# Citing a label that was never sent fails citations_valid.
def test_made_up_label_fails():
    check_result = citations.check_answer("Employees get 27 days. [S7]", make_labelled_hits(), "answer", LEAVE_EVIDENCE)
    assert check_result["invalid_labels"] == ["[S7]"]
    assert check_result["checks"]["citations_valid"] is False
    assert check_result["checks"]["answer_is_cited"] is False


# Citing a real label whose chunk lacks the evidence fails cites_evidence.
def test_citing_the_wrong_chunk_fails():
    check_result = citations.check_answer("Employees get 27 days. [S2]", make_labelled_hits(), "answer", LEAVE_EVIDENCE)
    assert check_result["checks"]["citations_valid"] is True
    assert check_result["checks"]["cites_evidence"] is False


# A clean refusal passes when refusal is expected.
def test_expected_refusal_passes():
    check_result = citations.check_answer(REFUSAL, make_labelled_hits(), "refuse", [])
    assert check_result["refused"] is True
    assert check_result["checks"]["behavior_matches"] is True
    assert check_result["checks"]["refusal_is_clean"] is True
    assert citations.all_checks_pass(check_result["checks"])


# A refusal that also cites a source is a half-answer and fails refusal_is_clean.
def test_refusal_with_citation_is_not_clean():
    answer = REFUSAL + " But leave is 27 days [S1]."
    check_result = citations.check_answer(answer, make_labelled_hits(), "refuse", [])
    assert check_result["checks"]["refusal_is_clean"] is False


# Refusing a question that should be answered fails behavior_matches.
def test_refusing_an_answerable_question_fails():
    check_result = citations.check_answer(REFUSAL, make_labelled_hits(), "answer", LEAVE_EVIDENCE)
    assert check_result["checks"]["behavior_matches"] is False
    assert check_result["checks"]["cites_evidence"] is None


# Answering a question that should be refused fails behavior_matches.
def test_answering_an_unanswerable_question_fails():
    check_result = citations.check_answer("Parking is free. [S1]", make_labelled_hits(), "refuse", [])
    assert check_result["checks"]["behavior_matches"] is False
```

A few lines explained:

- Each test feeds `check_answer()` a **hand-written answer** that breaks exactly one rule. That's how you test a checker: one known-bad input per rule, plus one known-good input. If a rule had no failing example, you couldn't know the check works.
- `make_labelled_hits()` builds the same two chunks for every test: `[S1]` contains the evidence, `[S2]` doesn't. So "cites `[S2]`" is a valid but **wrong** citation.
- `is False` / `is None` instead of `== False`: it distinguishes "failed" from "doesn't apply", which matters for `all_checks_pass()`.

Create `tests/unit/test_tracing.py`:

```python
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
```

A few lines explained:

- `make_result()` builds the dictionary `answer_question()` would return, by hand. The trace code is tested without any model.
- `list(trace["stages"].keys()) == [...]`: checks the stages are there **and in pipeline order** (Python dictionaries keep insertion order).
- `tmp_path` is a pytest fixture: a fresh, empty temporary folder per test. `"traces" / "run.jsonl"` inside it also tests that `write_traces()` creates a missing folder.

Create `tests/unit/test_errors.py`:

```python
from pathlib import Path

import pytest
import requests

from ragqa import generation
from ragqa import loaders
from ragqa.errors import DocumentLoadError
from ragqa.errors import GenerationError


# Bad documents raise our own DocumentLoadError, which is still a ValueError for older callers.
def test_bad_document_raises_document_load_error(tmp_path: Path):
    file_path = tmp_path / "notes.docx"
    file_path.write_text("x", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="unsupported file type"):
        loaders.load_document(file_path)
    assert issubclass(DocumentLoadError, ValueError)


# A folder with good and bad files: the good one loads, each bad one is skipped with its reason.
def test_bad_files_are_skipped_and_the_run_continues(tmp_path: Path):
    (tmp_path / "good.md").write_text("Annual leave: 27 days.", encoding="utf-8")
    (tmp_path / "empty.md").write_text("", encoding="utf-8")
    (tmp_path / "corrupted.pdf").write_bytes(b"this is not really a pdf")
    (tmp_path / "notes.docx").write_text("x", encoding="utf-8")

    pages, skipped_files = loaders.load_folder(tmp_path)

    assert len(pages) == 1
    assert pages[0].source == "good.md"
    assert len(skipped_files) == 3
    assert "corrupted.pdf: not a readable PDF" in skipped_files[0]
    assert skipped_files[1] == "empty.md: file has no text"
    assert skipped_files[2] == "notes.docx: unsupported file type '.docx'"


# An unexpected error (a bug, not a bad document) is not skipped: it stops the run.
def test_unexpected_error_is_not_swallowed(tmp_path: Path, monkeypatch):
    (tmp_path / "good.md").write_text("Annual leave: 27 days.", encoding="utf-8")

    # Pretend the loader has a bug.
    def broken_loader(file_path):
        raise KeyError("a bug in our code")

    monkeypatch.setattr(loaders, "load_document", broken_loader)
    with pytest.raises(KeyError):
        loaders.load_folder(tmp_path)


# Ollama not running becomes a GenerationError that says how to fix it.
def test_ollama_down_becomes_generation_error(monkeypatch):

    # Pretend nothing is listening on Ollama's port.
    def refuse_connection(url, json, timeout):
        raise requests.ConnectionError("Connection refused")

    monkeypatch.setattr(generation.requests, "post", refuse_connection)
    with pytest.raises(GenerationError, match="ollama serve"):
        generation.call_generator("prompt text")


# A slow Ollama becomes a GenerationError that names the time limit.
def test_ollama_timeout_becomes_generation_error(monkeypatch):

    # Pretend Ollama took too long.
    def time_out(url, json, timeout):
        raise requests.ReadTimeout("read timed out")

    monkeypatch.setattr(generation.requests, "post", time_out)
    with pytest.raises(GenerationError, match="did not answer within 180 s"):
        generation.call_generator("prompt text")
```

A few lines explained:

- `test_bad_files_are_skipped_and_the_run_continues`: the plan's robustness rule "explicit error type, run continues, file logged as skipped" in one test. Files are loaded in sorted name order, so the skip reasons come back in a known order.
- `test_unexpected_error_is_not_swallowed`: `monkeypatch` replaces `load_document()` with a function that raises `KeyError`, like a bug would. The test passes only if `load_folder()` lets it through. With Phase 1's `except ValueError` this would still pass, but with a bare `except Exception` it would fail: this test guards against "just catch everything".
- The two Ollama tests use the Phase 6 fake technique on `requests.post`, but this time the fake **raises** instead of returning. `requests.ReadTimeout` is a kind of `requests.Timeout`, so the `Timeout` branch catches it.

Create `tests/integration/test_check_golden.py`:

```python
import pytest

from ragqa import ask
from ragqa.citations import all_checks_pass
from ragqa.citations import check_answer
from ragqa.embeddings import load_embedding_model
from ragqa.golden import GOLDEN_V0_PATH
from ragqa.golden import load_golden_set
from ragqa.retrieval import DEFAULT_TOP_K


# Build the real index once for all tests in this file.
@pytest.fixture(scope="module")
def index():
    model = load_embedding_model()
    chunks, chunk_vectors = ask.build_index(model)
    return model, chunks, chunk_vectors


# Find one golden item by its id.
def golden_item(item_id: str):
    for item in load_golden_set(GOLDEN_V0_PATH):
        if item.id == item_id:
            return item
    raise KeyError(item_id)


# Answer one golden item with the real pipeline and run the Phase 7 checks on it.
def run_checks(index, item_id: str) -> dict:
    model, chunks, chunk_vectors = index
    item = golden_item(item_id)
    result = ask.answer_question(model, chunks, chunk_vectors, item.question, DEFAULT_TOP_K)
    return check_answer(result["answer"], result["labelled_hits"], item.expected_behavior, item.expected_evidence)


# Every answerable item whose evidence is retrieved gets a valid citation to that evidence.
@pytest.mark.integration
@pytest.mark.parametrize("item_id", ["q001", "q002", "q004", "q006"])
def test_answer_cites_the_evidence(index, item_id):
    check_result = run_checks(index, item_id)
    assert all_checks_pass(check_result["checks"]), check_result


# The unanswerable item refuses cleanly, with no citations.
@pytest.mark.integration
def test_unanswerable_item_refuses(index):
    check_result = run_checks(index, "q008")
    assert check_result["refused"] is True
    assert check_result["checks"]["refusal_is_clean"] is True
```

A few lines explained:

- `@pytest.fixture(scope="module")`: a *fixture* is setup code pytest runs for you and passes in by name (`index`). `scope="module"` builds the index **once** for the whole file, not once per test (it takes a few seconds).
- `@pytest.mark.parametrize("item_id", [...])` runs the same test 4 times, once per id; pytest shows them as `test_answer_cites_the_evidence[q001]` and so on.
- Why only q001, q002, q004, q006? They're the single-fact items whose evidence Phase 5 retrieves at rank 1. q003 is a known retrieval fail, and q005/q007 are judged "partial" in Phase 6. An integration test should check behaviour you **expect to be stable**; the full golden run reports the rest.
- `assert ..., check_result`: the text after the comma is printed when the assert fails, so you see which check broke without re-running.

**What the tests check, and why each matters**

| Test | What it proves | What would go wrong without it |
| --- | --- | --- |
| `cited_labels_are_found_once_in_order` | Each label counted once, first-seen order | Duplicate citations inflate counts |
| `other_spellings_are_not_citations` | Only the exact `[S#]` format counts | `[s1]` would pass but not map to a sent label |
| `label_that_was_not_sent_is_invalid` | Made-up labels are caught | A hallucinated source looks like evidence |
| `refusal_with_curly_apostrophe_is_detected` | `’` and `'` both count | Real refusals counted as answers |
| `refusal_in_the_middle_does_not_start_the_answer` | Opening vs mentioning the sentinel | Half-answers counted as clean refusals |
| `good_answer_passes_all_checks` | The happy path: all 4 applicable checks pass, the 5th is `None` | A checker that fails everything would go unnoticed |
| `uncited_answer_fails` | No citation → fail | Untraceable answers pass |
| `made_up_label_fails` | `[S7]` → invalid, and not counted as a citation | Fake sources pass |
| `citing_the_wrong_chunk_fails` | Valid label, wrong chunk → `cites_evidence` fails | Citations that point at unrelated text pass |
| `expected_refusal_passes` | A clean refusal on q008-style items passes | Correct refusals flagged as fails |
| `refusal_with_citation_is_not_clean` | Refuse + cite → fail | Mixed answers slip through |
| `refusing_an_answerable_question_fails` | Over-refusal is caught | A model that refuses everything scores 100% on "no hallucinations" |
| `answering_an_unanswerable_question_fails` | Guessing is caught | Hallucinated answers to out-of-scope questions pass |
| `normal_call_has_no_warnings` | No false alarms | Warnings everyone learns to ignore |
| `prompt_near_the_window_is_flagged` | ≥90% of `num_ctx` warns | Silent truncation (Phase 6's "2 days") goes unnoticed |
| `cut_short_answer_is_flagged` | `done_reason: length` warns | Answers cut mid-sentence look complete |
| `missing_evidence_points_at_retrieval` / `present_evidence_points_at_generation` / `passed_question_has_no_stage` | The three stage rules | Fixing the prompt when retrieval is broken (and the reverse) |
| `trace_has_every_stage` | IDs, scores, prompt, raw output, times, in order | Traces that can't explain a failure |
| `error_trace_names_the_stage` | Errors are recorded, typed, with a message | A failed question with no clue why |
| `traces_are_written_as_json_lines` | Settings first, one trace per line | Later tools (Phase 12) can't read the file |
| `bad_document_raises_document_load_error` | Typed error, still a `ValueError` | The API Lab's `except ValueError` breaks |
| `bad_files_are_skipped_and_the_run_continues` | 3 bad files skipped with reasons, good file loaded | One bad file stops indexing, or vanishes silently |
| `unexpected_error_is_not_swallowed` | Bugs still crash | Real bugs hidden as "skipped" |
| `ollama_down_becomes_generation_error` / `ollama_timeout_becomes_generation_error` | Clear, typed errors with a fix hint | 40-line tracebacks; one error kills the whole run |
| `answer_cites_the_evidence[...]` | The real model + pipeline pass all checks on 4 stable items | A prompt or model change silently breaks citations |
| `unanswerable_item_refuses` | The real model refuses q008 cleanly | Guessing creeps in after a change |

### QA role for this phase

**How important is testing here: very high.** These checks become the **ruler** for every later phase: Phase 12's regression gate and your before/after reports will trust them. A bug in a checker is worse than a bug in the pipeline, because it hides other bugs: if `cites_evidence` always returned `True`, every wrong citation from Phase 13 on would pass silently. And a missing error type here means a bad file or a stopped Ollama can make a run look "fine" with fewer documents or fewer answers.

**What to test**

| What can go wrong | How to check it | Manual or automated | Covered by |
| --- | --- | --- | --- |
| Citation parser misses or over-matches labels | Known strings in, exact lists out | Automated | `cited_labels_are_found_once_in_order`, `other_spellings_are_not_citations` |
| Made-up source accepted | `[S7]` with 3 chunks sent | Automated | `made_up_label_fails`, `label_that_was_not_sent_is_invalid` |
| Valid label, wrong chunk | Cite `[S2]` that lacks the quote | Automated | `citing_the_wrong_chunk_fails` |
| Uncited answer passes | Answer with no label | Automated | `uncited_answer_fails` |
| Wrong behaviour (guess / over-refuse) | Both directions | Automated | `answering_an_unanswerable_question_fails`, `refusing_an_answerable_question_fails`, `unanswerable_item_refuses` |
| Refusal format drifts | Curly apostrophe; extra words; refusal + citation | Automated | `refusal_with_curly_apostrophe_is_detected`, `refusal_in_the_middle_does_not_start_the_answer`, `refusal_with_citation_is_not_clean` |
| Checker too strict (false fails) | A good answer passes | Automated + manual | `good_answer_passes_all_checks`; read every FAIL row in 7.7 |
| Checker too loose (false passes) | Read every "ok" answer once | Manual | 7.7 golden table; Exercise 7.9 Part A (q005) |
| Bad file stops the run or vanishes | Mixed folder; `data/adversarial/` | Automated + manual | `bad_files_are_skipped_and_the_run_continues`; 7.6 adversarial run |
| Real bug hidden as "skipped" | Inject a `KeyError` | Automated | `unexpected_error_is_not_swallowed` |
| Ollama down crashes the run | Fake `ConnectionError`; really stop Ollama | Automated + manual | `ollama_down_becomes_generation_error`; 7.6 Ollama-stopped run |
| Slow Ollama hangs forever | Fake `ReadTimeout` | Automated | `ollama_timeout_becomes_generation_error` |
| Trace missing a stage or field | Build one and compare | Automated | `trace_has_every_stage`, `error_trace_names_the_stage` |
| Silent truncation / cut-off answer | Warnings at ≥90% window or `done_reason` ≠ `stop` | Automated | `prompt_near_the_window_is_flagged`, `cut_short_answer_is_flagged` |
| Wrong stage blamed | Three rule cases | Automated + manual | the `diagnose_stage` tests; compare with your Phase 6 review |

**Manual testing: how**

1. Run `uv run python -m ragqa.check_golden` (7.6). Pass: 7 of 8 pass and q003 fails with stage `retrieval`; the exit code is 1.
2. For every **FAIL**, open its trace line and confirm the verdict is right (not a checker bug). Pass: you agree with the check and the stage.
3. For every **ok** answer, read the answer once and ask: "would I have passed this?" Pass: yes. Fail: the checker missed something (that's how you'll find q005's uncited extra sentences, Exercise 7.9 Part A).
4. Compare with your Phase 6 `review` verdicts. Any disagreement between you and the checker is a finding: either the checker or your review is wrong.
5. Run `uv run python -m ragqa.loaders data/adversarial`. Pass: `Loaded 0 pages` and three `SKIPPED` lines, each naming the file and the reason.
6. Stop Ollama and run `check_golden` again. Pass: 8 `ERROR at generation` lines with the "Start it with: ollama serve" hint, `8 errors`, a trace file still saved, no traceback.
7. Open one trace line and check you can answer, from the trace alone: which chunks were sent, what the model saw, how long each stage took, and why it passed or failed.

**Automated testing: how**

- `tests/unit/test_citations.py`, `test_tracing.py`, `test_errors.py` (unit, 27 tests): pure logic with hand-made inputs and fakes, no model, so they run in the quick check in under a second. Run them on every change.
- `tests/integration/test_check_golden.py` (integration, 5 tests, about 30 s on my Mac): needs the embedding model and Ollama. Run it before each commit and after any change to `SYSTEM_RULES`, the model, chunking or retrieval.
- `check_golden` itself is the **system test**: the whole golden set, with a pass/fail exit code. Run it after every change that could affect answers, and keep the trace files to compare runs.

**Not covered yet:** sentences without a citation inside an otherwise cited answer (q005; Exercise 7.9 Part A, then a proper check in Phase 11 with a judge); whether a citation **supports** the claim it's attached to, beyond containing the quote (Phases 10–11, human review); prompt injection in a document (Phase 8's canary); per-question pass/fail **comparisons between runs** (Phase 12); retries and timeouts tuned per stage (not planned; one timeout is enough locally).

**How a real system tests this phase, and how we test it now**

In a real system, citations, refusals, errors and traces are part of the **answer API's contract**. The response has machine-readable fields (`citations`, `refused`, error codes), every request gets a trace ID, and the traces go to an observability tool (logs, dashboards, alerts). Testers check the contract per request and watch rates (refusal rate, invalid-citation rate, error rate) across many requests.

| What is checked | In a real system | In our project now |
| --- | --- | --- |
| Citation validity | Response field per answer; a monitored rate; tests in CI | `citations_valid` per golden item; the API Lab's `invalid_citations` field |
| Citation accuracy | Sampled human review + LLM judge | `cites_evidence` against golden quotes |
| Refusal | `refused` flag; refusal rate tracked over time; spikes alert | `behavior_matches`, `refusal_is_clean` on 8 items |
| Error handling | Typed error codes (503, 504, 422); retries; alerts on error rate | `DocumentLoadError`, `GenerationError`, error traces; the API Lab maps errors to 422/502/503/504 |
| Tracing | A trace ID per request; spans per stage in a tracing tool (e.g. OpenTelemetry) | One JSON line per question in `results/traces/` |
| Per-stage latency | Percentiles (p50, p95) per stage on a dashboard | `seconds` per stage per question |
| Truncation / cut-off | Token counts logged; alerts when near limits | Warnings at ≥90% of `num_ctx` or `done_reason` ≠ `stop` |

**Example: Ollama's own per-stage timing and stop reason** (real API, run on my machine: `3 passed in 2.60s`). Our trace relies on two things Ollama reports: per-stage timings and `done_reason`. These contract tests pin them down:

```python
import pytest
import requests

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:3b"
SYSTEM_TEXT = "Use only the facts inside <context>. Cite [S1]. If the answer is missing, reply exactly: I don't know based on the documents."
PROMPT_TEXT = "<context>\n[S1] (source: faq.txt, page 1)\nPasswords expire every 180 days.\n</context>\n\nQuestion: How often do passwords expire?"


# Send one grounded request with extra options and return Ollama's JSON reply.
def ask_ollama(extra_options: dict) -> dict:
    options = {"temperature": 0, "num_ctx": 4096}
    for name, value in extra_options.items():
        options[name] = value
    request_body = {
        "model": MODEL_NAME,
        "system": SYSTEM_TEXT,
        "prompt": PROMPT_TEXT,
        "stream": False,
        "options": options,
    }
    response = requests.post(f"{OLLAMA_URL}/api/generate", json=request_body, timeout=180)
    response.raise_for_status()
    return response.json()


# Ollama reports per-stage times in nanoseconds; they add up to no more than the total.
@pytest.mark.integration
def test_reply_has_per_stage_timings():
    reply = ask_ollama({})
    for field_name in ["load_duration", "prompt_eval_duration", "eval_duration", "total_duration"]:
        assert reply[field_name] > 0
    stage_sum = reply["load_duration"] + reply["prompt_eval_duration"] + reply["eval_duration"]
    assert stage_sum <= reply["total_duration"]


# An answer stopped by a length limit says so in done_reason, which our trace turns into a warning.
@pytest.mark.integration
def test_cut_short_answer_reports_length():
    reply = ask_ollama({"num_predict": 3})
    assert reply["done_reason"] == "length"
    assert reply["eval_count"] == 3


# A normal answer stops on its own.
@pytest.mark.integration
def test_normal_answer_reports_stop():
    reply = ask_ollama({})
    assert reply["done_reason"] == "stop"
```

- `test_reply_has_per_stage_timings`: Ollama splits its own time into loading the model, reading the prompt (`prompt_eval_duration`) and writing the answer (`eval_duration`), in nanoseconds (1 s = 1,000,000,000 ns). On my run: 49 ms load, 812 ms prompt, 624 ms answer, 1,517 ms total. A real trace would keep these too, to split our `generation.seconds` further.
- `test_cut_short_answer_reports_length`: `num_predict` 3 caps the answer at 3 tokens. Ollama returned `'Passwords expire every'` with `done_reason: "length"`: a cut-off answer that looks like text. This is exactly the case `find_warnings()` flags.
- `test_normal_answer_reports_stop`: the normal case is `"stop"`, so the warning stays quiet in normal runs. (Side note: with this one-line system text the model answered `Passwords expire every 180 days.` with **no** `[S1]`, and `answer_is_cited` would fail it. Our full `SYSTEM_RULES` gets the citation.)

The API Lab's `/chat` endpoint already returns `citations`, `invalid_citations`, `refused` and a `trace` (see `tests/api/test_api_chat.py`), so that's the "behind an API" version of this phase in your own project.

**The gap, in one line:** we check 8 questions with 5 rules and keep traces in files; a real system checks every request, tracks the rates over time on dashboards, and alerts a person when they move.

## 7.6 Run it

Run these from your project folder. The first two need no Ollama.

```bash
uv run pytest tests/unit -v
```

- **What it does:** the quick check, now with the 27 new unit tests.
- **Why:** proves the checkers and error handling work before you trust what they say about the model.
- **The parts:** `-v` (verbose) prints one line per test.

```bash
uv run python -m ragqa.loaders data/adversarial
```

- **What it does:** runs the Phase 1 loader on the folder with the three bad files.
- **Why:** the plan's Phase 7 check "corrupted files skipped cleanly", seen with your own eyes.
- **The parts:** `data/adversarial` becomes `sys.argv[1]`, the folder to load instead of `data/raw`.

Now start Ollama (7.2 Step 3), then:

```bash
uv run python -m ragqa.check_golden
```

- **What it does:** answers all 8 golden questions, runs the 5 checks on each, prints a table, saves one trace per question to `results/traces/<date>_traces.jsonl`, and exits with code 1 if any question failed.
- **Why:** the plan's Phase 7 check: "citation-validity check passes; unanswerable items refuse".

```bash
echo $?
```

- **What it does:** prints the exit code of the last command.
- **Why:** to see the "red or green" signal a script would read. `$?` is a shell variable holding the last exit code. Run it right after `check_golden`, with nothing in between.

```bash
uv run pytest tests/integration/test_check_golden.py -v -m integration
```

- **What it does:** runs the 5 integration tests against the real model and embedding model.
- **The parts:** `-m integration` selects tests marked `@pytest.mark.integration`.

**See the error handling.** Stop Ollama (`Ctrl+C` in its tab, or quit the app), then run:

```bash
uv run python -m ragqa.ask "How many days of annual leave do employees get?"
```

- **What it does:** tries to answer one question with Ollama stopped.
- **Why:** you should see one clear `ERROR:` line, not a traceback.

```bash
uv run python -m ragqa.check_golden
```

- **What it does:** the golden check with Ollama stopped.
- **Why:** every question gets an error trace and the run still finishes and saves a file. Delete that trace file afterwards (it only shows the error case), or keep it to compare in the exercise.

When you're done, keep Ollama stopped to free about 2.5 GB.

## 7.7 Expected output

All output below is real, from your project on your Mac (2026-10-18, Ollama 0.40.3, `llama3.2:3b`, embedding model loaded offline). I started Ollama for the runs and stopped it afterwards.

### Unit tests

```text
77 passed in 0.42s
```

50 tests from earlier phases + 27 new. Under half a second, because nothing imports PyTorch.

### The bad-file folder

```text
Loaded 0 pages from 0 files in data/adversarial
SKIPPED corrupted.pdf: not a readable PDF (Failed to open file 'data/adversarial/corrupted.pdf' as type pdf.)
SKIPPED empty.md: file has no text
SKIPPED notes.docx: unsupported file type '.docx'
```

Three files, three skip reasons, each starting with the file name, and no traceback. `.gitkeep` is ignored as a hidden file (Phase 1).

### The golden check

```text
id    expect  behavior citations cited evidence refusal  stage       answer
q001  answer  ok       ok        ok    ok       -        -           'According to the context, employees get 27 da'
q002  answer  ok       ok        ok    ok       -        -           'According to the handbook, employees may work'
q003  answer  FAIL     ok        -     -        ok       retrieval   "I don't know based on the documents."
q004  answer  ok       ok        ok    ok       -        -           'According to the context, the nightly hotel l'
q005  answer  ok       ok        ok    ok       -        -           'The daily meal allowance is 45 euros. [S1] Al'
q006  answer  ok       ok        ok    ok       -        -           'According to the FAQ, passwords expire every '
q007  answer  ok       ok        ok    ok       -        -           'According to the policy, the Compass portal i'
q008  refuse  ok       ok        -     -        ok       -           "I don't know based on the documents. There is"

Passed 7 of 8 questions (0 errors)
Saved: /Users/nowgg/Automation/rag_framework/results/traces/20261018-151556_traces.jsonl
```

I ran the check twice, a minute apart. Both runs gave the **same 8 answers word for word** and the same verdicts. The exit code, timing, per-stage table and trace line below come from the second run (`20261018-151641_traces.jsonl`).

`echo $?` printed `1`. The whole run took 39 s of wall-clock time, of which 29 s was generation (the sum of the 8 generation times below); the rest is loading the embedding model and building the index.

| Row | What it means |
| --- | --- |
| q001, q002, q004, q006 | All four answer checks pass; `refusal` is `-` because they didn't refuse |
| q003 | `behavior FAIL`: it refused an answerable question. `stage retrieval`: the evidence wasn't in the top 3. Same verdict as your Phase 6 review, now automatic |
| q005 | All `ok`, but Phase 6 judged it **partial**: two extra facts without citations. The checks only need *one* valid citation to the evidence. A false pass you'll fix in Exercise 7.9 Part A |
| q007 | All `ok`; Phase 6 judged it **partial** (incomplete: misses the 30-day rule). `cites_evidence` needs *any* expected quote, and the page-1 quote is cited. Completeness needs a judge (Phases 10–11) |
| q008 | Refused as expected; the extra sentence is allowed because the answer starts with the sentinel and cites nothing (the decision in 7.1) |

**Checker vs your Phase 6 review:** the checker agrees on 6 of 8 and is **more lenient** on q005 and q007. That's typical: deterministic checks are precise about what they check and blind to the rest. Never read "7 of 8 passed" as "7 of 8 answers are good".

### Per-stage numbers from the trace

| ID | Retrieval (s) | Generation (s) | Prompt tokens | Output tokens | `done_reason` | Evidence rank | Cited |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q001 | 0.019 | 4.71 | 495 | 25 | stop | 1 | `[S1]` |
| q002 | 0.069 | 2.35 | 495 | 22 | stop | 1 | `[S1]` |
| q003 | 0.033 | 2.51 | 450 | 10 | stop | none | — |
| q004 | 0.057 | 4.32 | 419 | 21 | stop | 1 | `[S1]` |
| q005 | 0.026 | 5.86 | 420 | 37 | stop | 1 | `[S1]` |
| q006 | 0.041 | 1.48 | 485 | 17 | stop | 1 | `[S1]` |
| q007 | 0.020 | 3.39 | 368 | 23 | stop | 1 | `[S1]` |
| q008 | 0.050 | 4.39 | 442 | 23 | stop | none | — |

Retrieval takes 20–70 **milliseconds**; generation takes 1.5–5.9 **seconds**, about 100 times longer. So when the pipeline feels slow, the trace tells you not to optimise search. Prompts use at most 495 of 4,096 tokens (12%), and every answer ended with `stop`, so no warnings were printed. Generation was slower than in Phase 6 (1–2.5 s warm), and the first run was slower still (q001 took 11.6 s, a cold start), because other apps were using memory: the same answers, slower. Times vary between runs; answers at temperature 0 didn't.

### One trace line

Line 4 of the trace file (q003), pretty-printed, prompt shortened:

```json
{
  "id": "q003",
  "question": "How many days a year can I work from another country?",
  "status": "ok",
  "passed": false,
  "failing_stage": "retrieval",
  "answer": "I don't know based on the documents.",
  "stages": {
    "retrieval": {
      "seconds": 0.033,
      "hits": [
        {"label": "[S1]", "chunk_id": "brightwater_handbook.md:p1:c1", "source": "brightwater_handbook.md", "page": 1, "score": 0.7069},
        {"label": "[S2]", "chunk_id": "brightwater_travel_policy.pdf:p2:c0", "source": "brightwater_travel_policy.pdf", "page": 2, "score": 0.623},
        {"label": "[S3]", "chunk_id": "brightwater_handbook.md:p1:c2", "source": "brightwater_handbook.md", "page": 1, "score": 0.614}
      ],
      "evidence_rank": null
    },
    "prompt": {
      "characters": 1313,
      "text": "<context>\n[S1] (source: brightwater_handbook.md, page 1)\nrking days per calendar ..."
    },
    "generation": {
      "seconds": 2.51,
      "model": "llama3.2:3b",
      "prompt_tokens": 450,
      "output_tokens": 10,
      "context_window_tokens": 4096,
      "done_reason": "stop",
      "raw_output": "I don't know based on the documents."
    },
    "checks": {
      "refused": true,
      "cited_labels": [],
      "invalid_labels": [],
      "results": {
        "behavior_matches": false,
        "citations_valid": true,
        "answer_is_cited": null,
        "cites_evidence": null,
        "refusal_is_clean": true
      }
    }
  },
  "warnings": [],
  "error": null
}
```

Line 1 of the file holds the `settings`: as in Phase 6, plus `"skipped_files": []` (no file in `data/raw` was skipped).

### Integration tests

I ran this phase's and Phase 6's integration files together:

```text
tests/integration/test_check_golden.py::test_answer_cites_the_evidence[q001] PASSED
tests/integration/test_check_golden.py::test_answer_cites_the_evidence[q002] PASSED
tests/integration/test_check_golden.py::test_answer_cites_the_evidence[q004] PASSED
tests/integration/test_check_golden.py::test_answer_cites_the_evidence[q006] PASSED
tests/integration/test_check_golden.py::test_unanswerable_item_refuses PASSED
tests/integration/test_generation_ollama.py::test_answer_uses_the_context_and_cites_it PASSED
tests/integration/test_generation_ollama.py::test_model_refuses_when_the_context_lacks_the_answer PASSED
tests/integration/test_generation_ollama.py::test_answer_follows_the_context_not_world_knowledge PASSED
tests/integration/test_generation_ollama.py::test_same_prompt_gives_the_same_answer PASSED

9 passed in 32.56s
```

I also ran the API Lab suite (`uv run pytest tests/api -q -m "api or integration"`): `58 passed in 39.82s`, so the changes to `loaders.py` and `citations.py` didn't break the API.

### With Ollama stopped

`ask`:

```text
ERROR: Ollama is not reachable at http://localhost:11434. Start it with: ollama serve
```

Exit code 2. `check_golden`:

```text
id    expect  behavior citations cited evidence refusal  stage       answer
q001  ERROR at generation: Ollama is not reachable at http://localhost:11434. Start it with: ollama serve
q002  ERROR at generation: Ollama is not reachable at http://localhost:11434. Start it with: ollama serve
...
q008  ERROR at generation: Ollama is not reachable at http://localhost:11434. Start it with: ollama serve

Passed 0 of 8 questions (8 errors)
Saved: /Users/nowgg/Automation/rag_framework/results/traces/20261018-151914_traces.jsonl
```

The error trace for q001:

```json
{
  "id": "q001",
  "question": "How many days of annual leave do employees get?",
  "status": "error",
  "passed": false,
  "failing_stage": "generation",
  "answer": null,
  "stages": {},
  "warnings": [],
  "error": {
    "stage": "generation",
    "type": "GenerationError",
    "message": "Ollama is not reachable at http://localhost:11434. Start it with: ollama serve"
  }
}
```

Compare with Phase 6, where the same situation printed a long `requests.exceptions.ConnectionError: HTTPConnectionPool(...)` traceback and stopped at the first question.

## 7.8 Common errors and debugging

| Error or symptom | Cause | Fix |
| --- | --- | --- |
| `ERROR: Ollama is not reachable at http://localhost:11434. Start it with: ollama serve` (seen on my Mac) | Ollama isn't running | `ollama serve` in its own tab, or open the app |
| `ERROR at generation: Ollama returned an error: 404 Client Error: Not Found…` | Model name not on disk | `ollama list`; fix `DEFAULT_GENERATOR_MODEL` or `ollama pull llama3.2:3b` |
| `ERROR at generation: Ollama did not answer within 180 s` | Memory pressure, the Mac is swapping (not reproduced here) | Close Chrome/Slack; `ollama ps`; Activity Monitor |
| `ImportError: cannot import name 'call_generator' from 'ragqa.generation'` | `ask.py` updated, `generation.py` not | Add `call_generator()` to `generation.py` (7.3) |
| `ImportError: cannot import name 'DocumentLoadError'` / `No module named 'ragqa.errors'` | `errors.py` not created | Create `src/ragqa/errors.py` (7.3) |
| `KeyError: 'retrieval_seconds'` (or `'done_reason'`, `'raw_output'`) in `tracing.py` | `answer_question()` still the Phase 6 version | Replace it with the 7.3 version |
| Unit tests take \~5 s instead of under 1 s | A unit test imports `ask.py` (and so PyTorch) | Import only `generation`, `citations`, `tracing`, `loaders`; check with `--durations=5` |
| A run "passes" but a document is missing | A file was skipped and nobody looked | Read the `SKIPPED` lines and `settings.skipped_files` in the trace file |
| A traceback from inside `load_folder()` instead of a `SKIPPED` line | A real bug (not a `DocumentLoadError`): working as intended | Read the traceback; fix the bug. Don't widen the `except` |
| `check_golden` exits 1 but every row says `ok` | A question had an error (`ERROR at generation`) | Scroll up; the summary says `(N errors)` |
| Everything fails with `behavior FAIL` and `stage generation` | Model refuses everything, e.g. a broken prompt or empty context | Open one trace: is the context empty? Did `SYSTEM_RULES` change? |
| `cites_evidence` fails though the answer looks right | The model cited a different chunk than the one with the quote, or the golden quote has different spacing | Compare `stages.checks.cited_labels` with `stages.retrieval.hits` and `evidence_rank` |

**Debugging techniques**

1. **Start from the trace, not from a rerun.** Every question's full story is saved. Print one question's trace, pretty-printed:

```bash
uv run python -c "import json, sys; lines = open(sys.argv[1]).read().splitlines(); print(json.dumps(json.loads(lines[3]), indent=2))" results/traces/20261018-151556_traces.jsonl
```

- **What it does:** prints line 4 of the trace file (q003; `lines[0]` is the settings) with indentation.
- **The parts:** `python -c "..."` runs the code in quotes; `sys.argv[1]` is the file path given after it (use your own file name: `ls results/traces`); `json.dumps(..., indent=2)` pretty-prints.

2. **List only the failures.** With 20 questions in Phase 9, you'll want just the broken ones:

```bash
grep '"passed": false' results/traces/20261018-151556_traces.jsonl
```

- **What it does:** prints only the trace lines of failed questions.
- **Why it works:** JSON Lines puts one question per line, so a plain text search finds whole records.

3. **Walk the stages in order** (the plan's "Finding the failing component"): `retrieval.evidence_rank` → `prompt.text` → `generation.raw_output` and `done_reason` → `checks.results`. Stop at the first stage that looks wrong.
4. **Test a checker by hand** on a made-up answer, without the model:

```bash
uv run python -c "from ragqa.citations import find_cited_labels, is_refusal, starts_with_refusal; print(find_cited_labels('Leave is 27 days [S1][S3].'), is_refusal('I don’t know based on the documents.'), starts_with_refusal('27 days. I don’t know based on the documents.'))"
```

- **What it does:** runs three helpers on hand-written text. On my Mac: `['[S1]', '[S3]'] True False`.
- **Why:** when a check gives a surprising verdict, this shows in seconds whether the checker or the model is the problem.

5. **Compare two runs:** keep the trace files of two runs and compare `passed` per `id`. Phase 12 automates this; for now, `grep '"passed": false'` on both files and compare the ids.

## 7.9 Exercise: catch a false pass, and test the tester

**Part A — catch q005's uncited sentences.** The checker passed q005, but its answer was:

```text
The daily meal allowance is 45 euros. [S1] Alcohol is never reimbursed. Submit receipts in Compass within 30 days of returning; late claims are rejected automatically.
```

Two of three sentences have no citation. Rule 2 says "after every fact, cite its source label". Write a function that returns the sentences without a citation:

1. Predict first: which sentences of each of the 8 answers should it return?
2. Write `find_uncited_sentences(answer_text)` in `citations.py`. Hint: split the answer into sentences, then use `CITATION_PATTERN.search(sentence)` (`None` = no label in it).
3. Watch out: the model writes the label **after** the full stop (`45 euros. [S1] Alcohol…`). A naive split after every `.` puts `[S1]` at the start of the *next* sentence and gets both sentences wrong.

My version (try yours first):

```python
# Split an answer into sentences and return the ones with no [S#] label.
def find_uncited_sentences(answer_text: str) -> list[str]:
    # Split after . ! ? unless a label follows, and also after a label.
    sentences = re.split(r"(?<=[.!?])\s+(?!\[S\d)|(?<=\])\s+", answer_text.strip())
    uncited = []
    for sentence in sentences:
        if sentence.startswith("[S"):
            continue
        if CITATION_PATTERN.search(sentence) is None:
            uncited.append(sentence)
    return uncited
```

The split pattern, part by part: `(?<=[.!?])` = "right after `.`, `!` or `?`" (a *lookbehind*: it checks without consuming the character); `\s+` = one or more spaces; `(?!\[S\d)` = "but not if a label comes next" (a *negative lookahead*); `|` = or; `(?<=\])\s+` = spaces right after a `]`. So `euros. [S1] Alcohol` splits after `[S1]`, not before it.

My results on the 8 answers of the run in 7.7:

| ID | Uncited sentences |
| --- | --- |
| q001, q002, q004, q006, q007 | none |
| q003 | `I don't know based on the documents.` (a refusal, so it shouldn't count) |
| q005 | `Alcohol is never reimbursed.` and `Submit receipts in Compass within 30 days of returning; late claims are rejected automatically.` |
| q008 | both sentences (again a refusal) |

Now decide, and write down why: should this become a sixth check (`every_sentence_cited`, `None` for refusals) that **fails** q005, or a **warning** in the trace? Think about q005's extra facts: they're true and in `[S1]`, just not labelled. If you add it as a check, add two unit tests (one answer that passes, one that fails) and rerun `check_golden`: with the same answers you'd expect 6 of 8 (I didn't run this variant; check it yourself).

**Part B — test the tester (mutation testing).** A test suite is only useful if it fails when the code is wrong. *Mutation testing* means: break the code on purpose, run the tests, and see whether any test notices. Try these two, one at a time, and **undo each change afterwards**:

| Mutation in `citations.py` | What it simulates | Tests that failed on my Mac |
| --- | --- | --- |
| In `all_checks_pass()`: `if result is False:` → `if not result:` | Treating "doesn't apply" (`None`) as a fail | 2: `good_answer_passes_all_checks`, `expected_refusal_passes` |
| In `is_refusal()`: `return refusal_core in ...` → `return normalize_for_refusal(answer_text) == refusal_core + "."` | Making the refusal match exact (nothing else allowed) | 1: `refusal_with_citation_is_not_clean` |

```bash
uv run pytest tests/unit/test_citations.py -q
```

- **What it does:** runs only the citation tests, quickly, after each mutation.

The second row is the interesting one. With an exact match, q008's real answer (`I don't know… There is no mention of a parking policy…`) would no longer count as a refusal and `check_golden` would fail it. Yet only **one** unit test noticed, and for a different reason. The suite has a gap: no test says "a refusal followed by an explanation still counts as a refusal". Write that test (use q008's real answer as the input), check it passes on the real code, then check it fails on the mutation.

**Part C — read a trace like an investigator.** Open the q005 line of your trace file (debugging technique 1, with `lines[5]`). From the trace alone, answer: which chunk was `[S1]`? Does that chunk contain "Alcohol is never reimbursed"? So is q005's extra text a hallucination, or a model repeating its context? Write a three-line note: symptom, evidence from the trace, which stage owns it. That's the format of the plan's investigation write-ups (Phase 14).

## 7.10 Done criteria

- [ ] `errors.py`, `tracing.py` and `check_golden.py` exist; `loaders.py`, `generation.py`, `ask.py` and `citations.py` match 7.3
- [ ] `uv run pytest tests/unit -v` passes, including the 27 new tests, in under 1 second
- [ ] `uv run python -m ragqa.loaders data/adversarial` shows `Loaded 0 pages` and three `SKIPPED` lines with reasons (corrupted files skipped cleanly)
- [ ] `uv run python -m ragqa.check_golden` reports `citations ok` for all 8 questions (citation-validity check passes)
- [ ] q008 shows `behavior ok` and `refusal ok` (unanswerable items refuse)
- [ ] q003 fails with stage `retrieval`, and `echo $?` prints `1`
- [ ] A trace file exists in `results/traces/` and you can explain one line of it stage by stage
- [ ] With Ollama stopped, `ask` prints one `ERROR:` line and `check_golden` saves 8 error traces, with no traceback
- [ ] `uv run pytest tests/integration/test_check_golden.py -v -m integration` passes (5 tests)
- [ ] Exercise Part A done, with your decision (check or warning) written down; Part B's missing test added; Part C's three-line note written
- [ ] Ollama stopped afterwards; work committed with a message like `phase 7 complete`

When these pass, share your Part A decision and your Part C note, and we'll start Phase 8: one consolidated pytest suite with clear markers, robustness fixtures, and a prompt-injection canary.
