# RAG Quality Lab — Phase 2: Chunking

Oct 13, 2026 · @Akshit

**Goal of this phase:** split every `Page` from Phase 1 into smaller, overlapping pieces called **chunks**. Each chunk keeps its page's metadata and records exactly where it starts and ends in the page text. No LLM is needed, so Ollama can stay off. Previous phase: [Phase 1](https://claude.ai/code/artifact/7df3b55e-6623-464f-8990-7106ec3af43f). Tracker: [Phase 0 doc](https://claude.ai/code/artifact/b32269ff-ce8f-4186-b984-8cf68f206b1b).

## 2.1 Concepts

Retrieval doesn't search whole documents. It searches small pieces and hands only the best few to the LLM. Chunking makes those pieces. A good chunk is small enough to be about one thing, and big enough to make sense on its own.

**Words you'll meet in this phase**

| Term | Plain meaning | In our project |
| --- | --- | --- |
| Chunk | A short piece of a page's text, plus labels | Our `Chunk` record |
| Chunk size | The longest a chunk may be | 500 characters (letters, digits, spaces and line breaks all count as 1) |
| Overlap | How many characters two neighbouring chunks share | 100 characters |
| Step | How far the window moves forward each time: chunk size minus overlap | 500 − 100 = 400 characters |
| Sliding window | Cutting text by moving a fixed-size "window" along it, one step at a time | How `split_text()` works |
| Offset | A position in the text, counted in characters from 0 | `start_char` and `end_char` on each chunk |
| Slice | Python's way to cut text by offsets: `text[start:end]` takes characters from `start` up to, but not including, `end` | `page.text[chunk.start_char:chunk.end_char]` gives back the chunk text exactly |
| Token | The unit an AI model reads; roughly ¾ of an English word, or about 4 characters | 500 characters ≈ 125 tokens |

**How the sliding window works.** A tiny example: the text `abcdefghij` (10 characters), chunk size 4, overlap 1, so step 3.

| Chunk | `start_char` | `end_char` | Text | Shares with the previous chunk |
| --- | --- | --- | --- | --- |
| 0 | 0 | 4 | `abcd` | — |
| 1 | 3 | 7 | `defg` | `d` |
| 2 | 6 | 10 | `ghij` | `g` |

Each new chunk starts 3 characters after the previous one, so the last character of one chunk is also the first of the next. The last chunk stops at the end of the text, so no chunk is wasted on text the previous one already covered.

**Why overlap?** Without it, a sentence that crosses a chunk boundary is cut in two, and neither half fully contains it. With an overlap of 100 characters, **any piece of text up to 100 characters long is guaranteed to sit whole inside at least one chunk.** That matters for evaluation: the golden set (Phase 5) checks whether a retrieved chunk *contains* an evidence quote. Short quotes (under 100 characters) can never be lost to a boundary. The tests in section 2.4 prove this.

**How it connects to the architecture.** Phase 2 is step 2 of the indexing flow:

1. **Load** (Phase 1): files → `Page` records. `chunking.py` reuses `load_folder()` to get them.
2. **Chunk** (Phase 2): pages → `Chunk` records. Each chunk copies its page's `source` and `page`, so citations still point to the right file and page.
3. **Embed** (Phase 3): each chunk's text becomes a list of numbers. The embedding model `bge-small-en-v1.5` reads at most 512 tokens, so our \~125-token chunks fit easily.
4. **Store** (Phase 4): chunks and their numbers go into Chroma, keyed by chunk ID.

**Design choices for this phase**

| Question | Our choice | Why |
| --- | --- | --- |
| Cut by characters, words or tokens? | Characters | Simplest to write, test and reason about. Offsets map straight back to `page.text` |
| Chunk size | 500 characters | About one paragraph of our handbook: big enough for context, small enough to be about one topic |
| Overlap | 100 characters (20% of the size) | Keeps short quotes whole; common starting point is 10–20% |
| Do we avoid cutting words in half? | Not yet | Fixed cuts are predictable and easy to test. You'll *see* mid-word cuts in the output; we only add smarter cutting if evaluation (Phase 12–13) shows it hurts |
| Can chunks cross from one page to the next? | No, each page is chunked on its own | A chunk then has exactly one page number for its citation |
| Do we clean the text first? | No | Offsets must point into the exact text Phase 1 stored |
| Chunk ID | `source:p<page>:c<index>`, like `brightwater_handbook.md:p1:c2` | Easy to read in output and tests. Phase 4 adds content-hash IDs to stop duplicates |

Chunk size and overlap are the first **tunable settings** in this project. They're exactly the kind of setting Phase 13 will change one at a time and compare with evaluation numbers.

## 2.2 Setup steps

**Nothing new to install.** Chunking is plain Python: slicing text and a loop. No `uv add` in this phase.

Phase 2 builds directly on Phase 1, so check that Phase 1 works first. Run from your project folder (`/Users/nowgg/Automation/rag_framework`):

```bash
uv run python -m ragqa.loaders
```

- **What it does:** runs the Phase 1 loader on `data/raw/`.
- **Why:** the chunker calls the same `load_folder()` function. If this command fails, fix Phase 1 first (see its section 1.6).

The last line must be:

```text
Loaded 5 pages from 3 files in /Users/nowgg/Automation/rag_framework/data/raw
```

You need these files from Phase 1:

| File | Why Phase 2 needs it |
| --- | --- |
| `src/ragqa/loaders.py` | `chunking.py` imports `Page`, `load_folder` and `DEFAULT_DATA_FOLDER` from it |
| `data/raw/brightwater_handbook.md` | The only document long enough to be split into several chunks (1,106 characters) |
| `data/raw/brightwater_faq.txt` and `brightwater_travel_policy.pdf` | Short pages that each fit in one chunk, a useful contrast |

## 2.3 Code: the chunker

This file turns `Page` records into `Chunk` records. It has three small working functions, each building on the one before: `split_text()` works out *where* to cut, `chunk_page()` cuts one page, and `chunk_pages()` cuts a whole list. `main()` reuses Phase 1's `load_folder()` and prints every chunk's edges so you can inspect the boundaries by eye. Phase 3 (embeddings) will import `chunk_pages()` and `Chunk` from here.

Create `src/ragqa/chunking.py`:

```python
"""Phase 2: split pages into overlapping chunks that remember where they came from."""

import sys
from dataclasses import dataclass

from ragqa.loaders import DEFAULT_DATA_FOLDER
from ragqa.loaders import Page
from ragqa.loaders import load_folder

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100
EDGE_LENGTH = 60


# One piece of a page's text, plus its position and source.
@dataclass
class Chunk:
    chunk_id: str
    source: str
    page: int
    chunk_index: int
    start_char: int
    end_char: int
    text: str


# Work out the (start, end) positions of overlapping windows.
def split_text(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be above 0, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap cannot be negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    step = chunk_size - overlap
    spans = []
    start = 0
    while True:
        end = min(start + chunk_size, len(text))
        spans.append((start, end))
        if end == len(text):
            break
        start = start + step
    return spans


# Cut one page into chunks.
def chunk_page(page: Page, chunk_size: int, overlap: int) -> list[Chunk]:
    chunks = []
    spans = split_text(page.text, chunk_size, overlap)
    for chunk_index, (start_char, end_char) in enumerate(spans):
        chunk_id = f"{page.source}:p{page.page}:c{chunk_index}"
        chunk = Chunk(
            chunk_id=chunk_id,
            source=page.source,
            page=page.page,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            text=page.text[start_char:end_char],
        )
        chunks.append(chunk)
    return chunks


# Cut every page into chunks, one page at a time.
def chunk_pages(pages: list[Page], chunk_size: int, overlap: int) -> list[Chunk]:
    all_chunks = []
    for page in pages:
        page_chunks = chunk_page(page, chunk_size, overlap)
        all_chunks.extend(page_chunks)
    return all_chunks


# Join all words with single spaces, for one-line printing.
def flatten(text: str) -> str:
    return " ".join(text.split())


# Chunk the documents and print each chunk's edges and overlap.
def main() -> None:
    chunk_size = DEFAULT_CHUNK_SIZE
    overlap = DEFAULT_OVERLAP
    if len(sys.argv) == 3:
        chunk_size = int(sys.argv[1])
        overlap = int(sys.argv[2])
    elif len(sys.argv) != 1:
        print("Usage: uv run python -m ragqa.chunking [CHUNK_SIZE OVERLAP]")
        sys.exit(1)

    pages, skipped_files = load_folder(DEFAULT_DATA_FOLDER)
    chunks = chunk_pages(pages, chunk_size, overlap)

    previous_chunk = None
    for chunk in chunks:
        length = chunk.end_char - chunk.start_char
        print(f"--- {chunk.chunk_id} | chars {chunk.start_char}-{chunk.end_char} ({length})")
        print(f"  starts: {flatten(chunk.text[:EDGE_LENGTH])!r}")
        print(f"  ends:   {flatten(chunk.text[-EDGE_LENGTH:])!r}")
        same_page = (
            previous_chunk is not None
            and previous_chunk.source == chunk.source
            and previous_chunk.page == chunk.page
        )
        if same_page:
            shared_text = chunk.text[: previous_chunk.end_char - chunk.start_char]
            print(f"  shares with previous: {flatten(shared_text)!r}")
        previous_chunk = chunk

    lengths = []
    for chunk in chunks:
        lengths.append(len(chunk.text))

    print()
    print(f"chunk_size={chunk_size}, overlap={overlap}")
    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    if len(lengths) > 0:
        average_length = sum(lengths) / len(lengths)
        print(f"Chunk length: min {min(lengths)}, average {average_length:.0f}, max {max(lengths)}")
    for reason in skipped_files:
        print(f"SKIPPED {reason}")


if __name__ == "__main__":
    main()
```

**What the code does, part by part**

- **The imports.** `sys` for command-line words and `sys.exit`; `dataclass` for the `Chunk` record. The three `from ragqa.loaders import ...` lines **reuse Phase 1**: the `Page` record, the `load_folder()` function and `DEFAULT_DATA_FOLDER`, the absolute path to `data/raw` (built from `PROJECT_ROOT`, so the chunker works from any folder). Nothing from Phase 1 is copied or rewritten.
- **The CAPITAL settings.** `DEFAULT_CHUNK_SIZE = 500` and `DEFAULT_OVERLAP = 100` are the defaults from section 2.1. `EDGE_LENGTH = 60` is only for display: how many characters of each chunk's start and end to print.
- **`class Chunk`.** Seven fields. Three are new compared with `Page`:
  - `chunk_index`: the chunk's position within its page (0, 1, 2...).
  - `start_char` / `end_char`: the chunk's offsets in `page.text`.
  - `chunk_id`: a readable unique name, built from source, page and index.
  - `file_type` is left out on purpose: no later phase needs it per chunk. Keep records as small as the next phases need.
- **`split_text(text, chunk_size, overlap)`.** Input: a string and the two settings. Output: a list of `(start, end)` pairs, one per chunk. It only does arithmetic; it never cuts text itself, which makes it very easy to test.
  - The three `if ... raise ValueError` lines reject settings that can't work. If `overlap` were equal to `chunk_size`, the step would be 0 and the window would never move: an endless loop. Checking first turns that hang into a clear error.
  - `step = chunk_size - overlap` is how far each new window moves (400 by default).
  - `while True:` repeats until `break`. Each round:
    1. `end = min(start + chunk_size, len(text))` — the window ends `chunk_size` characters later, but never past the end of the text. `min` picks the smaller of the two numbers.
    2. `spans.append((start, end))` stores this window. `(start, end)` is a **tuple**: two values kept together.
    3. If this window reached the end of the text, stop. This check is why there's never an extra, useless chunk at the end.
    4. Otherwise move `start` forward by one step and go round again.
- **`chunk_page(page, chunk_size, overlap)`.** Input: one `Page`. Output: its list of `Chunk`s.
  - `enumerate(spans)` gives a counter with each span. `(start_char, end_char)` in the `for` line **unpacks** each tuple into two named variables.
  - `page.text[start_char:end_char]` slices the exact chunk text out of the page. The chunk text is never edited, so `page.text[start:end] == chunk.text` always holds. Phase 9's quote-checking relies on that.
  - `source` and `page` are copied straight from the `Page`. This is how metadata survives chunking.
- **`chunk_pages(pages, chunk_size, overlap)`.** Runs `chunk_page` on every page and collects all chunks in one list, in the same order as the pages. Each page is chunked on its own, so a chunk never spans two pages.
- **`flatten(text)`.** Joins all words with single spaces, so line breaks don't spoil the one-line display. Same idea as Phase 1's `make_preview()`, minus the 200-character cut.
- **`main()`**
  - **Command-line settings.** With exactly two extra words (`len(sys.argv) == 3`, because `sys.argv[0]` is the program itself), they become the chunk size and overlap; `int(...)` turns the text `"300"` into the number 300. With no extra words, the defaults are used. Anything else prints a usage line and stops with exit code 1.
  - **Load, then chunk.** Phase 1's `load_folder()` returns pages and skip reasons; `chunk_pages()` turns the pages into chunks.
  - **Print each chunk.** One header line (ID, offsets, length), then its first and last 60 characters. In `chunk.text[-EDGE_LENGTH:]`, a negative number counts from the end: "the last 60 characters". In an f-string, `!r` prints the value with quotes around it, so spaces at the start or end become visible.
  - **Show the overlap.** `previous_chunk` remembers the chunk printed last time round. `same_page` is `True` only when there *was* a previous chunk and it came from the same file and page. Then `previous_chunk.end_char - chunk.start_char` is the number of shared characters (100), and `chunk.text[:that_number]` is the shared text. Seeing it printed is the "inspect overlap by eye" check from the plan.
  - **Summary.** The settings, how many pages became how many chunks, and the shortest, average and longest chunk. `:.0f` formats the average with no decimals. The `if len(lengths) > 0` check avoids dividing by zero when there are no chunks.

## 2.4 Tests

Eight unit tests cover the plan's three Phase 2 checks: **no lost text, correct overlap, and offsets that map back to the source**. They also cover the quote guarantee from section 2.1, metadata, and bad settings. All are unit tests: no files, no Ollama, under a second.

These tests don't touch the disk at all. A small helper, `make_page()`, builds a `Page` in memory, using the `Page` class from Phase 1. That's simpler than Phase 1's `tmp_path` files, because the chunker only needs text, not a file.

Two test ideas are new here:

- **Testing a rule, not one example.** Most tests check one input and one expected output. `test_every_quote_up_to_overlap_length_sits_inside_one_chunk` instead checks **every** possible 40-character quote in a long text (over 1,000 of them). That's how you test a promise like "no short quote is ever split".
- **Repeated text as test data.** `"The daily meal allowance is 45 euros. " * 40` repeats a sentence 40 times (1,520 characters). It's an easy way to get text long enough for many chunks.

Create `tests/unit/test_chunking.py`:

```python
import pytest

from ragqa import chunking
from ragqa.loaders import Page


# Build a test page in memory.
def make_page(text: str) -> Page:
    return Page(source="handbook.md", page=3, file_type="md", text=text)


# Text shorter than a chunk gives exactly one chunk.
def test_short_text_becomes_one_chunk():
    spans = chunking.split_text("Annual leave is 27 days.", chunk_size=500, overlap=100)
    assert spans == [(0, 24)]


# Spans match the abcdefghij example from section 2.1.
def test_sliding_window_spans_match_the_worked_example():
    spans = chunking.split_text("abcdefghij", chunk_size=4, overlap=1)
    assert spans == [(0, 4), (3, 7), (6, 10)]


# Each chunk's text is exactly page.text[start:end].
def test_chunk_text_matches_page_text_at_its_offsets():
    page = make_page("The daily meal allowance is 45 euros. " * 40)

    chunks = chunking.chunk_page(page, chunk_size=200, overlap=50)

    for chunk in chunks:
        assert chunk.text == page.text[chunk.start_char:chunk.end_char]


# Chunks cover the whole page with no gaps.
def test_no_text_is_lost_between_chunks():
    page = make_page("The daily meal allowance is 45 euros. " * 40)

    chunks = chunking.chunk_page(page, chunk_size=200, overlap=50)

    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(page.text)
    for index in range(1, len(chunks)):
        previous_chunk = chunks[index - 1]
        current_chunk = chunks[index]
        assert current_chunk.start_char <= previous_chunk.end_char


# Neighbouring chunks share exactly the overlap.
def test_neighbouring_chunks_overlap_by_exactly_the_overlap():
    page = make_page("x" * 1000)

    chunks = chunking.chunk_page(page, chunk_size=300, overlap=60)

    for index in range(1, len(chunks)):
        shared_characters = chunks[index - 1].end_char - chunks[index].start_char
        assert shared_characters == 60


# Every quote up to the overlap length fits whole in some chunk.
def test_every_quote_up_to_overlap_length_sits_inside_one_chunk():
    text = ""
    for number in range(300):
        text = text + str(number) + " "
    page = make_page(text)
    overlap = 40

    chunks = chunking.chunk_page(page, chunk_size=150, overlap=overlap)

    for quote_start in range(len(text) - overlap + 1):
        quote = text[quote_start:quote_start + overlap]
        found = False
        for chunk in chunks:
            if quote in chunk.text:
                found = True
                break
        assert found, f"quote at {quote_start} was split: {quote!r}"


# Chunks copy source and page, and get unique IDs.
def test_chunks_copy_page_metadata_and_get_unique_ids():
    page = make_page("y" * 1200)

    chunks = chunking.chunk_page(page, chunk_size=500, overlap=100)

    chunk_ids = []
    for chunk in chunks:
        assert chunk.source == "handbook.md"
        assert chunk.page == 3
        chunk_ids.append(chunk.chunk_id)
    assert chunk_ids == ["handbook.md:p3:c0", "handbook.md:p3:c1", "handbook.md:p3:c2"]


# An overlap as big as the chunk size is rejected.
def test_overlap_not_smaller_than_chunk_size_raises():
    with pytest.raises(ValueError, match="must be smaller than chunk_size"):
        chunking.split_text("some text", chunk_size=100, overlap=100)
```

**What the tests check, and why each matters**

| Test | What it proves | What would break later without it |
| --- | --- | --- |
| `short_text_becomes_one_chunk` | Text shorter than a chunk gives exactly one chunk, `(0, 24)` | Empty or duplicate chunks for short pages like our PDF pages |
| `sliding_window_spans_match_the_worked_example` | The code gives the same spans as the table in section 2.1 | The doc and the code could quietly disagree |
| `chunk_text_matches_page_text_at_its_offsets` | `page.text[start:end]` gives back each chunk's text exactly | Evidence quotes and citations couldn't be traced back to the page |
| `no_text_is_lost_between_chunks` | First chunk starts at 0, last ends at the text's end, and there are no gaps between neighbours | Facts in a gap would be unfindable, a silent retrieval failure |
| `neighbouring_chunks_overlap_by_exactly_the_overlap` | Every pair of neighbours shares exactly 60 characters when overlap is 60 | The overlap setting wouldn't mean what it says |
| `every_quote_up_to_overlap_length_sits_inside_one_chunk` | All 40-character quotes appear whole in some chunk | Short golden-set quotes could be split, giving false retrieval failures |
| `chunks_copy_page_metadata_and_get_unique_ids` | `source` and `page` are copied, and IDs are `handbook.md:p3:c0`, `c1`, `c2` | Wrong citations; clashing IDs in the vector store (Phase 4) |
| `overlap_not_smaller_than_chunk_size_raises` | Overlap 100 with size 100 is refused with a clear message | An endless loop that freezes the program |

A few lines explained:

- `"x" * 1000` makes a string of 1,000 x's.
- `range(1, len(chunks))` counts 1, 2, ... up to the last chunk's position, so `index - 1` is always the previous chunk.
- `chunks[-1]` is the last item of a list.
- `assert found, f"..."`: the text after the comma is printed only if the assert fails. It tells you *which* quote was split, so a failure is easy to debug.
- The `break` stops searching once a quote is found, which keeps this test fast even though it checks over 1,000 quotes.

## 2.5 Run it

Run these from your project folder. Ollama doesn't need to be running.

```bash
uv run python -m ragqa.chunking
```

- **What it does:** loads `data/raw/` with the Phase 1 loader, splits every page into chunks of up to 500 characters with 100 characters of overlap, and prints each chunk's edges and its shared text.
- **The parts:** `uv run` uses the project's environment; `python -m ragqa.chunking` runs `chunking.py` inside the `ragqa` package. With no extra words, the default settings are used.
- **Why:** this is the plan's Phase 2 check: inspect the boundaries and the overlap by eye before Phase 3 turns chunks into numbers.

```bash
uv run pytest tests/unit/test_chunking.py -v
```

- **What it does:** runs only the 8 chunking tests.
- **The parts:** the file path after `pytest` limits the run to that one file; `-v` prints each test's name and result.

```bash
uv run pytest -m "not integration" -v
```

- **What it does:** runs every fast test, Phase 1's and Phase 2's together.
- **Why:** proves the new code didn't break the old code. Run this before every commit.

**Expected output of the chunker** (real output from my test run):

```text
--- brightwater_faq.txt:p1:c0 | chars 0-419 (419)
  starts: 'Brightwater Robotics - IT help desk FAQ Q: How do I reset m'
  ends:   'e in the Software Center app. Anything else needs a ticket.'
--- brightwater_handbook.md:p1:c0 | chars 0-500 (500)
  starts: '# Brightwater Robotics - Employee Handbook Brightwater Robo'
  ends:   'pproval from People Operations. ## Leave - Annual leave: 2'
--- brightwater_handbook.md:p1:c1 | chars 400-900 (500)
  starts: 'rking days per calendar year and needs approval from People'
  ends:   'get of 600 euros. Laptops are replaced every 3 years. Lost e'
  shares with previous: 'rking days per calendar year and needs approval from People Operations. ## Leave - Annual leave: 2'
--- brightwater_handbook.md:p1:c2 | chars 800-1106 (306)
  starts: 'a laptop and a one-time home-office budget of 600 euros. La'
  ends:   'rses, books and conferences. The budget does not roll over.'
  shares with previous: 'a laptop and a one-time home-office budget of 600 euros. Laptops are replaced every 3 years. Lost e'
--- brightwater_travel_policy.pdf:p1:c0 | chars 0-306 (306)
  starts: 'Brightwater Robotics - Travel and Expenses Policy Version 2.'
  ends:   'y your team lead at least 10 working days before departure.'
--- brightwater_travel_policy.pdf:p2:c0 | chars 0-272 (272)
  starts: '2. Booking travel Trains are the default for trips under 4 h'
  ends:   's in most cities and 190 euros in London, Paris and Zurich.'
--- brightwater_travel_policy.pdf:p4:c0 | chars 0-193 (193)
  starts: '4. Daily allowance and claims The daily meal allowance is 45'
  ends:   'days of returning; late claims are rejected automatically.'

chunk_size=500, overlap=100
5 pages -> 7 chunks
Chunk length: min 193, average 357, max 500
```

What to notice:

| What you see | What it means |
| --- | --- |
| `5 pages -> 7 chunks` | Only the handbook (1,106 characters) is longer than 500, so only it is split: into 3 chunks. Every other page fits in 1 chunk |
| Handbook chunks at `0-500`, `400-900`, `800-1106` | Each window starts 400 characters (the step) after the previous one. The last stops at the end of the text, so it's shorter (306) |
| `shares with previous` is 100 characters | The overlap, visible. The shared text is exactly the end of the previous chunk |
| `c0` ends with `Annual leave: 2` | **A fact cut in half.** "27 days" is split across a boundary. Thanks to the overlap, `c1` holds the whole sentence; I checked that "Annual leave: 27 days per year" is inside `c1` and not inside `c0` |
| `c1` starts with `rking days...` and ends with `Lost e` | Mid-word cuts. Expected with fixed character windows (design choice in 2.1). Embeddings cope with this reasonably well; evaluation will tell us if it hurts |
| PDF chunk IDs `p1`, `p2`, `p4` | Page numbers from Phase 1 survive chunking. There's no `p3` because the blank page never became a `Page` |
| `min 193, average 357, max 500` | No chunk is bigger than the chunk size, and no chunk is empty |

**Expected output of the chunking tests** (real):

```text
tests/unit/test_chunking.py::test_short_text_becomes_one_chunk PASSED    [ 12%]
tests/unit/test_chunking.py::test_sliding_window_spans_match_the_worked_example PASSED [ 25%]
tests/unit/test_chunking.py::test_chunk_text_matches_page_text_at_its_offsets PASSED [ 37%]
tests/unit/test_chunking.py::test_no_text_is_lost_between_chunks PASSED  [ 50%]
tests/unit/test_chunking.py::test_neighbouring_chunks_overlap_by_exactly_the_overlap PASSED [ 62%]
tests/unit/test_chunking.py::test_every_quote_up_to_overlap_length_sits_inside_one_chunk PASSED [ 75%]
tests/unit/test_chunking.py::test_chunks_copy_page_metadata_and_get_unique_ids PASSED [ 87%]
tests/unit/test_chunking.py::test_overlap_not_smaller_than_chunk_size_raises PASSED [100%]
============================== 8 passed in 0.48s ===============================
```

The full run (`-m "not integration"`) gave `18 passed in 0.51s` on my test copy: 8 chunking tests, 9 loader tests, and the Phase 1 exercise test. Your total depends on which earlier tests you've written.

## 2.6 Common errors and debugging

The error lines below are real outputs from my test run (the last line of each traceback).

| What you see | What it means | How to fix it |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'ragqa.loaders'` | Phase 1's `loaders.py` is missing or misnamed | Finish Phase 1 first; check `src/ragqa/loaders.py` exists |
| `ImportError: cannot import name 'Page' from 'ragqa.loaders'` | Your Phase 1 code uses a different name for the record | Use the exact names from Phase 1: `Page`, `load_folder`, `DEFAULT_DATA_FOLDER` |
| `ValueError: overlap (100) must be smaller than chunk_size (100)` | You ran `ragqa.chunking 100 100`. The window would never move | Pick an overlap smaller than the size, e.g. `500 100` |
| `ValueError: chunk_size must be above 0, got 0` | Chunk size 0 can't hold any text | Use a positive size |
| `ValueError: overlap cannot be negative, got -5` | A negative overlap would leave gaps between chunks | Use 0 or more |
| `ValueError: invalid literal for int() with base 10: 'abc'` | A setting on the command line isn't a whole number | Type numbers only: `ragqa.chunking 300 50` |
| `Usage: uv run python -m ragqa.chunking [CHUNK_SIZE OVERLAP]` | You gave one setting instead of two (or three or more) | Give both numbers, or none for the defaults |
| A chunk count other than `5 pages -> 7 chunks` | Your sample text differs from Phase 1's. The handbook gives 3 chunks only if it's longer than 900 characters | Compare it with Phase 1, Step 2; the exact count depends on the exact text |
| A test fails with `assert [(0, 4), (4, 8), ...] == [(0, 4), (3, 7), ...]` | `step` is computed as `chunk_size` instead of `chunk_size - overlap`, so there's no overlap | Fix the `step` line. The worked-example test exists to catch exactly this |
| The program never finishes (no output, CPU busy) | The `while True` loop never reaches `break`, usually because `start` isn't moved forward | Press Ctrl + C to stop it. Check the `start = start + step` line and the overlap check |

**Debugging techniques for this phase**

```bash
uv run python -c "from ragqa.chunking import split_text; print(split_text('abcdefghij', 4, 1))"
```

- **What it does:** runs `split_text` on the tiny example from section 2.1 and prints the spans.
- **Why:** when chunk counts look wrong, test the arithmetic on text so small you can check it by hand. Real output: `[(0, 4), (3, 7), (6, 10)]`.
- **The parts:** `-c "..."` runs the short Python code in the quotes; `;` separates the two statements.

```bash
uv run python -m ragqa.chunking 300 0
```

- **What it does:** runs the chunker with no overlap.
- **Why:** removing one feature at a time shows what it was doing. With overlap 0, every `shares with previous` line prints an empty `''`, and the count goes from 7 to 10 chunks (see the exercise).

```bash
uv run pytest tests/unit/test_chunking.py -k overlap -v
```

- **What it does:** runs only the chunking tests with "overlap" in their name (3 tests).
- **The parts:** `-k overlap` keeps only tests whose name matches `overlap`.
- **Why:** quick feedback while you work on one part of the code.

**Find which chunk holds a fact.** This prints the ID of every chunk that contains a phrase:

```bash
uv run python -c "from ragqa.chunking import chunk_pages; from ragqa.loaders import load_folder, DEFAULT_DATA_FOLDER; pages, skipped = load_folder(DEFAULT_DATA_FOLDER); print([c.chunk_id for c in chunk_pages(pages, 500, 100) if '27 days' in c.text])"
```

- **What it does:** loads and chunks the documents, then keeps only the IDs of chunks whose text contains "27 days".
- **Why:** this is a hand-made version of what Phase 5 automates: "is the evidence in a chunk, and which one?" Change `'27 days'` to any phrase you want to trace.
- **The parts:** `[... for c in ... if ...]` is a *list comprehension*, a one-line loop that builds a list. Fine for a quick check in the Terminal; in project code we write the loop out in full.

## 2.7 Exercise: try different settings, then write a test

**Part A — Change the settings and compare.** Run the chunker four times, with the settings passed on the command line as `CHUNK_SIZE OVERLAP`:

```bash
uv run python -m ragqa.chunking 500 100
```

```bash
uv run python -m ragqa.chunking 300 0
```

```bash
uv run python -m ragqa.chunking 200 40
```

```bash
uv run python -m ragqa.chunking 1000 100
```

- **What they do:** the same chunker each time, with a different chunk size (first number) and overlap (second number).
- **Why:** this is your first small **experiment**: change one setting, keep everything else the same, compare results. Phase 13 does exactly this, with evaluation numbers instead of eyeballing.

For each run, copy the last two lines into a table like this one. Then compare with my real results:

| Settings (size, overlap) | Chunks | Shortest | Average | Longest |
| --- | --- | --- | --- | --- |
| 500, 100 (default) | 7 | 193 | 357 | 500 |
| 300, 0 | 10 | 6 | 230 | 300 |
| 200, 40 | 15 | 99 | 180 | 200 |
| 1000, 100 | 6 | 193 | 399 | 1000 |

Now use the "find which chunk holds a fact" command from section 2.6 to look up these two handbook facts under `500 100` and under `300 0`. Change `'27 days'` to each phrase, and the `500, 100` inside the command to the settings you're testing:

- `up to 5 days can be carried over to the next year` (49 characters)
- `Lost equipment must be reported to the IT desk` (46 characters)

Answer these in a few words each:

1. Under `300 0`, which chunk holds each fact? (My result: **none**, for both. Under `500 100`, they're in `c1` and `c2`.) Why? Look at where the `300 0` chunks start and end.
2. `300 0` made a chunk of only 6 characters. Print it (hint: `brightwater_travel_policy.pdf:p1:c1`). What text is in it, and why would a chunk like that be useless or even harmful during retrieval?
3. `1000 100` gives the fewest chunks. What is lost when one chunk holds the whole handbook's first 1,000 characters? Think about what Phase 6 sends to the LLM, and how precise a citation can be.
4. Which setting would you choose, and what evidence would convince you? (Hint: Phase 5's hit@k is that evidence.)

**Part B — Write one test yourself.** Add a test to `tests/unit/test_chunking.py` called `test_chunk_pages_never_mixes_two_pages`. It proves the design rule "a chunk never spans two pages".

- Build two `Page`s from the same source with page numbers 1 and 2. Fill the first with `"A" * 450` and the second with `"B" * 450`.
- Run `chunking.chunk_pages([first_page, second_page], chunk_size=300, overlap=50)`.
- Assert that no page-1 chunk contains a `"B"`, no page-2 chunk contains an `"A"`, and that there are 4 chunks in total (2 per page).

Then break the rule on purpose to see the test catch it: in `chunk_pages`, change the loop so it chunks one long string made by joining all page texts. Run the test, read the failure, and change the code back.

## 2.8 Done criteria

- [ ] `uv run python -m ragqa.chunking` prints `5 pages -> 7 chunks`, with 100 characters of shared text between the handbook's chunks
- [ ] `uv run pytest tests/unit/test_chunking.py -v` passes all 8 tests (9 with your own)
- [ ] `uv run pytest -m "not integration" -v` still passes every Phase 1 test too
- [ ] Your settings table from Part A is filled in, and you can explain why `300 0` lost both handbook facts
- [ ] You can explain in your own words why overlap protects any quote shorter than the overlap
- [ ] Your work is committed: `git add .` then `git commit -m "Phase 2: chunking"`, then `git push`
- [ ] Phase 2 ticked in the tracker of the [Phase 0 doc](https://claude.ai/code/artifact/b32269ff-ce8f-4186-b984-8cf68f206b1b)

When these pass, share your settings table and answers, and we'll start Phase 3: turning chunks into embeddings.
