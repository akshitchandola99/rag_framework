# RAG Quality Lab — Phase 1: Load documents with metadata

Oct 13, 2026 · @Akshit

**Goal of this phase:** turn every file in `data/raw/` into a list of **pages**, where each page knows its text *and* where it came from (file name, page number, file type). No LLM is needed in this phase, so Ollama can stay off. Previous phase: Phase 0 [doc](https://claude.ai/code/artifact/b32269ff-ce8f-4186-b984-8cf68f206b1b).

## 1.1 Concepts

A RAG system can only answer from text it has loaded, and it can only cite a source if it remembers where each piece of text came from. This phase builds that first step.

**Words you'll meet in this phase**

| Term | Plain meaning | In our project |
| --- | --- | --- |
| Document | One file you want the chatbot to know about | `brightwater_handbook.md`, `brightwater_travel_policy.pdf` |
| Page | One piece of a document's text, plus labels | One page of a PDF; a whole `.md` or `.txt` file counts as page 1 |
| Metadata | "Data about data": labels that describe a piece of text, not the text itself | `source` (file name), `page` (page number), `file_type` |
| Text extraction | Pulling plain text out of a file format | For PDFs this needs a library; `.md` and `.txt` are already plain text |
| PyMuPDF | A Python library that opens PDFs and extracts their text, page by page | Imported as `pymupdf`. License: **AGPL-3.0** (see below) |
| Dataclass | A small Python class that just holds named fields, written with `@dataclass` | Our `Page` record |
| `Path` | Python's object for a file or folder location (from `pathlib`) | `PROJECT_ROOT / "data" / "raw"` |
| Encoding | The rule that turns bytes on disk into letters. **UTF-8** is the modern standard | We read `.md` and `.txt` as UTF-8 |
| Exception | Python's way of saying "something went wrong". `raise` creates one; `try`/`except` catches one | We raise `ValueError` for bad files |

**How it connects to the architecture.** The indexing flow from the plan runs in this order. Phase 1 is step 1; every later step works on the `Page` records it produces.

1. **Load** (Phase 1): files → `Page` records with text and metadata.
2. **Chunk** (Phase 2): split each page into smaller pieces. Each chunk copies its page's `source` and `page`.
3. **Embed** (Phase 3): turn each chunk into numbers for searching.
4. **Store** (Phase 4): save chunks, numbers and metadata in Chroma.

The metadata is what makes evaluation possible later. The golden set (Phase 5) labels evidence as `{source, page, quote}`, and citations (Phase 7) point back to a source and page. That's why our field names are exactly `source` and `page`.

**Design choices for this phase**

| Question | Our choice | Why |
| --- | --- | --- |
| How do we split a PDF? | One `Page` per PDF page | Page numbers are what a person checks when verifying a citation |
| How do we split `.md` / `.txt`? | The whole file is one `Page`, numbered 1 | These formats have no pages. Splitting into smaller pieces is Phase 2's job |
| What about blank PDF pages? | Skip them, but keep counting | The meal-allowance rule is on page 4 of the PDF, so its citation must say page 4, even if page 3 is blank |
| Do we clean the text? | No, keep it exactly as extracted | Golden-set quotes must match the stored text word for word. Phase 2 decides what cleaning is safe |
| What happens with a bad file (empty, broken, wrong type)? | The loader raises `ValueError`; the folder loader skips the file and reports why | One bad file must never stop the whole run (the plan's robustness rule) |
| Hidden files like `.gitkeep`, `.DS_Store`? | Ignored silently | They're not documents |

**About the PyMuPDF license.** PyMuPDF is free and open source under **AGPL-3.0**. That's fine for a public portfolio repo. It becomes a problem only if you build closed-source or commercial software with it. All PDF code lives in one function, `load_pdf()`, so swapping to `pypdf` (BSD license, permissive) later is a five-minute change.

## 1.2 Setup steps

Run every command from your project folder (`/Users/nowgg/Automation/rag_framework`).

### Step 1 — Install PyMuPDF

```bash
uv add pymupdf
```

- **What it does:** downloads PyMuPDF, adds it to `dependencies` in `pyproject.toml`, and records the exact version in `uv.lock`.
- **Why:** we need it to read text out of PDFs. `.md` and `.txt` need nothing extra; Python reads them on its own.
- **The parts:** `uv add` installs a package *and* saves it as a project requirement. No `--dev` here, because the app itself needs PyMuPDF, not just the tests.

On my test run this installed `pymupdf==1.28.2`, and `pyproject.toml` now lists `"pymupdf>=1.28.2"` next to `requests`.

**Old tutorials write `import fitz`.** That's the library's old name. Current versions use `import pymupdf`, and that's what we use.

### Step 2 — Add two sample documents

The plan asks for **fictional** documents. If the model answers a question about an imaginary company correctly, it must have used our text, not something it memorised during training. That's what we want to test.

Our imaginary company is **Brightwater Robotics**. Create `data/raw/brightwater_handbook.md` in VS Code and paste:

```markdown
# Brightwater Robotics - Employee Handbook

Brightwater Robotics builds warehouse robots. This handbook is version 3, updated in January 2026.

## Working hours

Core hours are 10:00 to 15:00, Monday to Thursday. Fridays are "focus Fridays": no meetings are allowed before 14:00.

## Remote work

Employees may work remotely up to 3 days per week. Remote work from another country is limited to 20 working days per calendar year and needs approval from People Operations.

## Leave

- Annual leave: 27 days per year, plus public holidays.
- Unused leave: up to 5 days can be carried over to the next year. Carried-over days expire on 31 March.
- Sick leave: tell your team lead before 10:00 on the first day. A doctor's note is needed after 4 days in a row.

## Equipment

Every new employee receives a laptop and a one-time home-office budget of 600 euros. Laptops are replaced every 3 years. Lost equipment must be reported to the IT desk within 24 hours.

## Learning budget

Each employee has a learning budget of 1,200 euros per year for courses, books and conferences. The budget does not roll over.
```

Then create `data/raw/brightwater_faq.txt`:

```text
Brightwater Robotics - IT help desk FAQ

Q: How do I reset my password?
A: Use the self-service page at id.brightwater.example. Passwords expire every 180 days.

Q: Who do I call outside office hours?
A: The on-call IT number is extension 4400. It is staffed from 07:00 to 22:00 every day.

Q: Can I install my own software?
A: Only from the approved catalogue in the Software Center app. Anything else needs a ticket.
```

Every fact here is specific (a number, a time, a name). Specific facts make good golden-set questions later, because an answer is clearly right or wrong.

### Step 3 — Create the sample PDF with a script

Instead of downloading a PDF, we write a tiny script that *creates* one. Three reasons: you know exactly what text is inside, anyone can rebuild it, and the tests in section 1.4 reuse the same function to make test PDFs. The PDF has **4 pages, and page 3 is blank on purpose**, to prove that page numbers stay correct.

Create `src/ragqa/make_sample_pdf.py`:

```python
"""Phase 1 helper: write the fictional sample PDF into data/raw/."""

from pathlib import Path

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "brightwater_travel_policy.pdf"

PAGE_TEXTS = [
    (
        "Brightwater Robotics - Travel and Expenses Policy\n"
        "Version 2.1, effective 1 March 2026\n\n"
        "1. Who this policy covers\n"
        "This policy applies to every Brightwater employee and contractor who travels "
        "for work. Travel must be approved in the Compass portal by your team lead "
        "at least 10 working days before departure."
    ),
    (
        "2. Booking travel\n"
        "Trains are the default for trips under 4 hours. Flights must be economy class, "
        "except for flights longer than 8 hours, which may be premium economy.\n\n"
        "3. Hotels\n"
        "The nightly hotel limit is 140 euros in most cities and 190 euros in London, "
        "Paris and Zurich."
    ),
    "",
    (
        "4. Daily allowance and claims\n"
        "The daily meal allowance is 45 euros. Alcohol is never reimbursed. "
        "Submit receipts in Compass within 30 days of returning; late claims are "
        "rejected automatically."
    ),
]


def write_pdf(output_path: Path, page_texts: list[str]) -> None:
    pdf_document = pymupdf.open()
    for page_text in page_texts:
        new_page = pdf_document.new_page()
        if page_text != "":
            text_area = pymupdf.Rect(72, 72, 523, 770)
            new_page.insert_textbox(text_area, page_text, fontsize=11, fontname="helv")
    pdf_document.save(output_path)
    pdf_document.close()


def main() -> None:
    write_pdf(OUTPUT_PATH, PAGE_TEXTS)
    print(f"Wrote {OUTPUT_PATH} with {len(PAGE_TEXTS)} pages (page 3 is blank on purpose)")


if __name__ == "__main__":
    main()
```

**What the script does, part by part**

- **`PROJECT_ROOT` and `OUTPUT_PATH`** build an **absolute path**: the full location from the top of your disk, like `/Users/nowgg/Automation/rag_framework/data/raw/brightwater_travel_policy.pdf`. Read `PROJECT_ROOT` left to right:
  - `__file__` is the location of this Python file: `.../rag_framework/src/ragqa/make_sample_pdf.py`.
  - `.resolve()` turns it into a full, clean absolute path.
  - `.parents[2]` walks up the folder tree: `parents[0]` is `src/ragqa`, `parents[1]` is `src`, and `parents[2]` is the project folder.
  - `PROJECT_ROOT / "data" / "raw" / ...` joins folder names with `/`, the same way as `tmp_path / "file.md"` in the tests.
  - **Why not just `Path("data/raw/...")`?** That's a *relative* path: Python looks for it inside whatever folder the Terminal is in. Run the script from anywhere else and it fails or writes to the wrong place. The absolute path works from any folder. This project always uses this pattern for its own files.
- **`PAGE_TEXTS`** is a list with one string per page. Strings inside brackets `( ... )` that sit next to each other are glued into one string, which keeps long lines readable. `\n` means "new line". The third item is `""` (empty), so page 3 stays blank.
- **`write_pdf(output_path, page_texts)`** takes a file location and a list of page texts, and writes a PDF. It returns nothing.
  - `pymupdf.open()` with no file name creates a new, empty PDF in memory.
  - `new_page()` adds a blank A4-sized page (595 × 842 points; 1 point = 1/72 inch).
  - `pymupdf.Rect(72, 72, 523, 770)` is the box the text goes in: left, top, right, bottom, in points. 72 points = 1 inch of margin on each side.
  - `insert_textbox(...)` writes the text into that box and wraps long lines. `"helv"` is Helvetica, a font every PDF reader has built in.
  - `save(...)` writes the file to disk; `close()` frees the memory.
- **`main()`** calls `write_pdf` with our sample texts and prints a confirmation.

Run it once:

```bash
uv run python -m ragqa.make_sample_pdf
```

- **What it does:** runs the script, which writes `data/raw/brightwater_travel_policy.pdf`.
- **The parts:** `uv run` uses the project's environment; `python -m ragqa.make_sample_pdf` runs the file `make_sample_pdf.py` inside the `ragqa` package.

Real output on my test run (with your project path; it prints the full path because OUTPUT\_PATH is absolute):

```text
Wrote /Users/nowgg/Automation/rag_framework/data/raw/brightwater_travel_policy.pdf with 4 pages (page 3 is blank on purpose)
```

Open the PDF in Preview to see it. Running the script again simply overwrites the file with the same content.

## 1.3 Code: the document loader

This file reads every document in a folder and returns a list of `Page` records. It has one small function per job: one for text files, one for PDFs, one that picks the right one by file extension, and one for a whole folder. Phase 2 (chunking) will import `load_folder()` and `Page` from here.

Create `src/ragqa/loaders.py`:

```python
"""Phase 1: load .pdf, .md and .txt files into pages that carry metadata."""

import sys
from dataclasses import dataclass
from pathlib import Path

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_FOLDER = PROJECT_ROOT / "data" / "raw"
PREVIEW_LENGTH = 200


@dataclass
class Page:
    source: str
    page: int
    file_type: str
    text: str


def load_text_file(file_path: Path) -> list[Page]:
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{file_path.name}: not valid UTF-8 text ({error})") from error

    if text.strip() == "":
        raise ValueError(f"{file_path.name}: file has no text")

    file_type = file_path.suffix.lower().lstrip(".")
    whole_file_page = Page(source=file_path.name, page=1, file_type=file_type, text=text)
    return [whole_file_page]


def load_pdf(file_path: Path) -> list[Page]:
    try:
        pdf_document = pymupdf.open(file_path)
    except pymupdf.FileDataError as error:
        raise ValueError(f"{file_path.name}: not a readable PDF ({error})") from error

    pages = []
    with pdf_document:
        for page_index, pdf_page in enumerate(pdf_document):
            text = pdf_page.get_text()
            if text.strip() == "":
                # Blank page, or a scanned image with no text layer. Skip it,
                # but keep counting so later pages keep their real numbers.
                continue
            page_number = page_index + 1
            pages.append(Page(source=file_path.name, page=page_number, file_type="pdf", text=text))

    if len(pages) == 0:
        raise ValueError(f"{file_path.name}: PDF has no extractable text (empty or scanned)")
    return pages


def load_document(file_path: Path) -> list[Page]:
    extension = file_path.suffix.lower()
    if extension == ".pdf":
        return load_pdf(file_path)
    if extension == ".md" or extension == ".txt":
        return load_text_file(file_path)
    raise ValueError(f"{file_path.name}: unsupported file type '{extension}'")


def load_folder(folder_path: Path) -> tuple[list[Page], list[str]]:
    all_pages = []
    skipped_files = []

    for file_path in sorted(folder_path.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            # Hidden files such as .gitkeep and .DS_Store are not documents.
            continue

        try:
            pages = load_document(file_path)
        except ValueError as error:
            skipped_files.append(str(error))
            continue
        all_pages.extend(pages)

    return all_pages, skipped_files


def make_preview(text: str) -> str:
    one_line_text = " ".join(text.split())
    return one_line_text[:PREVIEW_LENGTH]


def main() -> None:
    if len(sys.argv) > 1:
        folder_path = Path(sys.argv[1])
    else:
        folder_path = DEFAULT_DATA_FOLDER

    if not folder_path.is_dir():
        print(f"Folder not found: {folder_path.resolve()}")
        print("Run this from the project folder, or pass a folder path.")
        sys.exit(1)

    pages, skipped_files = load_folder(folder_path)

    for page in pages:
        print(f"--- {page.source} | page {page.page} | {page.file_type} | {len(page.text)} chars")
        print(make_preview(page.text))
        print()

    source_names = set()
    for page in pages:
        source_names.add(page.source)

    print(f"Loaded {len(pages)} pages from {len(source_names)} files in {folder_path}")
    for reason in skipped_files:
        print(f"SKIPPED {reason}")


if __name__ == "__main__":
    main()
```

**What the code does, part by part**

- **The imports.** `sys` gives us the command-line words (`sys.argv`) and `sys.exit`. `dataclass` builds our record type. `Path` handles file locations. `pymupdf` reads PDFs.
- **The three CAPITAL settings.** `PROJECT_ROOT` is the project folder, worked out from this file's own location exactly as in `make_sample_pdf.py` (Step 3): `src/ragqa/loaders.py` → `parents[2]` → `/Users/nowgg/Automation/rag_framework`. `DEFAULT_DATA_FOLDER` is where documents live: `PROJECT_ROOT / "data" / "raw"`, an *absolute* path, so the loader finds `data/raw` whichever folder you run it from. `PREVIEW_LENGTH` is how many characters of each page we print: 200, as the plan's Phase 1 check asks.
- **`class Page` with `@dataclass`.** A record with four labelled fields. Writing `@dataclass` above the class makes Python create the setup code for us, so `Page(source="a.md", page=1, file_type="md", text="...")` just works, and printing a `Page` shows all four fields. Every later phase passes these records around.
- **`load_text_file(file_path)`** — input: the location of a `.md` or `.txt` file. Output: a list holding exactly one `Page`.
  - `read_text(encoding="utf-8")` reads the whole file as one string. If the bytes aren't valid UTF-8, Python raises `UnicodeDecodeError`; we turn it into a `ValueError` that names the file, so the skip message tells you *which* file failed.
  - `text.strip() == ""` checks for an empty file. `strip()` removes spaces and blank lines from both ends, so a file of only spaces also counts as empty.
  - `file_path.suffix` is the extension, like `".md"`. `.lower()` turns `.MD` into `.md`; `.lstrip(".")` removes the dot, giving `"md"`.
  - `file_path.name` is just the file name (`brightwater_handbook.md`), not the full folder path. That's our `source`.
  - We return a *list* even though it holds one page, so every loader has the same output shape. Code that calls a loader never has to care which one ran.
- **`load_pdf(file_path)`** — input: a PDF location. Output: one `Page` per PDF page that has text.
  - `pymupdf.open(file_path)` opens the PDF. A broken file makes it raise `FileDataError` (an empty 0-byte file raises `EmptyFileError`, a kind of `FileDataError`). We turn both into a `ValueError` with the file name.
  - `with pdf_document:` closes the PDF automatically when the block ends, even if an error happens inside.
  - `enumerate(pdf_document)` walks through the pages and gives each one a counter, `page_index`, starting at **0**. People count pages from **1**, so `page_number = page_index + 1`.
  - `pdf_page.get_text()` extracts that page's text as one string, with `\n` at each line end.
  - A page with no text is skipped with `continue` ("go straight to the next page"). The counter still moves on, so page 4 stays page 4.
  - If *no* page had text, the whole PDF is useless for us, so we raise `ValueError`. Scanned PDFs (photos of paper) end up here: they contain pictures, not text. Reading them needs OCR (optical character recognition), which is out of scope.
- **`load_document(file_path)`** looks at the extension and calls the right loader. Anything else raises `ValueError("... unsupported file type ...")`. Adding `.docx` support one day means adding one `if` here.
- **`load_folder(folder_path)`** — input: a folder. Output: *two* things, returned together as a tuple (a fixed-size group of values): all pages, and a list of skip reasons.
  - `sorted(folder_path.iterdir())` lists the folder's contents in alphabetical order, so the output is the same every run. Repeatable order matters for testing.
  - `is_file()` skips sub-folders; `startswith(".")` skips hidden files.
  - The `try` / `except ValueError` block is the robustness rule in action: a bad file adds a reason to `skipped_files`, and the loop carries on with the next file.
  - `all_pages.extend(pages)` adds every page from this file to the big list (`append` would add the list itself as one item).
- **`make_preview(text)`** turns the page into one short line for printing. `text.split()` breaks the text at every space and line break; `" ".join(...)` glues the words back with single spaces; `[:PREVIEW_LENGTH]` keeps the first 200 characters. This is only for display: the stored `Page.text` is untouched.
- **`main()`**
  - `sys.argv` is the list of words on the command line. `sys.argv[0]` is the program itself, so `len(sys.argv) > 1` means "the user typed a folder name". A typed folder is used as typed, so a relative one like `data/adversarial` is read from the folder you're in. Otherwise we use `DEFAULT_DATA_FOLDER`, the absolute path to `data/raw`.
  - If the folder doesn't exist, we print where we looked (`resolve()` turns a relative path into the full one) and stop with exit code 1, instead of a long traceback.
  - For each page we print one header line (source, page, type, character count) and the preview.
  - `set()` is a collection that keeps only unique values. Adding the same file name three times still counts once, which gives us the number of files.
  - Finally we print the totals and every skip reason.

## 1.4 Tests

Nine unit tests: one per file type, one for page numbering, and one for each kind of bad file. They're all **unit** tests: no Ollama, no network, done in under a second. This phase has no integration test, because nothing here talks to a model.

Two new pytest tools appear here:

- **`tmp_path`** is a *fixture*: a ready-made helper that pytest hands to any test that names it as a parameter. It's a brand-new empty folder, different for every test and deleted later. Each test writes its own small files there, so tests never depend on (or change) your real `data/raw/` folder.
- **`pytest.raises(ValueError, match="...")`** means "the code inside this `with` block must raise a `ValueError`, and its message must contain this text". If no error happens, or a different one does, the test fails. This is how we test that bad input is *rejected* properly.

The PDF tests reuse `write_pdf()` from `make_sample_pdf.py` to build small PDFs, instead of writing new PDF code.

Create `tests/unit/test_loaders.py`:

```python
from pathlib import Path

import pytest

from ragqa import loaders
from ragqa.make_sample_pdf import write_pdf


def test_markdown_file_becomes_one_page_with_metadata(tmp_path: Path):
    file_path = tmp_path / "handbook.md"
    file_path.write_text("# Leave\n\nAnnual leave is 27 days.", encoding="utf-8")

    pages = loaders.load_document(file_path)

    assert len(pages) == 1
    assert pages[0].source == "handbook.md"
    assert pages[0].page == 1
    assert pages[0].file_type == "md"
    assert "27 days" in pages[0].text


def test_txt_file_is_loaded(tmp_path: Path):
    file_path = tmp_path / "faq.txt"
    file_path.write_text("Passwords expire every 180 days.", encoding="utf-8")

    pages = loaders.load_document(file_path)

    assert pages[0].file_type == "txt"
    assert pages[0].text == "Passwords expire every 180 days."


def test_pdf_keeps_real_page_numbers_and_skips_blank_pages(tmp_path: Path):
    file_path = tmp_path / "policy.pdf"
    write_pdf(file_path, ["First page text", "", "Third page text"])

    pages = loaders.load_document(file_path)

    page_numbers = []
    for page in pages:
        page_numbers.append(page.page)
    assert page_numbers == [1, 3]
    assert "Third page text" in pages[1].text
    assert pages[1].source == "policy.pdf"


def test_empty_text_file_raises(tmp_path: Path):
    file_path = tmp_path / "empty.md"
    file_path.write_text("   \n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no text"):
        loaders.load_document(file_path)


def test_non_utf8_text_file_raises(tmp_path: Path):
    file_path = tmp_path / "menu.txt"
    file_path.write_bytes(b"Caf\xe9 menu")

    with pytest.raises(ValueError, match="not valid UTF-8"):
        loaders.load_document(file_path)


def test_pdf_with_only_blank_pages_raises(tmp_path: Path):
    file_path = tmp_path / "blank.pdf"
    write_pdf(file_path, ["", ""])

    with pytest.raises(ValueError, match="no extractable text"):
        loaders.load_document(file_path)


def test_corrupted_pdf_raises(tmp_path: Path):
    file_path = tmp_path / "broken.pdf"
    file_path.write_bytes(b"this is not really a pdf")

    with pytest.raises(ValueError, match="not a readable PDF"):
        loaders.load_document(file_path)


def test_unsupported_file_type_raises(tmp_path: Path):
    file_path = tmp_path / "notes.docx"
    file_path.write_text("some text", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported file type"):
        loaders.load_document(file_path)


def test_folder_loading_skips_bad_and_hidden_files(tmp_path: Path):
    (tmp_path / "good.md").write_text("Real content", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    (tmp_path / "notes.docx").write_text("x", encoding="utf-8")
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")

    pages, skipped_files = loaders.load_folder(tmp_path)

    assert len(pages) == 1
    assert pages[0].source == "good.md"
    assert len(skipped_files) == 2
```

**What the tests check, and why each matters**

| Test | What it proves | What would break later without it |
| --- | --- | --- |
| `markdown_file_becomes_one_page_with_metadata` | All four fields are filled correctly | Citations would show the wrong source or type |
| `txt_file_is_loaded` | Text is kept exactly, character for character | Golden-set quotes wouldn't match stored text |
| `pdf_keeps_real_page_numbers_and_skips_blank_pages` | Pages 1 and 3 come back as `[1, 3]`, not `[1, 2]` | Every citation after a blank page would point at the wrong page |
| `empty_text_file_raises` | A file of only spaces is rejected | Empty pages would become empty chunks in Phase 2 |
| `non_utf8_text_file_raises` | Bytes like `\xe9` ("é" in an old encoding) are reported with the file name | A confusing crash with no file name |
| `pdf_with_only_blank_pages_raises` | A PDF with no text (like a scanned one) is rejected | Silent "success" with zero pages |
| `corrupted_pdf_raises` | A fake or damaged PDF gives a clear `ValueError` | The whole run would crash on one bad file |
| `unsupported_file_type_raises` | `.docx` is refused with a clear reason | Unknown files would be ignored without a word |
| `folder_loading_skips_bad_and_hidden_files` | 1 good page, 2 skip reasons, `.gitkeep` ignored | One bad file would stop indexing the folder |

In the code, `tmp_path / "handbook.md"` uses `/` to join a folder and a file name into one `Path`. `b"..."` is a *bytes* value: raw bytes written as-is, which lets us create a file that isn't valid UTF-8 on purpose.

**Hide a harmless PyMuPDF warning.** Without this, pytest prints 5 lines like `DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute` after every run. They come from inside PyMuPDF's compiled code, not from ours, and they're harmless. Add this to the end of the `[tool.pytest.ini_options]` section in `pyproject.toml`, under `markers`:

```toml
filterwarnings = [
    "ignore:builtin type [Ss]wig:DeprecationWarning",
]
```

- **What it does:** tells pytest to ignore `DeprecationWarning`s whose message starts with "builtin type Swig" (or "swig").
- **The parts:** each entry is `action:message:category`. `ignore` is the action; `builtin type [Ss]wig` is the start of the message (`[Ss]` means "capital or small s"); `DeprecationWarning` is the warning type. Other warnings still show, which is what we want.

## 1.5 Run it

Run these from your project folder. Ollama doesn't need to be running.

```bash
uv run python -m ragqa.loaders
```

- **What it does:** loads every document in `data/raw/` and prints each page's source, page number, type, length and first 200 characters, then a summary.
- **The parts:** `uv run` uses the project's environment; `python -m ragqa.loaders` runs `loaders.py` inside the `ragqa` package. With no extra word, it reads the default folder `data/raw`.
- **Why:** this is the plan's Phase 1 check. You *see* what the loader produced before any later phase builds on it.

```bash
uv run pytest -m "not integration" -v
```

- **What it does:** runs all fast tests, including the 9 new ones.
- **The parts:** `-m "not integration"` skips tests labelled `integration` (they need Ollama); `-v` prints each test's name and result.
- **Why:** proves the loader handles every file type and every bad-file case, in under a second.

**Expected output of the loader** (real output from my test run):

```text
--- brightwater_faq.txt | page 1 | txt | 419 chars
Brightwater Robotics - IT help desk FAQ Q: How do I reset my password? A: Use the self-service page at id.brightwater.example. Passwords expire every 180 days. Q: Who do I call outside office hours? A

--- brightwater_handbook.md | page 1 | md | 1106 chars
# Brightwater Robotics - Employee Handbook Brightwater Robotics builds warehouse robots. This handbook is version 3, updated in January 2026. ## Working hours Core hours are 10:00 to 15:00, Monday to 

--- brightwater_travel_policy.pdf | page 1 | pdf | 306 chars
Brightwater Robotics - Travel and Expenses Policy Version 2.1, effective 1 March 2026 1. Who this policy covers This policy applies to every Brightwater employee and contractor who travels for work. T

--- brightwater_travel_policy.pdf | page 2 | pdf | 272 chars
2. Booking travel Trains are the default for trips under 4 hours. Flights must be economy class, except for flights longer than 8 hours, which may be premium economy. 3. Hotels The nightly hotel limit

--- brightwater_travel_policy.pdf | page 4 | pdf | 193 chars
4. Daily allowance and claims The daily meal allowance is 45 euros. Alcohol is never reimbursed. Submit receipts in Compass within 30 days of returning; late claims are rejected automatically.

Loaded 5 pages from 3 files in /Users/nowgg/Automation/rag_framework/data/raw
```

What to notice:

- **Files come in alphabetical order** (`faq`, `handbook`, `travel_policy`), because of `sorted(...)`.
- **The PDF jumps from page 2 to page 4.** Page 3 is blank, so it's skipped, and page 4 keeps its real number. This is the most important line in the output.
- **`.gitkeep` isn't listed**, and there are no `SKIPPED` lines, because every real file loaded.
- **Previews are cut at exactly 200 characters**, sometimes mid-word ("...Monday to"). That's only the display; the stored text is complete.
- **Character counts** (419, 1106, 306...) should match yours exactly, if you copied the sample text exactly. A different number means a typo or extra blank lines.

**What one `Page` record looks like inside Python.** This prints the third record from the PDF:

```bash
uv run python -c "from ragqa.loaders import DEFAULT_DATA_FOLDER, load_document; print(load_document(DEFAULT_DATA_FOLDER / 'brightwater_travel_policy.pdf')[2])"
```

- **What it does:** loads the PDF and prints item `[2]` of the result, which is the third `Page` (lists count from 0).
- **The parts:** `-c "..."` runs the short Python code in the quotes; `;` separates statements on one line. Handy for quick checks, but real code goes in a file.

Real output:

```text
Page(source='brightwater_travel_policy.pdf', page=4, file_type='pdf', text='4. Daily allowance and claims\nThe daily meal allowance is 45 euros. Alcohol is never reimbursed. Submit receipts in\nCompass within 30 days of returning; late claims are rejected automatically.\n')
```

| Field | Value | Meaning |
| --- | --- | --- |
| `source` | `brightwater_travel_policy.pdf` | File name only, no folder |
| `page` | `4` | Real PDF page number, counted from 1 (the third record, but the fourth page) |
| `file_type` | `pdf` | Extension without the dot |
| `text` | 193 characters | Raw extracted text. `\n` marks where each line ended on the PDF page, including in the middle of a sentence ("...receipts in\\nCompass") |

Those mid-sentence `\n` breaks are normal for PDFs: the PDF stores lines, not sentences. Phase 2 deals with them when it splits text into chunks.

**Expected output of the tests** (real, shortened):

```text
tests/unit/test_loaders.py::test_markdown_file_becomes_one_page_with_metadata PASSED [ 11%]
tests/unit/test_loaders.py::test_txt_file_is_loaded PASSED               [ 22%]
tests/unit/test_loaders.py::test_pdf_keeps_real_page_numbers_and_skips_blank_pages PASSED [ 33%]
tests/unit/test_loaders.py::test_empty_text_file_raises PASSED           [ 44%]
tests/unit/test_loaders.py::test_non_utf8_text_file_raises PASSED        [ 55%]
tests/unit/test_loaders.py::test_pdf_with_only_blank_pages_raises PASSED [ 66%]
tests/unit/test_loaders.py::test_corrupted_pdf_raises PASSED             [ 77%]
tests/unit/test_loaders.py::test_unsupported_file_type_raises PASSED     [ 88%]
tests/unit/test_loaders.py::test_folder_loading_skips_bad_and_hidden_files PASSED [100%]

============================== 9 passed in 0.48s ===============================
```

If you also have the Phase 0 unit tests in `tests/unit/`, they run too, so your total will be higher than 9.

## 1.6 Common errors and debugging

The `SKIPPED` and `Folder not found` messages below are real outputs from my test run.

| What you see | What it means | How to fix it |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'pymupdf'` | PyMuPDF isn't installed in the project's environment | Run `uv add pymupdf` (Step 1); always start commands with `uv run` |
| `warning: The fitz API is deprecated...` | Code (yours or copied) uses the old name `import fitz` | Change it to `import pymupdf` |
| `Folder not found: /private/tmp/data/adversarial` | You typed a relative folder (`data/adversarial`) while the Terminal was in another folder (`/tmp`). Typed folders are read from where you are; only the default `data/raw` is absolute | `cd` into the project folder, or type the full path: `uv run python -m ragqa.loaders /Users/nowgg/Automation/rag_framework/data/adversarial` |
| `Loaded 0 pages from 0 files in /Users/nowgg/Automation/rag_framework/data/raw` | The folder only has `.gitkeep` | Do Steps 2 and 3 to add the sample documents |
| `SKIPPED menu.txt: not valid UTF-8 text ('utf-8' codec can't decode byte 0xe9 ...)` | The file was saved in an older encoding (for example Latin-1, where `é` is byte `0xe9`) | Open it in VS Code, click the encoding name at the bottom right, choose **Save with Encoding → UTF-8** |
| `SKIPPED corrupted.pdf: not a readable PDF (Failed to open file ... as type pdf.)` | The file is damaged, or isn't really a PDF (only its name ends in `.pdf`) | Open it in Preview. If Preview fails too, get a fresh copy |
| `SKIPPED x.pdf: PDF has no extractable text (empty or scanned)` | The PDF holds pictures of pages, not text | Use a PDF with real text. OCR is out of scope for this project |
| `SKIPPED empty.md: file has no text` | The file is empty or only spaces | Add content, or delete the file |
| `SKIPPED notes.docx: unsupported file type '.docx'` | We only load `.pdf`, `.md` and `.txt` | Export it as PDF, or save as `.txt` |
| Character counts differ from section 1.5 | Your sample text differs from the guide (typo, extra blank line) | Fine for learning; just know your golden-set quotes must match *your* text |
| PDF page numbers are off by one (page 0 appears) | `+ 1` is missing from `page_number = page_index + 1` | Add it back; the page-number test catches this |
| Words glued together or in odd order in PDF text | Multi-column layouts or unusual fonts confuse text extraction | Check with the debugging command below; not an issue for our sample PDF |
| `SwigPyPacked` / `swigvarlink` `DeprecationWarning` lines in pytest | Harmless warnings from inside PyMuPDF | Add the `filterwarnings` lines from section 1.4 |

**Debugging techniques for this phase**

```bash
file data/raw/*
```

- **What it does:** macOS's `file` command looks at each file's actual bytes and says what it really is, ignoring the name.
- **Why:** spots fake PDFs and wrong encodings before Python does. On my run: the `.md` and `.txt` files showed `ASCII text` (plain English letters, valid UTF-8), the PDF showed `PDF document, version 1.7, 4 pages`, a fake PDF showed `ASCII text`, an empty file showed `empty`, and the Latin-1 test file showed `ISO-8859 text` (not UTF-8).
- **The parts:** `data/raw/*` means "every file in `data/raw`" (the `*` matches any name; hidden files like `.gitkeep` aren't matched).

```bash
uv run python -c "from ragqa.loaders import DEFAULT_DATA_FOLDER, load_pdf; print(repr(load_pdf(DEFAULT_DATA_FOLDER / 'brightwater_travel_policy.pdf')[1].text))"
```

- **What it does:** prints the raw text of the PDF's second loaded page, with `repr(...)`.
- **Why:** `repr` shows hidden characters, so you can see every `\n`, tab or odd symbol exactly as it's stored. Use it whenever extracted text looks wrong. Swap `[1]` for another position to see other pages.

```bash
uv run pytest tests/unit/test_loaders.py -k pdf -v
```

- **What it does:** runs only the loader tests whose names contain "pdf" (3 tests).
- **The parts:** a file path after `pytest` runs just that file; `-k pdf` keeps only tests whose name matches `pdf`.
- **Why:** when one area breaks, rerun only its tests for fast feedback.

**When a test fails,** pytest prints the failing `assert` line and both values it compared, for example `assert [1, 2] == [1, 3]`. Read that line first: the left side is what your code returned, the right side is what the test expected.

## 1.7 Exercise: bad files and a test of your own

**Part A — Feed the loader bad files.** The plan's `data/adversarial/` folder holds deliberately broken files. Phase 8 reuses them, so create the first three now.

```bash
touch data/adversarial/empty.md
```

- **What it does:** creates an empty file (0 bytes). `touch` makes a file if it doesn't exist.

```bash
echo "not a pdf" > data/adversarial/corrupted.pdf
```

- **What it does:** writes the words "not a pdf" into a file whose name ends in `.pdf`, a fake PDF.
- **The parts:** `echo` prints text; `>` sends that text into a file instead of the screen (replacing anything already in it).

```bash
echo "hello" > data/adversarial/notes.docx
```

- **What it does:** creates a file with an extension we don't support.

Now run the loader on that folder, passing the folder name as the extra word that `main()` reads from `sys.argv`:

```bash
uv run python -m ragqa.loaders data/adversarial
```

Real output from my test run:

```text
Loaded 0 pages from 0 files in data/adversarial
SKIPPED corrupted.pdf: not a readable PDF (Failed to open file 'data/adversarial/corrupted.pdf' as type pdf.)
SKIPPED empty.md: file has no text
SKIPPED notes.docx: unsupported file type '.docx'
```

Three bad files, three clear reasons, and no crash. Now answer these in a few words each (in your notes, or under the hardware baseline in `docs/limitations.md`):

1. Why do the SKIPPED lines name the file *first*? Think about a folder with 200 files.
2. Copy `brightwater_faq.txt` into `data/adversarial/` and run again. What changes in the summary line?
3. The meal allowance is on the PDF's **fourth** page, but it's the **third** `Page` record. Which number must a citation show, and why?

**Part B — Write one test yourself.** Add a tenth test to `tests/unit/test_loaders.py` called `test_sample_documents_load`. It should load the real `data/raw` folder with `loaders.load_folder(loaders.DEFAULT_DATA_FOLDER)` and check:

- there are 5 pages,
- there are no skipped files,
- the PDF's page numbers are `[1, 2, 4]`.

Hints: copy the loop from `test_pdf_keeps_real_page_numbers_and_skips_blank_pages`, and add an `if page.source == "brightwater_travel_policy.pdf":` inside it. Use the absolute `loaders.DEFAULT_DATA_FOLDER`, never the relative path `data/raw`, so the test passes from any folder. Then break the code on purpose (remove `+ 1` from `load_pdf`), run the tests, read the failure, and put the `+ 1` back.

## 1.8 Done criteria

- [ ] `uv run python -m ragqa.make_sample_pdf` writes the 4-page PDF
- [ ] `uv run python -m ragqa.loaders` prints "Loaded 5 pages from 3 files in /Users/nowgg/Automation/rag\_framework/data/raw", with the PDF going from page 2 to page 4
- [ ] `uv run pytest -m "not integration" -v` passes all loader tests (9, or 10 with your own) with no warnings
- [ ] `uv run python -m ragqa.loaders data/adversarial` shows three SKIPPED lines and no crash
- [ ] You can explain why blank pages are skipped but still counted
- [ ] Your work is committed: `git add .` then `git commit -m "Phase 1: load documents with metadata"`, then `git push`
- [ ] Phase 1 ticked in the tracker of the [Phase 0 doc](https://claude.ai/code/artifact/b32269ff-ce8f-4186-b984-8cf68f206b1b)

When these pass, share your loader output and your answers to the three questions, and we'll start Phase 2: splitting pages into chunks.
