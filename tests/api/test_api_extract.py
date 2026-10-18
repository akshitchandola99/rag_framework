"""Phase 1 through the API: extracting a document's text page by page, including multi-page PDFs."""

from pathlib import Path

import pymupdf
import pytest
import requests

from api_helpers import REQUEST_TIMEOUT_SECONDS
from ragqa import loaders
from ragqa.make_sample_pdf import write_pdf

pytestmark = [pytest.mark.integration, pytest.mark.api]

ONE_BYTE_OVER_LIMIT = 5 * 1024 * 1024 + 1


# Send a file to POST /extract and return the raw HTTP response.
def extract(api_url: str, file_name: str, content: bytes) -> requests.Response:
    files = {"file": (file_name, content)}
    return requests.post(f"{api_url}/extract", files=files, timeout=REQUEST_TIMEOUT_SECONDS)


# Build a PDF with one page per text (an empty text gives a blank page) and return its bytes.
def make_pdf_bytes(tmp_path: Path, page_texts: list[str]) -> bytes:
    file_path = tmp_path / "made.pdf"
    write_pdf(file_path, page_texts)
    return file_path.read_bytes()


# Build a password-protected one-page PDF and return its bytes.
def make_password_pdf_bytes(tmp_path: Path) -> bytes:
    file_path = tmp_path / "salaries.pdf"
    pdf_document = pymupdf.open()
    pdf_page = pdf_document.new_page()
    pdf_page.insert_text((72, 72), "Secret salary bands.", fontsize=12)
    pdf_document.save(file_path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    pdf_document.close()
    return file_path.read_bytes()


# Return the page numbers in an extract reply, in the order they came back.
def page_numbers_of(reply: dict) -> list[int]:
    numbers = []
    for page in reply["pages"]:
        numbers.append(page["page"])
    return numbers


# The real sample PDF: every page with text comes back, with real numbers and the right facts on each.
def test_extract_returns_text_for_every_page(api_url):
    content = (loaders.DEFAULT_DATA_FOLDER / "brightwater_travel_policy.pdf").read_bytes()

    response = extract(api_url, "brightwater_travel_policy.pdf", content)
    reply = response.json()

    assert response.status_code == 200, reply
    assert reply["source"] == "brightwater_travel_policy.pdf"
    assert reply["file_type"] == "pdf"
    assert reply["page_count"] == 3
    assert page_numbers_of(reply) == [1, 2, 4]
    pages_by_number = {}
    for page in reply["pages"]:
        pages_by_number[page["page"]] = page
        assert page["char_count"] == len(page["text"])
    assert "Travel and Expenses Policy" in pages_by_number[1]["text"]
    assert "140 euros" in pages_by_number[2]["text"]
    assert "within 30 days" in " ".join(pages_by_number[4]["text"].split())


# A Markdown file comes back as one page with its text unchanged.
def test_extract_markdown_is_one_page_with_exact_text(api_url):
    text = "# Leave\n\nAnnual leave is 27 days."

    reply = extract(api_url, "handbook.md", text.encode("utf-8")).json()

    assert reply["page_count"] == 1
    assert reply["pages"][0]["page"] == 1
    assert reply["pages"][0]["text"] == text


# Multi-page: each page keeps its own text, in page order, and blank pages keep later numbers real.
def test_extract_multi_page_keeps_each_page_separate(tmp_path, api_url):
    page_texts = []
    for page_number in range(1, 7):
        if page_number == 3:
            page_texts.append("")
        else:
            page_texts.append(f"MARKER-{page_number} This text belongs to page {page_number} only.")

    reply = extract(api_url, "six_pages.pdf", make_pdf_bytes(tmp_path, page_texts)).json()

    assert page_numbers_of(reply) == [1, 2, 4, 5, 6]
    for page in reply["pages"]:
        assert f"MARKER-{page['page']} " in page["text"]
        for other_number in range(1, 7):
            if other_number != page["page"]:
                assert f"MARKER-{other_number} " not in page["text"]


# Multi-page: a sentence split by a page break is in no single page, only in the pages joined.
def test_extract_sentence_split_across_a_page_break(tmp_path, api_url):
    page_texts = ["4. Claims\nSubmit receipts in Compass within", "30 days of returning; late claims are rejected."]

    reply = extract(api_url, "split.pdf", make_pdf_bytes(tmp_path, page_texts)).json()

    sentence = "Submit receipts in Compass within 30 days of returning"
    joined_text = ""
    for page in reply["pages"]:
        assert sentence not in " ".join(page["text"].split())
        joined_text = joined_text + " " + page["text"]
    assert sentence in " ".join(joined_text.split())


# Multi-page: a long PDF (50 pages) comes back complete, first page to last.
def test_extract_long_pdf_returns_every_page(tmp_path, api_url):
    page_texts = []
    for page_number in range(1, 51):
        page_texts.append(f"MARKER-{page_number} Page {page_number} of a long report.")

    reply = extract(api_url, "long_report.pdf", make_pdf_bytes(tmp_path, page_texts)).json()

    assert reply["page_count"] == 50
    assert page_numbers_of(reply) == list(range(1, 51))
    assert "MARKER-1 " in reply["pages"][0]["text"]
    assert "MARKER-50 " in reply["pages"][-1]["text"]


# Extracting is a dry run: nothing is added to the stored documents.
def test_extract_stores_nothing(tmp_path, api_url):
    before = requests.get(f"{api_url}/documents", timeout=REQUEST_TIMEOUT_SECONDS).json()["documents"]

    response = extract(api_url, "dry_run.pdf", make_pdf_bytes(tmp_path, ["Only a dry run."]))

    assert response.status_code == 200
    after = requests.get(f"{api_url}/documents", timeout=REQUEST_TIMEOUT_SECONDS).json()["documents"]
    assert len(after) == len(before)


# Bad files get the same codes and messages as an upload.
@pytest.mark.parametrize(
    ("file_name", "content", "expected_code", "expected_message"),
    [
        ("notes.docx", b"some bytes", 415, "notes.docx: unsupported file type '.docx'"),
        ("empty.md", b"", 422, "empty.md: file is empty"),
        ("spaces.md", b"   \n", 422, "spaces.md: file has no text"),
        ("broken.pdf", b"this is not a pdf", 422, "broken.pdf: not a readable PDF"),
        ("huge.txt", b"a" * ONE_BYTE_OVER_LIMIT, 413, "huge.txt: file is larger than 5242880 bytes"),
    ],
    ids=["unsupported", "empty", "only-spaces", "broken-pdf", "too-big"],
)
def test_extract_bad_files_are_refused(api_url, file_name, content, expected_code, expected_message):
    response = extract(api_url, file_name, content)

    assert response.status_code == expected_code
    assert response.json()["error"].startswith(expected_message)
    assert "/var/" not in response.json()["error"]


# A password-protected PDF is refused with a message that names the file.
def test_extract_password_protected_pdf_is_refused(tmp_path, api_url):
    response = extract(api_url, "salaries.pdf", make_password_pdf_bytes(tmp_path))

    assert response.status_code == 422
    assert response.json()["error"] == "salaries.pdf: PDF is password-protected"


# A stored document's pages are exactly what a dry-run extract of the same file gives.
def test_stored_pages_match_extract(api_url, sample_documents):
    document_id = sample_documents["brightwater_travel_policy.pdf"]
    content = (loaders.DEFAULT_DATA_FOLDER / "brightwater_travel_policy.pdf").read_bytes()

    stored = requests.get(f"{api_url}/documents/{document_id}/pages", timeout=REQUEST_TIMEOUT_SECONDS)
    extracted = extract(api_url, "brightwater_travel_policy.pdf", content)

    assert stored.status_code == 200
    assert stored.json()["page_count"] == 3
    assert stored.json()["pages"] == extracted.json()["pages"]


# Asking for the pages of a document that does not exist gives 404.
def test_pages_of_unknown_document_give_404(api_url):
    response = requests.get(f"{api_url}/documents/doesnotexist/pages", timeout=REQUEST_TIMEOUT_SECONDS)

    assert response.status_code == 404
    assert response.json()["error"] == "no document with id doesnotexist"
