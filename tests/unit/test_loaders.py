from pathlib import Path

import pytest

from ragqa import loaders
from ragqa.make_sample_pdf import write_pdf


# A .md file becomes one page with correct metadata.
def test_markdown_file_becomes_one_page_with_metadata(tmp_path: Path):
    file_path = tmp_path / "handbook.md"
    file_path.write_text("# Leave\n\nAnnual leave is 27 days.", encoding="utf-8")

    pages = loaders.load_document(file_path)

    assert len(pages) == 1
    assert pages[0].source == "handbook.md"
    assert pages[0].page == 1
    assert pages[0].file_type == "md"
    assert "27 days" in pages[0].text


# A .txt file is loaded with its text unchanged.
def test_txt_file_is_loaded(tmp_path: Path):
    file_path = tmp_path / "faq.txt"
    file_path.write_text("Passwords expire every 180 days.", encoding="utf-8")

    pages = loaders.load_document(file_path)

    assert pages[0].file_type == "txt"
    assert pages[0].text == "Passwords expire every 180 days."


# Blank PDF pages are skipped, but page numbers stay real.
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


# A file with only whitespace is rejected.
def test_empty_text_file_raises(tmp_path: Path):
    file_path = tmp_path / "empty.md"
    file_path.write_text("   \n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no text"):
        loaders.load_document(file_path)


# A file that isn't valid UTF-8 is rejected.
def test_non_utf8_text_file_raises(tmp_path: Path):
    file_path = tmp_path / "menu.txt"
    file_path.write_bytes(b"Caf\xe9 menu")

    with pytest.raises(ValueError, match="not valid UTF-8"):
        loaders.load_document(file_path)


# A PDF with no text at all is rejected.
def test_pdf_with_only_blank_pages_raises(tmp_path: Path):
    file_path = tmp_path / "blank.pdf"
    write_pdf(file_path, ["", ""])

    with pytest.raises(ValueError, match="no extractable text"):
        loaders.load_document(file_path)


# A broken PDF is rejected with a clear error.
def test_corrupted_pdf_raises(tmp_path: Path):
    file_path = tmp_path / "broken.pdf"
    file_path.write_bytes(b"this is not really a pdf")

    with pytest.raises(ValueError, match="not a readable PDF"):
        loaders.load_document(file_path)


# Unsupported file types like .docx are rejected.
def test_unsupported_file_type_raises(tmp_path: Path):
    file_path = tmp_path / "notes.docx"
    file_path.write_text("some text", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported file type"):
        loaders.load_document(file_path)


# Bad files are skipped and hidden files are ignored.
def test_folder_loading_skips_bad_and_hidden_files(tmp_path: Path):
    (tmp_path / "good.md").write_text("Real content", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    (tmp_path / "notes.docx").write_text("x", encoding="utf-8")
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")

    pages, skipped_files = loaders.load_folder(tmp_path)

    assert len(pages) == 1
    assert pages[0].source == "good.md"
    assert len(skipped_files) == 2