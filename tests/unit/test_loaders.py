from pathlib import Path

import pymupdf
import pytest

from ragqa import loaders
from ragqa.make_sample_pdf import write_pdf

IMAGE_WORDS = "APPROVAL NEEDED 10 DAYS BEFORE"


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


# The tests below pin down how today's loader treats complex PDF content.
# They describe current behaviour, good or bad, so any change to it is noticed on purpose.


# Make a small picture (a pixmap) that shows some words, like a scanned figure.
def make_picture_of_words(words: str) -> pymupdf.Pixmap:
    picture_document = pymupdf.open()
    picture_page = picture_document.new_page(width=300, height=80)
    picture_page.insert_text((10, 40), words, fontsize=14)
    picture = picture_page.get_pixmap()
    picture_document.close()
    return picture


# A link keeps its visible words; the hidden URL behind it is dropped. A typed-out URL is kept.
def test_pdf_link_keeps_visible_text_but_drops_hidden_url(tmp_path: Path):
    file_path = tmp_path / "link.pdf"
    pdf_document = pymupdf.open()
    pdf_page = pdf_document.new_page()
    pdf_page.insert_text((72, 72), "Read the full policy on the intranet.", fontsize=12)
    link = {"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(72, 60, 300, 76), "uri": "https://intranet.example/policy"}
    pdf_page.insert_link(link)
    pdf_page.insert_text((72, 100), "Or visit https://brightwater.example/help for help.", fontsize=12)
    pdf_document.save(file_path)
    pdf_document.close()

    text = loaders.load_document(file_path)[0].text

    assert "Read the full policy on the intranet." in text
    assert "https://intranet.example/policy" not in text
    assert "https://brightwater.example/help" in text


# A table comes out one cell per line, row by row: the words stay, the rows and columns don't.
def test_pdf_table_comes_out_one_cell_per_line(tmp_path: Path):
    file_path = tmp_path / "table.pdf"
    rows = [
        ["City", "Hotel limit", "Meal allowance"],
        ["London", "190 euros", "45 euros"],
        ["Madrid", "140 euros", "45 euros"],
    ]
    pdf_document = pymupdf.open()
    pdf_page = pdf_document.new_page()
    for row_index, row in enumerate(rows):
        for column_index, cell in enumerate(row):
            x = 72 + column_index * 150
            y = 100 + row_index * 25
            pdf_page.draw_rect(pymupdf.Rect(x - 4, y - 16, x + 146, y + 9))
            pdf_page.insert_text((x, y), cell, fontsize=11)
    pdf_document.save(file_path)
    pdf_document.close()

    text = loaders.load_document(file_path)[0].text

    expected_lines = []
    for row in rows:
        for cell in row:
            expected_lines.append(cell)
    assert text.splitlines() == expected_lines


# Words inside an image are silently lost; the normal text around the image is kept.
def test_pdf_text_inside_an_image_is_ignored(tmp_path: Path):
    file_path = tmp_path / "figure.pdf"
    pdf_document = pymupdf.open()
    pdf_page = pdf_document.new_page()
    pdf_page.insert_text((72, 72), "Figure 1 shows the travel approval steps.", fontsize=12)
    pdf_page.insert_image(pymupdf.Rect(72, 90, 372, 170), pixmap=make_picture_of_words(IMAGE_WORDS))
    pdf_document.save(file_path)
    pdf_document.close()

    text = loaders.load_document(file_path)[0].text

    assert "Figure 1 shows the travel approval steps." in text
    assert IMAGE_WORDS not in text


# A page that is only a picture (like a scanned page) has no text, so the PDF is refused.
def test_pdf_with_only_an_image_raises(tmp_path: Path):
    file_path = tmp_path / "scanned.pdf"
    pdf_document = pymupdf.open()
    pdf_page = pdf_document.new_page()
    pdf_page.insert_image(pymupdf.Rect(72, 90, 372, 170), pixmap=make_picture_of_words(IMAGE_WORDS))
    pdf_document.save(file_path)
    pdf_document.close()

    with pytest.raises(ValueError, match="scanned.pdf: PDF has no extractable text"):
        loaders.load_document(file_path)


# A password-protected PDF is refused with a clear error that names the file.
def test_password_protected_pdf_raises_with_file_name(tmp_path: Path):
    file_path = tmp_path / "salaries.pdf"
    pdf_document = pymupdf.open()
    pdf_page = pdf_document.new_page()
    pdf_page.insert_text((72, 72), "Secret salary bands.", fontsize=12)
    pdf_document.save(file_path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    pdf_document.close()

    with pytest.raises(ValueError, match="salaries.pdf: PDF is password-protected"):
        loaders.load_document(file_path)


# A header and a footer repeat on every page; the loader keeps them on every page.
def test_pdf_header_and_footer_repeat_on_every_page(tmp_path: Path):
    file_path = tmp_path / "report.pdf"
    pdf_document = pymupdf.open()
    for page_number in range(1, 3):
        pdf_page = pdf_document.new_page()
        pdf_page.insert_text((72, 40), "Brightwater Robotics - CONFIDENTIAL", fontsize=9)
        pdf_page.insert_text((72, 120), f"Body text of page {page_number}.", fontsize=12)
        pdf_page.insert_text((72, 800), f"Page {page_number} of 2", fontsize=9)
    pdf_document.save(file_path)
    pdf_document.close()

    pages = loaders.load_document(file_path)

    assert len(pages) == 2
    for page in pages:
        assert page.text.startswith("Brightwater Robotics - CONFIDENTIAL")
        assert f"Page {page.page} of 2" in page.text
