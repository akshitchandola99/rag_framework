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