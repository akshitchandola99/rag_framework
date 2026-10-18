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
