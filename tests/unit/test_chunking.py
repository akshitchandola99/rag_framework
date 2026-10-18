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
