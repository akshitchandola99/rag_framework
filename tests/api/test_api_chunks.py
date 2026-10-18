"""Phase 2 through the API: the stored chunks of a document."""

import pytest
import requests

from api_helpers import REQUEST_TIMEOUT_SECONDS

pytestmark = [pytest.mark.integration, pytest.mark.api]


# Fetch a document's metadata and its chunks.
def get_document_and_chunks(api_url: str, document_id: str) -> tuple[dict, list[dict]]:
    document = requests.get(f"{api_url}/documents/{document_id}", timeout=REQUEST_TIMEOUT_SECONDS).json()
    response = requests.get(f"{api_url}/documents/{document_id}/chunks", timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return document, response.json()["chunks"]


# Read one setting from /ready.
def get_setting(api_url: str, name: str):
    return requests.get(f"{api_url}/ready", timeout=REQUEST_TIMEOUT_SECONDS).json()["settings"][name]


# On every page, the chunks start at 0, end at the page's end, and leave no gap.
@pytest.mark.parametrize("file_name", ["brightwater_handbook.md", "brightwater_travel_policy.pdf"])
def test_chunks_cover_every_page_with_no_gap(api_url, sample_documents, file_name):
    document, chunks = get_document_and_chunks(api_url, sample_documents[file_name])

    for page in document["pages"]:
        page_chunks = []
        for chunk in chunks:
            if chunk["page"] == page["page"]:
                page_chunks.append(chunk)
        assert page_chunks, f"page {page['page']} has no chunks"
        assert page_chunks[0]["start"] == 0
        assert page_chunks[-1]["end"] == page["char_count"]
        for previous_chunk, next_chunk in zip(page_chunks, page_chunks[1:]):
            assert next_chunk["start"] <= previous_chunk["end"]


# Neighbouring chunks on a page share exactly the overlap the service reports.
def test_neighbouring_chunks_overlap_by_the_setting(api_url, sample_documents):
    overlap = get_setting(api_url, "chunk_overlap_chars")
    document, chunks = get_document_and_chunks(api_url, sample_documents["brightwater_handbook.md"])

    assert len(chunks) >= 2, "the handbook should need more than one chunk"
    for previous_chunk, next_chunk in zip(chunks, chunks[1:]):
        if previous_chunk["page"] == next_chunk["page"]:
            assert previous_chunk["end"] - next_chunk["start"] == overlap


# No chunk is longer than the embedding model can read (longer text would be silently cut).
def test_no_chunk_is_too_long_for_the_embedding_model(api_url, sample_documents):
    max_tokens = get_setting(api_url, "embedding_max_tokens")

    for document_id in sample_documents.values():
        document, chunks = get_document_and_chunks(api_url, document_id)
        for chunk in chunks:
            assert chunk["token_count"] <= max_tokens, chunk["chunk_id"]


# Every chunk keeps the source file name in its chunk ID.
def test_chunks_keep_their_source(api_url, sample_documents):
    document, chunks = get_document_and_chunks(api_url, sample_documents["brightwater_faq.txt"])

    for chunk in chunks:
        assert chunk["chunk_id"].startswith("brightwater_faq.txt:p1:")


# Asking for the chunks of a document that does not exist gives 404.
def test_chunks_of_unknown_document_give_404(api_url):
    response = requests.get(f"{api_url}/documents/doesnotexist/chunks", timeout=REQUEST_TIMEOUT_SECONDS)

    assert response.status_code == 404
