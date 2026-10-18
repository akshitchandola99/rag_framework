"""Phase 1 through the API: uploading documents, their metadata, bad files and deleting."""

import pytest
import requests

from api_helpers import REQUEST_TIMEOUT_SECONDS
from api_helpers import search
from api_helpers import unique_text
from api_helpers import upload
from api_helpers import upload_and_wait
from api_helpers import wait_for_job

pytestmark = [pytest.mark.integration, pytest.mark.api]

ONE_BYTE_OVER_LIMIT = 5 * 1024 * 1024 + 1


# A good Markdown file is accepted, processed in the background, and gets the right metadata.
def test_markdown_upload_has_correct_metadata(api_url):
    content = unique_text("# Leave\n\nAnnual leave is 27 days.")
    response = upload(api_url, "handbook_copy.md", content)

    assert response.status_code == 201, response.text
    assert response.json()["status"] == "processing"
    job = wait_for_job(api_url, response.json()["job_id"])
    assert job["status"] == "ready"

    document_id = response.json()["id"]
    document = requests.get(f"{api_url}/documents/{document_id}", timeout=REQUEST_TIMEOUT_SECONDS).json()
    assert document["source"] == "handbook_copy.md"
    assert document["file_type"] == "md"
    assert document["owner"] == "public"
    assert document["page_count"] == 1
    assert document["pages"][0]["char_count"] == len(content.decode("utf-8"))
    assert document["chunk_count"] >= 1


# The sample PDF keeps its real page numbers; the blank page 3 is skipped.
def test_pdf_keeps_real_page_numbers(api_url, sample_documents):
    document_id = sample_documents["brightwater_travel_policy.pdf"]
    document = requests.get(f"{api_url}/documents/{document_id}", timeout=REQUEST_TIMEOUT_SECONDS).json()

    page_numbers = []
    for page in document["pages"]:
        page_numbers.append(page["page"])
    assert page_numbers == [1, 2, 4]


# A fake PDF gets a client error that names the file and does not leak server paths.
def test_corrupted_pdf_is_rejected_cleanly(api_url):
    response = upload(api_url, "broken.pdf", b"this is not a pdf")

    assert response.status_code == 422
    error_message = response.json()["error"]
    assert "broken.pdf" in error_message
    assert "uploads" not in error_message
    assert "/Users/" not in error_message


# A file type we can't read gets 415 Unsupported Media Type.
def test_unsupported_file_type_gives_415(api_url):
    response = upload(api_url, "notes.docx", b"some bytes")

    assert response.status_code == 415
    assert ".docx" in response.json()["error"]


# An empty file gets 422.
def test_empty_file_gives_422(api_url):
    response = upload(api_url, "empty.md", b"")

    assert response.status_code == 422
    assert "empty" in response.json()["error"]


# A file over the size limit gets 413 Content Too Large.
def test_too_big_file_gives_413(api_url):
    response = upload(api_url, "huge.txt", b"a" * ONE_BYTE_OVER_LIMIT)

    assert response.status_code == 413


# A blank owner is refused with 422 and nothing is stored (it must not quietly become "public").
@pytest.mark.parametrize("blank_owner", ["", "   "])
def test_blank_owner_gives_422(api_url, blank_owner):
    content = unique_text("Private notes that must not become public.")

    response = upload(api_url, "private_notes.md", content, owner=blank_owner)

    assert response.status_code == 422, response.text
    assert response.json()["error"] == "owner must not be empty"
    listing = requests.get(f"{api_url}/documents", timeout=REQUEST_TIMEOUT_SECONDS).json()
    for document in listing["documents"]:
        assert document["source"] != "private_notes.md"


# Uploading without any owner field is still fine: the document is public.
def test_missing_owner_means_public(api_url):
    files = {"file": ("public_notes.md", unique_text("Notes for everyone."))}
    response = requests.post(f"{api_url}/documents", files=files, timeout=REQUEST_TIMEOUT_SECONDS)

    assert response.status_code == 201, response.text
    document_id = response.json()["id"]
    document = requests.get(f"{api_url}/documents/{document_id}", timeout=REQUEST_TIMEOUT_SECONDS).json()
    assert document["owner"] == "public"


# Uploading the same file twice gives 409 Conflict and points to the first copy.
def test_same_file_twice_gives_409(api_url):
    content = unique_text("Laptops are replaced every 3 years.")
    first_id = upload_and_wait(api_url, "equipment.md", content)

    second_response = upload(api_url, "equipment.md", content)

    assert second_response.status_code == 409
    assert second_response.json()["id"] == first_id


# A file name with folders in it is cut down to the bare name (no writing outside our folder).
def test_path_in_file_name_is_stripped(api_url):
    document_id = upload_and_wait(api_url, "../../evil.md", unique_text("Nothing evil here."))

    document = requests.get(f"{api_url}/documents/{document_id}", timeout=REQUEST_TIMEOUT_SECONDS).json()
    assert document["source"] == "evil.md"


# Asking for a document that does not exist gives 404.
def test_unknown_document_gives_404(api_url):
    response = requests.get(f"{api_url}/documents/doesnotexist", timeout=REQUEST_TIMEOUT_SECONDS)

    assert response.status_code == 404


# A deleted document is gone from the registry and from search results.
def test_deleted_document_disappears_from_search(api_url):
    document_id = upload_and_wait(api_url, "lab.md", unique_text("The Zorbex lab door code is 7781."))
    hits_before = search(api_url, "Zorbex lab door code")["hits"]
    found_before = False
    for hit in hits_before:
        if hit["document_id"] == document_id:
            found_before = True
    assert found_before

    delete_response = requests.delete(f"{api_url}/documents/{document_id}", timeout=REQUEST_TIMEOUT_SECONDS)
    assert delete_response.status_code == 204

    get_response = requests.get(f"{api_url}/documents/{document_id}", timeout=REQUEST_TIMEOUT_SECONDS)
    assert get_response.status_code == 404
    for hit in search(api_url, "Zorbex lab door code")["hits"]:
        assert hit["document_id"] != document_id
