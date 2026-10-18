"""Fixtures for the API tests: one real server with clean storage, and the sample documents uploaded."""

import pytest

from api_helpers import start_server
from api_helpers import stop_server
from api_helpers import upload_and_wait
from ragqa import loaders

SAMPLE_FILES = ["brightwater_handbook.md", "brightwater_travel_policy.pdf", "brightwater_faq.txt"]


# Start one API server for the whole test run, on a free port, with an empty temporary storage folder.
@pytest.fixture(scope="session")
def api_url(tmp_path_factory):
    storage_folder = tmp_path_factory.mktemp("api_storage")
    process, base_url = start_server(storage_folder, {})
    yield base_url
    stop_server(process)


# Upload the three Brightwater sample files once; return {file name: document ID}.
@pytest.fixture(scope="session")
def sample_documents(api_url):
    document_ids = {}
    for file_name in SAMPLE_FILES:
        content = (loaders.DEFAULT_DATA_FOLDER / file_name).read_bytes()
        document_ids[file_name] = upload_and_wait(api_url, file_name, content)
    return document_ids
