from pathlib import Path

import pytest
import requests

from ragqa import generation
from ragqa import loaders
from ragqa.errors import DocumentLoadError
from ragqa.errors import GenerationError


# Bad documents raise our own DocumentLoadError, which is still a ValueError for older callers.
def test_bad_document_raises_document_load_error(tmp_path: Path):
    file_path = tmp_path / "notes.docx"
    file_path.write_text("x", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="unsupported file type"):
        loaders.load_document(file_path)
    assert issubclass(DocumentLoadError, ValueError)


# A folder with good and bad files: the good one loads, each bad one is skipped with its reason.
def test_bad_files_are_skipped_and_the_run_continues(tmp_path: Path):
    (tmp_path / "good.md").write_text("Annual leave: 27 days.", encoding="utf-8")
    (tmp_path / "empty.md").write_text("", encoding="utf-8")
    (tmp_path / "corrupted.pdf").write_bytes(b"this is not really a pdf")
    (tmp_path / "notes.docx").write_text("x", encoding="utf-8")

    pages, skipped_files = loaders.load_folder(tmp_path)

    assert len(pages) == 1
    assert pages[0].source == "good.md"
    assert len(skipped_files) == 3
    assert "corrupted.pdf: not a readable PDF" in skipped_files[0]
    assert skipped_files[1] == "empty.md: file has no text"
    assert skipped_files[2] == "notes.docx: unsupported file type '.docx'"


# An unexpected error (a bug, not a bad document) is not skipped: it stops the run.
def test_unexpected_error_is_not_swallowed(tmp_path: Path, monkeypatch):
    (tmp_path / "good.md").write_text("Annual leave: 27 days.", encoding="utf-8")

    # Pretend the loader has a bug.
    def broken_loader(file_path):
        raise KeyError("a bug in our code")

    monkeypatch.setattr(loaders, "load_document", broken_loader)
    with pytest.raises(KeyError):
        loaders.load_folder(tmp_path)


# Ollama not running becomes a GenerationError that says how to fix it.
def test_ollama_down_becomes_generation_error(monkeypatch):

    # Pretend nothing is listening on Ollama's port.
    def refuse_connection(url, json, timeout):
        raise requests.ConnectionError("Connection refused")

    monkeypatch.setattr(generation.requests, "post", refuse_connection)
    with pytest.raises(GenerationError, match="ollama serve"):
        generation.call_generator("prompt text")


# A slow Ollama becomes a GenerationError that names the time limit.
def test_ollama_timeout_becomes_generation_error(monkeypatch):

    # Pretend Ollama took too long.
    def time_out(url, json, timeout):
        raise requests.ReadTimeout("read timed out")

    monkeypatch.setattr(generation.requests, "post", time_out)
    with pytest.raises(GenerationError, match="did not answer within 180 s"):
        generation.call_generator("prompt text")
