"""Phase 8: shared, expensive fixtures for integration tests, built once per test run."""

import pytest
import requests

from ragqa import ask
from ragqa import embeddings
from ragqa import generation

OLLAMA_CHECK_TIMEOUT_SECONDS = 5


# Load the embedding model once for the whole run (about 5 s), not once per file.
@pytest.fixture(scope="session")
def embedding_model():
    return embeddings.load_embedding_model()


# Load, chunk and embed data/raw once; every test that needs the real index shares it.
@pytest.fixture(scope="session")
def real_index(embedding_model):
    chunks, chunk_vectors = ask.build_index(embedding_model)
    return embedding_model, chunks, chunk_vectors


# Check once that Ollama is up and has our model; otherwise fail fast with the fix, not a long traceback.
@pytest.fixture(scope="session")
def ollama_ready():
    problem = None
    try:
        response = requests.get(f"{generation.DEFAULT_OLLAMA_URL}/api/tags", timeout=OLLAMA_CHECK_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as error:
        problem = f"Ollama is not reachable at {generation.DEFAULT_OLLAMA_URL} ({type(error).__name__}). Start it with: ollama serve"
    # Fail outside the except block, so pytest prints only our one-line message.
    if problem is not None:
        pytest.fail(problem, pytrace=False)

    installed_models = []
    for model_info in response.json()["models"]:
        installed_models.append(model_info["name"])
    if generation.DEFAULT_GENERATOR_MODEL not in installed_models:
        pytest.fail(f"Model {generation.DEFAULT_GENERATOR_MODEL} is missing. Get it with: ollama pull {generation.DEFAULT_GENERATOR_MODEL}", pytrace=False)
