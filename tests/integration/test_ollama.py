import pytest

from ragqa import env_check


# Ollama must list our generator model.
@pytest.mark.integration
def test_generator_model_is_installed():
    model_names = env_check.list_installed_models()
    assert env_check.GENERATOR_MODEL in model_names