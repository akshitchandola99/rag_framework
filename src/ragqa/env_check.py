"""Phase 0 environment check: Python, CPU architecture, and the local Ollama server."""

import platform
import sys

import requests

OLLAMA_URL = "http://localhost:11434"
GENERATOR_MODEL = "llama3.2:3b"
MINIMUM_PYTHON = (3, 12)


def check_python_version() -> bool:
    current_version = sys.version_info[:2]
    print(f"Python version: {sys.version.split()[0]}")
    if current_version < MINIMUM_PYTHON:
        print(f"  FAIL: need Python {MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]} or newer")
        return False
    print("  OK")
    return True


def check_architecture() -> bool:
    machine = platform.machine()
    print(f"CPU architecture: {machine}")
    if machine != "arm64":
        print("  FAIL: expected arm64; Python may be running under Rosetta (x86_64)")
        return False
    print("  OK")
    return True


def list_installed_models() -> list[str]:
    response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    response.raise_for_status()
    response_data = response.json()

    model_names = []
    for model_info in response_data["models"]:
        model_names.append(model_info["name"])
    return model_names


def check_ollama_server() -> bool:
    print(f"Ollama server at {OLLAMA_URL}")
    try:
        model_names = list_installed_models()
    except requests.exceptions.ConnectionError:
        print("  FAIL: cannot reach Ollama. Is 'ollama serve' or the Ollama app running?")
        return False
    print(f"  OK, installed models: {model_names}")
    if GENERATOR_MODEL not in model_names:
        print(f"  FAIL: {GENERATOR_MODEL} not found. Run: ollama pull {GENERATOR_MODEL}")
        return False
    return True


def measure_generation_speed() -> None:
    print(f"Test generation with {GENERATOR_MODEL} (first call includes model load time)")
    request_data = {
        "model": GENERATOR_MODEL,
        "prompt": "In one sentence, what is retrieval-augmented generation?",
        "stream": False,
        "options": {"temperature": 0},
    }
    response = requests.post(f"{OLLAMA_URL}/api/generate", json=request_data, timeout=120)
    response.raise_for_status()
    response_data = response.json()

    # Ollama reports all durations in nanoseconds.
    load_seconds = response_data["load_duration"] / 1_000_000_000
    generated_tokens = response_data["eval_count"]
    generation_seconds = response_data["eval_duration"] / 1_000_000_000
    tokens_per_second = generated_tokens / generation_seconds

    print(f"  Answer: {response_data['response'].strip()}")
    print(f"  Load time: {load_seconds:.1f} s")
    print(f"  Generated {generated_tokens} tokens at {tokens_per_second:.1f} tokens/s")


def main() -> None:
    python_ok = check_python_version()
    architecture_ok = check_architecture()
    ollama_ok = check_ollama_server()

    if ollama_ok:
        measure_generation_speed()

    if python_ok and architecture_ok and ollama_ok:
        print("\nEnvironment ready for Phase 1.")
    else:
        print("\nSome checks failed. Fix the FAIL lines above and run again.")
        sys.exit(1)


if __name__ == "__main__":
    main()