import requests, json

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def ask_llm(question: str) -> dict:
    request_data = {
        "model": MODEL_NAME,
        "prompt": question,
        "stream": False,
        "options": {"temperature": 0 , "num_predict" : 100},
    }
    response = requests.post(OLLAMA_GENERATE_URL, json=request_data, timeout=120)
    response.raise_for_status()
    response_data = response.json()
    return response_data


def main() -> None:
    question = "What is retrieval-augmented generation? Answer in two sentences."
    print("Question:", question)

    response_data = ask_llm(question)

    print("Answer:", response_data["response"].strip())
    print("Tokens generated:", response_data["eval_count"])
    print("Total time (seconds):", round(response_data["total_duration"] / 1_000_000_000, 1))
    print(json.dumps(response_data,indent=4))


if __name__ == "__main__":
    main()