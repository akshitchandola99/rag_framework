import requests,json

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"
CONTEXT = "Context : I am 29 years old"
QUESTION = "Question : what is my name?"
SYSTEM = "Answer using only the context. If the context does not contain the answer, say: I don't know"



request_data = {
    "model": MODEL_NAME,
    "prompt": f"{CONTEXT}\n{QUESTION}",  # must be a string, not a tuple
    "system": SYSTEM,
    "stream": False,  # one JSON object; default True returns NDJSON lines
}

response = requests.post(OLLAMA_GENERATE_URL, json=request_data, timeout=120)
response.raise_for_status()
response_data = response.json()
#print(response_data.get("response", response_data))
#print(json.dumps(response_data,indent=4))

user_response = response_data["response"]
print(user_response)


