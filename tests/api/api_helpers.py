"""Small helpers the API tests share: start/stop a server, upload a file, wait for a job."""

import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests

SERVER_START_TIMEOUT_SECONDS = 120
JOB_TIMEOUT_SECONDS = 60
REQUEST_TIMEOUT_SECONDS = 30
CHAT_TIMEOUT_SECONDS = 240


# Ask the operating system for a free port number.
def find_free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# Start the API server in its own process, with its own storage, and wait until /health answers.
def start_server(storage_folder: Path, extra_env: dict) -> tuple[subprocess.Popen, str]:
    port = find_free_port()
    env = dict(os.environ)
    env["RAGQA_STORAGE_DIR"] = str(storage_folder)
    env["RAGQA_PORT"] = str(port)
    env.update(extra_env)

    log_path = storage_folder / "server.log"
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            [sys.executable, "-m", "ragqa.api"],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + SERVER_START_TIMEOUT_SECONDS
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"API server stopped during startup; see {log_path}")
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                return process, base_url
        except requests.ConnectionError:
            pass
        time.sleep(0.5)

    stop_server(process)
    raise TimeoutError(f"API server did not start in {SERVER_START_TIMEOUT_SECONDS} s; see {log_path}")


# Stop a server process, forcefully if it does not stop by itself.
def stop_server(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()


# Upload one file and return the raw HTTP response.
def upload(api_url: str, file_name: str, content: bytes, owner: str = "public") -> requests.Response:
    files = {"file": (file_name, content)}
    data = {"owner": owner}
    return requests.post(f"{api_url}/documents", files=files, data=data, timeout=REQUEST_TIMEOUT_SECONDS)


# Ask for a job's status every half second until it is ready or failed.
def wait_for_job(api_url: str, job_id: str) -> dict:
    deadline = time.time() + JOB_TIMEOUT_SECONDS
    while time.time() < deadline:
        response = requests.get(f"{api_url}/jobs/{job_id}", timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        job = response.json()
        if job["status"] in ("ready", "failed"):
            return job
        time.sleep(0.5)
    raise TimeoutError(f"job {job_id} did not finish in {JOB_TIMEOUT_SECONDS} s")


# Upload a file, wait for its job, and return the document ID (fails the test if anything goes wrong).
def upload_and_wait(api_url: str, file_name: str, content: bytes, owner: str = "public") -> str:
    response = upload(api_url, file_name, content, owner)
    assert response.status_code == 201, response.text
    job = wait_for_job(api_url, response.json()["job_id"])
    assert job["status"] == "ready", job
    return response.json()["id"]


# Make text that no other test uploads, so the duplicate check never gets in the way.
def unique_text(sentence: str) -> bytes:
    return f"{sentence}\n\nTest marker: {uuid.uuid4().hex}\n".encode("utf-8")


# Send a search request and return the JSON reply.
def search(api_url: str, query: str, top_k: int = 3, user_id: str = "public") -> dict:
    request_body = {"query": query, "top_k": top_k, "user_id": user_id}
    response = requests.post(f"{api_url}/search", json=request_body, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


# Send a chat request and return the raw HTTP response.
def chat(api_url: str, question: str, user_id: str = "public", include_trace: bool = False) -> requests.Response:
    request_body = {"question": question, "user_id": user_id, "include_trace": include_trace}
    return requests.post(f"{api_url}/chat", json=request_body, timeout=CHAT_TIMEOUT_SECONDS)
