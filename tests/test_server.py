"""Smoke tests for the OpenAI-style HTTP API (random weights, tiny model)."""

import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from flashserve import InferenceEngine, LlamaConfig  # noqa: E402
from flashserve.serving.server import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    engine = InferenceEngine(LlamaConfig.tiny(), device="cpu")
    return TestClient(create_app(engine=engine))


def test_health(client):
    assert client.get("/health").status_code == 200


def test_completion(client):
    r = client.post("/v1/completions", json={"model": "flashserve-tiny", "prompt": "Hello", "max_tokens": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"] and isinstance(body["choices"][0]["text"], str)


def test_streaming_completion(client):
    with client.stream("POST", "/v1/completions", json={"model": "flashserve-tiny", "prompt": "Hi", "max_tokens": 4, "stream": True}) as r:
        assert r.status_code == 200
        chunks = [line for line in r.iter_lines() if line.startswith("data:")]
    assert chunks and chunks[-1].strip() == "data: [DONE]"
    json.loads(chunks[0][len("data:"):])  # first chunk is valid JSON


def test_chat_completion(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "flashserve-tiny", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4},
    )
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["role"] == "assistant"
