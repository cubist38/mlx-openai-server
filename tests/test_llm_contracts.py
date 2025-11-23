"""Universal API contract test suite for the MLX OpenAI-compatible server."""

from __future__ import annotations

from collections.abc import Iterable
import json
import os
from typing import Literal

import pytest

try:  # Dependency guard keeps script self-explanatory when deps are missing.
    import httpx
except ImportError as exc:  # pragma: no cover - runtime dependency defense
    pytest.fail(f"Missing required dependency: httpx - {exc}")

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError as exc:  # pragma: no cover
    pytest.fail(f"Missing required dependency: pydantic - {exc}")


# --------------------------------------------------------------------------------------
# OpenAI-compatible response models (trimmed to required fields, tolerant to extras).


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Server reported status string")

    model_config = ConfigDict(extra="allow")


class ModelData(BaseModel):
    """Represents a model in the models list."""

    id: str
    object: Literal["model"]
    created: int | None = None
    owned_by: str | None = None

    model_config = ConfigDict(extra="allow")


class ModelList(BaseModel):
    """Represents the response from the models endpoint."""

    object: Literal["list"]
    data: list[ModelData]

    model_config = ConfigDict(extra="allow")


class ChatMessage(BaseModel):
    """Represents a chat message."""

    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str | None

    model_config = ConfigDict(extra="allow")


class ChatChoice(BaseModel):
    """Represents a choice in a chat completion."""

    index: int
    message: ChatMessage
    finish_reason: str | None = None

    model_config = ConfigDict(extra="allow")


class Usage(BaseModel):
    """Represents token usage."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    model_config = ConfigDict(extra="allow")


class ChatCompletion(BaseModel):
    """Represents a chat completion response."""

    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage | None = None

    model_config = ConfigDict(extra="allow")


class DeltaMessage(BaseModel):
    """Represents a delta message in streaming."""

    role: Literal["system", "user", "assistant", "tool"] | None = None
    content: str | None = None

    model_config = ConfigDict(extra="allow")


class ChatChunkChoice(BaseModel):
    """Represents a choice in a streaming chunk."""

    index: int
    delta: DeltaMessage
    finish_reason: str | None = None

    model_config = ConfigDict(extra="allow")


class ChatCompletionChunk(BaseModel):
    """Represents a streaming chat completion chunk."""

    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: list[ChatChunkChoice]

    model_config = ConfigDict(extra="allow")


# --------------------------------------------------------------------------------------


def env_base_url() -> str:
    """
    Get the base URL for the MLX server from environment variables.

    Returns
    -------
    str
        The base URL with trailing slash removed.
    """
    raw = os.getenv("MLX_URL", "http://127.0.0.1:8000")
    return raw.rstrip("/")


def build_headers() -> dict[str, str]:
    """
    Build HTTP headers for API requests.

    Uses API key from OPENAI_API_KEY or MLX_API_KEY environment variables.

    Returns
    -------
    dict[str, str]
        Dictionary containing Authorization header if API key is available.
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("MLX_API_KEY")
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


@pytest.fixture(scope="session")
def base_url() -> str:
    """Fixture providing the base URL for the MLX server."""
    return env_base_url()


@pytest.fixture(scope="session")
def headers() -> dict[str, str]:
    """Fixture providing HTTP headers for API requests."""
    return build_headers()


@pytest.fixture(scope="session")
def http_client(base_url: str, headers: dict[str, str]) -> httpx.Client:
    """Fixture providing an HTTP client configured for the MLX server."""
    client = httpx.Client(base_url=base_url, timeout=30.0, headers=headers)
    yield client
    client.close()


@pytest.fixture(scope="session")
def model_id() -> str | None:
    """Fixture providing the model ID from environment variables."""
    return os.getenv("MLX_MODEL_ID")


@pytest.fixture(scope="session")
def server_available(http_client: httpx.Client) -> bool:
    """Fixture that checks if the MLX server is available."""
    try:
        response = http_client.get("/health", timeout=5.0)
    except (httpx.ConnectError, httpx.TimeoutException):
        return False
    else:
        return response.status_code == 200


class TestLLMContract:
    """Test suite for LLM API contract compliance."""

    @pytest.mark.integration
    def test_health_endpoint(self, http_client: httpx.Client, server_available: bool) -> None:
        """Test the health endpoint."""
        if not server_available:
            pytest.skip("MLX server is not available - start server to run integration tests")

        response = http_client.get("/health")
        assert response.status_code == 200

        payload = response.json()
        HealthResponse.model_validate(payload)

        status = payload.get("status", "unknown")
        model_status = payload.get("model_status", "unknown")
        model_id = payload.get("model_id", None)

        # Build detailed status message
        if model_id:
            assert status in ["ok", "healthy", "ready"]  # Basic health check
            assert model_status in [
                "loaded",
                "ready",
                "ok",
                "unknown",
            ]  # Model status can vary
        else:
            assert status in ["ok", "healthy", "ready"]

    @pytest.mark.integration
    def test_models_endpoint(
        self, http_client: httpx.Client, model_id: str | None, server_available: bool
    ) -> None:
        """Test the models endpoint."""
        if not server_available:
            pytest.skip("MLX server is not available - start server to run integration tests")

        response = http_client.get("/v1/models")
        assert response.status_code == 200

        payload = response.json()
        model_list = ModelList.model_validate(payload)
        assert len(model_list.data) > 0, "Model registry returned an empty list"

        # Verify metadata presence and required fields (Phase 02)
        raw_model = payload["data"][0]  # Get raw dict for metadata check
        if "metadata" in raw_model:
            metadata = raw_model["metadata"]
            assert "context_length" in metadata, "Model metadata missing 'context_length' field"
            assert metadata.get("backend") == "mlx", (
                f"Expected backend='mlx', got '{metadata.get('backend')}'"
            )

    @pytest.mark.integration
    def test_chat_completion(
        self, http_client: httpx.Client, model_id: str | None, server_available: bool
    ) -> None:
        """Test chat completion endpoint."""
        if not server_available:
            pytest.skip("MLX server is not available - start server to run integration tests")

        if not model_id:
            # Try to get model_id from models endpoint
            response = http_client.get("/v1/models")
            if response.status_code == 200:
                payload = response.json()
                model_list = ModelList.model_validate(payload)
                if model_list.data:
                    model_id = model_list.data[0].id

        assert model_id, "Model id unavailable. Ensure GET /v1/models succeeds or set MLX_MODEL_ID."

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a latency test assistant."},
                {"role": "user", "content": "Respond with a short acknowledgement."},
            ],
            "temperature": 0,
        }
        response = http_client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200

        completion = ChatCompletion.model_validate(response.json())
        assert len(completion.choices) > 0, "Chat completion returned no choices"

        top_choice = completion.choices[0]
        assert top_choice.message.content or top_choice.finish_reason, (
            "Chat completion choice missing content and finish_reason"
        )

    @pytest.mark.integration
    def test_chat_completion_streaming(
        self, http_client: httpx.Client, model_id: str | None, server_available: bool
    ) -> None:
        """Test streaming chat completion endpoint."""
        if not server_available:
            pytest.skip("MLX server is not available - start server to run integration tests")

        if not model_id:
            # Try to get model_id from models endpoint
            response = http_client.get("/v1/models")
            if response.status_code == 200:
                payload = response.json()
                model_list = ModelList.model_validate(payload)
                if model_list.data:
                    model_id = model_list.data[0].id

        assert model_id, "Model id unavailable. Ensure GET /v1/models succeeds or set MLX_MODEL_ID."

        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": "Stream a single friendly sentence."},
            ],
            "stream": True,
            "temperature": 0,
        }
        chunk_counter = 0
        content_tokens = 0
        with http_client.stream("POST", "/v1/chat/completions", json=payload) as response:
            assert response.status_code == 200
            for data in self._iter_sse_payloads(response):
                if data == "[DONE]":
                    break
                chunk = ChatCompletionChunk.model_validate(json.loads(data))
                chunk_counter += 1
                for choice in chunk.choices:
                    if choice.delta.content:
                        content_tokens += len(choice.delta.content.strip())

        assert chunk_counter > 0, "No streaming chunks received"
        assert content_tokens > 0, "Streaming response did not include assistant tokens"

    @staticmethod
    def _iter_sse_payloads(response: httpx.Response) -> Iterable[str]:
        """Iterate over SSE payloads from a response."""
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            payload = line.split("data:", 1)[1].lstrip()
            if payload:
                yield payload
