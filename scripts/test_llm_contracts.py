#!/usr/bin/env python3
"""Universal API contract test suite for the MLX OpenAI-compatible server."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

try:  # Dependency guard keeps script self-explanatory when deps are missing.
    import httpx
except ImportError as exc:  # pragma: no cover - runtime dependency defense
    logger.error("Missing required dependency: httpx")
    raise SystemExit(1) from exc

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
except ImportError as exc:  # pragma: no cover
    logger.error("Missing required dependency: pydantic")
    raise SystemExit(1) from exc


# --------------------------------------------------------------------------------------
# OpenAI-compatible response models (trimmed to required fields, tolerant to extras).


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Server reported status string")

    model_config = ConfigDict(extra="allow")


class ModelData(BaseModel):
    """Data model for a language model."""

    id: str = Field(..., description="Unique identifier for the model")
    object: str = Field(..., description="Object type, always 'model'")  # e.g., "model"
    created: int = Field(..., description="Unix timestamp of model creation")
    owned_by: str = Field(..., description="Organization that owns the model")

    model_config = ConfigDict(extra="allow")


class ModelList(BaseModel):
    """Response model for the models list endpoint."""

    object: str = Field(..., description="Object type, always 'list'")  # e.g., "list"
    data: list[ModelData] = Field(..., description="List of available models")

    model_config = ConfigDict(extra="allow")


class ChatMessage(BaseModel):
    """A message in a chat conversation."""

    role: str = Field(
        ..., description="Role of the message sender (user, assistant, system)"
    )  # e.g., "user", "assistant", "system"
    content: str = Field(..., description="Content of the message")

    model_config = ConfigDict(extra="allow")


class ChatChoice(BaseModel):
    """A choice in a chat completion response."""

    index: int
    message: ChatMessage
    finish_reason: str | None = None

    model_config = ConfigDict(extra="allow")


class Usage(BaseModel):
    """Token usage statistics for a completion."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    model_config = ConfigDict(extra="allow")


class ChatCompletion(BaseModel):
    """Response model for chat completion endpoint."""

    id: str
    object: str  # e.g., "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage | None = None

    model_config = ConfigDict(extra="allow")


class DeltaMessage(BaseModel):
    """Delta message for streaming responses."""

    role: str | None = None  # e.g., "assistant"
    content: str | None = None

    model_config = ConfigDict(extra="allow")


class ChatChunkChoice(BaseModel):
    """A choice in a streaming chat completion response."""

    index: int
    delta: DeltaMessage
    finish_reason: str | None = None

    model_config = ConfigDict(extra="allow")


class ChatCompletionChunk(BaseModel):
    """Response model for streaming chat completion chunks."""

    id: str
    object: str  # e.g., "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatChunkChoice]

    model_config = ConfigDict(extra="allow")


# --------------------------------------------------------------------------------------


def env_base_url() -> str:
    """Get the base URL from environment variable or default.

    Returns
    -------
    str
        The base URL for the API server, with trailing slashes removed.
    """
    raw = os.getenv("MLX_URL", "http://127.0.0.1:8000")
    return raw.rstrip("/")


def build_headers() -> dict[str, str]:
    """Build HTTP headers for API requests including authorization.

    Returns
    -------
    dict[str, str]
        Dictionary of HTTP headers, including Authorization header if API key is available.
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("MLX_API_KEY")
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


@dataclass
class TestCase:
    """Test case for LLM contract validation."""

    name: str
    handler: callable


class LLMContractTester:
    """Runs OpenAI contract checks against a target server."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        """Initialize the contract tester.

        Parameters
        ----------
        base_url : str
            Base URL for the API server.
        timeout : float, optional
            Request timeout in seconds, by default 30.0.
        """
        self.base_url = base_url
        self.headers = build_headers()
        self.client = httpx.Client(base_url=base_url, timeout=timeout, headers=self.headers)
        self.model_id: str | None = os.getenv("MLX_MODEL_ID")
        self.results: list[tuple[str, bool, str]] = []

    def close(self) -> None:
        """Close the HTTP client connection.

        This method should be called when the tester is no longer needed
        to properly clean up network resources.
        """
        self.client.close()

    # Individual tests -----------------------------------------------------------------

    def test_health(self) -> str:
        """Check the server health endpoint and validate its response.

        Sends a GET request to the /health endpoint, validates the returned JSON
        against the HealthResponse model, and returns a concise status summary.

        Returns
        -------
        str
            Human-readable status summary including model status and model id when available.

        Raises
        ------
        httpx.HTTPError
            If the HTTP request fails or returns a non-2xx status.
        ValidationError
            If the response payload does not conform to the HealthResponse schema.
        """
        response = self.client.get("/health")
        response.raise_for_status()
        payload = response.json()
        HealthResponse.model_validate(payload)
        status = payload.get("status", "unknown")
        model_status = payload.get("model_status", "unknown")
        model_id = payload.get("model_id", None)

        # Build detailed status message
        if model_id:
            return f"status={status}, model_status={model_status}, model_id={model_id}"
        return f"status={status}, model_status={model_status}"

    def test_models(self) -> str:
        """Test the models endpoint.

        Validates the /v1/models endpoint response and checks for required
        model metadata fields.

        Returns
        -------
        str
            Summary of model count and metadata validation results.

        Raises
        ------
        httpx.HTTPError
            If the HTTP request fails.
        AssertionError
            If model registry is empty or metadata validation fails.
        """
        response = self.client.get("/v1/models")
        response.raise_for_status()
        payload = response.json()
        model_list = ModelList.model_validate(payload)
        if not model_list.data:
            raise AssertionError("Model registry returned an empty list")
        if not self.model_id:
            self.model_id = model_list.data[0].id

        # Verify metadata presence and required fields (Phase 02)
        raw_model = payload["data"][0]  # Get raw dict for metadata check
        if "metadata" in raw_model:
            metadata = raw_model["metadata"]
            if "context_length" not in metadata:
                raise AssertionError("Model metadata missing 'context_length' field")
            if metadata.get("backend") != "mlx":
                raise AssertionError(f"Expected backend='mlx', got '{metadata.get('backend')}'")
            return f"{len(model_list.data)} models, metadata: backend={metadata.get('backend')}, context={metadata.get('context_length')}"
        # Metadata is optional for backward compat, but warn if missing
        return f"{len(model_list.data)} models detected (no metadata)"

    def test_chat_completion(self) -> str:
        """Test the chat completion endpoint.

        Sends a chat completion request and validates the response structure.

        Returns
        -------
        str
            Summary of completion results including choice count.

        Raises
        ------
        httpx.HTTPError
            If the HTTP request fails.
        AssertionError
            If the response lacks required fields or choices.
        """
        model_id = self.ensure_model_id()
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a latency test assistant."},
                {"role": "user", "content": "Respond with a short acknowledgement."},
            ],
            "temperature": 0,
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        completion = ChatCompletion.model_validate(response.json())
        if not completion.choices:
            raise AssertionError("Chat completion returned no choices")
        top_choice = completion.choices[0]
        if not (top_choice.message.content or top_choice.finish_reason):
            raise AssertionError("Chat completion choice missing content and finish_reason")
        return f"choice_count={len(completion.choices)}"

    def test_chat_completion_stream(self) -> str:
        """Test the streaming chat completion endpoint.

        Sends a streaming chat completion request and validates the response chunks.

        Returns
        -------
        str
            Summary of streaming results including chunk and token counts.

        Raises
        ------
        httpx.HTTPError
            If the HTTP request fails.
        AssertionError
            If no chunks are received or no content tokens are found.
        """
        model_id = self.ensure_model_id()
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
        with self.client.stream("POST", "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            for data in self.iter_sse_payloads(response):
                if data == "[DONE]":
                    break
                chunk = ChatCompletionChunk.model_validate(json.loads(data))
                chunk_counter += 1
                for choice in chunk.choices:
                    if choice.delta.content:
                        content_tokens += len(choice.delta.content.strip())
        if chunk_counter == 0:
            raise AssertionError("No streaming chunks received")
        if content_tokens == 0:
            raise AssertionError("Streaming response did not include assistant tokens")
        return f"chunks={chunk_counter} tokens~{content_tokens}"

    # Utility methods ------------------------------------------------------------------

    def ensure_model_id(self) -> str:
        """Ensure a model ID is available for testing.

        Returns
        -------
        str
            The model ID to use for testing.

        Raises
        ------
        AssertionError
            If no model ID is available from environment or previous model discovery.
        """
        if not self.model_id:
            raise AssertionError(
                "Model id unavailable. Ensure GET /v1/models succeeds or set MLX_MODEL_ID."
            )
        return self.model_id

    @staticmethod
    def iter_sse_payloads(response: httpx.Response) -> Iterable[str]:
        """Iterate over Server-Sent Events payloads from a streaming response.

        Parameters
        ----------
        response : httpx.Response
            The streaming HTTP response containing SSE data.

        Yields
        ------
        str
            Individual SSE payload strings (data after "data: " prefix).
        """
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            payload = line.split("data:", 1)[1].lstrip()
            if payload:
                yield payload

    # Runner ---------------------------------------------------------------------------

    def run(self) -> bool:
        """Run all test cases and return overall success status.

        Executes all defined test cases and logs their results.

        Returns
        -------
        bool
            True if all tests passed, False if any test failed.
        """
        tests = [
            TestCase("GET /health", LLMContractTester.test_health),
            TestCase("GET /v1/models", LLMContractTester.test_models),
            TestCase("POST /v1/chat/completions", LLMContractTester.test_chat_completion),
            TestCase(
                "POST /v1/chat/completions (stream)",
                LLMContractTester.test_chat_completion_stream,
            ),
        ]
        all_passed = True
        for test in tests:
            ok, message = self.run_single(test)
            indicator = "✓" if ok else "✗"
            detail = f" - {message}" if message else ""
            logger.info("%s %s%s", indicator, test.name, detail)
            all_passed &= ok
        return all_passed

    def run_single(self, test: TestCase) -> tuple[bool, str]:
        """Run a single test case and return the result.

        Parameters
        ----------
        test : TestCase
            The test case to execute.

        Returns
        -------
        tuple[bool, str]
            A tuple of (success: bool, message: str) where success indicates
            if the test passed and message provides details.
        """
        try:
            detail = test.handler(self)
        except ValidationError as exc:
            return False, f"schema validation failed: {exc.errors()[:2]}"
        except (httpx.HTTPError, AssertionError, ValueError) as exc:
            return False, str(exc)
        except Exception as exc:  # pragma: no cover - safety net
            return False, f"unexpected error: {exc}"
        else:
            return True, detail


# --------------------------------------------------------------------------------------


def main() -> None:
    """Main entry point for the LLM contract testing script.

    Initializes the tester, runs all tests, and exits with appropriate status code.
    """
    base_url = env_base_url()
    tester = LLMContractTester(base_url)
    try:
        success = tester.run()
    finally:
        tester.close()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
