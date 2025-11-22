#!/usr/bin/env python3
"""Universal API contract test suite for the MLX OpenAI-compatible server."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import json
import logging
import os
import sys
from typing import Literal

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

try:  # Dependency guard keeps script self-explanatory when deps are missing.
    import httpx
except ImportError as e:  # pragma: no cover - runtime dependency defense
    logger.error("✗ Missing required dependency: httpx")
    raise SystemExit(1) from e

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
except ImportError as e:  # pragma: no cover
    logger.error("✗ Missing required dependency: pydantic")
    raise SystemExit(1) from e


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


@dataclass
class TestCase:
    """Represents a test case."""

    name: str
    handler: Callable[[LLMContractTester], str]


class LLMContractTester:
    """Runs OpenAI contract checks against a target server."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url
        self.headers = build_headers()
        self.client = httpx.Client(base_url=base_url, timeout=timeout, headers=self.headers)
        self.model_id: str | None = os.getenv("MLX_MODEL_ID")
        self.results: list[tuple[str, bool, str]] = []

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()

    # Individual tests -----------------------------------------------------------------

    def test_health(self) -> str:
        """Test the health endpoint."""
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
        """Test the models endpoint."""
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
        """Test chat completion endpoint."""
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
        """Test streaming chat completion endpoint."""
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
        """Ensure a model ID is available."""
        if not self.model_id:
            raise AssertionError(
                "Model id unavailable. Ensure GET /v1/models succeeds or set MLX_MODEL_ID."
            )
        return self.model_id

    @staticmethod
    def iter_sse_payloads(response: httpx.Response) -> Iterable[str]:
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

    # Runner ---------------------------------------------------------------------------

    def run(self) -> bool:
        """Run all tests."""
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
        """Run a single test."""
        try:
            detail = test.handler(self)
        except ValidationError as e:
            logger.exception("schema validation failed")
            return False, f"schema validation failed: {e.errors()[:2]}"
        except (httpx.HTTPError, AssertionError, ValueError) as e:
            logger.exception("request failed")
            return False, f"{type(e).__name__}: {e}"
        except Exception as e:  # pragma: no cover - safety net
            logger.exception("unexpected error")
            return False, f"unexpected error: {type(e).__name__}: {e}"
        else:
            return True, detail


# --------------------------------------------------------------------------------------


def main() -> None:
    """Run the LLM contract tests."""
    base_url = env_base_url()
    tester = LLMContractTester(base_url)
    try:
        success = tester.run()
    finally:
        tester.close()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
