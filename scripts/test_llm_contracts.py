#!/usr/bin/env python3
"""Universal API contract test suite for the MLX OpenAI-compatible server."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Literal, Optional

try:  # Dependency guard keeps script self-explanatory when deps are missing.
    import httpx
except ImportError as exc:  # pragma: no cover - runtime dependency defense
    print("✗ Missing required dependency: httpx", file=sys.stderr)
    raise SystemExit(1) from exc

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
except ImportError as exc:  # pragma: no cover
    print("✗ Missing required dependency: pydantic", file=sys.stderr)
    raise SystemExit(1) from exc


# --------------------------------------------------------------------------------------
# OpenAI-compatible response models (trimmed to required fields, tolerant to extras).


class HealthResponse(BaseModel):
    status: str = Field(..., description="Server reported status string")

    model_config = ConfigDict(extra="allow")


class ModelData(BaseModel):
    id: str
    object: Literal["model"]
    created: Optional[int] = None
    owned_by: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class ModelList(BaseModel):
    object: Literal["list"]
    data: List[ModelData]

    model_config = ConfigDict(extra="allow")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: Optional[str]

    model_config = ConfigDict(extra="allow")


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    model_config = ConfigDict(extra="allow")


class ChatCompletion(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Optional[Usage] = None

    model_config = ConfigDict(extra="allow")


class DeltaMessage(BaseModel):
    role: Optional[Literal["system", "user", "assistant", "tool"]] = None
    content: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class ChatChunkChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: List[ChatChunkChoice]

    model_config = ConfigDict(extra="allow")


# --------------------------------------------------------------------------------------


def env_base_url() -> str:
    raw = os.getenv("MLX_URL", "http://127.0.0.1:8000")
    return raw.rstrip("/")


def build_headers() -> dict[str, str]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("MLX_API_KEY")
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


@dataclass
class TestCase:
    name: str
    handler: Callable[["LLMContractTester"], str]


class LLMContractTester:
    """Runs OpenAI contract checks against a target server."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url
        self.headers = build_headers()
        self.client = httpx.Client(base_url=base_url, timeout=timeout, headers=self.headers)
        self.model_id: Optional[str] = os.getenv("MLX_MODEL_ID")
        self.results: list[tuple[str, bool, str]] = []

    def close(self) -> None:
        self.client.close()

    # Individual tests -----------------------------------------------------------------

    def test_health(self) -> str:
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
        else:
            return f"status={status}, model_status={model_status}"

    def test_models(self) -> str:
        response = self.client.get("/v1/models")
        response.raise_for_status()
        payload = response.json()
        model_list = ModelList.model_validate(payload)
        if not model_list.data:
            raise AssertionError("Model registry returned an empty list")
        if not self.model_id:
            self.model_id = model_list.data[0].id

        # Verify metadata presence and required fields (Phase 02)
        first_model = model_list.data[0]
        raw_model = payload["data"][0]  # Get raw dict for metadata check
        if "metadata" in raw_model:
            metadata = raw_model["metadata"]
            if "context_length" not in metadata:
                raise AssertionError("Model metadata missing 'context_length' field")
            if metadata.get("backend") != "mlx":
                raise AssertionError(f"Expected backend='mlx', got '{metadata.get('backend')}'")
            return f"{len(model_list.data)} models, metadata: backend={metadata.get('backend')}, context={metadata.get('context_length')}"
        else:
            # Metadata is optional for backward compat, but warn if missing
            return f"{len(model_list.data)} models detected (no metadata)"

    def test_chat_completion(self) -> str:
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
        if not self.model_id:
            raise AssertionError(
                "Model id unavailable. Ensure GET /v1/models succeeds or set MLX_MODEL_ID."
            )
        return self.model_id

    @staticmethod
    def iter_sse_payloads(response: httpx.Response) -> Iterable[str]:
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
            print(f"{indicator} {test.name}{detail}")
            all_passed &= ok
        return all_passed

    def run_single(self, test: TestCase) -> tuple[bool, str]:
        try:
            detail = test.handler(self)
            return True, detail
        except ValidationError as exc:
            return False, f"schema validation failed: {exc.errors()[:2]}"
        except (httpx.HTTPError, AssertionError, ValueError) as exc:
            return False, str(exc)
        except Exception as exc:  # pragma: no cover - safety net
            return False, f"unexpected error: {exc}"


# --------------------------------------------------------------------------------------


def main() -> None:
    base_url = env_base_url()
    tester = LLMContractTester(base_url)
    try:
        success = tester.run()
    finally:
        tester.close()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
