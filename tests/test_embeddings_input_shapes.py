"""Tests for the accepted shapes of ``/v1/embeddings`` input."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.handler.mlx_embeddings import MLXEmbeddingsHandler
from app.schemas.openai import EmbeddingRequest


class _FakeTokenizer:
    """Tokenizer stub that renders token ids as ``tok<id>`` words."""

    def decode(self, token_ids: list[int]) -> str:
        """Return a readable stand-in for the decoded text."""
        return " ".join(f"tok{token_id}" for token_id in token_ids)


def _handler() -> MLXEmbeddingsHandler:
    """Build a handler without loading a real embedding model."""
    handler = MLXEmbeddingsHandler.__new__(MLXEmbeddingsHandler)
    handler.model = SimpleNamespace(tokenizer=_FakeTokenizer())
    return handler


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("hello", ["hello"]),
        (["hello", "world"], ["hello", "world"]),
        ([1985, 78], ["tok1985 tok78"]),
        ([[1985, 78], [42]], ["tok1985 tok78", "tok42"]),
        ([], []),
    ],
)
def test_normalize_input_shapes(value: Any, expected: list[str]) -> None:
    """Strings and pre-tokenized input should both reduce to a list of strings."""
    assert _handler()._normalize_input(value) == expected


def test_normalize_input_rejects_mixed_shapes() -> None:
    """A list that is neither all text nor all token ids should be rejected."""
    with pytest.raises(ValueError, match="Embedding input must be"):
        _handler()._normalize_input([{"unsupported": True}])


@pytest.mark.parametrize(
    "value",
    [
        "hello",
        ["hello", "world"],
        [1985, 78, 3067],
        [[1985, 78], [3067]],
    ],
)
def test_request_schema_accepts_pretokenized_input(value: Any) -> None:
    """Clients that tokenize before sending must not be rejected at validation.

    LangChain's ``OpenAIEmbeddings`` counts tokens client-side and posts token
    id arrays, which the OpenAI embeddings API accepts.
    """
    assert EmbeddingRequest(model="m", input=value).input == value


def test_numeric_strings_are_not_coerced_to_token_ids() -> None:
    """Digit-only text must stay text rather than validating as token ids."""
    request = EmbeddingRequest(model="m", input=["1985", "78"])

    assert request.input == ["1985", "78"]
    assert _handler()._normalize_input(request.input) == ["1985", "78"]
