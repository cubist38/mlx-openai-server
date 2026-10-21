"""Tests for ``verbose_json`` transcriptions with segment and word timestamps."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import FastAPI, Form
from fastapi.testclient import TestClient
import pytest

from app.handler import mlx_whisper as whisper_handler
from app.handler.mlx_whisper import (
    MLXWhisperHandler,
    build_transcription_response,
    pop_response_options,
)
from app.schemas.openai import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseFormat,
    TranscriptionTimestampGranularity,
    TranscriptionVerboseResponse,
)

WORD = TranscriptionTimestampGranularity.WORD
SEGMENT = TranscriptionTimestampGranularity.SEGMENT


def _segment(seek: int, start: float, end: float, words: list[tuple[str, float, float]]) -> dict:
    """Build a segment shaped like ``mlx_whisper.transcribe`` output."""
    return {
        "seek": seek,
        "start": start,
        "end": end,
        "text": "".join(w for w, _, _ in words),
        "tokens": [50364, 1, 2, 50414],
        "temperature": 0.0,
        "avg_logprob": -0.2,
        "compression_ratio": 1.3,
        "no_speech_prob": 0.01,
        "words": [{"word": w, "start": s, "end": e, "probability": 0.9} for w, s, e in words],
    }


RESULT: dict[str, Any] = {
    "text": " Hello there. Walking on.",
    "language": "en",
    "segments": [
        _segment(0, 0.0, 1.2, [(" Hello", 0.0, 0.5), (" there.", 0.5, 1.2)]),
        _segment(0, 1.4, 2.6, [(" Walking", 1.4, 2.0), (" on.", 2.0, 2.6)]),
    ],
}


def _form_app() -> TestClient:
    """A minimal app that parses the transcription form the way the endpoint does."""
    app = FastAPI()

    @app.post("/form")
    async def form(request: Annotated[TranscriptionRequest, Form()]) -> dict[str, Any]:
        return {
            "response_format": request.response_format,
            "timestamp_granularities": request.timestamp_granularities,
        }

    return TestClient(app)


@pytest.mark.parametrize("key", ["timestamp_granularities", "timestamp_granularities[]"])
def test_form_accepts_plain_and_bracketed_granularity_keys(key: str) -> None:
    """Both spellings parse; the OpenAI SDKs send the bracketed one."""
    response = _form_app().post(
        "/form",
        data={"response_format": "verbose_json", key: ["word", "segment"]},
        files={"file": ("a.wav", b"RIFF", "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json() == {
        "response_format": "verbose_json",
        "timestamp_granularities": ["word", "segment"],
    }


def test_form_rejects_unknown_granularity() -> None:
    """An unknown granularity is a validation error, not silently ignored."""
    response = _form_app().post(
        "/form",
        data={"response_format": "verbose_json", "timestamp_granularities": ["sentence"]},
        files={"file": ("a.wav", b"RIFF", "audio/wav")},
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("response_format", "granularities", "expected_granularities", "word_timestamps"),
    [
        ("json", None, {SEGMENT}, False),
        ("json", ["word"], {WORD}, False),
        ("verbose_json", None, {SEGMENT}, False),
        ("verbose_json", ["segment"], {SEGMENT}, False),
        ("verbose_json", ["word"], {WORD}, True),
        ("verbose_json", ["word", "segment"], {WORD, SEGMENT}, True),
    ],
)
def test_pop_response_options(
    response_format: str,
    granularities: list[str] | None,
    expected_granularities: set[TranscriptionTimestampGranularity],
    word_timestamps: bool,
) -> None:
    """Options are removed, and word timing is only computed when it is returned."""
    request_data = {
        "verbose": False,
        "language": "en",
        "response_format": response_format,
        "timestamp_granularities": granularities,
    }
    fmt, found = pop_response_options(request_data)
    assert fmt == TranscriptionResponseFormat(response_format)
    assert found == expected_granularities
    assert "response_format" not in request_data
    assert "timestamp_granularities" not in request_data
    assert request_data.get("word_timestamps", False) is word_timestamps


def test_json_response_is_text_only() -> None:
    """``json`` keeps its existing shape."""
    response = build_transcription_response(
        RESULT, 2.6, TranscriptionResponseFormat.JSON, {SEGMENT}
    )
    assert isinstance(response, TranscriptionResponse)
    assert response.model_dump() == {
        "text": " Hello there. Walking on.",
        "usage": {"type": "duration", "seconds": 2},
    }


def test_verbose_json_with_segments() -> None:
    """Segments are numbered and carry Whisper's decoding statistics."""
    response = build_transcription_response(
        RESULT, 2.6, TranscriptionResponseFormat.VERBOSE_JSON, {SEGMENT}
    )
    assert isinstance(response, TranscriptionVerboseResponse)
    assert response.task == "transcribe"
    assert response.language == "english"
    assert response.duration == 2.6
    assert response.words is None
    assert [(s.id, s.start, s.end, s.text) for s in response.segments] == [
        (0, 0.0, 1.2, " Hello there."),
        (1, 1.4, 2.6, " Walking on."),
    ]
    assert response.segments[0].no_speech_prob == 0.01
    assert "words" not in response.segments[0].model_dump()


def test_verbose_json_with_words_only() -> None:
    """Words are flattened across segments and trimmed, as OpenAI returns them."""
    response = build_transcription_response(
        RESULT, 2.6, TranscriptionResponseFormat.VERBOSE_JSON, {WORD}
    )
    assert response.segments is None
    assert [w.model_dump() for w in response.words] == [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "there.", "start": 0.5, "end": 1.2},
        {"word": "Walking", "start": 1.4, "end": 2.0},
        {"word": "on.", "start": 2.0, "end": 2.6},
    ]


def test_verbose_json_with_no_speech() -> None:
    """Silent audio gives empty lists, and an unknown language code is kept as is."""
    result = {"text": "", "language": "xx", "segments": []}
    response = build_transcription_response(
        result, 1.0, TranscriptionResponseFormat.VERBOSE_JSON, {WORD, SEGMENT}
    )
    assert response.segments == []
    assert response.words == []
    assert response.language == "xx"


class _FakeWorker:
    """Records what reaches ``mlx_whisper.transcribe`` and returns ``RESULT``."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def submit(self, model: Any, **kwargs: Any) -> dict[str, Any]:
        self.kwargs = kwargs
        return RESULT


@pytest.mark.parametrize(
    ("response_format", "granularities", "word_timestamps", "response_type"),
    [
        ("json", None, None, TranscriptionResponse),
        ("verbose_json", ["word"], True, TranscriptionVerboseResponse),
    ],
)
def test_transcribe_from_data_forwards_only_model_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    response_format: str,
    granularities: list[str] | None,
    word_timestamps: bool | None,
    response_type: type,
) -> None:
    """The multi-model path builds the requested shape and passes clean kwargs on."""
    monkeypatch.setattr(whisper_handler, "calculate_audio_duration", lambda path: 2.6)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    handler = MLXWhisperHandler("mlx-community/whisper-tiny")
    worker = _FakeWorker()
    handler.inference_worker = worker

    response = asyncio.run(
        handler.transcribe_from_data(
            {
                "audio_path": str(audio),
                "verbose": False,
                "response_format": response_format,
                "timestamp_granularities": granularities,
            }
        )
    )

    assert isinstance(response, response_type)
    assert "response_format" not in worker.kwargs
    assert "timestamp_granularities" not in worker.kwargs
    assert worker.kwargs.get("word_timestamps") is word_timestamps
    assert not audio.exists()
