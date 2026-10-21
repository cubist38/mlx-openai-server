from collections.abc import AsyncGenerator
import gc
from http import HTTPStatus
import json
import os
import tempfile
import time
from typing import Any
import uuid

from fastapi import HTTPException
from loguru import logger
from mlx_whisper.tokenizer import LANGUAGES

from ..core import InferenceWorker
from ..models.mlx_whisper import MLX_Whisper, calculate_audio_duration
from ..schemas.openai import (
    Delta,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseFormat,
    TranscriptionResponseStream,
    TranscriptionResponseStreamChoice,
    TranscriptionSegment,
    TranscriptionTimestampGranularity,
    TranscriptionUsageAudio,
    TranscriptionVerboseResponse,
    TranscriptionWord,
)
from ..utils.errors import create_error_response


def pop_response_options(
    request_data: dict[str, Any],
) -> tuple[TranscriptionResponseFormat, set[TranscriptionTimestampGranularity]]:
    """Take the response options out of prepared request data.

    What remains is passed to ``mlx_whisper.transcribe``. When word
    timestamps are requested, ``word_timestamps`` is turned on for it.

    Parameters
    ----------
    request_data : dict[str, Any]
        Prepared request data, modified in place.

    Returns
    -------
    tuple[TranscriptionResponseFormat, set[TranscriptionTimestampGranularity]]
        The response format, and the timestamp granularities for
        ``verbose_json`` (segments when none were asked for).
    """
    response_format = TranscriptionResponseFormat(
        request_data.pop("response_format", None) or TranscriptionResponseFormat.JSON
    )
    granularities = {
        TranscriptionTimestampGranularity(g)
        for g in request_data.pop("timestamp_granularities", None) or []
    } or {TranscriptionTimestampGranularity.SEGMENT}
    if (
        response_format == TranscriptionResponseFormat.VERBOSE_JSON
        and TranscriptionTimestampGranularity.WORD in granularities
    ):
        request_data["word_timestamps"] = True
    return response_format, granularities


def build_transcription_response(
    result: dict[str, Any],
    duration: float,
    response_format: TranscriptionResponseFormat,
    granularities: set[TranscriptionTimestampGranularity],
) -> TranscriptionResponse | TranscriptionVerboseResponse:
    """Build the API response from an ``mlx_whisper.transcribe`` result.

    Parameters
    ----------
    result : dict[str, Any]
        The result of ``mlx_whisper.transcribe``: ``text``, ``segments``
        and ``language``.
    duration : float
        The duration of the audio in seconds.
    response_format : TranscriptionResponseFormat
        The requested response format.
    granularities : set[TranscriptionTimestampGranularity]
        The timestamp granularities to include with ``verbose_json``.

    Returns
    -------
    TranscriptionResponse | TranscriptionVerboseResponse
        A verbose response with timestamps for ``verbose_json``,
        otherwise the text alone.
    """
    usage = TranscriptionUsageAudio(type="duration", seconds=int(duration))
    if response_format != TranscriptionResponseFormat.VERBOSE_JSON:
        return TranscriptionResponse(text=result["text"], usage=usage)

    raw_segments = result.get("segments") or []
    segments = None
    if TranscriptionTimestampGranularity.SEGMENT in granularities:
        segments = [
            TranscriptionSegment(
                id=i,
                seek=s["seek"],
                start=s["start"],
                end=s["end"],
                text=s["text"],
                tokens=s["tokens"],
                temperature=s["temperature"],
                avg_logprob=s["avg_logprob"],
                compression_ratio=s["compression_ratio"],
                no_speech_prob=s["no_speech_prob"],
            )
            for i, s in enumerate(raw_segments)
        ]
    words = None
    if TranscriptionTimestampGranularity.WORD in granularities:
        words = [
            TranscriptionWord(word=w["word"].strip(), start=w["start"], end=w["end"])
            for s in raw_segments
            for w in s.get("words") or []
        ]
    code = result.get("language") or ""
    return TranscriptionVerboseResponse(
        language=LANGUAGES.get(code, code),
        duration=duration,
        text=result["text"],
        segments=segments,
        words=words,
        usage=usage,
    )


class MLXWhisperHandler:
    """
    Handler class for making requests to the underlying MLX Whisper model service.
    Provides request queuing, metrics tracking, and robust error handling for audio transcription.
    """

    handler_type: str = "whisper"

    def __init__(self, model_path: str):
        """
        Initialize the handler with the specified model path.

        Args:
            model_path (str): Path to the model directory.
        """
        self.model_path = model_path
        self.model = MLX_Whisper(model_path)
        self.model_created = int(time.time())  # Store creation time when model is loaded

        # Dedicated inference thread — keeps the event loop free during
        # blocking MLX model computation.
        self.inference_worker = InferenceWorker()

        logger.info(f"Initialized MLXWhisperHandler with model path: {model_path}")

    async def get_models(self) -> list[dict[str, Any]]:
        """
        Get list of available models with their metadata.
        """
        try:
            return [
                {
                    "id": self.model_path,
                    "object": "model",
                    "created": self.model_created,
                    "owned_by": "local",
                }
            ]
        except Exception as e:
            logger.error(f"Error getting models: {e!s}")
            return []

    async def initialize(self, queue_config: dict[str, Any] | None = None) -> None:
        """Initialize the handler and start the inference worker.

        Parameters
        ----------
        queue_config : dict, optional
            Dictionary with ``queue_size`` and ``timeout`` keys used
            to configure the inference worker's internal queue.
        """
        if not queue_config:
            queue_config = {
                "timeout": 600,  # Longer timeout for audio processing
                "queue_size": 50,
            }
        self.inference_worker = InferenceWorker(
            queue_size=queue_config.get("queue_size", 50),
            timeout=queue_config.get("timeout", 600),
        )
        self.inference_worker.start()
        logger.info("Initialized MLXWhisperHandler and started inference worker")

    async def generate_transcription_response(
        self, request: TranscriptionRequest
    ) -> TranscriptionResponse:
        """
        Generate a transcription response for the given request.
        """
        temp_file_path = None

        try:
            request_data = await self._prepare_transcription_request(request)
            temp_file_path = request_data.get("audio_path")

            # Submit to the inference thread
            audio_path = request_data.pop("audio_path")
            response_format, granularities = pop_response_options(request_data)
            response = await self.inference_worker.submit(
                self.model,
                audio_path=audio_path,
                **request_data,
            )
            response_data = build_transcription_response(
                response, calculate_audio_duration(temp_file_path), response_format, granularities
            )
            if response_format != TranscriptionResponseFormat.TEXT:
                return response_data
            # dump to string for text response
            return json.dumps(response_data.model_dump())
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file_path}: {e!s}")

    async def generate_transcription_stream_from_data(
        self, request_data: dict[str, Any], response_format: TranscriptionResponseFormat
    ) -> AsyncGenerator[str, None]:
        """
        Generate a transcription stream from prepared request data.
        Yields SSE-formatted chunks with timing information.

        Args:
            request_data: Prepared request data with audio_path already saved
            response_format: The response format (json or text)
        """
        request_id = f"transcription-{uuid.uuid4()}"
        created_time = int(time.time())
        temp_file_path = request_data.get("audio_path")

        try:
            # Set stream mode and submit to inference thread
            request_data["stream"] = True
            audio_path = request_data.pop("audio_path")
            request_data.pop("stream")
            # Streamed chunks carry text only, so timestamps aren't computed.
            request_data.pop("response_format", None)
            request_data.pop("timestamp_granularities", None)

            generator = self.inference_worker.submit_stream(
                self.model,
                audio_path=audio_path,
                stream=True,
                **request_data,
            )

            # Stream each chunk (async — keeps event loop free)
            async for chunk in generator:
                # Create streaming response
                stream_response = TranscriptionResponseStream(
                    id=request_id,
                    object="transcription.chunk",
                    created=created_time,
                    model=self.model_path,
                    choices=[
                        TranscriptionResponseStreamChoice(
                            delta=Delta(content=chunk.get("text", "")), finish_reason=None
                        )
                    ],
                )

                # Yield as SSE format
                yield f"data: {stream_response.model_dump_json()}\n\n"

            # Send final chunk with finish_reason
            final_response = TranscriptionResponseStream(
                id=request_id,
                object="transcription.chunk",
                created=created_time,
                model=self.model_path,
                choices=[
                    TranscriptionResponseStreamChoice(delta=Delta(content=""), finish_reason="stop")
                ],
            )
            yield f"data: {final_response.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error during transcription streaming: {e!s}")
            raise
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file_path}: {e!s}")

    async def _save_uploaded_file(self, file) -> str:
        """
        Save the uploaded file to a temporary location.

        Args:
            file: The uploaded file object.

        Returns:
            str: Path to the temporary file.
        """
        try:
            # Create a temporary file with the same extension as the uploaded file
            file_extension = os.path.splitext(file.filename)[1] if file.filename else ".wav"

            print("file_extension", file_extension)

            # Read file content first (this can only be done once with FastAPI uploads)
            content = await file.read()

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                # Write the file contents
                temp_file.write(content)
                temp_path = temp_file.name

            logger.debug(f"Saved uploaded file to temporary location: {temp_path}")
            return temp_path

        except Exception as e:
            logger.error(f"Error saving uploaded file: {e!s}")
            raise

    async def _prepare_transcription_request(self, request: TranscriptionRequest) -> dict[str, Any]:
        """
        Prepare a transcription request by parsing model parameters.

        Args:
            request: TranscriptionRequest object.
            audio_path: Path to the audio file.

        Returns:
            Dict containing the request data ready for the model.
        """
        try:
            file = request.file

            file_path = await self._save_uploaded_file(file)
            request_data = {
                "audio_path": file_path,
                "verbose": False,
                "response_format": request.response_format,
                "timestamp_granularities": request.timestamp_granularities,
            }

            # Add optional parameters if provided
            if request.temperature is not None:
                request_data["temperature"] = request.temperature

            if request.language is not None:
                request_data["language"] = request.language

            if request.prompt is not None:
                request_data["initial_prompt"] = request.prompt

            # Map additional parameters if they exist
            decode_options = {}
            if request.language is not None:
                decode_options["language"] = request.language

            # Add decode options to request data
            request_data.update(decode_options)

            logger.debug(f"Prepared transcription request: {request_data}")

            return request_data

        except Exception as e:
            logger.error(f"Failed to prepare transcription request: {e!s}")
            content = create_error_response(
                f"Failed to process request: {e!s}", "bad_request", HTTPStatus.BAD_REQUEST
            )
            raise HTTPException(status_code=400, detail=content)

    async def transcribe_from_data(self, request_data: dict[str, Any]) -> TranscriptionResponse:
        """Run transcription from pre-processed request data.

        This method is used by ``HandlerProcessProxy`` for IPC: the
        proxy saves the uploaded file in the main process and sends
        a plain dict with the file path here.

        Parameters
        ----------
        request_data : dict[str, Any]
            Dictionary containing ``audio_path``, the response options
            (``response_format``, ``timestamp_granularities``) and optional
            model parameters (``temperature``, ``language``, etc.).

        Returns
        -------
        TranscriptionResponse | TranscriptionVerboseResponse
            The transcription result with usage info, and timestamps for
            ``verbose_json``.
        """
        temp_file_path = request_data.get("audio_path")
        try:
            audio_path = request_data.pop("audio_path")
            response_format, granularities = pop_response_options(request_data)
            response = await self.inference_worker.submit(
                self.model,
                audio_path=audio_path,
                **request_data,
            )
            return build_transcription_response(
                response, calculate_audio_duration(temp_file_path), response_format, granularities
            )
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_file_path}: {e}")

    async def transcribe_stream_from_data(
        self,
        request_data: dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """Run streaming transcription from pre-processed request data.

        This method is used by ``HandlerProcessProxy`` for IPC: the
        proxy saves the uploaded file in the main process and sends
        a plain dict with the file path here.

        Parameters
        ----------
        request_data : dict[str, Any]
            Dictionary containing ``audio_path`` and optional model
            parameters.

        Yields
        ------
        str
            SSE-formatted transcription chunks.
        """
        request_id = f"transcription-{uuid.uuid4()}"
        created_time = int(time.time())
        temp_file_path = request_data.get("audio_path")

        try:
            request_data["stream"] = True
            audio_path = request_data.pop("audio_path")
            request_data.pop("stream")
            # Streamed chunks carry text only, so timestamps aren't computed.
            request_data.pop("response_format", None)
            request_data.pop("timestamp_granularities", None)

            generator = self.inference_worker.submit_stream(
                self.model,
                audio_path=audio_path,
                stream=True,
                **request_data,
            )

            async for chunk in generator:
                stream_response = TranscriptionResponseStream(
                    id=request_id,
                    object="transcription.chunk",
                    created=created_time,
                    model=self.model_path,
                    choices=[
                        TranscriptionResponseStreamChoice(
                            delta=Delta(content=chunk.get("text", "")),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {stream_response.model_dump_json()}\n\n"

            final_response = TranscriptionResponseStream(
                id=request_id,
                object="transcription.chunk",
                created=created_time,
                model=self.model_path,
                choices=[
                    TranscriptionResponseStreamChoice(
                        delta=Delta(content=""),
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {final_response.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error during transcription streaming: {e}")
            raise
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_file_path}: {e}")

    async def get_queue_stats(self) -> dict[str, Any]:
        """Get statistics from the inference worker.

        Returns
        -------
        dict[str, Any]
            Dictionary with ``queue_stats`` sub-dictionary.
        """
        return {
            "queue_stats": self.inference_worker.get_stats(),
        }

    async def cleanup(self) -> None:
        """Cleanup resources and stop the inference worker before shutdown.

        This method ensures all pending requests are properly completed
        and resources are released.
        """
        try:
            logger.info("Cleaning up MLXWhisperHandler resources")
            if hasattr(self, "inference_worker"):
                self.inference_worker.stop()
            # Force garbage collection
            gc.collect()
            logger.info("MLXWhisperHandler cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during MLXWhisperHandler cleanup: {e!s}")
            raise
