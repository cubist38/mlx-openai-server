"""
Handler for MLX Whisper audio transcription models.

This module provides the MLXWhisperHandler class for processing audio transcription
requests using MLX Whisper models, with support for streaming responses and
request queuing.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import gc
from http import HTTPStatus
from pathlib import Path
import tempfile
import time
from typing import Any
import uuid

from fastapi import HTTPException, UploadFile
from loguru import logger

from ..core.queue import RequestQueue
from ..models.mlx_whisper import MLX_Whisper, calculate_audio_duration
from ..schemas.openai import (
    Delta,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseFormat,
    TranscriptionResponseStream,
    TranscriptionResponseStreamChoice,
    TranscriptionUsageAudio,
)
from ..utils.errors import create_error_response


class MLXWhisperHandler:
    """
    Handler class for making requests to the underlying MLX Whisper model service.

    Provides request queuing, metrics tracking, and robust error handling for audio transcription.
    """

    def __init__(self, model_path: str, max_concurrency: int = 1) -> None:
        """
        Initialize the handler with the specified model path.

        Args:
            model_path (str): Path to the model directory.
            max_concurrency (int): Maximum number of concurrent model inference tasks.
        """
        self.model_path = model_path
        self.model = MLX_Whisper(model_path)
        self.model_created = int(time.time())  # Store creation time when model is loaded

        # Initialize request queue for audio transcription tasks
        self.request_queue = RequestQueue(max_concurrency=max_concurrency)

        logger.info(f"Initialized MLXWhisperHandler with model path: {model_path}")

    async def get_models(self) -> list[dict[str, Any]]:
        """Get list of available models with their metadata."""
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
        """Initialize the handler and start the request queue."""
        if not queue_config:
            queue_config = {
                "max_concurrency": self.request_queue.max_concurrency,
                "timeout": 600,  # Longer timeout for audio processing
                "queue_size": 50,
            }
        self.request_queue = RequestQueue(
            max_concurrency=queue_config.get("max_concurrency", 1),
            timeout=queue_config.get("timeout", 600),
            queue_size=queue_config.get("queue_size", 50),
        )
        await self.request_queue.start(self._process_request)
        logger.info("Initialized MLXWhisperHandler and started request queue")

    async def generate_transcription_response(
        self, request: TranscriptionRequest
    ) -> TranscriptionResponse | str:
        """Generate a transcription response for the given request."""
        request_id = f"transcription-{uuid.uuid4()}"
        temp_file_path = None

        try:
            request_data = await self.prepare_transcription_request(request)
            temp_file_path = request_data.get("audio_path")
            response = await self.request_queue.submit(request_id, request_data)
            response_data = TranscriptionResponse(
                text=response["text"],
                usage=TranscriptionUsageAudio(
                    type="duration", seconds=int(calculate_audio_duration(temp_file_path))
                ),
            )
            if request.response_format == TranscriptionResponseFormat.JSON:
                return response_data
            # Return plain text for text response format
            return response_data.text
        finally:
            # Clean up temporary file
            if temp_file_path and Path(temp_file_path).exists():
                try:
                    Path(temp_file_path).unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file_path}: {e!s}")
            # Force garbage collection
            gc.collect()

    async def generate_transcription_stream_from_data(
        self, request_data: dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Generate a transcription stream from prepared request data.

        Yields SSE-formatted chunks with timing information.

        Args:
            request_data: Prepared request data with audio_path already saved
        """
        request_id = f"transcription-{uuid.uuid4()}"
        created_time = int(time.time())
        temp_file_path = request_data.get("audio_path")

        try:
            # Set stream mode
            request_data["stream"] = True

            # Offload synchronous generator to thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            audio_path = request_data.pop("audio_path")

            def collect_chunks() -> list[dict[str, Any]]:
                """Collect all chunks from the synchronous generator."""
                generator = self.model(audio_path=audio_path, **request_data)
                return list(generator)

            # Run the synchronous generator collection in a thread
            chunks = await loop.run_in_executor(None, collect_chunks)

            # Stream each chunk asynchronously
            for chunk in chunks:
                # Create streaming response
                stream_response = TranscriptionResponseStream(
                    id=request_id,
                    object="transcription.chunk",
                    created=created_time,
                    model=self.model_path,
                    choices=[
                        TranscriptionResponseStreamChoice(
                            delta=Delta(content=chunk.get("text", "")),  # type: ignore[call-arg]
                            finish_reason=None,
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
                    TranscriptionResponseStreamChoice(delta=Delta(content=""), finish_reason="stop")  # type: ignore[call-arg]
                ],
            )
            yield f"data: {final_response.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Error during transcription streaming: {e!s}")
            raise
        finally:
            # Clean up temporary file
            if temp_file_path and Path(temp_file_path).exists():
                try:
                    Path(temp_file_path).unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file_path}: {e!s}")
            # Clean up
            gc.collect()

    async def _process_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """
        Process an audio transcription request. This is the worker function for the request queue.

        Args:
            request_data: Dictionary containing the request data.

        Returns
        -------
            Dict: The model's response containing transcribed text.
        """
        try:
            # Extract request parameters
            audio_path = request_data.pop("audio_path")

            # Call the model with the audio file
            result = self.model(audio_path=audio_path, **request_data)

            # Force garbage collection after model inference
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing audio transcription request: {e!s}")
            # Clean up on error
            gc.collect()
            raise
        else:
            return result

    async def _save_uploaded_file(self, file: UploadFile) -> str:
        """
        Save the uploaded file to a temporary location.

        Args:
            file: The uploaded file object.

        Returns
        -------
            str: Path to the temporary file.
        """
        try:
            # Create a temporary file with the same extension as the uploaded file
            file_extension = Path(file.filename).suffix if file.filename else ".wav"

            logger.debug(f"file_extension: {file_extension}")

            # Read file content first (this can only be done once with FastAPI uploads)
            content = await file.read()

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                # Write the file contents
                temp_file.write(content)
                temp_path = temp_file.name

            logger.debug(f"Saved uploaded file to temporary location: {temp_path}")

        except OSError as e:
            logger.error(f"Error saving uploaded file ({e.__class__.__name__}): {e}")
            raise HTTPException(
                status_code=500, detail="Internal server error while saving uploaded file"
            ) from e
        except Exception as e:
            logger.error(f"Error saving uploaded file ({e.__class__.__name__}): {e!s}")
            raise
        else:
            return temp_path

    async def prepare_transcription_request(self, request: TranscriptionRequest) -> dict[str, Any]:
        """
        Prepare a transcription request by parsing model parameters.

        The function saves the uploaded file and returns its path along with other
        request fields ready for the model.

        Args:
            request: TranscriptionRequest object.

        Returns
        -------
            Dict containing the saved audio path and other request data ready for the model.
        """
        try:
            file = request.file

            file_path = await self._save_uploaded_file(file)
            request_data = {
                "audio_path": file_path,
                "verbose": False,
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

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to prepare transcription request: {e!s}")
            content = create_error_response(
                f"Failed to process request: {e!s}", "bad_request", HTTPStatus.BAD_REQUEST
            )
            raise HTTPException(status_code=400, detail=content) from e
        else:
            return request_data

    async def get_queue_stats(self) -> dict[str, Any]:
        """
        Get statistics from the request queue.

        Returns
        -------
            Dict with queue statistics.
        """
        return self.request_queue.get_queue_stats()

    async def cleanup(self) -> None:
        """
        Cleanup resources and stop the request queue before shutdown.

        This method ensures all pending requests are properly cancelled
        and resources are released.
        """
        try:
            logger.info("Cleaning up MLXWhisperHandler resources")
            if hasattr(self, "request_queue"):
                await self.request_queue.stop()
            logger.info("MLXWhisperHandler cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during MLXWhisperHandler cleanup: {e!s}")
