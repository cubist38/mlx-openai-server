import gc
import json
import os
import tempfile
import time
import uuid
from http import HTTPStatus
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import HTTPException
from loguru import logger

from ..core import InferenceWorker
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

    def __init__(self, model_path: str, max_concurrency: int = 1):
        """
        Initialize the handler with the specified model path.
        
        Args:
            model_path (str): Path to the model directory.
            max_concurrency (int): Maximum number of concurrent model inference tasks.
        """
        self.model_path = model_path
        self.model = MLX_Whisper(model_path)
        self.model_created = int(time.time())  # Store creation time when model is loaded

        # Dedicated inference thread — keeps the event loop free during
        # blocking MLX model computation.
        self.inference_worker = InferenceWorker()

        logger.info(f"Initialized MLXWhisperHandler with model path: {model_path}")
    
    async def get_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models with their metadata.
        """
        try:
            return [{
                "id": self.model_path,
                "object": "model",
                "created": self.model_created,
                "owned_by": "local"
            }]
        except Exception as e:
            logger.error(f"Error getting models: {str(e)}")
            return []
    
    async def initialize(self, queue_config: Optional[Dict[str, Any]] = None) -> None:
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

    async def generate_transcription_response(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """
        Generate a transcription response for the given request.
        """
        request_id = f"transcription-{uuid.uuid4()}"
        temp_file_path = None
        
        try:
            request_data = await self._prepare_transcription_request(request)
            temp_file_path = request_data.get("audio_path")

            # Submit to the inference thread
            audio_path = request_data.pop("audio_path")
            response = await self.inference_worker.submit(
                self.model,
                audio_path=audio_path,
                **request_data,
            )
            response_data = TranscriptionResponse(
                text=response["text"],
                usage=TranscriptionUsageAudio(
                    type="duration",
                    seconds=int(calculate_audio_duration(temp_file_path))
                )
            )
            if request.response_format == TranscriptionResponseFormat.JSON:
                return response_data
            else:
                # dump to string for text response
                return json.dumps(response_data.model_dump())
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file_path}: {str(e)}")
            # Force garbage collection
            gc.collect()

    async def generate_transcription_stream_from_data(
        self, 
        request_data: Dict[str, Any],
        response_format: TranscriptionResponseFormat
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
                            delta=Delta(
                                content=chunk.get("text", "")
                            ),
                            finish_reason=None
                        )
                    ]
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
                    TranscriptionResponseStreamChoice(
                        delta=Delta(content=""),
                        finish_reason="stop"
                    )
                ]
            )
            yield f"data: {final_response.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error during transcription streaming: {str(e)}")
            raise
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file_path}: {str(e)}")
            # Clean up
            gc.collect()



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
            logger.error(f"Error saving uploaded file: {str(e)}")
            raise

    async def _prepare_transcription_request(
        self, 
        request: TranscriptionRequest    
        ) -> Dict[str, Any]:
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
            logger.error(f"Failed to prepare transcription request: {str(e)}")
            content = create_error_response(
                f"Failed to process request: {str(e)}", 
                "bad_request", 
                HTTPStatus.BAD_REQUEST
            )
            raise HTTPException(status_code=400, detail=content)

    async def get_queue_stats(self) -> Dict[str, Any]:
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
            if hasattr(self, 'inference_worker'):
                self.inference_worker.stop()
            logger.info("MLXWhisperHandler cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during MLXWhisperHandler cleanup: {str(e)}")
            raise

