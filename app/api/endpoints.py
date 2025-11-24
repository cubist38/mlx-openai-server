"""API endpoints for the MLX OpenAI server."""

import base64
from collections.abc import AsyncGenerator
from http import HTTPStatus
import json
import random
import time
from typing import Annotated, Any, Literal, TypeAlias

from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
import numpy as np

from ..handler import MFLUX_AVAILABLE, MLXFluxHandler
from ..handler.mlx_embeddings import MLXEmbeddingsHandler
from ..handler.mlx_lm import MLXLMHandler
from ..handler.mlx_vlm import MLXVLMHandler
from ..handler.mlx_whisper import MLXWhisperHandler
from ..schemas.openai import (
    ChatCompletionChunk,
    ChatCompletionMessageToolCall,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
    Delta,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResponseData,
    FunctionCall,
    HealthCheckResponse,
    HealthCheckStatus,
    ImageEditRequest,
    ImageEditResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    Message,
    Model,
    ModelsResponse,
    StreamingChoice,
    TranscriptionRequest,
    TranscriptionResponse,
    UsageInfo,
)
from ..utils.errors import create_error_response
from .hub_routes import (
    get_cached_model_metadata,
    get_configured_model_id,
    get_running_hub_models,
    hub_load_model,
    hub_router,
    hub_start_model,
    hub_status,
    hub_status_page,
    hub_stop_model,
    hub_unload_model,
)

router = APIRouter()
router.include_router(hub_router)

__all__ = [
    "router",
    "hub_status",
    "hub_status_page",
    "hub_start_model",
    "hub_stop_model",
    "hub_load_model",
    "hub_unload_model",
]


MLXHandlerType: TypeAlias = (
    MLXVLMHandler | MLXLMHandler | MLXFluxHandler | MLXEmbeddingsHandler | MLXWhisperHandler
)


async def _get_handler_or_error(
    raw_request: Request,
    reason: str,
    *,
    model_name: str | None = None,
) -> tuple[MLXHandlerType | None, JSONResponse | None]:
    """Return a loaded handler or an error response if unavailable.

    Parameters
    ----------
    raw_request : Request
        Incoming FastAPI request.
    reason : str
        Context string used for logging and load tracking.
    model_name : str | None, optional
        Explicit model identifier supplied by the caller.

    Returns
    -------
    tuple[MLXHandlerType | None, JSONResponse | None]
        A handler/error tuple where exactly one entry is ``None``.
    """

    controller = getattr(raw_request.app.state, "hub_controller", None)
    if controller is not None and model_name is not None:
        target = model_name.strip()
        if not target:
            return (
                None,
                JSONResponse(
                    content=create_error_response(
                        "Model name is required when running the hub server",
                        "invalid_request_error",
                        HTTPStatus.BAD_REQUEST,
                    ),
                    status_code=HTTPStatus.BAD_REQUEST,
                ),
            )
        try:
            handler = await controller.acquire_handler(target, reason=reason)
        except Exception as exc:
            status = getattr(exc, "status_code", HTTPStatus.INTERNAL_SERVER_ERROR)
            return (
                None,
                JSONResponse(
                    content=create_error_response(str(exc), "service_unavailable", status),
                    status_code=status,
                ),
            )
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception(
                f"Unable to load handler for model '{target}'. {type(exc).__name__}: {exc}"
            )
            return (
                None,
                JSONResponse(
                    content=create_error_response(
                        f"Failed to load model '{target}' handler",
                        "server_error",
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                    ),
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                ),
            )
        else:
            return handler, None

    handler_manager = getattr(raw_request.app.state, "handler_manager", None)
    if handler_manager is not None:
        try:
            handler = await handler_manager.ensure_loaded(reason)
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover - defensive logging
            logger.exception(
                f"Unable to load handler via JIT for {reason}. {type(e).__name__}: {e}"
            )
            return (
                None,
                JSONResponse(
                    content=create_error_response(
                        "Failed to load model handler",
                        "server_error",
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                    ),
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                ),
            )

        if handler is not None:
            return handler, None

    handler = getattr(raw_request.app.state, "handler", None)
    if handler is None:
        return (
            None,
            JSONResponse(
                content=create_error_response(
                    "Model handler not initialized",
                    "service_unavailable",
                    HTTPStatus.SERVICE_UNAVAILABLE,
                ),
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            ),
        )
    return handler, None


def _hub_model_required_error() -> JSONResponse:
    """Return a standardized 400 response for missing hub model selections.

    Returns
    -------
    JSONResponse
        A JSON response with error details.
    """

    return JSONResponse(
        content=create_error_response(
            "Model selection is required when the server runs in hub mode. "
            "Include the 'model' field (or query parameter) to choose a registered model. "
            "Call /v1/models to list available names.",
            "invalid_request_error",
            HTTPStatus.BAD_REQUEST,
        ),
        status_code=HTTPStatus.BAD_REQUEST,
    )


def _resolve_model_name(
    raw_request: Request,
    model_name: str | None,
    *,
    provided_explicitly: bool,
) -> tuple[str | None, JSONResponse | None]:
    """Normalize ``model_name`` and enforce hub requirements when needed.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.
    model_name : str or None
        The model name to resolve.
    provided_explicitly : bool
        Whether the model was explicitly provided in the request.

    Returns
    -------
    tuple[str or None, JSONResponse or None]
        A tuple of (resolved_model_name, error_response) where exactly one is None.
    """

    normalized = (model_name or "").strip() or None
    controller = getattr(raw_request.app.state, "hub_controller", None)
    if controller is None:
        return normalized, None

    if not provided_explicitly or normalized is None:
        return None, _hub_model_required_error()

    running_models = get_running_hub_models(raw_request)
    if running_models is not None and normalized not in running_models:
        return (
            None,
            JSONResponse(
                content=create_error_response(
                    f"Model '{normalized}' is not started. Start the process before sending requests.",
                    "invalid_request_error",
                    HTTPStatus.NOT_FOUND,
                ),
                status_code=HTTPStatus.NOT_FOUND,
            ),
        )
    return normalized, None


def _model_field_was_provided(payload: Any) -> bool:
    """Return True when the ``model`` field was explicitly provided.

    Parameters
    ----------
    payload : Any
        The request payload object.

    Returns
    -------
    bool
        True if the model field was explicitly set.
    """

    fields_set = getattr(payload, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(payload, "__fields_set__", None)
    if not isinstance(fields_set, set):
        return False
    return "model" in fields_set


# =============================================================================
# Critical/Monitoring Endpoints - Defined first to ensure priority matching
# =============================================================================


@router.get("/health", response_model=None)
async def health(raw_request: Request) -> HealthCheckResponse | JSONResponse:
    """Health check endpoint aware of JIT/auto-unload state.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    HealthCheckResponse or JSONResponse
        Health status information.
    """
    handler_manager = getattr(raw_request.app.state, "handler_manager", None)
    configured_model_id = get_configured_model_id(raw_request)
    controller = getattr(raw_request.app.state, "hub_controller", None)

    if handler_manager is not None:
        handler = getattr(handler_manager, "current_handler", None)
        if handler is not None:
            model_id = getattr(handler, "model_path", configured_model_id or "unknown")
            return HealthCheckResponse(
                status=HealthCheckStatus.OK, model_id=model_id, model_status="initialized"
            )
        return HealthCheckResponse(
            status=HealthCheckStatus.OK,
            model_id=configured_model_id,
            model_status="unloaded",
        )

    if controller is not None:
        return HealthCheckResponse(
            status=HealthCheckStatus.OK,
            model_id=configured_model_id,
            model_status="controller",
        )

    handler = getattr(raw_request.app.state, "handler", None)
    if handler is None:
        # Handler not initialized - return 503 with degraded status
        return JSONResponse(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "model_id": None, "model_status": "uninitialized"},
        )

    model_id = getattr(handler, "model_path", configured_model_id or "unknown")
    return HealthCheckResponse(
        status=HealthCheckStatus.OK, model_id=model_id, model_status="initialized"
    )


@router.get("/v1/models", response_model=None)
async def models(raw_request: Request) -> ModelsResponse | JSONResponse:
    """Get list of available models with cached response for instant delivery.

    This endpoint is defined early to ensure it's not blocked by other routes.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    ModelsResponse or JSONResponse
        List of available models or error response.
    """
    # Try registry first (Phase 1+), fall back to handler for backward compat
    registry = getattr(raw_request.app.state, "registry", None)
    if registry is not None:
        try:
            models_data = registry.list_models()
            running_models = get_running_hub_models(raw_request)
            if running_models is not None:
                models_data = [model for model in models_data if model.get("id") in running_models]
            return ModelsResponse(object="list", data=[Model(**model) for model in models_data])
        except Exception as e:
            logger.error(f"Error retrieving models from registry. {type(e).__name__}: {e}")
            return JSONResponse(
                content=create_error_response(
                    f"Failed to retrieve models: {e}",
                    "server_error",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                ),
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    cached_metadata = get_cached_model_metadata(raw_request)
    if cached_metadata is not None:
        return ModelsResponse(object="list", data=[Model(**cached_metadata)])

    # Fallback to handler (Phase 0 compatibility)
    handler, error = await _get_handler_or_error(raw_request, "models")
    if error is not None:
        return error
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )

    try:
        models_data = await handler.get_models()
        return ModelsResponse(object="list", data=[Model(**model) for model in models_data])
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving models. {type(e).__name__}: {e}")
        return JSONResponse(
            content=create_error_response(
                f"Failed to retrieve models: {e}",
                "server_error",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            ),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@router.get("/v1/queue/stats", response_model=None)
async def queue_stats(
    raw_request: Request, model: str | None = Query(None, description="Optional model name")
) -> dict[str, Any] | JSONResponse:
    """Get queue statistics.

    Note: queue_stats shape is handler-dependent (Flux vs LM/VLM/Whisper)
    so callers know keys may vary.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request.
    model : str or None, optional
        Optional model name to filter statistics.

    Returns
    -------
    dict[str, Any] or JSONResponse
        Queue statistics or error response.
    """
    resolved_model, validation_error = _resolve_model_name(
        raw_request,
        model,
        provided_explicitly=model is not None,
    )
    if validation_error is not None:
        return validation_error

    handler, error = await _get_handler_or_error(
        raw_request, "queue_stats", model_name=resolved_model
    )
    if error is not None:
        return error
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )

    try:
        stats = await handler.get_queue_stats()
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        return JSONResponse(
            content=create_error_response(
                "Failed to get queue stats", "server_error", HTTPStatus.INTERNAL_SERVER_ERROR
            ),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
    else:
        return {"status": "ok", "queue_stats": stats}


# =============================================================================
# API Endpoints - Core functionality
# =============================================================================


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest, raw_request: Request
) -> ChatCompletionResponse | StreamingResponse | JSONResponse:
    """Handle chat completion requests.

    Parameters
    ----------
    request : ChatCompletionRequest
        The chat completion request payload.
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    ChatCompletionResponse or StreamingResponse or JSONResponse
        Chat completion response, stream, or error response.
    """
    resolved_model, validation_error = _resolve_model_name(
        raw_request,
        request.model,
        provided_explicitly=_model_field_was_provided(request),
    )
    if validation_error is not None:
        return validation_error
    selected_model = resolved_model or request.model
    if selected_model is not None:
        request.model = selected_model

    handler, error = await _get_handler_or_error(
        raw_request, "chat_completions", model_name=selected_model
    )
    if error is not None:
        return error
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )

    if not isinstance(handler, MLXVLMHandler) and not isinstance(handler, MLXLMHandler):
        return JSONResponse(
            content=create_error_response(
                "Unsupported model type", "unsupported_request", HTTPStatus.BAD_REQUEST
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    request_id = getattr(raw_request.state, "request_id", None)

    try:
        if isinstance(handler, MLXVLMHandler):
            return await process_multimodal_request(handler, request, request_id)
        return await process_text_request(handler, request, request_id)
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - defensive logging
        logger.exception(f"Error processing chat completion request: {type(e).__name__}: {e}")
        return JSONResponse(
            content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )


@router.post("/v1/embeddings", response_model=None)
async def embeddings(
    request: EmbeddingRequest, raw_request: Request
) -> EmbeddingResponse | JSONResponse:
    """Handle embedding requests.

    Parameters
    ----------
    request : EmbeddingRequest
        The embedding request payload.
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    EmbeddingResponse or JSONResponse
        Embedding response or error response.
    """
    resolved_model, validation_error = _resolve_model_name(
        raw_request,
        request.model,
        provided_explicitly=_model_field_was_provided(request),
    )
    if validation_error is not None:
        return validation_error
    selected_model = resolved_model or request.model
    if selected_model is not None:
        request.model = selected_model

    handler, error = await _get_handler_or_error(
        raw_request, "embeddings", model_name=selected_model
    )
    if error is not None:
        return error
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )

    if not isinstance(handler, MLXEmbeddingsHandler):
        return JSONResponse(
            content=create_error_response(
                "Embedding requests require an embeddings model. Use --model-type embeddings.",
                "unsupported_request",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    try:
        embeddings_response = await handler.generate_embeddings_response(request)
        return create_response_embeddings(
            embeddings_response, request.model, request.encoding_format
        )
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - defensive logging
        logger.exception(f"Error processing embedding request: {type(e).__name__}: {e}")
        return JSONResponse(
            content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )


@router.post("/v1/images/generations", response_model=None)
async def image_generations(
    request: ImageGenerationRequest, raw_request: Request
) -> ImageGenerationResponse | JSONResponse:
    """Handle image generation requests.

    Parameters
    ----------
    request : ImageGenerationRequest
        The image generation request payload.
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    ImageGenerationResponse or JSONResponse
        Image generation response or error response.
    """
    resolved_model, validation_error = _resolve_model_name(
        raw_request,
        request.model,
        provided_explicitly=_model_field_was_provided(request),
    )
    if validation_error is not None:
        return validation_error
    selected_model = resolved_model or request.model
    if selected_model is not None:
        request.model = selected_model

    handler, error = await _get_handler_or_error(
        raw_request, "image_generation", model_name=selected_model
    )
    if error is not None:
        return error
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )

    # Check if the handler is an MLXFluxHandler
    if not MFLUX_AVAILABLE or not isinstance(handler, MLXFluxHandler):
        return JSONResponse(
            content=create_error_response(
                "Image generation requests require an image generation model. Use --model-type image-generation.",
                "unsupported_request",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    try:
        image_response: ImageGenerationResponse = await handler.generate_image(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing image generation request: {type(e).__name__}: {e}")
        return JSONResponse(
            content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )
    else:
        return image_response


@router.post("/v1/images/edits", response_model=None)
async def create_image_edit(
    request: Annotated[ImageEditRequest, Form()], raw_request: Request
) -> ImageEditResponse | JSONResponse:
    """Handle image editing requests with dynamic provider routing.

    Parameters
    ----------
    request : ImageEditRequest
        The image edit request payload.
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    ImageEditResponse or JSONResponse
        Image edit response or error response.
    """
    resolved_model, validation_error = _resolve_model_name(
        raw_request,
        request.model,
        provided_explicitly=_model_field_was_provided(request),
    )
    if validation_error is not None:
        return validation_error
    selected_model = resolved_model or request.model
    if selected_model is not None:
        request.model = selected_model

    handler, error = await _get_handler_or_error(
        raw_request, "image_edit", model_name=selected_model
    )
    if error is not None:
        return error
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )

    # Check if the handler is an MLXFluxHandler
    if not MFLUX_AVAILABLE or not isinstance(handler, MLXFluxHandler):
        return JSONResponse(
            content=create_error_response(
                "Image editing requests require an image generation model. Use --model-type image-generation.",
                "unsupported_request",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )
    try:
        image_response: ImageEditResponse = await handler.edit_image(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing image edit request: {type(e).__name__}: {e}")
        return JSONResponse(
            content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )
    else:
        return image_response


@router.post("/v1/audio/transcriptions", response_model=None)
async def create_audio_transcriptions(
    request: Annotated[TranscriptionRequest, Form()], raw_request: Request
) -> StreamingResponse | TranscriptionResponse | JSONResponse | str:
    """Handle audio transcription requests.

    Parameters
    ----------
    request : TranscriptionRequest
        The transcription request payload.
    raw_request : Request
        The incoming FastAPI request.

    Returns
    -------
    StreamingResponse or TranscriptionResponse or JSONResponse or str
        Transcription response, stream, or error response.
    """
    resolved_model, validation_error = _resolve_model_name(
        raw_request,
        request.model,
        provided_explicitly=_model_field_was_provided(request),
    )
    if validation_error is not None:
        return validation_error
    selected_model = resolved_model or request.model
    if selected_model is not None:
        request.model = selected_model

    handler, error = await _get_handler_or_error(
        raw_request, "audio_transcriptions", model_name=selected_model
    )
    if error is not None:
        return error
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )

    if not isinstance(handler, MLXWhisperHandler):
        return JSONResponse(
            content=create_error_response(
                "Audio transcription requests require a whisper model. Use --model-type whisper.",
                "unsupported_request",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    try:
        if request.stream:
            # process the request before sending to the handler
            request_data = await handler.prepare_transcription_request(request)
            return StreamingResponse(
                handler.generate_transcription_stream_from_data(request_data),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        transcription_response: (
            TranscriptionResponse | str
        ) = await handler.generate_transcription_response(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing transcription request: {type(e).__name__}: {e}")
        return JSONResponse(
            content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )
    else:
        return transcription_response


def create_response_embeddings(
    embeddings: list[list[float]], model: str, encoding_format: Literal["float", "base64"] = "float"
) -> EmbeddingResponse:
    """Create embedding response data from embeddings list.

    Parameters
    ----------
    embeddings : list[list[float]]
        List of embedding vectors.
    model : str
        Model name used for embeddings.
    encoding_format : Literal["float", "base64"], optional
        Encoding format for embeddings, by default "float".

    Returns
    -------
    EmbeddingResponse
        Formatted embedding response.
    """
    embeddings_response = []
    for index, embedding in enumerate(embeddings):
        if encoding_format == "base64":
            # Convert list/array to bytes before base64 encoding
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
            embeddings_response.append(
                EmbeddingResponseData(
                    embedding=base64.b64encode(embedding_bytes).decode("utf-8"), index=index
                )
            )
        else:
            embeddings_response.append(EmbeddingResponseData(embedding=embedding, index=index))
    return EmbeddingResponse(object="list", data=embeddings_response, model=model, usage=None)


def create_response_chunk(
    chunk: str | dict[str, Any],
    model: str,
    *,
    is_final: bool = False,
    finish_reason: str | None = "stop",
    chat_id: str | None = None,
    created_time: int | None = None,
    request_id: str | None = None,
    tool_call_id: str | None = None,
) -> ChatCompletionChunk:
    """Create a formatted response chunk for streaming.

    Parameters
    ----------
    chunk : str or dict[str, Any]
        The chunk content to format.
    model : str
        Model name for the response.
    is_final : bool, optional
        Whether this is the final chunk, by default False.
    finish_reason : str or None, optional
        Finish reason for the completion, by default "stop".
    chat_id : str or None, optional
        Chat completion ID, by default None.
    created_time : int or None, optional
        Creation timestamp, by default None.
    request_id : str or None, optional
        Request ID, by default None.
    tool_call_id : str or None, optional
        Tool call ID, by default None.

    Returns
    -------
    ChatCompletionChunk
        Formatted chat completion chunk.
    """
    chat_id = chat_id or get_id()
    created_time = created_time or int(time.time())

    # Handle string chunks (text content)
    if isinstance(chunk, str):
        return ChatCompletionChunk(
            id=chat_id,
            object="chat.completion.chunk",
            created=created_time,
            model=model,
            choices=[
                StreamingChoice(
                    index=0,
                    delta=Delta(content=chunk, role="assistant"),  # type: ignore[call-arg]
                    finish_reason=finish_reason if is_final else None,  # type: ignore[arg-type]
                )
            ],
            request_id=request_id,
        )

    # Handle reasoning content chunks
    if "reasoning_content" in chunk:
        return ChatCompletionChunk(
            id=chat_id,
            object="chat.completion.chunk",
            created=created_time,
            model=model,
            choices=[
                StreamingChoice(
                    index=0,
                    delta=Delta(
                        reasoning_content=chunk["reasoning_content"],
                        role="assistant",
                        content=chunk.get("content", None),
                    ),  # type: ignore[call-arg]
                    finish_reason=finish_reason if is_final else None,  # type: ignore[arg-type]
                )
            ],
            request_id=request_id,
        )

    # Handle dict chunks with only content (no reasoning or tool calls)
    if "content" in chunk and isinstance(chunk["content"], str):
        return ChatCompletionChunk(
            id=chat_id,
            object="chat.completion.chunk",
            created=created_time,
            model=model,
            choices=[
                StreamingChoice(
                    index=0,
                    delta=Delta(content=chunk["content"], role="assistant"),  # type: ignore[call-arg]
                    finish_reason=finish_reason if is_final else None,  # type: ignore[arg-type]
                )
            ],
            request_id=request_id,
        )

    # Handle tool/function call chunks
    function_call = None
    if "name" in chunk:
        function_call = ChoiceDeltaFunctionCall(name=chunk["name"], arguments=None)
        if "arguments" in chunk:
            function_call.arguments = chunk["arguments"]
    elif "arguments" in chunk:
        # Handle case where arguments come before name (streaming)
        function_call = ChoiceDeltaFunctionCall(name=None, arguments=chunk["arguments"])

    if function_call:
        # Validate index exists before accessing
        tool_index = chunk.get("index", 0)
        tool_identifier = tool_call_id or get_tool_call_id()
        tool_chunk = ChoiceDeltaToolCall(
            index=tool_index,
            type="function",
            id=tool_identifier,
            function=function_call,
        )

        delta = Delta(content=None, role="assistant", tool_calls=[tool_chunk])  # type: ignore[call-arg]
    else:
        # Fallback: create empty delta if no recognized chunk type
        delta = Delta(role="assistant")  # type: ignore[call-arg]

    return ChatCompletionChunk(
        id=chat_id,
        object="chat.completion.chunk",
        created=created_time,
        model=model,
        choices=[
            StreamingChoice(index=0, delta=delta, finish_reason=finish_reason if is_final else None)  # type: ignore[arg-type]
        ],
        request_id=request_id,
    )


def _yield_sse_chunk(data: dict[str, Any] | ChatCompletionChunk) -> str:
    """Format and yield SSE chunk data.

    Parameters
    ----------
    data : dict[str, Any] or ChatCompletionChunk
        The data to format as SSE chunk.

    Returns
    -------
    str
        Formatted SSE chunk string.
    """
    if isinstance(data, ChatCompletionChunk):
        return f"data: {json.dumps(data.model_dump())}\n\n"
    return f"data: {json.dumps(data)}\n\n"


async def handle_stream_response(
    generator: AsyncGenerator[Any, None], model: str, request_id: str | None = None
) -> AsyncGenerator[str, None]:
    """Handle streaming response generation (OpenAI-compatible).

    Parameters
    ----------
    generator : AsyncGenerator[Any, None]
        The async generator yielding response chunks.
    model : str
        Model name for the response.
    request_id : str or None, optional
        Request ID, by default None.

    Yields
    ------
    str
        SSE-formatted response chunks.
    """
    chat_index = get_id()
    created_time = int(time.time())
    finish_reason = "stop"
    next_tool_call_index = 0
    current_implicit_tool_index: int | None = None
    tool_call_ids: dict[int, str] = {}
    usage_info = None

    try:
        # First chunk: role-only delta, as per OpenAI
        first_chunk = ChatCompletionChunk(
            id=chat_index,
            object="chat.completion.chunk",
            created=created_time,
            model=model,
            choices=[StreamingChoice(index=0, delta=Delta(role="assistant"))],  # type: ignore[call-arg]
            request_id=request_id,
        )
        yield _yield_sse_chunk(first_chunk)

        async for chunk in generator:
            if not chunk:
                continue

            if isinstance(chunk, str):
                response_chunk = create_response_chunk(
                    chunk,
                    model,
                    chat_id=chat_index,
                    created_time=created_time,
                    request_id=request_id,
                )
                yield _yield_sse_chunk(response_chunk)

            elif isinstance(chunk, dict):
                # Check if this is usage info from the handler
                if "__usage__" in chunk:
                    usage_info = chunk["__usage__"]
                    continue

                # Handle tool call chunks
                payload = dict(chunk)  # Create a copy to avoid mutating the original
                current_tool_id = None

                has_name = bool(payload.get("name"))
                has_arguments = "arguments" in payload
                payload_index = payload.get("index")

                if has_name:
                    finish_reason = "tool_calls"
                    if payload_index is None:
                        if current_implicit_tool_index is not None:
                            payload_index = current_implicit_tool_index
                        else:
                            payload_index = next_tool_call_index
                            next_tool_call_index += 1
                        payload["index"] = payload_index
                    current_implicit_tool_index = payload_index
                    # Keep the implicit index available for additional argument chunks
                elif has_arguments:
                    if payload_index is None:
                        if current_implicit_tool_index is not None:
                            payload_index = current_implicit_tool_index
                        else:
                            payload_index = next_tool_call_index
                            next_tool_call_index += 1
                        payload["index"] = payload_index
                    current_implicit_tool_index = payload_index
                elif payload_index is not None:
                    current_implicit_tool_index = payload_index

                payload_index = payload.get("index")
                if payload_index is not None:
                    if payload_index not in tool_call_ids:
                        tool_call_ids[payload_index] = get_tool_call_id()
                    current_tool_id = tool_call_ids[payload_index]

                response_chunk = create_response_chunk(
                    payload,
                    model,
                    chat_id=chat_index,
                    created_time=created_time,
                    request_id=request_id,
                    tool_call_id=current_tool_id,
                )
                yield _yield_sse_chunk(response_chunk)

            else:
                error_response = create_error_response(
                    f"Invalid chunk type: {type(chunk)}",
                    "server_error",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                yield _yield_sse_chunk(error_response)

    except HTTPException as e:
        logger.exception(f"HTTPException in stream wrapper: {type(e).__name__}: {e}")
        detail = e.detail if isinstance(e.detail, dict) else {"message": str(e)}
        error_response = detail  # type: ignore[assignment]
        yield _yield_sse_chunk(error_response)
    except Exception as e:
        logger.exception(f"Error in stream wrapper: {type(e).__name__}: {e}")
        error_response = create_error_response(
            str(e), "server_error", HTTPStatus.INTERNAL_SERVER_ERROR
        )
        yield _yield_sse_chunk(error_response)
    finally:
        # Final chunk: finish_reason with usage info, as per OpenAI
        final_chunk = ChatCompletionChunk(
            id=chat_index,
            object="chat.completion.chunk",
            created=created_time,
            model=model,
            choices=[StreamingChoice(index=0, delta=Delta(), finish_reason=finish_reason)],  # type: ignore[call-arg,arg-type]
            usage=usage_info,
            request_id=request_id,
        )
        yield _yield_sse_chunk(final_chunk)
        yield "data: [DONE]\n\n"


async def process_multimodal_request(
    handler: MLXVLMHandler, request: ChatCompletionRequest, request_id: str | None = None
) -> ChatCompletionResponse | StreamingResponse | JSONResponse:
    """Process multimodal-specific requests.

    Parameters
    ----------
    handler : MLXVLMHandler
        The multimodal handler instance.
    request : ChatCompletionRequest
        The chat completion request.
    request_id : str or None, optional
        Request ID for logging, by default None.

    Returns
    -------
    ChatCompletionResponse or StreamingResponse or JSONResponse
        Processed response.
    """
    if request_id:
        logger.info(f"Processing multimodal request [request_id={request_id}]")

    if request.stream:
        return StreamingResponse(
            handle_stream_response(
                handler.generate_multimodal_stream(request), request.model, request_id
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    # Extract response and usage from handler
    result = await handler.generate_multimodal_response(request)
    if isinstance(result, dict) and "response" in result and "usage" in result:
        response_data = result.get("response")
        usage = result.get("usage")
        return format_final_response(response_data, request.model, request_id, usage)  # type: ignore[arg-type]

    # Fallback for backward compatibility or if structure is different
    return format_final_response(result, request.model, request_id)


async def process_text_request(
    handler: MLXLMHandler | MLXVLMHandler,
    request: ChatCompletionRequest,
    request_id: str | None = None,
) -> ChatCompletionResponse | StreamingResponse | JSONResponse:
    """Process text-only requests.

    Parameters
    ----------
    handler : MLXLMHandler or MLXVLMHandler
        The text handler instance.
    request : ChatCompletionRequest
        The chat completion request.
    request_id : str or None, optional
        Request ID for logging, by default None.

    Returns
    -------
    ChatCompletionResponse or StreamingResponse or JSONResponse
        Processed response.
    """
    if request_id:
        logger.info(f"Processing text request [request_id={request_id}]")

    if request.stream:
        return StreamingResponse(
            handle_stream_response(
                handler.generate_text_stream(request),  # type: ignore[union-attr]
                request.model,
                request_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Extract response and usage from handler
    result = await handler.generate_text_response(request)  # type: ignore[union-attr]
    response_data = result.get("response")
    usage = result.get("usage")
    return format_final_response(response_data, request.model, request_id, usage)  # type: ignore[arg-type]


def get_id() -> str:
    """Generate a unique ID for chat completions with timestamp and random component.

    Returns
    -------
    str
        Unique chat completion ID.
    """
    timestamp = int(time.time())
    random_suffix = random.randint(0, 999999)
    return f"chatcmpl_{timestamp}{random_suffix:06d}"


def get_tool_call_id() -> str:
    """Generate a unique ID for tool calls with timestamp and random component.

    Returns
    -------
    str
        Unique tool call ID.
    """
    timestamp = int(time.time())
    random_suffix = random.randint(0, 999999)
    return f"call_{timestamp}{random_suffix:06d}"


def format_final_response(
    response: str | dict[str, Any],
    model: str,
    request_id: str | None = None,
    usage: UsageInfo | None = None,
) -> ChatCompletionResponse:
    """Format the final non-streaming response.

    Parameters
    ----------
    response : str or dict[str, Any]
        The response content.
    model : str
        Model name.
    request_id : str or None, optional
        Request ID, by default None.
    usage : UsageInfo or None, optional
        Usage information, by default None.

    Returns
    -------
    ChatCompletionResponse
        Formatted chat completion response.
    """
    if isinstance(response, str):
        return ChatCompletionResponse(
            id=get_id(),
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=response,
                        refusal=None,
                        reasoning_content=None,
                        tool_calls=None,
                        tool_call_id=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=usage,
            request_id=request_id,
        )

    reasoning_content = response.get("reasoning_content", None)
    response_content = response.get("content", None)
    tool_calls = response.get("tool_calls", None)
    tool_call_responses = []
    if tool_calls is None or len(tool_calls) == 0:
        return ChatCompletionResponse(
            id=get_id(),
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=response_content,
                        reasoning_content=reasoning_content,
                        refusal=None,
                        tool_calls=None,
                        tool_call_id=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=usage,
            request_id=request_id,
        )
    for idx, tool_call in enumerate(tool_calls):
        arguments = tool_call.get("arguments")
        # If arguments is already a string, use it directly; otherwise serialize it
        if isinstance(arguments, str):
            arguments_str = arguments
        else:
            arguments_str = json.dumps(arguments)
        function_call = FunctionCall(name=tool_call.get("name"), arguments=arguments_str)
        tool_call_response = ChatCompletionMessageToolCall(
            id=get_tool_call_id(), type="function", function=function_call, index=idx
        )
        tool_call_responses.append(tool_call_response)

    message = Message(
        role="assistant",
        content=response_content,
        reasoning_content=reasoning_content,
        tool_calls=tool_call_responses,
        refusal=None,
        tool_call_id=None,
    )

    return ChatCompletionResponse(
        id=get_id(),
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[Choice(index=0, message=message, finish_reason="tool_calls")],
        usage=usage,
        request_id=request_id,
    )
