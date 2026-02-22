"""API endpoints for the MLX OpenAI server."""

from __future__ import annotations

import os
import base64
from collections.abc import AsyncGenerator
from http import HTTPStatus
import json
import random
import time
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
import numpy as np

from ..handler.mlx_lm import MLXLMHandler
from ..handler.mlx_vlm import MLXVLMHandler
from ..schemas.openai import (
    ChatCompletionContentPartImage,
    ChatCompletionContentPartText,
    ChatCompletionChunk,
    ChatCompletionMessageToolCall,
    ChatCompletionRequest,
    Config,
    ChatCompletionResponse,
    Choice,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
    Delta,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResponseData,
    FunctionCall,
    ImageURL,
    HealthCheckResponse,
    HealthCheckStatus,
    ImageEditRequest,
    ImageEditResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    Message,
    Model,
    ModelsResponse,
    ResponsesRequest,
    ResponsesResponse,
    ResponseUsage,
    InputTokensDetails,
    OutputTokensDetails,
    StreamingChoice,
    TranscriptionRequest,
    TranscriptionResponse,
    UsageInfo,
    random_uuid,
)
from openai.types.responses import FunctionTool
from openai.types.responses.response_output_message import (
    ResponseOutputText,
    ResponseOutputMessage
)
from openai.types.responses.response_function_tool_call import (
    ResponseFunctionToolCall
)
from openai.types.responses.response_reasoning_item import (
    Summary,
    Content,
    ResponseReasoningItem
)
from ..utils.errors import create_error_response

router = APIRouter()


def _get_handler_type(handler: Any) -> str:
    """Return the handler type string for a handler or proxy.

    Uses the ``handler_type`` class/instance attribute present on all
    concrete handler classes and on ``HandlerProcessProxy``.

    Parameters
    ----------
    handler : Any
        A handler instance or proxy.

    Returns
    -------
    str
        Handler type string (``"lm"``, ``"multimodal"``, ``"embeddings"``,
        ``"image"``, ``"whisper"``), or ``""`` if not determinable.
    """
    return getattr(handler, "handler_type", "")


def _resolve_handler(
    raw_request: Request,
    model_id: str | None = None,
) -> Any | None:
    """Resolve the correct handler for a request.

    In multi-handler mode (``app.state.registry`` is set) the handler
    is looked up by ``model_id``.  In single-handler mode the global
    ``app.state.handler`` is returned.

    Parameters
    ----------
    raw_request : Request
        The incoming FastAPI request (used to access ``app.state``).
    model_id : str | None, optional
        The model identifier from the request body.  Only used when a
        ``ModelRegistry`` is attached to the app state.

    Returns
    -------
    Any | None
        A handler instance, or ``None`` if no handler could be resolved.

    Raises
    ------
    HTTPException
        404 when a ``model_id`` is provided but not found in the
        registry.
    """
    registry = getattr(raw_request.app.state, "registry", None)
    if registry is not None and model_id is not None:
        try:
            return registry.get_handler(model_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail={
                    "error": {
                        "message": str(exc),
                        "type": "model_not_found",
                        "code": HTTPStatus.NOT_FOUND,
                    }
                },
            ) from exc

    # Fallback: single-handler mode
    return getattr(raw_request.app.state, "handler", None)


# =============================================================================
# Critical/Monitoring Endpoints - Defined first to ensure priority matching
# =============================================================================


@router.get("/health", response_model=None)
async def health(raw_request: Request) -> HealthCheckResponse | JSONResponse:
    """Health check endpoint - verifies handler initialization status.

    Returns 503 if handler is not initialized, 200 otherwise.
    In multi-handler mode reports the number of loaded models.
    """
    registry = getattr(raw_request.app.state, "registry", None)
    if registry is not None:
        model_count = registry.get_model_count()
        if model_count == 0:
            return JSONResponse(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                content={"status": "unhealthy", "model_id": None, "model_status": "no_models"},
            )
        model_ids = [m["id"] for m in registry.list_models()]
        return HealthCheckResponse(
            status=HealthCheckStatus.OK,
            model_id=", ".join(model_ids),
            model_status=f"initialized ({model_count} model(s))",
        )

    handler = getattr(raw_request.app.state, "handler", None)

    if handler is None:
        # Handler not initialized - return 503 with degraded status
        return JSONResponse(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "model_id": None, "model_status": "uninitialized"},
        )

    # Handler initialized - extract model_id
    model_id = getattr(handler, "model_path", "unknown")

    return HealthCheckResponse(
        status=HealthCheckStatus.OK, model_id=model_id, model_status="initialized"
    )


@router.get("/v1/models", response_model=None)
async def models(raw_request: Request) -> ModelsResponse | JSONResponse:
    """
    Get list of available models with cached response for instant delivery.

    This endpoint is defined early to ensure it's not blocked by other routes.
    """
    # Try registry first (Phase 1+), fall back to handler for backward compat
    registry = getattr(raw_request.app.state, "registry", None)
    if registry is not None:
        try:
            models_data = registry.list_models()
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

    # Fallback to handler (Phase 0 compatibility)
    handler = getattr(raw_request.app.state, "handler", None)
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
async def queue_stats(raw_request: Request) -> dict[str, Any] | JSONResponse:
    """
    Get queue statistics.

    Note: queue_stats shape is handler-dependent (Flux vs LM/VLM/Whisper)
    so callers know keys may vary.
    """
    handler = getattr(raw_request.app.state, "handler", None)
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


def _parse_env_float(key: str, default: float | None = None) -> float | None:
    """Parse a float from an environment variable, or return default."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _parse_env_int(key: str, default: int | None = None) -> int | None:
    """Parse an int from an environment variable, or return default."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def refine_chat_completion_request(
    request: ChatCompletionRequest,
) -> ChatCompletionRequest:
    """Refine chat completion request with defaults from env for sampling params and model."""
    if request.temperature is None:
        request.temperature = _parse_env_float("DEFAULT_TEMPERATURE")
    if request.top_p is None:
        request.top_p = _parse_env_float("DEFAULT_TOP_P")
    if request.top_k is None:
        request.top_k = _parse_env_int("DEFAULT_TOP_K")
    if request.seed is None:
        request.seed = _parse_env_int("DEFAULT_SEED")
    if request.repetition_penalty is None:
        request.repetition_penalty = _parse_env_float("DEFAULT_REPETITION_PENALTY")
    if request.max_completion_tokens is None and request.max_tokens is None:
        request.max_completion_tokens = _parse_env_int("DEFAULT_MAX_TOKENS")
    if request.xtc_probability is None:
        request.xtc_probability = _parse_env_float("DEFAULT_XTC_PROBABILITY")
    if request.xtc_threshold is None:
        request.xtc_threshold = _parse_env_float("DEFAULT_XTC_THRESHOLD")
    if request.presence_penalty is None:
        request.presence_penalty = _parse_env_float("DEFAULT_PRESENCE_PENALTY")
    if request.repetition_context_size is None:
        request.repetition_context_size = _parse_env_int("DEFAULT_REPETITION_CONTEXT_SIZE")
    if not request.model:
        request.model = Config.TEXT_MODEL
    return request


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest, raw_request: Request
) -> ChatCompletionResponse | StreamingResponse | JSONResponse:
    """Handle chat completion requests."""
    request = refine_chat_completion_request(request)
    handler = _resolve_handler(raw_request, model_id=request.model)
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )

    handler_type = _get_handler_type(handler)
    if handler_type not in ("lm", "multimodal"):
        return JSONResponse(
            content=create_error_response(
                "Unsupported model type for chat completions. "
                f"Handler for '{request.model}' is {type(handler).__name__} "
                f"(handler_type={handler_type!r}).",
                "unsupported_request",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    # Get request ID from middleware
    request_id = getattr(raw_request.state, "request_id", None)

    try:
        if handler_type == "multimodal":
            return await process_multimodal_request(handler, request, request_id)
        return await process_text_request(handler, request, request_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing chat completion request: {type(e).__name__}: {e}")
        return JSONResponse(
            content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )

@router.post("/v1/embeddings", response_model=None)
async def embeddings(
    request: EmbeddingRequest, raw_request: Request
) -> EmbeddingResponse | JSONResponse:
    """Handle embedding requests."""
    handler = _resolve_handler(raw_request, model_id=request.model)
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )

    if _get_handler_type(handler) != "embeddings":
        return JSONResponse(
            content=create_error_response(
                "Unsupported model type for embeddings. "
                f"Handler for '{request.model}' is {type(handler).__name__}.",
                "unsupported_request",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    try:
        embeddings = await handler.generate_embeddings_response(request)
        return create_response_embeddings(embeddings, request.model, request.encoding_format)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing embedding request: {type(e).__name__}: {e}")
        return JSONResponse(
            content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )


@router.post("/v1/images/generations", response_model=None)
async def image_generations(
    request: ImageGenerationRequest, raw_request: Request
) -> ImageGenerationResponse | JSONResponse:
    """Handle image generation requests."""
    handler = _resolve_handler(raw_request, model_id=request.model)
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )

    # Check if the handler supports image generation
    if _get_handler_type(handler) != "image":
        return JSONResponse(
            content=create_error_response(
                "Image generation requests require an image generation model. "
                f"Handler for '{request.model}' is {type(handler).__name__}.",
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
    """Handle image editing requests with dynamic provider routing."""
    handler = _resolve_handler(raw_request, model_id=request.model)
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )

    # Check if the handler supports image editing
    if _get_handler_type(handler) != "image":
        return JSONResponse(
            content=create_error_response(
                "Image editing requests require an image editing model. "
                f"Handler for '{request.model}' is {type(handler).__name__}.",
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
    """Handle audio transcription requests."""
    try:
        handler = _resolve_handler(raw_request, model_id=request.model)
        if handler is None:
            return JSONResponse(
                content=create_error_response(
                    "Model handler not initialized",
                    "service_unavailable",
                    HTTPStatus.SERVICE_UNAVAILABLE,
                ),
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            )

        if request.stream:
            # procoess the request before sending to the handler
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
) -> ChatCompletionChunk:
    """Create a formatted response chunk for streaming."""
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
        tool_chunk = ChoiceDeltaToolCall(
            index=tool_index, type="function", id=get_tool_call_id(), function=function_call
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
    """Format and yield SSE chunk data."""
    # Assuming ResponseChunk was added recently, we support it too
    if hasattr(data, "model_dump"):
        return f"data: {json.dumps(data.model_dump(exclude_none=True))}\n\n"
    return f"data: {json.dumps(data)}\n\n"


async def handle_stream_response(
    generator: AsyncGenerator[Any, None], model: str, request_id: str | None = None
) -> AsyncGenerator[str, None]:
    """Handle streaming response generation (OpenAI-compatible)."""
    chat_index = get_id()
    created_time = int(time.time())
    finish_reason = "stop"
    tool_call_index = -1
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
                if payload.get("name"):
                    finish_reason = "tool_calls"
                    tool_call_index += 1
                    payload["index"] = tool_call_index
                elif payload.get("arguments") and "index" not in payload:
                    payload["index"] = tool_call_index

                response_chunk = create_response_chunk(
                    payload,
                    model,
                    chat_id=chat_index,
                    created_time=created_time,
                    request_id=request_id,
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
    """Process multimodal-specific requests."""
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
    result = await handler.generate_multimodal_response(request)
    response_data = result.get("response")
    usage = result.get("usage")
    final_response = format_final_response(response_data, request.model, request_id, usage)
    return JSONResponse(content=final_response.model_dump(exclude_none=True))
    


async def process_text_request(
    handler: MLXLMHandler | MLXVLMHandler,
    request: ChatCompletionRequest,
    request_id: str | None = None,
) -> ChatCompletionResponse | StreamingResponse | JSONResponse:
    """Process text-only requests."""
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
    final_response = format_final_response(response_data, request.model, request_id, usage)
    return JSONResponse(content=final_response.model_dump(exclude_none=True))


def get_id() -> str:
    """Generate a unique ID for chat completions with timestamp and random component."""
    return f"chatcmpl_{random_uuid()}"


def get_tool_call_id() -> str:
    """Generate a unique ID for tool calls with timestamp and random component."""
    timestamp = int(time.time())
    random_suffix = random.randint(0, 999999)
    return f"call_{timestamp}{random_suffix:06d}"


def format_final_response(
    response: str | dict[str, Any],
    model: str,
    request_id: str | None = None,
    usage: UsageInfo | None = None,
) -> ChatCompletionResponse:
    """Format the final non-streaming response."""
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

# =============================================================================
# Responses API Handlers
# =============================================================================

def _normalize_responses_item(item: Any) -> dict[str, Any]:
    """Normalize TypedDict/BaseModel response item to a plain dictionary."""
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        dumped = item.model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    return {}


def _serialize_responses_tool_output(output: Any) -> str:
    """Serialize function_call_output payloads into tool message text."""
    if isinstance(output, str):
        return output

    if isinstance(output, list):
        text_parts: list[str] = []
        for output_item in output:
            normalized = _normalize_responses_item(output_item)
            if normalized.get("type") in {"input_text", "output_text", "text"} and normalized.get("text"):
                text_parts.append(str(normalized["text"]))
        if text_parts:
            return "\n".join(text_parts)

    return json.dumps(output)


def _convert_responses_content(
    role: str, content: Any
) -> str | list[ChatCompletionContentPartText | ChatCompletionContentPartImage]:
    """Convert Responses message content into chat completion content."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    if role != "user":
        text_parts: list[str] = []
        for part in content:
            normalized = _normalize_responses_item(part)
            part_type = normalized.get("type")
            if part_type in {"input_text", "output_text", "text", "reasoning_text", "summary_text"}:
                text = normalized.get("text")
                if text:
                    text_parts.append(str(text))
        return "\n".join(text_parts)

    converted: list[ChatCompletionContentPartText | ChatCompletionContentPartImage] = []
    for part in content:
        normalized = _normalize_responses_item(part)
        part_type = normalized.get("type")
        if part_type in {"input_text", "output_text", "text"}:
            text = normalized.get("text")
            if text:
                converted.append(ChatCompletionContentPartText(type="text", text=str(text)))
        elif part_type in {"input_image", "image_url"} and normalized.get("image_url"):
            converted.append(
                ChatCompletionContentPartImage(
                    type="image_url",
                    image_url=ImageURL(url=str(normalized["image_url"])),
                )
            )
    return converted if converted else ""


def _convert_responses_tools(tools: list[Any] | None) -> list[dict[str, Any]] | None:
    """Convert Responses function tools into chat-completions tool schema."""
    if not tools:
        return None

    converted_tools: list[dict[str, Any]] = []
    for tool in tools:
        normalized = _normalize_responses_item(tool)
        if normalized.get("type") != "function":
            continue

        function_name = normalized.get("name")
        if not function_name and isinstance(tool, FunctionTool):
            function_name = tool.name
        if not function_name:
            continue

        converted_tools.append(
            {
                "type": "function",
                "function": {
                    "name": function_name,
                    "description": normalized.get("description"),
                    "parameters": normalized.get("parameters"),
                },
            }
        )
    return converted_tools or None


def _convert_responses_tool_choice(tool_choice: Any) -> Any:
    """Convert Responses tool_choice to chat-completions tool_choice shape."""
    if tool_choice in {None, "none", "auto", "required"}:
        return tool_choice

    normalized = _normalize_responses_item(tool_choice)
    if normalized.get("type") == "function" and normalized.get("name"):
        return {
            "type": "function",
            "function": {"name": normalized["name"]},
        }
    return "auto"


def convert_responses_request_to_chat_request(
    request: ResponsesRequest
) -> ChatCompletionRequest:
    """Convert a Responses request into a ChatCompletionRequest with full turn history."""
    chat_messages: list[Message] = []
    pending_tool_calls: list[ChatCompletionMessageToolCall] = []
    pending_user_parts: list[ChatCompletionContentPartText | ChatCompletionContentPartImage] = []

    def flush_pending_user_parts() -> None:
        if pending_user_parts:
            chat_messages.append(Message(role="user", content=list(pending_user_parts)))
            pending_user_parts.clear()

    def flush_pending_tool_calls() -> None:
        flush_pending_user_parts()
        if pending_tool_calls:
            chat_messages.append(
                Message(role="assistant", content="", tool_calls=list(pending_tool_calls))
            )
            pending_tool_calls.clear()

    if request.instructions:
        chat_messages.append(Message(role="system", content=request.instructions))

    input_items = request.input
    if isinstance(input_items, str):
        chat_messages.append(Message(role="user", content=input_items))
    elif isinstance(input_items, list):
        for raw_item in input_items:
            item = _normalize_responses_item(raw_item)
            if not item:
                continue

            item_type = item.get("type")
            if item_type == "function_call":
                flush_pending_user_parts()
                pending_tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=item.get("call_id") or get_tool_call_id(),
                        type="function",
                        function=FunctionCall(
                            name=item.get("name", ""),
                            arguments=item.get("arguments", "{}"),
                        ),
                    )
                )
                continue

            flush_pending_tool_calls()

            if item_type == "function_call_output":
                flush_pending_user_parts()
                chat_messages.append(
                    Message(
                        role="tool",
                        tool_call_id=item.get("call_id"),
                        content=_serialize_responses_tool_output(item.get("output", "")),
                    )
                )
                continue

            if item_type == "reasoning":
                flush_pending_user_parts()
                reasoning_parts = item.get("content") or item.get("summary") or []
                reasoning_text = _convert_responses_content("assistant", reasoning_parts)
                if reasoning_text:
                    chat_messages.append(
                        Message(
                            role="assistant",
                            content="",
                            reasoning_content=str(reasoning_text),
                        )
                    )
                continue

            role = item.get("role")
            if role:
                flush_pending_user_parts()
                mapped_role = "system" if role == "developer" else role
                if mapped_role not in {"system", "user", "assistant", "tool"}:
                    mapped_role = "user"
                chat_messages.append(
                    Message(
                        role=mapped_role,
                        content=_convert_responses_content(mapped_role, item.get("content", "")),
                    )
                )
                continue

            if item_type in {"input_text", "text"}:
                text = item.get("text")
                if text:
                    pending_user_parts.append(ChatCompletionContentPartText(type="text", text=str(text)))
            elif item_type in {"input_image", "image_url"} and item.get("image_url"):
                pending_user_parts.append(
                    ChatCompletionContentPartImage(
                        type="image_url",
                        image_url=ImageURL(url=str(item["image_url"])),
                    )
                )

        flush_pending_tool_calls()
        flush_pending_user_parts()
    else:
        raise ValueError(f"Unsupported Responses input format: {type(input_items)}")

    chat_request_payload: dict[str, Any] = {
        "model": request.model,
        "messages": chat_messages,
        "tools": _convert_responses_tools(request.tools),
        "tool_choice": _convert_responses_tool_choice(request.tool_choice),
        "top_p": request.top_p,
        "top_k": request.top_k,
        "min_p": request.min_p,
        "temperature": request.temperature,
        "max_completion_tokens": request.max_output_tokens,
        "repetition_penalty": request.repetition_penalty,
        "seed": request.seed,
    }

    if request.text and request.text.format:
        # Responses API `text.format` is flat: {"type":"json_schema","name":"...","schema":{...}}
        # Chat Completions `response_format` nests those fields under "json_schema".
        fmt = request.text.format
        fmt_type = fmt.get("type")
        if fmt_type == "json_schema":
            chat_request_payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": fmt.get("name", ""),
                    "schema": fmt.get("schema", {}),
                },
            }
        elif fmt_type in {"json_object", "text"}:
            chat_request_payload["response_format"] = {"type": fmt_type}

    if request.reasoning:
        # Best-effort translation from Responses reasoning to template kwargs.
        reasoning_data = _normalize_responses_item(request.reasoning)
        reasoning_effort = str(reasoning_data.get("effort", "")).lower()
        if reasoning_effort in {"none", "minimal"}:
            chat_request_payload["chat_template_kwargs"] = {
                "enable_thinking": False,
            }
        elif reasoning_effort in {"low", "medium", "high"}:
            chat_request_payload["chat_template_kwargs"] = {
                "enable_thinking": True,
                "reasoning_effort": reasoning_effort,
            }
        elif reasoning_effort == "xhigh":
            chat_request_payload["chat_template_kwargs"] = {
                "enable_thinking": True,
                "reasoning_effort": "high",
            }

    return ChatCompletionRequest(**chat_request_payload)

def format_final_responses_response(
    response: str | dict[str, Any],
    request: ResponsesRequest,
    usage: UsageInfo | None = None
) -> ResponsesResponse:
    """Format the final non-streaming response."""
    response_payload: dict[str, Any]
    if isinstance(response, str):
        response_payload = {"reasoning_content": None, "tool_calls": None, "content": response}
    else:
        response_payload = response

    unique_id = random_uuid()

    reasoning_content = response_payload.get("reasoning_content")
    tool_calls = response_payload.get("tool_calls")
    content = response_payload.get("content")
    output_items = []

    if reasoning_content:
        reasoning_item = ResponseReasoningItem(
            id=f"rs_{unique_id}",
            summary=[Summary(text=reasoning_content, type="summary_text")],
            type="reasoning",
            content=[Content(text=reasoning_content, type="reasoning_text")],
            status="completed",
        )
        output_items.append(reasoning_item)

    if tool_calls:
        for idx, tool_call in enumerate(tool_calls):
            tool_call_name = tool_call.get("name", "")
            tool_call_arguments = tool_call.get("arguments", "{}")
            if not isinstance(tool_call_arguments, str):
                tool_call_arguments = json.dumps(tool_call_arguments)
            tool_call_item = ResponseFunctionToolCall(
                id=f"fc_{unique_id}_{idx}",
                name=tool_call_name,
                arguments=tool_call_arguments,
                call_id=tool_call.get("id", f"call_{unique_id}_{idx}"),
                type="function_call",
                status="completed",
            )
            output_items.append(tool_call_item)

    if content:
        content_item = ResponseOutputText(
            annotations=[],
            logprobs=[],
            text=content,
            type="output_text",
        )
        msg_item = ResponseOutputMessage(
            id=f"msg_{unique_id}",
            content=[content_item],
            type="message",
            role="assistant",
            status="completed",
        )
        output_items.append(msg_item)

    response_usage = None
    if usage is not None:
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens or 0
        response_usage = ResponseUsage(
            input_tokens=input_tokens,
            input_tokens_details=InputTokensDetails(
                cached_tokens=usage.prompt_tokens_details.cached_tokens
                if usage.prompt_tokens_details
                and usage.prompt_tokens_details.cached_tokens is not None
                else 0
            ),
            output_tokens=output_tokens,
            output_tokens_details=OutputTokensDetails(),
            total_tokens=usage.total_tokens,
        )

    responses_response = ResponsesResponse(
        id=f"resp_{unique_id}",
        created_at=int(time.time()),
        status="completed",
        incomplete_details=None,
        instructions=request.instructions,
        model=request.model,       
        object="response",
        top_p=request.top_p,
        temperature=request.temperature,
        output=output_items,
        usage=response_usage,
        tool_choice=request.tool_choice,
        tools=request.tools,
        text=request.text,
        reasoning=request.reasoning,
    )
    return responses_response


def refine_responses_request(
    request: ResponsesRequest,
) -> ResponsesRequest:
    """Refine Responses API request with defaults from env for sampling params and model."""
    if request.temperature is None:
        request.temperature = _parse_env_float("DEFAULT_TEMPERATURE")
    if request.top_p is None:
        request.top_p = _parse_env_float("DEFAULT_TOP_P")
    if request.top_k is None:
        request.top_k = _parse_env_int("DEFAULT_TOP_K")
    if request.min_p is None:
        request.min_p = _parse_env_float("DEFAULT_MIN_P")
    if request.seed is None:
        request.seed = _parse_env_int("DEFAULT_SEED")
    if request.repetition_penalty is None:
        request.repetition_penalty = _parse_env_float("DEFAULT_REPETITION_PENALTY")
    if request.max_output_tokens is None:
        request.max_output_tokens = _parse_env_int("DEFAULT_MAX_TOKENS")
    if not request.model:
        request.model = Config.TEXT_MODEL
    return request

async def handle_responses_stream_response(
    generator: AsyncGenerator[Any, None],
    request: ResponsesRequest,
    model: str | None = None,
) -> AsyncGenerator[str, None]:
    """Handle streaming response generation for Responses API.

    Emits the full sequence of server-sent events defined by the OpenAI
    Responses API streaming protocol:
    response.created → response.in_progress → (per output item) →
    response.output_item.added → response.content_part.added →
    response.output_text.delta* → response.output_text.done →
    response.content_part.done → response.output_item.done →
    response.completed

    Reasoning chunks emitted by the handler are forwarded as
    ``response.reasoning_summary_text.delta`` events.  Tool-call chunks
    are forwarded as ``response.function_call_arguments.delta`` events.
    """
    unique_id = random_uuid()
    resp_id = f"resp_{unique_id}"
    msg_item_id = f"msg_{unique_id}"
    created_time = int(time.time())

    sequence_number = 0

    def _next_seq() -> int:
        nonlocal sequence_number
        seq = sequence_number
        sequence_number += 1
        return seq

    def _create_base_response(
        status: str, output: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        if output is None:
            output = []
        reasoning_val = None
        if request.reasoning:
            reasoning_val = (
                request.reasoning.model_dump()
                if hasattr(request.reasoning, "model_dump")
                else request.reasoning
            )
        text_val = None
        if request.text:
            text_val = (
                request.text.model_dump()
                if hasattr(request.text, "model_dump")
                else request.text
            )
        return {
            "id": resp_id,
            "created_at": created_time,
            "incomplete_details": None,
            "instructions": request.instructions,
            "metadata": None,
            "model": request.model,
            "object": "response",
            "output": output,
            "parallel_tool_calls": True,
            "temperature": request.temperature,
            "tool_choice": request.tool_choice,
            "tools": [_normalize_responses_item(t) for t in (request.tools or [])],
            "top_p": request.top_p,
            "background": False,
            "max_output_tokens": request.max_output_tokens,
            "max_tool_calls": None,
            "previous_response_id": request.previous_response_id,
            "prompt": None,
            "reasoning": reasoning_val,
            "service_tier": "auto",
            "status": status,
            "text": text_val,
            "top_logprobs": None,
            "truncation": "disabled",
            "usage": None,
            "user": None,
        }

    # ── Lifecycle: response.created / response.in_progress ────────────────
    base_in_progress = _create_base_response("in_progress")
    yield (
        f"event: response.created\n"
        f"data: {json.dumps({'response': base_in_progress, 'sequence_number': _next_seq(), 'type': 'response.created'})}\n\n"
    )
    yield (
        f"event: response.in_progress\n"
        f"data: {json.dumps({'response': base_in_progress, 'sequence_number': _next_seq(), 'type': 'response.in_progress'})}\n\n"
    )

    # ── Output item: message ───────────────────────────────────────────────
    # output_index=0 is the message item; reasoning items (if present) are
    # emitted inline before the message item is opened.
    output_index = 0

    # Reasoning item state (output_index=0 when reasoning precedes message)
    reasoning_item_id = f"rs_{unique_id}"
    reasoning_output_index: int | None = None
    full_reasoning = ""

    # Message item state
    full_text = ""
    usage_info = None
    # track pending tool-call streams keyed by tool_call_index
    tool_call_output_indices: dict[int, int] = {}  # tool_call_index → output_index
    tool_call_ids: dict[int, str] = {}  # tool_call_index → call item id
    tool_call_names: dict[int, str] = {}
    tool_call_args: dict[int, str] = {}
    msg_item_opened = False

    def _open_message_item() -> str:
        nonlocal msg_item_opened, output_index
        if msg_item_opened:
            return ""
        msg_item_opened = True
        events = []
        events.append(
            f"event: response.output_item.added\n"
            f"data: {json.dumps({'item': {'id': msg_item_id, 'content': [], 'role': 'assistant', 'status': 'in_progress', 'type': 'message'}, 'output_index': output_index, 'sequence_number': _next_seq(), 'type': 'response.output_item.added'})}\n\n"
        )
        events.append(
            f"event: response.content_part.added\n"
            f"data: {json.dumps({'content_index': 0, 'item_id': msg_item_id, 'output_index': output_index, 'part': {'annotations': [], 'text': '', 'type': 'output_text', 'logprobs': []}, 'sequence_number': _next_seq(), 'type': 'response.content_part.added'})}\n\n"
        )
        return "".join(events)

    try:
        async for chunk in generator:
            if not chunk:
                continue

            if isinstance(chunk, str):
                # Plain text delta → output_text
                pending = _open_message_item()
                if pending:
                    yield pending
                full_text += chunk
                yield (
                    f"event: response.output_text.delta\n"
                    f"data: {json.dumps({'content_index': 0, 'delta': chunk, 'item_id': msg_item_id, 'logprobs': [], 'output_index': output_index, 'sequence_number': _next_seq(), 'type': 'response.output_text.delta'})}\n\n"
                )

            elif isinstance(chunk, dict):
                if "__usage__" in chunk:
                    usage_info = chunk["__usage__"]
                    continue

                if "reasoning_content" in chunk:
                    # Reasoning delta — emitted before the message item
                    reasoning_text = chunk["reasoning_content"]
                    if reasoning_text:
                        if reasoning_output_index is None:
                            # Open reasoning item once
                            reasoning_output_index = output_index
                            output_index += 1
                            yield (
                                f"event: response.output_item.added\n"
                                f"data: {json.dumps({'item': {'id': reasoning_item_id, 'summary': [], 'type': 'reasoning', 'status': 'in_progress'}, 'output_index': reasoning_output_index, 'sequence_number': _next_seq(), 'type': 'response.output_item.added'})}\n\n"
                            )
                        full_reasoning += reasoning_text
                        yield (
                            f"event: response.reasoning_summary_text.delta\n"
                            f"data: {json.dumps({'delta': reasoning_text, 'item_id': reasoning_item_id, 'output_index': reasoning_output_index, 'summary_index': 0, 'sequence_number': _next_seq(), 'type': 'response.reasoning_summary_text.delta'})}\n\n"
                        )
                    # Also forward any content portion that may accompany reasoning
                    content_part = chunk.get("content")
                    if content_part:
                        pending = _open_message_item()
                        if pending:
                            yield pending
                        full_text += content_part
                        yield (
                            f"event: response.output_text.delta\n"
                            f"data: {json.dumps({'content_index': 0, 'delta': content_part, 'item_id': msg_item_id, 'logprobs': [], 'output_index': output_index, 'sequence_number': _next_seq(), 'type': 'response.output_text.delta'})}\n\n"
                        )

                elif "name" in chunk or "arguments" in chunk:
                    # Tool-call delta
                    tc_index = chunk.get("index", 0)
                    if tc_index not in tool_call_output_indices:
                        tc_item_id = f"fc_{unique_id}_{tc_index}"
                        tc_call_id = get_tool_call_id()
                        tc_out_idx = output_index
                        output_index += 1
                        tool_call_output_indices[tc_index] = tc_out_idx
                        tool_call_ids[tc_index] = tc_item_id
                        tool_call_names[tc_index] = chunk.get("name", "")
                        tool_call_args[tc_index] = ""
                        yield (
                            f"event: response.output_item.added\n"
                            f"data: {json.dumps({'item': {'id': tc_item_id, 'call_id': tc_call_id, 'name': tool_call_names[tc_index], 'arguments': '', 'type': 'function_call', 'status': 'in_progress'}, 'output_index': tc_out_idx, 'sequence_number': _next_seq(), 'type': 'response.output_item.added'})}\n\n"
                        )
                    tc_item_id = tool_call_ids[tc_index]
                    tc_out_idx = tool_call_output_indices[tc_index]
                    if "name" in chunk and chunk["name"] and not tool_call_names.get(tc_index):
                        tool_call_names[tc_index] = chunk["name"]
                    args_delta = chunk.get("arguments", "")
                    if args_delta:
                        tool_call_args[tc_index] = tool_call_args.get(tc_index, "") + args_delta
                        yield (
                            f"event: response.function_call_arguments.delta\n"
                            f"data: {json.dumps({'delta': args_delta, 'item_id': tc_item_id, 'output_index': tc_out_idx, 'sequence_number': _next_seq(), 'type': 'response.function_call_arguments.delta'})}\n\n"
                        )

        # ── Close reasoning item ───────────────────────────────────────────
        if reasoning_output_index is not None and full_reasoning:
            yield (
                f"event: response.reasoning_summary_text.done\n"
                f"data: {json.dumps({'item_id': reasoning_item_id, 'output_index': reasoning_output_index, 'summary_index': 0, 'text': full_reasoning, 'sequence_number': _next_seq(), 'type': 'response.reasoning_summary_text.done'})}\n\n"
            )
            yield (
                f"event: response.output_item.done\n"
                f"data: {json.dumps({'item': {'id': reasoning_item_id, 'summary': [{'type': 'summary_text', 'text': full_reasoning}], 'type': 'reasoning', 'status': 'completed'}, 'output_index': reasoning_output_index, 'sequence_number': _next_seq(), 'type': 'response.output_item.done'})}\n\n"
            )

        # ── Close tool-call items ──────────────────────────────────────────
        for tc_index, tc_out_idx in tool_call_output_indices.items():
            tc_item_id = tool_call_ids[tc_index]
            full_args = tool_call_args.get(tc_index, "")
            yield (
                f"event: response.function_call_arguments.done\n"
                f"data: {json.dumps({'arguments': full_args, 'item_id': tc_item_id, 'output_index': tc_out_idx, 'sequence_number': _next_seq(), 'type': 'response.function_call_arguments.done'})}\n\n"
            )
            yield (
                f"event: response.output_item.done\n"
                f"data: {json.dumps({'item': {'id': tc_item_id, 'name': tool_call_names.get(tc_index, ''), 'arguments': full_args, 'type': 'function_call', 'status': 'completed'}, 'output_index': tc_out_idx, 'sequence_number': _next_seq(), 'type': 'response.output_item.done'})}\n\n"
            )

        # ── Close message item (text) ──────────────────────────────────────
        if msg_item_opened:
            yield (
                f"event: response.output_text.done\n"
                f"data: {json.dumps({'content_index': 0, 'item_id': msg_item_id, 'logprobs': [], 'output_index': output_index, 'sequence_number': _next_seq(), 'text': full_text, 'type': 'response.output_text.done'})}\n\n"
            )
            yield (
                f"event: response.content_part.done\n"
                f"data: {json.dumps({'content_index': 0, 'item_id': msg_item_id, 'output_index': output_index, 'part': {'annotations': [], 'text': full_text, 'type': 'output_text', 'logprobs': None}, 'sequence_number': _next_seq(), 'type': 'response.content_part.done'})}\n\n"
            )
            yield (
                f"event: response.output_item.done\n"
                f"data: {json.dumps({'item': {'id': msg_item_id, 'content': [{'annotations': [], 'text': full_text, 'type': 'output_text', 'logprobs': None}], 'role': 'assistant', 'status': 'completed', 'summary': [], 'type': 'message'}, 'output_index': output_index, 'sequence_number': _next_seq(), 'type': 'response.output_item.done'})}\n\n"
            )

    except Exception as e:
        logger.exception(f"Error in Responses stream wrapper: {e}")
    finally:
        # ── Build final completed output list ─────────────────────────────
        final_output: list[dict[str, Any]] = []
        if reasoning_output_index is not None:
            final_output.insert(
                reasoning_output_index,
                {
                    "id": reasoning_item_id,
                    "summary": [{"type": "summary_text", "text": full_reasoning}],
                    "type": "reasoning",
                    "status": "completed",
                },
            )
        for tc_index, tc_out_idx in tool_call_output_indices.items():
            final_output.insert(
                tc_out_idx,
                {
                    "id": tool_call_ids[tc_index],
                    "name": tool_call_names.get(tc_index, ""),
                    "arguments": tool_call_args.get(tc_index, ""),
                    "type": "function_call",
                    "status": "completed",
                },
            )
        if msg_item_opened:
            final_output.append(
                {
                    "id": msg_item_id,
                    "content": [
                        {"annotations": [], "text": full_text, "type": "output_text", "logprobs": None}
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            )

        final_response_obj = _create_base_response("completed", final_output)

        if usage_info:
            input_tokens = usage_info.prompt_tokens
            output_tokens = usage_info.completion_tokens or 0
            cached_tokens = (
                usage_info.prompt_tokens_details.cached_tokens
                if usage_info.prompt_tokens_details
                and usage_info.prompt_tokens_details.cached_tokens is not None
                else 0
            )
            final_response_obj["usage"] = {
                "input_tokens": input_tokens,
                "input_tokens_details": {
                    "cached_tokens": cached_tokens,
                    "input_tokens_per_turn": [],
                    "cached_tokens_per_turn": [],
                },
                "output_tokens": output_tokens,
                "output_tokens_details": {
                    "reasoning_tokens": 0,
                    "tool_output_tokens": 0,
                    "output_tokens_per_turn": [],
                    "tool_output_tokens_per_turn": [],
                },
                "total_tokens": usage_info.total_tokens,
            }

        yield (
            f"event: response.completed\n"
            f"data: {json.dumps({'response': final_response_obj, 'sequence_number': _next_seq(), 'type': 'response.completed'})}\n\n"
        )

async def process_text_responses_request(
    handler: MLXLMHandler,
    request: ResponsesRequest
) -> ResponsesResponse | StreamingResponse | JSONResponse:
    """Handle text-only Responses API requests."""
    refined_request = refine_responses_request(request)
    chat_request = convert_responses_request_to_chat_request(refined_request)
    if refined_request.stream:
        return StreamingResponse(
            handle_responses_stream_response(
                handler.generate_text_stream(chat_request),
                refined_request,
                refined_request.model,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    result = await handler.generate_text_response(chat_request)
    response_data = result.get("response")
    usage = result.get("usage")
    final_response = format_final_responses_response(
        response_data,
        refined_request,
        usage,
    )
    return JSONResponse(content=final_response.model_dump(exclude_none=True))


async def process_multimodal_responses_request(
    handler: MLXVLMHandler,
    request: ResponsesRequest,
) -> ResponsesResponse | StreamingResponse | JSONResponse:
    """Handle multimodal Responses API requests."""
    refined_request = refine_responses_request(request)
    chat_request = convert_responses_request_to_chat_request(refined_request)

    if refined_request.stream:
        return StreamingResponse(
            handle_responses_stream_response(
                handler.generate_multimodal_stream(chat_request),
                refined_request,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    result = await handler.generate_multimodal_response(chat_request)
    response_data = result.get("response")
    usage = result.get("usage")
    final_response = format_final_responses_response(
        response_data,
        refined_request,
        usage
    )
    return JSONResponse(content=final_response.model_dump(exclude_none=True))

@router.post("/v1/responses", response_model=None)
async def responses_endpoint(
    request: ResponsesRequest, raw_request: Request
) -> ResponsesResponse | StreamingResponse | JSONResponse:
    """Handle Responses API requests (OpenAI-compatible)."""
    handler = _resolve_handler(raw_request, model_id=request.model)
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized",
                "service_unavailable",
                HTTPStatus.SERVICE_UNAVAILABLE,
            ),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )

    handler_type = _get_handler_type(handler)
    if handler_type not in ("lm", "multimodal"):
        return JSONResponse(
            content=create_error_response(
                "Unsupported model type for responses. "
                f"Handler for '{request.model}' is {type(handler).__name__} "
                f"(handler_type={handler_type!r}).",
                "unsupported_request",
                HTTPStatus.BAD_REQUEST,
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    try:
        if handler_type == "multimodal":
            return await process_multimodal_responses_request(handler, request)
        return await process_text_responses_request(handler, request)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing responses request: {type(e).__name__}: {e}")
        return JSONResponse(
            content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )