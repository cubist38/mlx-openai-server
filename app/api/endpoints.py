"""API endpoints for the MLX OpenAI server."""

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

from ..handler import MFLUX_AVAILABLE, MLXFluxHandler
from ..handler.mlx_lm import MLXLMHandler
from ..handler.mlx_vlm import MLXVLMHandler
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

router = APIRouter()


# =============================================================================
# Critical/Monitoring Endpoints - Defined first to ensure priority matching
# =============================================================================


@router.get("/health", response_model=None)
async def health(raw_request: Request) -> HealthCheckResponse | JSONResponse:
    """
    Health check endpoint - verifies handler initialization status.

    Returns 503 if handler is not initialized, 200 otherwise.
    """
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


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest, raw_request: Request
) -> ChatCompletionResponse | StreamingResponse | JSONResponse:
    """Handle chat completion requests."""
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

    if not isinstance(handler, MLXVLMHandler) and not isinstance(handler, MLXLMHandler):
        return JSONResponse(
            content=create_error_response(
                "Unsupported model type", "unsupported_request", HTTPStatus.BAD_REQUEST
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )

    # Get request ID from middleware
    request_id = getattr(raw_request.state, "request_id", None)

    try:
        if isinstance(handler, MLXVLMHandler):
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
    """Handle image editing requests with dynamic provider routing."""
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
    """Handle audio transcription requests."""
    try:
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
    if isinstance(data, ChatCompletionChunk):
        return f"data: {json.dumps(data.model_dump())}\n\n"
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
    return format_final_response(response_data, request.model, request_id, usage)  # type: ignore[arg-type]


def get_id() -> str:
    """Generate a unique ID for chat completions with timestamp and random component."""
    timestamp = int(time.time())
    random_suffix = random.randint(0, 999999)
    return f"chatcmpl_{timestamp}{random_suffix:06d}"


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
