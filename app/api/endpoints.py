"""FastAPI router endpoints for the MLX OpenAI Server.

This module defines all the API endpoints that provide OpenAI-compatible
interfaces for language models, vision models, embeddings, image generation,
and other MLX-powered services.
"""

import base64
from collections.abc import AsyncGenerator
from http import HTTPStatus
import json
import random
import time
from typing import Annotated, Any

from fastapi import APIRouter, Form, Request
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
    ImageGenerationRequest,
    Message,
    Model,
    ModelsResponse,
    StreamingChoice,
    TranscriptionRequest,
)
from ..utils.errors import create_error_response

router = APIRouter()


# =============================================================================
# Critical/Monitoring Endpoints - Defined first to ensure priority matching
# =============================================================================


@router.get("/health")
async def health():
    """Health check endpoint.

    Always responds immediately without dependencies.

    Returns
    -------
    HealthCheckResponse
        Response indicating service health status.
    """
    return HealthCheckResponse(status=HealthCheckStatus.OK)


@router.get("/v1/models")
async def models(raw_request: Request):
    """Get list of available models.

    Provides a cached response for instant delivery. This endpoint is
    defined early to ensure it's not blocked by other routes.

    Parameters
    ----------
    raw_request : Request
        The FastAPI request object.

    Returns
    -------
    ModelsResponse or JSONResponse
        List of available models or error response.
    """
    handler = raw_request.app.state.handler
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized", "service_unavailable", 503
            ),
            status_code=503,
        )

    try:
        models_data = await handler.get_models()
        return ModelsResponse(data=[Model(**model) for model in models_data])
    except Exception as e:
        logger.error(f"Error retrieving models: {e!s}")
        return JSONResponse(
            content=create_error_response(f"Failed to retrieve models: {e!s}", "server_error", 500),
            status_code=500,
        )


@router.get("/v1/queue/stats")
async def queue_stats(raw_request: Request):
    """Get queue statistics."""
    handler = raw_request.app.state.handler
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized", "service_unavailable", 503
            ),
            status_code=503,
        )

    try:
        stats = await handler.get_queue_stats()
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e!s}")
        return JSONResponse(
            content=create_error_response("Failed to get queue stats", "server_error", 500),
            status_code=500,
        )
    else:
        return {"status": "ok", "queue_stats": stats}


# =============================================================================
# API Endpoints - Core functionality
# =============================================================================


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """Handle chat completion requests."""
    handler = raw_request.app.state.handler
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized", "service_unavailable", 503
            ),
            status_code=503,
        )

    if not isinstance(handler, MLXVLMHandler) and not isinstance(handler, MLXLMHandler):
        return JSONResponse(
            content=create_error_response("Unsupported model type", "unsupported_request", 400),
            status_code=400,
        )

    try:
        if isinstance(handler, MLXVLMHandler):
            return await process_multimodal_request(handler, request)
        return await process_text_request(handler, request)
    except Exception as e:
        logger.error(f"Error processing chat completion request: {e!s}", exc_info=True)
        return JSONResponse(
            content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )


@router.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest, raw_request: Request):
    """Handle embedding requests."""
    handler = raw_request.app.state.handler
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized", "service_unavailable", 503
            ),
            status_code=503,
        )

    try:
        embeddings = await handler.generate_embeddings_response(request)
        return create_response_embeddings(embeddings, request.model, request.encoding_format)
    except Exception as e:
        logger.error(f"Error processing embedding request: {e!s}", exc_info=True)
        return JSONResponse(
            content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )


@router.post("/v1/images/generations")
async def image_generations(request: ImageGenerationRequest, raw_request: Request):
    """Handle image generation requests."""
    handler = raw_request.app.state.handler
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized", "service_unavailable", 503
            ),
            status_code=503,
        )

    # Check if the handler is an MLXFluxHandler
    if not MFLUX_AVAILABLE or not isinstance(handler, MLXFluxHandler):
        return JSONResponse(
            content=create_error_response(
                "Image generation requests require an image generation model. Use --model-type image-generation.",
                "unsupported_request",
                400,
            ),
            status_code=400,
        )

    try:
        return await handler.generate_image(request)
    except Exception as e:
        logger.error(f"Error processing image generation request: {e!s}", exc_info=True)
        return JSONResponse(
            content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )


@router.post("/v1/images/edits")
async def create_image_edit(request: Annotated[ImageEditRequest, Form()], raw_request: Request):
    """Handle image editing requests with dynamic provider routing."""
    handler = raw_request.app.state.handler
    if handler is None:
        return JSONResponse(
            content=create_error_response(
                "Model handler not initialized", "service_unavailable", 503
            ),
            status_code=503,
        )

    # Check if the handler is an MLXFluxHandler
    if not MFLUX_AVAILABLE or not isinstance(handler, MLXFluxHandler):
        return JSONResponse(
            content=create_error_response(
                "Image editing requests require an image generation model. Use --model-type image-generation.",
                "unsupported_request",
                400,
            ),
            status_code=400,
        )
    try:
        return await handler.edit_image(request)
    except Exception as e:
        logger.error(f"Error processing image edit request: {e!s}", exc_info=True)
        return JSONResponse(
            content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )


@router.post("/v1/audio/transcriptions")
async def create_audio_transcriptions(
    request: Annotated[TranscriptionRequest, Form()], raw_request: Request
):
    """Handle audio transcription requests."""
    try:
        handler = raw_request.app.state.handler
        if handler is None:
            return JSONResponse(
                content=create_error_response(
                    "Model handler not initialized", "service_unavailable", 503
                ),
                status_code=503,
            )

        if request.stream:
            # procoess the request before sending to the handler
            request_data = await handler._prepare_transcription_request(request)
            return StreamingResponse(
                handler.generate_transcription_stream_from_data(
                    request_data, request.response_format
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        return await handler.generate_transcription_response(request)
    except Exception as e:
        logger.error(f"Error processing transcription request: {e!s}", exc_info=True)
        return JSONResponse(
            content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )


def create_response_embeddings(
    embeddings: list[float], model: str, encoding_format: str = "float"
) -> EmbeddingResponse:
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
    return EmbeddingResponse(data=embeddings_response, model=model)


def create_response_chunk(
    chunk: str | dict[str, Any],
    model: str,
    is_final: bool = False,
    finish_reason: str | None = "stop",
    chat_id: str | None = None,
    created_time: int | None = None,
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
                    delta=Delta(content=chunk, role="assistant"),
                    finish_reason=finish_reason if is_final else None,
                )
            ],
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
                    ),
                    finish_reason=finish_reason if is_final else None,
                )
            ],
        )

    # Handle tool/function call chunks
    function_call = None
    if "name" in chunk:
        function_call = ChoiceDeltaFunctionCall(name=chunk["name"])
        if "arguments" in chunk:
            function_call.arguments = chunk["arguments"]
    elif "arguments" in chunk:
        # Handle case where arguments come before name (streaming)
        function_call = ChoiceDeltaFunctionCall(arguments=chunk["arguments"])

    if function_call:
        # Validate index exists before accessing
        tool_index = chunk.get("index", 0)
        tool_chunk = ChoiceDeltaToolCall(
            index=tool_index, type="function", id=get_tool_call_id(), function=function_call
        )

        delta = Delta(content=None, role="assistant", tool_calls=[tool_chunk])
    else:
        # Fallback: create empty delta if no recognized chunk type
        delta = Delta(role="assistant")

    return ChatCompletionChunk(
        id=chat_id,
        object="chat.completion.chunk",
        created=created_time,
        model=model,
        choices=[
            StreamingChoice(index=0, delta=delta, finish_reason=finish_reason if is_final else None)
        ],
    )


def _yield_sse_chunk(data: dict[str, Any] | ChatCompletionChunk) -> str:
    """Helper function to format and yield SSE chunk data."""
    if isinstance(data, ChatCompletionChunk):
        return f"data: {json.dumps(data.model_dump())}\n\n"
    return f"data: {json.dumps(data)}\n\n"


async def handle_stream_response(generator: AsyncGenerator, model: str):
    """Handle streaming response generation (OpenAI-compatible)."""
    chat_index = get_id()
    created_time = int(time.time())
    finish_reason = "stop"
    tool_call_index = -1

    try:
        # First chunk: role-only delta, as per OpenAI
        first_chunk = ChatCompletionChunk(
            id=chat_index,
            object="chat.completion.chunk",
            created=created_time,
            model=model,
            choices=[StreamingChoice(index=0, delta=Delta(role="assistant"))],
        )
        yield _yield_sse_chunk(first_chunk)

        async for chunk in generator:
            if not chunk:
                continue

            if isinstance(chunk, str):
                response_chunk = create_response_chunk(
                    chunk, model, chat_id=chat_index, created_time=created_time
                )
                yield _yield_sse_chunk(response_chunk)

            elif isinstance(chunk, dict):
                # Handle tool call chunks
                payload = dict(chunk)  # Create a copy to avoid mutating the original
                if payload.get("name"):
                    finish_reason = "tool_calls"
                    tool_call_index += 1
                    payload["index"] = tool_call_index
                elif payload.get("arguments") and "index" not in payload:
                    payload["index"] = tool_call_index

                response_chunk = create_response_chunk(
                    payload, model, chat_id=chat_index, created_time=created_time
                )
                yield _yield_sse_chunk(response_chunk)

            else:
                error_response = create_error_response(
                    f"Invalid chunk type: {type(chunk)}",
                    "server_error",
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                yield _yield_sse_chunk(error_response)

    except Exception as e:
        logger.error(f"Error in stream wrapper: {e!s}", exc_info=True)
        error_response = create_error_response(
            str(e), "server_error", HTTPStatus.INTERNAL_SERVER_ERROR
        )
        yield _yield_sse_chunk(error_response)
    finally:
        # Final chunk: finish_reason and [DONE], as per OpenAI
        final_chunk = create_response_chunk(
            "", model, is_final=True, finish_reason=finish_reason, chat_id=chat_index
        )
        yield _yield_sse_chunk(final_chunk)
        yield "data: [DONE]\n\n"


async def process_multimodal_request(handler, request: ChatCompletionRequest):
    """Process multimodal-specific requests."""
    if request.stream:
        return StreamingResponse(
            handle_stream_response(handler.generate_multimodal_stream(request), request.model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    return format_final_response(await handler.generate_multimodal_response(request), request.model)


async def process_text_request(handler, request: ChatCompletionRequest):
    """Process text-only requests."""
    if request.stream:
        return StreamingResponse(
            handle_stream_response(handler.generate_text_stream(request), request.model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    return format_final_response(await handler.generate_text_response(request), request.model)


def get_id():
    """Generate a unique ID for chat completions with timestamp and random component."""
    timestamp = int(time.time())
    random_suffix = random.randint(0, 999999)
    return f"chatcmpl_{timestamp}{random_suffix:06d}"


def get_tool_call_id():
    """Generate a unique ID for tool calls with timestamp and random component."""
    timestamp = int(time.time())
    random_suffix = random.randint(0, 999999)
    return f"call_{timestamp}{random_suffix:06d}"


def format_final_response(response: str | dict[str, Any], model: str) -> ChatCompletionResponse:
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
                        function_call=None,
                        reasoning_content=None,
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
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
                    ),
                    finish_reason="stop",
                )
            ],
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
    )

    return ChatCompletionResponse(
        id=get_id(),
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[Choice(index=0, message=message, finish_reason="tool_calls")],
    )
