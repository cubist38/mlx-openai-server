import base64
import json
import random
import time
from http import HTTPStatus
from typing import Annotated, Any, AsyncGenerator, Dict, List, Optional, Union

import numpy as np
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

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


def _get_handler_manager(raw_request: Request):
    return getattr(raw_request.app.state, "handler_manager", None)


def _handler_unavailable_response(message: str = "Model handler not initialized") -> JSONResponse:
    return JSONResponse(
        content=create_error_response(message, "service_unavailable", 503),
        status_code=503,
    )


async def _ensure_active_handler(raw_request: Request, reason: str):
    handler_manager = _get_handler_manager(raw_request)
    if handler_manager is None:
        return None

    handler = await handler_manager.ensure_loaded(reason)
    if handler is None:
        return None

    auto_unload_controller = getattr(raw_request.app.state, "auto_unload_controller", None)
    if auto_unload_controller:
        auto_unload_controller.notify_activity()

    return handler


# =============================================================================
# Critical/Monitoring Endpoints - Defined first to ensure priority matching
# =============================================================================


@router.get("/health")
async def health(raw_request: Request):
    """
    Health check endpoint - verifies handler initialization status.
    Returns 503 if handler is not initialized, 200 otherwise.
    """
    handler = getattr(raw_request.app.state, "handler", None)

    if handler is None:
        # Handler not initialized - return 503 with degraded status
        return JSONResponse(
            status_code=503,
            content={"status": "ok", "model_id": None, "model_status": "uninitialized"},
        )

    # Handler initialized - extract model_id
    model_id = getattr(handler, "model_path", "unknown")

    return HealthCheckResponse(
        status=HealthCheckStatus.OK, model_id=model_id, model_status="initialized"
    )


@router.get("/v1/models")
async def models(raw_request: Request):
    """
    Get list of available models with cached response for instant delivery.
    This endpoint is defined early to ensure it's not blocked by other routes.
    """
    handler_manager = _get_handler_manager(raw_request)
    handler = handler_manager.current_handler if handler_manager else None

    if handler is not None:
        try:
            models_data = await handler.get_models()
            return ModelsResponse(data=[Model(**model) for model in models_data])
        except (HTTPException, ValueError) as e:
            logger.error(f"Error retrieving models: {str(e)}")
            return JSONResponse(
                content=create_error_response(
                    f"Failed to retrieve models: {str(e)}", "server_error", 500
                ),
                status_code=500,
            )
        except:
            # Re-raise unexpected exceptions to avoid masking them
            raise

    config = getattr(raw_request.app.state, "server_config", None)
    metadata = getattr(raw_request.app.state, "model_metadata", {})
    created_ts = metadata.get("created", int(time.time()))

    if config is None:
        return _handler_unavailable_response()

    fallback_model = Model(
        id=config.model_path,
        object="model",
        created=created_ts,
        owned_by="local",
    )
    return ModelsResponse(data=[fallback_model])


@router.get("/v1/queue/stats")
async def queue_stats(raw_request: Request):
    """
    Get queue statistics.
    """
    handler_manager = _get_handler_manager(raw_request)
    handler = handler_manager.current_handler if handler_manager else None
    if handler is None:
        return _handler_unavailable_response(
            "Model handler is currently unloaded. Send a request to load it."
        )

    try:
        stats = await handler.get_queue_stats()
        return {"status": "ok", "queue_stats": stats}
    except Exception as e:
        logger.error(f"Failed to get queue stats: {str(e)}")
        return JSONResponse(
            content=create_error_response("Failed to get queue stats", "server_error", 500),
            status_code=500,
        )


# =============================================================================
# API Endpoints - Core functionality
# =============================================================================


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """Handle chat completion requests."""

    handler = await _ensure_active_handler(raw_request, "chat-completions")
    if handler is None:
        return _handler_unavailable_response()

    if not isinstance(handler, MLXVLMHandler) and not isinstance(handler, MLXLMHandler):
        return JSONResponse(
            content=create_error_response("Unsupported model type", "unsupported_request", 400),
            status_code=400,
        )

    # Get request ID from middleware
    request_id = getattr(raw_request.state, "request_id", None)

    try:
        if isinstance(handler, MLXVLMHandler):
            return await process_multimodal_request(handler, request, request_id)
        else:
            return await process_text_request(handler, request, request_id)
    except Exception as e:
        logger.error(f"Error processing chat completion request: {str(e)}", exc_info=True)
        return JSONResponse(
            content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )
    except:
        # Re-raise unexpected exceptions to avoid masking them
        raise


@router.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest, raw_request: Request):
    """Handle embedding requests."""
    handler = await _ensure_active_handler(raw_request, "embeddings")
    if handler is None:
        return _handler_unavailable_response()

    try:
        embeddings = await handler.generate_embeddings_response(request)
        return create_response_embeddings(embeddings, request.model, request.encoding_format)
    except Exception as e:
        logger.error(f"Error processing embedding request: {str(e)}", exc_info=True)
        return JSONResponse(
            content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )


@router.post("/v1/images/generations")
async def image_generations(request: ImageGenerationRequest, raw_request: Request):
    """Handle image generation requests."""
    handler = await _ensure_active_handler(raw_request, "image-generations")
    if handler is None:
        return _handler_unavailable_response()

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
        image_response = await handler.generate_image(request)
        return image_response
    except Exception as e:
        logger.error(f"Error processing image generation request: {str(e)}", exc_info=True)
        return JSONResponse(
            content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )


@router.post("/v1/images/edits")
async def create_image_edit(request: Annotated[ImageEditRequest, Form()], raw_request: Request):
    """Handle image editing requests with dynamic provider routing."""

    handler = await _ensure_active_handler(raw_request, "image-edits")
    if handler is None:
        return _handler_unavailable_response()

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
        image_response = await handler.edit_image(request)
        return image_response
    except Exception as e:
        logger.error(f"Error processing image edit request: {str(e)}", exc_info=True)
        return JSONResponse(
            content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )


@router.post("/v1/audio/transcriptions")
async def create_audio_transcriptions(
    request: Annotated[TranscriptionRequest, Form()], raw_request: Request
):
    """Handle audio transcription requests."""
    try:
        handler = await _ensure_active_handler(raw_request, "audio-transcriptions")
        if handler is None:
            return _handler_unavailable_response()

        if request.stream:
            # process the request before sending to the handler
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
        else:
            transcription_response = await handler.generate_transcription_response(request)
            return transcription_response
    except Exception as e:
        logger.error(f"Error processing transcription request: {str(e)}", exc_info=True)
        return JSONResponse(
            content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )


def create_response_embeddings(
    embeddings: List[float], model: str, encoding_format: str = "float"
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
    chunk: Union[str, Dict[str, Any]],
    model: str,
    is_final: bool = False,
    finish_reason: Optional[str] = "stop",
    chat_id: Optional[str] = None,
    created_time: Optional[int] = None,
    request_id: str = None,
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
                    ),
                    finish_reason=finish_reason if is_final else None,
                )
            ],
            request_id=request_id,
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
        request_id=request_id,
    )


def _yield_sse_chunk(data: Union[Dict[str, Any], ChatCompletionChunk]) -> str:
    """Helper function to format and yield SSE chunk data."""
    if isinstance(data, ChatCompletionChunk):
        return f"data: {json.dumps(data.model_dump())}\n\n"
    return f"data: {json.dumps(data)}\n\n"


async def handle_stream_response(generator: AsyncGenerator, model: str, request_id: str = None):
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
            choices=[StreamingChoice(index=0, delta=Delta(role="assistant"))],
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

    except Exception as e:
        logger.error(f"Error in stream wrapper: {str(e)}", exc_info=True)
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
            choices=[StreamingChoice(index=0, delta=Delta(), finish_reason=finish_reason)],
            usage=usage_info,
            request_id=request_id,
        )
        yield _yield_sse_chunk(final_chunk)
        yield "data: [DONE]\n\n"


async def process_multimodal_request(
    handler, request: ChatCompletionRequest, request_id: str = None
):
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
        return format_final_response(response_data, request.model, request_id, usage)

    # Fallback for backward compatibility or if structure is different
    return format_final_response(result, request.model, request_id)


async def process_text_request(handler, request: ChatCompletionRequest, request_id: str = None):
    """Process text-only requests."""
    if request_id:
        logger.info(f"Processing text request [request_id={request_id}]")

    if request.stream:
        return StreamingResponse(
            handle_stream_response(
                handler.generate_text_stream(request), request.model, request_id
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Extract response and usage from handler
    result = await handler.generate_text_response(request)
    response_data = result.get("response")
    usage = result.get("usage")
    return format_final_response(response_data, request.model, request_id, usage)


def get_id():
    """
    Generate a unique ID for chat completions with timestamp and random component.
    """
    timestamp = int(time.time())
    random_suffix = random.randint(0, 999999)
    return f"chatcmpl_{timestamp}{random_suffix:06d}"


def get_tool_call_id():
    """
    Generate a unique ID for tool calls with timestamp and random component.
    """
    timestamp = int(time.time())
    random_suffix = random.randint(0, 999999)
    return f"call_{timestamp}{random_suffix:06d}"


def format_final_response(
    response: Union[str, Dict[str, Any]], model: str, request_id: str = None, usage=None
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
                        function_call=None,
                        reasoning_content=None,
                        tool_calls=None,
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
