import json
import random
import time
import base64
import numpy as np
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Annotated

from fastapi import APIRouter, Request, Form
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from app.handler.mlx_lm import MLXLMHandler
from app.handler.mlx_vlm import MLXVLMHandler
from app.handler import MLXFluxHandler, MFLUX_AVAILABLE
from app.schemas.openai import (HealthCheckResponse, HealthCheckStatus,
                                ChatCompletionChunk, ChatCompletionMessageToolCall,
                                ChatCompletionRequest, ChatCompletionResponse,
                                Choice, ChoiceDeltaFunctionCall,
                                ChoiceDeltaToolCall, Delta, EmbeddingResponseData,
                                EmbeddingRequest, EmbeddingResponse,
                                FunctionCall, Message, Model, ModelsResponse,
                                StreamingChoice, ImageGenerationRequest,
                                ImageEditRequest, TranscriptionRequest)
from app.utils.errors import create_error_response

router = APIRouter()


# =============================================================================
# Critical/Monitoring Endpoints - Defined first to ensure priority matching
# =============================================================================

@router.get("/health")
async def health():
    """
    Health check endpoint - always responds immediately without dependencies.
    """
    return HealthCheckResponse(status=HealthCheckStatus.OK)

@router.get("/v1/models")
async def models(raw_request: Request):
    """
    Get list of available models with cached response for instant delivery.
    This endpoint is defined early to ensure it's not blocked by other routes.
    """
    handler = raw_request.app.state.handler
    if handler is None:
        return JSONResponse(content=create_error_response("Model handler not initialized", "service_unavailable", 503), status_code=503)
    
    try:
        models_data = await handler.get_models()
        return ModelsResponse(data=[Model(**model) for model in models_data])
    except Exception as e:
        logger.error(f"Error retrieving models: {str(e)}")
        return JSONResponse(content=create_error_response(f"Failed to retrieve models: {str(e)}", "server_error", 500), status_code=500)

@router.get("/v1/queue/stats")
async def queue_stats(raw_request: Request):
    """
    Get queue statistics.
    """
    handler = raw_request.app.state.handler
    if handler is None:
        return JSONResponse(content=create_error_response("Model handler not initialized", "service_unavailable", 503), status_code=503)
    
    try:
        stats = await handler.get_queue_stats()
        return {
            "status": "ok",
            "queue_stats": stats
        }
    except Exception as e:
        logger.error(f"Failed to get queue stats: {str(e)}")
        return JSONResponse(content=create_error_response("Failed to get queue stats", "server_error", 500), status_code=500)


# =============================================================================
# API Endpoints - Core functionality
# =============================================================================

@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """Handle chat completion requests."""

    handler = raw_request.app.state.handler
    if handler is None:
        return JSONResponse(content=create_error_response("Model handler not initialized", "service_unavailable", 503), status_code=503)
    
    if not isinstance(handler, MLXVLMHandler) and not isinstance(handler, MLXLMHandler):
        return JSONResponse(content=create_error_response("Unsupported model type", "unsupported_request", 400), status_code=400)

    try:
        if isinstance(handler, MLXVLMHandler):
            return await process_multimodal_request(handler, request)
        else:
            return await process_text_request(handler, request)
    except Exception as e:
        logger.error(f"Error processing chat completion request: {str(e)}", exc_info=True)
        return JSONResponse(content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
    
@router.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest, raw_request: Request):
    """Handle embedding requests."""
    handler = raw_request.app.state.handler
    if handler is None:
        return JSONResponse(content=create_error_response("Model handler not initialized", "service_unavailable", 503), status_code=503)

    try:
        embeddings = await handler.generate_embeddings_response(request)
        return create_response_embeddings(embeddings, request.model, request.encoding_format)
    except Exception as e:
        logger.error(f"Error processing embedding request: {str(e)}", exc_info=True)
        return JSONResponse(content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

@router.post("/v1/images/generations")
async def image_generations(request: ImageGenerationRequest, raw_request: Request):
    """Handle image generation requests."""
    handler = raw_request.app.state.handler
    if handler is None:
        return JSONResponse(content=create_error_response("Model handler not initialized", "service_unavailable", 503), status_code=503)
    
    # Check if the handler is an MLXFluxHandler
    if not MFLUX_AVAILABLE or not isinstance(handler, MLXFluxHandler):
        return JSONResponse(
            content=create_error_response(
                "Image generation requests require an image generation model. Use --model-type image-generation.",
                "unsupported_request",
                400
            ),
            status_code=400
        )
    
    try:
        image_response = await handler.generate_image(request)
        return image_response
    except Exception as e:
        logger.error(f"Error processing image generation request: {str(e)}", exc_info=True)
        return JSONResponse(content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

@router.post("/v1/images/edits")
async def create_image_edit(request: Annotated[ImageEditRequest, Form()], raw_request: Request):
    """Handle image editing requests with dynamic provider routing."""

    handler = raw_request.app.state.handler
    if handler is None:
        return JSONResponse(content=create_error_response("Model handler not initialized", "service_unavailable", 503), status_code=503)
    
    # Check if the handler is an MLXFluxHandler
    if not MFLUX_AVAILABLE or not isinstance(handler, MLXFluxHandler):
        return JSONResponse(
            content=create_error_response(
                "Image editing requests require an image generation model. Use --model-type image-generation.",
                "unsupported_request",
                400
            ),
            status_code=400
        )
    try:
        image_response = await handler.edit_image(request)
        return image_response
    except Exception as e:
        logger.error(f"Error processing image edit request: {str(e)}", exc_info=True)
        return JSONResponse(content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
    
@router.post("/v1/audio/transcriptions")
async def create_audio_transcriptions(
    request: Annotated[TranscriptionRequest, Form()],
    raw_request: Request
):
    """Handle audio transcription requests."""
    try:
        handler = raw_request.app.state.handler
        if handler is None:
            return JSONResponse(content=create_error_response("Model handler not initialized", "service_unavailable", 503), status_code=503)
        
        if request.stream:

            # procoess the request before sending to the handler
            request_data = await handler._prepare_transcription_request(request)
            return StreamingResponse(
                handler.generate_transcription_stream_from_data(request_data, request.response_format),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
            )
        else:
            transcription_response = await handler.generate_transcription_response(request)
            return transcription_response
    except Exception as e:
        logger.error(f"Error processing transcription request: {str(e)}", exc_info=True)
        return JSONResponse(content=create_error_response(str(e)), status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
    
def create_response_embeddings(embeddings: List[float], model: str, encoding_format: str = "float") -> EmbeddingResponse:
    embeddings_response = []
    for index, embedding in enumerate(embeddings):
        if encoding_format == "base64":
            # Convert list/array to bytes before base64 encoding
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
            embeddings_response.append(EmbeddingResponseData(embedding=base64.b64encode(embedding_bytes).decode('utf-8'), index=index))
        else:
            embeddings_response.append(EmbeddingResponseData(embedding=embedding, index=index))
    return EmbeddingResponse(data=embeddings_response, model=model)

def create_response_chunk(chunk: Union[str, Dict[str, Any]], model: str, is_final: bool = False, finish_reason: Optional[str] = "stop", chat_id: Optional[str] = None, created_time: Optional[int] = None) -> ChatCompletionChunk:
    """Create a formatted response chunk for streaming."""
    chat_id = chat_id if chat_id else get_id()
    created_time = created_time if created_time else int(time.time())
    if isinstance(chunk, str):
        return ChatCompletionChunk(
            id=chat_id,
            object="chat.completion.chunk",
            created=created_time,
            model=model,
            choices=[StreamingChoice(
                index=0,
                delta=Delta(content=chunk, role="assistant"),
                finish_reason=finish_reason if is_final else None
            )]
        )
    if "reasoning_content" in chunk:
        return ChatCompletionChunk(
            id=chat_id,
            object="chat.completion.chunk",
            created=created_time,
            model=model,
            choices=[StreamingChoice(
                index=0,
                delta=Delta(reasoning_content=chunk["reasoning_content"], role="assistant", content=chunk.get("content", None)),
                finish_reason=finish_reason if is_final else None
            )]
        )

    if "name" in chunk and chunk["name"]:
        tool_chunk = ChoiceDeltaToolCall(
            index=chunk["index"],
            type="function",
            id=get_tool_call_id(),
            function=ChoiceDeltaFunctionCall(
                name=chunk["name"],
                arguments=chunk["arguments"]
            )
        )
    else:
        tool_chunk = ChoiceDeltaToolCall(
            index=chunk["index"],
            type="function",
            function= ChoiceDeltaFunctionCall(
                arguments=chunk["arguments"]
            )
        )
    delta = Delta(
        content = None,
        role = "assistant",
        tool_calls = [tool_chunk]
    )
    return ChatCompletionChunk(
        id=chat_id,
        object="chat.completion.chunk",
        created=created_time,
        model=model,
        choices=[StreamingChoice(index=0, delta=delta, finish_reason=None)]
    )

async def handle_stream_response(generator: AsyncGenerator, model: str):
    """Handle streaming response generation (OpenAI-compatible)."""
    chat_index = get_id()
    created_time = int(time.time())
    try:
        finish_reason = "stop"
        index = -1
        # First chunk: role-only delta, as per OpenAI
        first_chunk = ChatCompletionChunk(
            id=chat_index,
            object="chat.completion.chunk",
            created=created_time,
            model=model,
            choices=[StreamingChoice(index=0, delta=Delta(role="assistant"), finish_reason=None)]
        )
        yield f"data: {json.dumps(first_chunk.model_dump())}\n\n"
        async for chunk in generator:
            if chunk:
                if isinstance(chunk, str):
                    response_chunk = create_response_chunk(chunk, model, chat_id=chat_index, created_time=created_time)
                    yield f"data: {json.dumps(response_chunk.model_dump())}\n\n"
                elif isinstance(chunk, dict):
                    if "name" in chunk and chunk["name"]:
                        finish_reason = "tool_calls"
                        index += 1
                        payload = {
                            "index": index,
                            **chunk
                        }
                        response_chunk = create_response_chunk(payload, model, chat_id=chat_index, created_time=created_time, finish_reason=finish_reason)
                        yield f"data: {json.dumps(response_chunk.model_dump())}\n\n"
                    else:
                        response_chunk = create_response_chunk(chunk, model, chat_id=chat_index, created_time=created_time)
                        yield f"data: {json.dumps(response_chunk.model_dump())}\n\n"
                else:
                    error_response = create_error_response(f"Invalid chunk type: {type(chunk)}", "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
                    yield f"data: {json.dumps(error_response)}\n\n"
    except Exception as e:
        logger.error(f"Error in stream wrapper: {str(e)}")
        error_response = create_error_response(str(e), "server_error", HTTPStatus.INTERNAL_SERVER_ERROR)
        # Yield error as last chunk before [DONE]
        yield f"data: {json.dumps(error_response)}\n\n"
    finally:
        # Final chunk: finish_reason and [DONE], as per OpenAI
        final_chunk = create_response_chunk('', model, is_final=True, finish_reason=finish_reason, chat_id=chat_index)
        yield f"data: {json.dumps(final_chunk.model_dump())}\n\n"
        yield "data: [DONE]\n\n"

async def process_multimodal_request(handler, request: ChatCompletionRequest):
    """Process multimodal-specific requests."""
    if request.stream:
        return StreamingResponse(
            handle_stream_response(handler.generate_multimodal_stream(request), request.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )
    return format_final_response(await handler.generate_multimodal_response(request), request.model)

async def process_text_request(handler, request: ChatCompletionRequest):
    """Process text-only requests."""
    if request.stream:
        return StreamingResponse(
            handle_stream_response(handler.generate_text_stream(request), request.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )
    return format_final_response(await handler.generate_text_response(request), request.model)

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

def format_final_response(response: Union[str, Dict[str, Any]], model: str) -> ChatCompletionResponse:
    """Format the final non-streaming response."""
    
    if isinstance(response, str):
        return ChatCompletionResponse(
            id=get_id(),
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[Choice(
                index=0,
                message=Message(role="assistant", content=response, refusal=None, function_call=None, reasoning_content=None, tool_calls=None),
                finish_reason="stop"
            )]
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
            choices=[Choice(index=0, message=Message(role="assistant", content=response_content, reasoning_content=reasoning_content), finish_reason="stop")]
        )
    for idx, tool_call in enumerate(tool_calls):
        arguments = tool_call.get("arguments")
        # If arguments is already a string, use it directly; otherwise serialize it
        if isinstance(arguments, str):
            arguments_str = arguments
        else:
            arguments_str = json.dumps(arguments)
        function_call = FunctionCall(
            name=tool_call.get("name"),
            arguments=arguments_str
        )
        tool_call_response = ChatCompletionMessageToolCall(
            id=get_tool_call_id(),
            type="function",
            function=function_call,
            index=idx
        )
        tool_call_responses.append(tool_call_response)
    
    message = Message(role="assistant", content=response_content, reasoning_content=reasoning_content, tool_calls=tool_call_responses)
    
    return ChatCompletionResponse(
        id=get_id(),
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[Choice(
            index=0,
            message=message,
            finish_reason="tool_calls"
        )]
    )