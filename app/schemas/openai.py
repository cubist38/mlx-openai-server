"""OpenAI-compatible API schemas and models."""

from __future__ import annotations

from enum import Enum
import random
from typing import Any, ClassVar, Literal

from fastapi import UploadFile
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OpenAIBaseModel(BaseModel):
    """Base model for OpenAI-compatible API models with extra field logging."""

    # OpenAI API does allow extra fields
    model_config = ConfigDict(extra="allow")

    # Cache class field names
    field_names: ClassVar[set[str] | None] = None

    @model_validator(mode="wrap")
    @classmethod
    def __log_extra_fields__(cls, data: Any, handler: Any) -> Any:
        """Log any extra fields present in the request data."""
        result = handler(data)
        if not isinstance(data, dict):
            return result
        field_names = cls.field_names
        if field_names is None:
            # Get all class field names and their potential aliases
            field_names = set()
            for field_name, field in cls.model_fields.items():
                field_names.add(field_name)
                if alias := getattr(field, "alias", None):
                    field_names.add(alias)
            cls.field_names = field_names

        # Compare against both field names and aliases
        if any(k not in field_names for k in data):
            logger.warning(
                f"The following fields were present in the request but ignored: {data.keys() - field_names}"
            )
        return result


# Configuration
class Config:
    """Configuration class holding the default model names for different types of requests."""

    TEXT_MODEL = "local-text-model"  # Default model for text-based chat completions
    MULTIMODAL_MODEL = "local-multimodal-model"  # Model used for multimodal requests
    EMBEDDING_MODEL = "local-embedding-model"  # Model used for generating embeddings
    IMAGE_GENERATION_MODEL = "local-image-generation-model"
    IMAGE_EDIT_MODEL = "local-image-edit-model"
    TRANSCRIPTION_MODEL = "local-transcription-model"


class HealthCheckStatus(str, Enum):
    """Health check status."""

    OK = "ok"


class HealthCheckResponse(OpenAIBaseModel):
    """Response model for health check endpoint."""

    status: HealthCheckStatus = Field(..., description="The status of the health check.")
    model_id: str | None = Field(None, description="ID of the loaded model, if any.")
    model_status: str | None = Field(
        None, description="Status of the model handler (initialized/uninitialized)."
    )


class ErrorResponse(OpenAIBaseModel):
    """Response model for error responses."""

    object: str = Field("error", description="The object type, always 'error'.")
    message: str = Field(..., description="The error message.")
    type: str = Field(..., description="The type of error.")
    param: str | None = Field(None, description="The parameter related to the error, if any.")
    code: int = Field(..., description="The error code.")


# Common models used in both streaming and non-streaming contexts
class ImageURL(OpenAIBaseModel):
    """Represents an image URL or base64 encoded image data."""

    url: str = Field(..., description="Either a URL of the image or the base64 encoded image data.")


class ChatCompletionContentPartImage(OpenAIBaseModel):
    """Represents an image content part in a chat completion message."""

    image_url: ImageURL | None = Field(
        None, description="Either a URL of the image or the base64 encoded image data."
    )
    type: Literal["image_url"] = Field(..., description="The type of content, e.g., 'image_url'.")


class VideoURL(OpenAIBaseModel):
    """Represents a video URL or base64 encoded video data."""

    url: str = Field(..., description="Either a URL of the video or the base64 encoded video data.")


class ChatCompletionContentPartVideo(OpenAIBaseModel):
    """Represents a video content part in a chat completion message."""

    video_url: VideoURL | None = Field(
        None, description="Either a URL of the video or the base64 encoded video data."
    )
    type: Literal["video_url"] = Field(..., description="The type of content, e.g., 'video_url'.")


class InputAudio(OpenAIBaseModel):
    """Represents input audio data."""

    data: str = Field(
        ..., description="Either a URL of the audio or the base64 encoded audio data."
    )
    format: Literal["mp3", "wav"] = Field(..., description="The audio format.")


class ChatCompletionContentPartInputAudio(OpenAIBaseModel):
    """Represents an input audio content part in a chat completion message."""

    input_audio: InputAudio | None = Field(
        None, description="Either a URL of the audio or the base64 encoded audio data."
    )
    type: Literal["input_audio"] = Field(
        ..., description="The type of content, e.g., 'input_audio'."
    )


class ChatCompletionContentPartText(OpenAIBaseModel):
    """Represents a text content part in a chat completion message."""

    text: str = Field(..., description="The text content.")
    type: Literal["text"] = Field(..., description="The type of content, e.g., 'text'.")


ChatCompletionContentPart = (
    ChatCompletionContentPartImage
    | ChatCompletionContentPartVideo
    | ChatCompletionContentPartInputAudio
    | ChatCompletionContentPartText
)


class PromptTokenUsageInfo(OpenAIBaseModel):
    """Represents detailed information about prompt token usage."""

    cached_tokens: int | None = None


class UsageInfo(OpenAIBaseModel):
    """Represents token usage information for a request."""

    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = 0
    prompt_tokens_details: PromptTokenUsageInfo | None = None


class FunctionCall(OpenAIBaseModel):
    """Represents a function call in a message."""

    arguments: str = Field(..., description="The arguments for the function call.")
    name: str = Field(..., description="The name of the function to call.")


class ChatCompletionMessageToolCall(OpenAIBaseModel):
    """Represents a tool call in a message."""

    id: str | None = Field(None, description="The ID of the tool call.")
    function: FunctionCall = Field(..., description="The function call details.")
    type: Literal["function"] = Field(..., description="The type of tool call, always 'function'.")
    index: int | None = Field(None, description="The index of the tool call.")


class Message(OpenAIBaseModel):
    """Represents a message in a chat completion."""

    content: str | list[ChatCompletionContentPart] | None = Field(
        None,
        description="The content of the message, either text or a list of content items (vision, audio, or multimodal).",
    )
    refusal: str | None = Field(None, description="The refusal reason, if any.")
    role: Literal["system", "user", "assistant", "tool"] = Field(
        ..., description="The role of the message sender."
    )
    reasoning_content: str | None = Field(None, description="The reasoning content, if any.")
    tool_calls: list[ChatCompletionMessageToolCall] | None = Field(
        None, description="List of tool calls, if any."
    )
    tool_call_id: str | None = Field(None, description="The ID of the tool call, if any.")


# Common request base for both streaming and non-streaming
class ChatCompletionRequestBase(OpenAIBaseModel):
    """Base model for chat completion requests."""

    model: str = Field(Config.TEXT_MODEL, description="The model to use for completion.")
    messages: list[Message] = Field(..., description="The list of messages in the conversation.")
    tools: list[dict[str, Any]] | None = Field(
        None, description="List of tools available for the request."
    )
    tool_choice: str | dict[str, Any] | None = Field(
        "auto", description="Tool choice for the request."
    )
    max_tokens: int | None = Field(None, description="The maximum number of tokens to generate.")
    temperature: float | None = Field(0.7, description="Sampling temperature.")
    top_p: float | None = Field(1.0, description="Nucleus sampling probability.")
    top_k: int | None = Field(20, description="Top-k sampling parameter.")
    min_p: float | None = Field(0.0, description="Minimum probability for token generation.")
    frequency_penalty: float | None = Field(
        0.0, description="Frequency penalty for token generation."
    )
    presence_penalty: float | None = Field(
        0.0, description="Presence penalty for token generation."
    )
    stop: list[str] | None = Field(None, description="List of stop sequences.")
    n: int | None = Field(1, description="Number of completions to generate.")
    response_format: dict[str, Any] | None = Field(None, description="Format for the response.")
    seed: int = Field(
        42, description="Seed for reproducibility.",
    )
    user: str | None = Field(None, description="User identifier.")
    repetition_penalty: float | None = Field(
        1.05, description="Repetition penalty for token generation."
    )
    repetition_context_size: int | None = Field(
        20, description="Repetition context size for token generation."
    )

    @field_validator("messages")
    @classmethod
    def check_messages_not_empty(cls, v: list[Message]) -> list[Message]:
        """Ensure that the messages list is not empty and validate message structure."""
        if not v:
            raise ValueError("messages cannot be empty")

        # Validate message history length
        if len(v) > 100:  # OpenAI's limit is typically around 100 messages
            raise ValueError("message history too long")

        # Validate message roles
        valid_roles = {"user", "assistant", "system", "tool"}
        for msg in v:
            if msg.role not in valid_roles:
                raise ValueError(f"invalid role: {msg.role}")

        return v

    @field_validator("temperature")
    @classmethod
    def check_temperature(cls, v: float | None) -> float | None:
        """Validate temperature is between 0 and 2."""
        if v is not None and (v < 0 or v > 2):
            raise ValueError("temperature must be between 0 and 2")
        return v

    @field_validator("max_tokens")
    @classmethod
    def check_max_tokens(cls, v: int | None) -> int | None:
        """Validate max_tokens is positive and within reasonable limits."""
        if v is not None:
            if v <= 0:
                raise ValueError("max_tokens must be positive")
        return v


class ChatTemplateKwargs(OpenAIBaseModel):
    """Represents the arguments for a chat template."""

    enable_thinking: bool = Field(default=True, description="Whether to enable thinking.")
    reasoning_effort: Literal["low", "medium", "high"] = Field(
        default="medium", description="The reasoning effort level."
    )


# Non-streaming request and response
class ChatCompletionRequest(ChatCompletionRequestBase):
    """Model for non-streaming chat completion requests."""

    stream: bool = Field(False, description="Whether to stream the response.")
    chat_template_kwargs: ChatTemplateKwargs = Field(
        default_factory=ChatTemplateKwargs, description="Arguments for the chat template."
    )


class Choice(OpenAIBaseModel):
    """Represents a choice in a chat completion response."""

    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"] = (
        Field(..., description="The reason for the choice.")
    )
    index: int = Field(..., description="The index of the choice.")
    message: Message = Field(..., description="The message of the choice.")


class ChatCompletionResponse(OpenAIBaseModel):
    """Represents a complete chat completion response."""

    id: str = Field(..., description="The response ID.")
    object: Literal["chat.completion"] = Field(
        ..., description="The object type, always 'chat.completion'."
    )
    created: int = Field(..., description="The creation timestamp.")
    model: str = Field(..., description="The model used for completion.")
    choices: list[Choice] = Field(..., description="List of choices in the response.")
    usage: UsageInfo | None = Field(default=None, description="The usage of the completion.")
    request_id: str | None = Field(None, description="Request correlation ID for tracking.")


class ChoiceDeltaFunctionCall(OpenAIBaseModel):
    """Represents a function call delta in a streaming response."""

    arguments: str | None = Field(None, description="Arguments for the function call delta.")
    name: str | None = Field(None, description="Name of the function in the delta.")


class ChoiceDeltaToolCall(OpenAIBaseModel):
    """Represents a tool call delta in a streaming response."""

    index: int | None = Field(None, description="Index of the tool call delta.")
    id: str | None = Field(None, description="ID of the tool call delta.")
    function: ChoiceDeltaFunctionCall | None = Field(
        None, description="Function call details in the delta."
    )
    type: str | None = Field(None, description="Type of the tool call delta.")


class Delta(OpenAIBaseModel):
    """Represents a delta in a streaming response."""

    content: str | None = Field(None, description="Content of the delta.")
    function_call: ChoiceDeltaFunctionCall | None = Field(
        None, description="Function call delta, if any."
    )
    refusal: str | None = Field(None, description="Refusal reason, if any.")
    role: Literal["system", "user", "assistant", "tool"] | None = Field(
        None, description="Role in the delta."
    )
    tool_calls: list[ChoiceDeltaToolCall] | None = Field(
        None, description="List of tool call deltas, if any."
    )
    reasoning_content: str | None = Field(None, description="Reasoning content, if any.")


class StreamingChoice(OpenAIBaseModel):
    """Represents a choice in a streaming response."""

    delta: Delta | None = Field(None, description="The delta for this streaming choice.")
    finish_reason: (
        Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | None
    ) = Field(None, description="The reason for finishing, if any.")
    index: int = Field(..., description="The index of the streaming choice.")


class ChatCompletionChunk(OpenAIBaseModel):
    """Represents a chunk in a streaming chat completion response."""

    id: str = Field(..., description="The chunk ID.")
    choices: list[StreamingChoice] = Field(
        ..., description="List of streaming choices in the chunk."
    )
    created: int = Field(..., description="The creation timestamp of the chunk.")
    model: str = Field(..., description="The model used for the chunk.")
    object: Literal["chat.completion.chunk"] = Field(
        ..., description="The object type, always 'chat.completion.chunk'."
    )
    usage: UsageInfo | None = Field(default=None, description="The usage of the chunk.")
    request_id: str | None = Field(None, description="Request correlation ID for tracking.")


# Embedding models
class EmbeddingRequest(OpenAIBaseModel):
    """Model for embedding requests."""

    model: str = Field(Config.EMBEDDING_MODEL, description="The embedding model to use.")
    input: list[str] | str = Field(
        ..., description="List of text inputs for embedding or the image file to embed."
    )
    image_url: str | None = Field(default=None, description="Image URL to embed.")
    user: str | None = Field(default=None, description="User identifier.")
    encoding_format: Literal["float", "base64"] = Field(
        default="float", description="The encoding format for the embedding."
    )


class EmbeddingResponseData(OpenAIBaseModel):
    """Represents an embedding object in an embedding response."""

    embedding: list[float] | str = Field(
        ..., description="The embedding vector or the base64 encoded embedding."
    )
    index: int = Field(..., description="The index of the embedding in the list.")
    object: str = Field(default="embedding", description="The object type, always 'embedding'.")


class EmbeddingResponse(OpenAIBaseModel):
    """Represents an embedding response."""

    object: str = Field("list", description="The object type, always 'list'.")
    data: list[EmbeddingResponseData] = Field(..., description="List of embedding objects.")
    model: str = Field(..., description="The model used for embedding.")
    usage: UsageInfo | None = Field(default=None, description="The usage of the embedding.")


class Model(OpenAIBaseModel):
    """Represents a model in the models list response."""

    id: str = Field(..., description="The model ID.")
    object: str = Field("model", description="The object type, always 'model'.")
    created: int = Field(..., description="The creation timestamp.")
    owned_by: str = Field("openai", description="The owner of the model.")
    metadata: dict[str, Any] | None = Field(None, description="Additional model metadata.")


class ModelsResponse(OpenAIBaseModel):
    """Represents the response for the models list endpoint."""

    object: str = Field("list", description="The object type, always 'list'.")
    data: list[Model] = Field(..., description="List of models.")


class ImageSize(str, Enum):
    """Available image sizes."""

    SMALL = "256x256"
    MEDIUM = "512x512"
    LARGE = "1024x1024"
    # COSMOS_SIZE = "1024x1024"


class Priority(str, Enum):
    """Task priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class ImageEditQuality(str, Enum):
    """Image edit quality levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ImageResponseFormat(str, Enum):
    """Image edit response format."""

    # Only support b64_json for now
    B64_JSON = "b64_json"


class TranscriptionResponseFormat(str, Enum):
    """Audio response format."""

    JSON = "json"
    TEXT = "text"


class ImageGenerationRequest(OpenAIBaseModel):
    """Request schema for OpenAI-compatible image generation API."""

    prompt: str = Field(
        ...,
        description="A text description of the desired image(s). The maximum length is 1000 characters.",
        max_length=1000,
    )
    negative_prompt: str | None = Field(
        None,
        description="A text description of the desired image(s). The maximum length is 1000 characters.",
        max_length=1000,
    )
    model: str | None = Field(
        default=Config.IMAGE_GENERATION_MODEL, description="The model to use for image generation"
    )
    size: ImageSize | None = Field(
        default=ImageSize.LARGE, description="The size of the generated images"
    )
    guidance_scale: float | None = Field(
        default=4.5, description="The guidance scale for the image generation"
    )
    steps: int | None = Field(
        default=4, ge=1, le=50, description="The number of inference steps (1-50)"
    )
    seed: int | None = Field(42, description="Seed for reproducible generation")
    response_format: ImageResponseFormat | None = Field(
        default=ImageResponseFormat.B64_JSON,
        description="The format in which the generated images are returned",
    )


class ImageData(OpenAIBaseModel):
    """Individual image data in the response."""

    url: str | None = Field(
        None, description="The URL of the generated image, if response_format is url"
    )
    b64_json: str | None = Field(
        None,
        description="The base64-encoded JSON of the generated image, if response_format is b64_json",
    )


class ImageGenerationResponse(OpenAIBaseModel):
    """Response schema for OpenAI-compatible image generation API."""

    created: int = Field(
        ..., description="The Unix timestamp (in seconds) when the image was created"
    )
    data: list[ImageData] = Field(..., description="List of generated images")


class ImageGenerationError(OpenAIBaseModel):
    """Error response schema."""

    code: str = Field(
        ..., description="Error code (e.g., 'contentFilter', 'generation_error', 'queue_full')"
    )
    message: str = Field(..., description="Human-readable error message")
    type: str | None = Field(None, description="Error type")


class ImageEditRequest(OpenAIBaseModel):
    """Request data for OpenAI-compatible image edit API."""

    image: UploadFile | list[UploadFile] = Field(..., description="The image(s) to edit. Must be a file upload or a list of file uploads")
    prompt: str = Field(..., description="The prompt for the image edit")
    model: str | None = Field(
        default=Config.IMAGE_EDIT_MODEL, description="The model to use for image edit"
    )
    negative_prompt: str | None = Field(None, description="The negative prompt for the image edit")
    guidance_scale: float | None = Field(
        default=2.5, description="The guidance scale for the image edit"
    )
    response_format: ImageResponseFormat | None = Field(
        default=ImageResponseFormat.B64_JSON,
        description="The format in which the edited image is returned",
    )
    seed: int | None = Field(default=42, description="The seed for the image edit")
    size: ImageSize | None = Field(None, description="The size of the edited image")
    steps: int | None = Field(
        default=28, description="The number of inference steps for the image edit"
    )


class ImageEditResponse(OpenAIBaseModel):
    """Response schema for OpenAI-compatible image edit API."""

    created: int = Field(
        ..., description="The Unix timestamp (in seconds) when the image was edited"
    )
    data: list[ImageData] = Field(..., description="List of edited images")


class TranscriptionRequest(OpenAIBaseModel):
    """Request schema for OpenAI-compatible transcription API."""

    file: UploadFile = Field(..., description="The audio file to transcribe")
    model: str | None = Field(
        default=Config.TRANSCRIPTION_MODEL, description="The model to use for transcription"
    )
    language: str | None = Field(None, description="The language of the audio file")
    prompt: str | None = Field(None, description="The prompt for the transcription")
    response_format: TranscriptionResponseFormat | None = Field(
        default=TranscriptionResponseFormat.JSON,
        description="The format in which the transcription is returned",
    )
    stream: bool | None = Field(default=False, description="Whether to stream the transcription")
    temperature: float | None = Field(
        default=0.0, description="The temperature for the transcription"
    )
    top_p: float | None = Field(default=None, description="The top-p for the transcription")
    top_k: int | None = Field(default=None, description="The top-k for the transcription")
    min_p: float | None = Field(default=None, description="The min-p for the transcription")
    seed: int | None = Field(default=None, description="The seed for the transcription")
    frequency_penalty: float | None = Field(
        default=None, description="The frequency penalty for the transcription"
    )
    repetition_penalty: float | None = Field(
        default=None, description="The repetition penalty for the transcription"
    )
    presence_penalty: float | None = Field(
        default=None, description="Presence penalty for token generation"
    )


# Transcription response objects
class TranscriptionUsageAudio(OpenAIBaseModel):
    """Represents audio usage information for transcription."""

    type: Literal["duration"] = Field(..., description="The type of usage, always 'duration'")
    seconds: int = Field(..., description="The duration of the audio in seconds")


class TranscriptionResponse(OpenAIBaseModel):
    """Represents a transcription response."""

    text: str = Field(..., description="The transcribed text.")
    usage: TranscriptionUsageAudio = Field(..., description="The usage of the transcription.")


class TranscriptionResponseStreamChoice(OpenAIBaseModel):
    """Represents a choice in a streaming transcription response."""

    delta: Delta = Field(..., description="The delta for this streaming choice.")
    finish_reason: str | None = None
    stop_reason: int | str | None = None


class TranscriptionResponseStream(OpenAIBaseModel):
    """Represents a streaming transcription response."""

    id: str = Field(..., description="The ID of the transcription.")
    object: Literal["transcription.chunk"] = Field(
        ..., description="The object type, always 'transcription.chunk'."
    )
    created: int = Field(..., description="The creation timestamp of the chunk.")
    model: str = Field(..., description="The model used for the transcription.")
    choices: list[TranscriptionResponseStreamChoice] = Field(
        ..., description="The choices for this streaming response."
    )
    usage: TranscriptionUsageAudio | None = Field(
        default=None, description="The usage of the transcription."
    )