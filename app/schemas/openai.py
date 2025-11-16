import random
from enum import Enum
from app.core.queue import T
from fastapi import UploadFile

from typing import ClassVar, Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator, ConfigDict,model_validator
from typing_extensions import Literal, TypeAlias
from loguru import logger


class OpenAIBaseModel(BaseModel):
    # OpenAI API does allow extra fields
    model_config = ConfigDict(extra="allow")

    # Cache class field names
    field_names: ClassVar[Optional[set[str]]] = None

    @model_validator(mode="wrap")
    @classmethod
    def __log_extra_fields__(cls, data, handler):
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
                "The following fields were present in the request "
                "but ignored: %s",
                data.keys() - field_names,
            )
        return result

# Configuration
class Config:
    """
    Configuration class holding the default model names for different types of requests.
    """
    TEXT_MODEL = "local-text-model"          # Default model for text-based chat completions
    MULTIMODAL_MODEL = "local-multimodal-model"  # Model used for multimodal requests
    EMBEDDING_MODEL = "local-embedding-model"  # Model used for generating embeddings
    IMAGE_GENERATION_MODEL = "local-image-generation-model"
    IMAGE_EDIT_MODEL = "local-image-edit-model"
    TRANSCRIPTION_MODEL="local-transcription-model"

class HealthCheckStatus(str, Enum):
    """Health check status"""
    OK = "ok"

class HealthCheckResponse(OpenAIBaseModel):
    status: HealthCheckStatus = Field(..., description="The status of the health check.")

class ErrorResponse(OpenAIBaseModel):
    object: str = Field("error", description="The object type, always 'error'.")
    message: str = Field(..., description="The error message.")
    type: str = Field(..., description="The type of error.")
    param: Optional[str] = Field(None, description="The parameter related to the error, if any.")
    code: int = Field(..., description="The error code.")

# Common models used in both streaming and non-streaming contexts
class ImageURL(OpenAIBaseModel):
    url: str = Field(..., description="Either a URL of the image or the base64 encoded image data.")

class ChatCompletionContentPartImage(OpenAIBaseModel):
    image_url: Optional[ImageURL] = Field(None, description="Either a URL of the image or the base64 encoded image data.")
    type: Literal["image_url"] = Field(..., description="The type of content, e.g., 'image_url'.")

class VideoURL(OpenAIBaseModel):
    url: str = Field(..., description="Either a URL of the video or the base64 encoded video data.")

class ChatCompletionContentPartVideo(OpenAIBaseModel):
    video_url: Optional[VideoURL] = Field(None, description="Either a URL of the video or the base64 encoded video data.")
    type: Literal["video_url"] = Field(..., description="The type of content, e.g., 'video_url'.")

class InputAudio(OpenAIBaseModel):
    data: str = Field(..., description="Either a URL of the audio or the base64 encoded audio data.")
    format: Literal["mp3", "wav"] = Field(..., description="The audio format.")

class ChatCompletionContentPartInputAudio(OpenAIBaseModel):
    input_audio: Optional[InputAudio] = Field(None, description="Either a URL of the audio or the base64 encoded audio data.")
    type: Literal["input_audio"] = Field(..., description="The type of content, e.g., 'input_audio'.")

class ChatCompletionContentPartText(OpenAIBaseModel):
    text: str = Field(..., description="The text content.")
    type: Literal["text"] = Field(..., description="The type of content, e.g., 'text'.")

ChatCompletionContentPart: TypeAlias = Union[ChatCompletionContentPartImage, ChatCompletionContentPartVideo, ChatCompletionContentPartInputAudio, ChatCompletionContentPartText]

class PromptTokenUsageInfo(OpenAIBaseModel):
    cached_tokens: Optional[int] = None

class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0
    prompt_tokens_details: Optional[PromptTokenUsageInfo] = None

class FunctionCall(OpenAIBaseModel):
    """
    Represents a function call in a message.
    """
    arguments: str = Field(..., description="The arguments for the function call.")
    name: str = Field(..., description="The name of the function to call.")

class ChatCompletionMessageToolCall(OpenAIBaseModel):
    """
    Represents a tool call in a message.
    """
    id: Optional[str] = Field(None, description="The ID of the tool call.")
    function: FunctionCall = Field(..., description="The function call details.")
    type: Literal["function"] = Field(..., description="The type of tool call, always 'function'.")
    index: Optional[int] = Field(None, description="The index of the tool call.")

class Message(OpenAIBaseModel):
    """
    Represents a message in a chat completion.
    """
    content: Optional[Union[str, List[ChatCompletionContentPart]]] = Field(None, description="The content of the message, either text or a list of content items (vision, audio, or multimodal).")
    refusal: Optional[str] = Field(None, description="The refusal reason, if any.")
    role: Literal["system", "user", "assistant", "tool"] = Field(..., description="The role of the message sender.")
    reasoning_content: Optional[str] = Field(None, description="The reasoning content, if any.")
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = Field(None, description="List of tool calls, if any.")
    tool_call_id: Optional[str] = Field(None, description="The ID of the tool call, if any.")

# Common request base for both streaming and non-streaming
class ChatCompletionRequestBase(OpenAIBaseModel):
    """
    Base model for chat completion requests.
    """
    model: str = Field(Config.TEXT_MODEL, description="The model to use for completion.")
    messages: List[Message] = Field(..., description="The list of messages in the conversation.")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="List of tools available for the request.")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field("auto", description="Tool choice for the request.")
    max_tokens: Optional[int] = Field(None, description="The maximum number of tokens to generate.")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature.")
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling probability.")
    top_k: Optional[int] = Field(20, description="Top-k sampling parameter.")
    min_p: Optional[float] = Field(0.0, description="Minimum probability for token generation.")
    frequency_penalty: Optional[float] = Field(0.0, description="Frequency penalty for token generation.")
    presence_penalty: Optional[float] = Field(0.0, description="Presence penalty for token generation.")
    stop: Optional[List[str]] = Field(None, description="List of stop sequences.")
    n: Optional[int] = Field(1, description="Number of completions to generate.")
    response_format: Optional[Dict[str, Any]] = Field(None, description="Format for the response.")
    seed: Optional[int] = Field(random.randint(0, 1000000), description="Random seed for reproducibility.")
    user: Optional[str] = Field(None, description="User identifier.")
    repetition_penalty: Optional[float] = Field(1.05, description="Repetition penalty for token generation.")
    repetition_context_size: Optional[int] = Field(20, description="Repetition context size for token generation.")

    @validator("messages")
    def check_messages_not_empty(cls, v):
        """
        Ensure that the messages list is not empty and validate message structure.
        """
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

    @validator("temperature")
    def check_temperature(cls, v):
        """
        Validate temperature is between 0 and 2.
        """
        if v is not None and (v < 0 or v > 2):
            raise ValueError("temperature must be between 0 and 2")
        return v

    @validator("max_tokens")
    def check_max_tokens(cls, v):
        """
        Validate max_tokens is positive and within reasonable limits.
        """
        if v is not None:
            if v <= 0:
                raise ValueError("max_tokens must be positive")
        return v
    
class ChatTemplateKwargs(OpenAIBaseModel):
    """
    Represents the arguments for a chat template.
    """
    enable_thinking: bool = Field(True, description="Whether to enable thinking.")
    reasoning_effot: Literal["low", "medium", "high"] = Field("medium", description="The reasoning effort level.")

# Non-streaming request and response
class ChatCompletionRequest(ChatCompletionRequestBase):
    """
    Model for non-streaming chat completion requests.
    """
    stream: bool = Field(False, description="Whether to stream the response.")
    chat_template_kwargs: ChatTemplateKwargs = Field(ChatTemplateKwargs(), description="Arguments for the chat template.")

class Choice(OpenAIBaseModel):
    """
    Represents a choice in a chat completion response.
    """
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"] = Field(..., description="The reason for the choice.")
    index: int = Field(..., description="The index of the choice.")
    message: Message = Field(..., description="The message of the choice.")

class ChatCompletionResponse(OpenAIBaseModel):
    """
    Represents a complete chat completion response.
    """
    id: str = Field(..., description="The response ID.")
    object: Literal["chat.completion"] = Field(..., description="The object type, always 'chat.completion'.")
    created: int = Field(..., description="The creation timestamp.")
    model: str = Field(..., description="The model used for completion.")
    choices: List[Choice] = Field(..., description="List of choices in the response.")
    usage: Optional[UsageInfo] = Field(default=None, description="The usage of the completion.")
    request_id: Optional[str] = Field(None, description="Request correlation ID for tracking.")

class ChoiceDeltaFunctionCall(OpenAIBaseModel):
    """
    Represents a function call delta in a streaming response.
    """
    arguments: Optional[str] = Field(None, description="Arguments for the function call delta.")
    name: Optional[str] = Field(None, description="Name of the function in the delta.")

class ChoiceDeltaToolCall(OpenAIBaseModel):
    """
    Represents a tool call delta in a streaming response.
    """
    index: Optional[int] = Field(None, description="Index of the tool call delta.")
    id: Optional[str] = Field(None, description="ID of the tool call delta.")
    function: Optional[ChoiceDeltaFunctionCall] = Field(None, description="Function call details in the delta.")
    type: Optional[str] = Field(None, description="Type of the tool call delta.")

class Delta(OpenAIBaseModel):
    """
    Represents a delta in a streaming response.
    """
    content: Optional[str] = Field(None, description="Content of the delta.")
    function_call: Optional[ChoiceDeltaFunctionCall] = Field(None, description="Function call delta, if any.")
    refusal: Optional[str] = Field(None, description="Refusal reason, if any.")
    role: Optional[Literal["system", "user", "assistant", "tool"]] = Field(None, description="Role in the delta.")
    tool_calls: Optional[List[ChoiceDeltaToolCall]] = Field(None, description="List of tool call deltas, if any.")
    reasoning_content: Optional[str] = Field(None, description="Reasoning content, if any.")

class StreamingChoice(OpenAIBaseModel):
    """
    Represents a choice in a streaming response.
    """
    delta: Optional[Delta] = Field(None, description="The delta for this streaming choice.")
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = Field(None, description="The reason for finishing, if any.")
    index: int = Field(..., description="The index of the streaming choice.")
    
class ChatCompletionChunk(OpenAIBaseModel):
    """
    Represents a chunk in a streaming chat completion response.
    """
    id: str = Field(..., description="The chunk ID.")
    choices: List[StreamingChoice] = Field(..., description="List of streaming choices in the chunk.")
    created: int = Field(..., description="The creation timestamp of the chunk.")
    model: str = Field(..., description="The model used for the chunk.")
    object: Literal["chat.completion.chunk"] = Field(..., description="The object type, always 'chat.completion.chunk'.")
    usage: Optional[UsageInfo] = Field(default=None, description="The usage of the chunk.")
    request_id: Optional[str] = Field(None, description="Request correlation ID for tracking.")

# Embedding models
class EmbeddingRequest(OpenAIBaseModel):
    """
    Model for embedding requests.
    """
    model: str = Field(Config.EMBEDDING_MODEL, description="The embedding model to use.")
    input: Union[List[str], str] = Field(..., description="List of text inputs for embedding or the image file to embed.")
    image_url: Optional[str] = Field(default=None, description="Image URL to embed.")
    user: Optional[str] = Field(default=None, description="User identifier.")
    encoding_format: Literal["float", "base64"] = Field(default="float", description="The encoding format for the embedding.")

class EmbeddingResponseData(OpenAIBaseModel):
    """
    Represents an embedding object in an embedding response.
    """
    embedding: Union[List[float], str] = Field(..., description="The embedding vector or the base64 encoded embedding.")
    index: int = Field(..., description="The index of the embedding in the list.")
    object: str = Field(default="embedding", description="The object type, always 'embedding'.")

class EmbeddingResponse(OpenAIBaseModel):
    """
    Represents an embedding response.
    """
    object: str = Field("list", description="The object type, always 'list'.")
    data: List[EmbeddingResponseData] = Field(..., description="List of embedding objects.")
    model: str = Field(..., description="The model used for embedding.")
    usage: Optional[UsageInfo] = Field(default=None, description="The usage of the embedding.")

class Model(OpenAIBaseModel):
    """
    Represents a model in the models list response.
    """
    id: str = Field(..., description="The model ID.")
    object: str = Field("model", description="The object type, always 'model'.")
    created: int = Field(..., description="The creation timestamp.")
    owned_by: str = Field("openai", description="The owner of the model.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional model metadata.")

class ModelsResponse(OpenAIBaseModel):
    """
    Represents the response for the models list endpoint.
    """
    object: str = Field("list", description="The object type, always 'list'.")
    data: List[Model] = Field(..., description="List of models.")


class ImageSize(str, Enum):
    """Available image sizes"""
    SMALL = "256x256"
    MEDIUM = "512x512"
    LARGE = "1024x1024"
    COSMOS_SIZE = "1024x1024"

class Priority(str, Enum):
    """Task priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

class ImageEditQuality(str, Enum):
    """Image edit quality levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ImageResponseFormat(str, Enum):
    """Image edit response format"""

    # Only support b64_json for now
    B64_JSON = "b64_json" 

class TranscriptionResponseFormat(str, Enum):
    """Audio response format"""
    JSON = "json"
    TEXT = "text"

class ImageGenerationRequest(OpenAIBaseModel):
    """Request schema for OpenAI-compatible image generation API"""
    prompt: str = Field(..., description="A text description of the desired image(s). The maximum length is 1000 characters.", max_length=1000)
    negative_prompt: Optional[str] = Field(None, description="A text description of the desired image(s). The maximum length is 1000 characters.", max_length=1000)
    model: Optional[str] = Field(default=Config.IMAGE_GENERATION_MODEL, description="The model to use for image generation")
    size: Optional[ImageSize] = Field(default=ImageSize.LARGE, description="The size of the generated images")
    guidance_scale: Optional[float] = Field(default=4.5, description="The guidance scale for the image generation")
    steps: Optional[int] = Field(default=28, ge=1, le=50, description="The number of inference steps (1-50)")
    seed: Optional[int] = Field(42, description="Seed for reproducible generation")
    response_format: Optional[ImageResponseFormat] = Field(default=ImageResponseFormat.B64_JSON, description="The format in which the generated images are returned")

class ImageData(OpenAIBaseModel):
    """Individual image data in the response"""
    url: Optional[str] = Field(None, description="The URL of the generated image, if response_format is url")
    b64_json: Optional[str] = Field(None, description="The base64-encoded JSON of the generated image, if response_format is b64_json")

class ImageGenerationResponse(OpenAIBaseModel):
    """Response schema for OpenAI-compatible image generation API"""
    created: int = Field(..., description="The Unix timestamp (in seconds) when the image was created")
    data: List[ImageData] = Field(..., description="List of generated images")

class ImageGenerationError(OpenAIBaseModel):
    """Error response schema"""
    code: str = Field(..., description="Error code (e.g., 'contentFilter', 'generation_error', 'queue_full')")
    message: str = Field(..., description="Human-readable error message")
    type: Optional[str] = Field(None, description="Error type")

class ImageEditRequest(OpenAIBaseModel):
    """Request data for OpenAI-compatible image edit API"""
    image: UploadFile = Field(..., description="The image to edit")
    prompt: str = Field(..., description="The prompt for the image edit")
    model: Optional[str] = Field(default=Config.IMAGE_EDIT_MODEL, description="The model to use for image edit")
    negative_prompt: Optional[str] = Field(None, description="The negative prompt for the image edit")
    guidance_scale: Optional[float] = Field(default=2.5, description="The guidance scale for the image edit")
    response_format: Optional[ImageResponseFormat] = Field(default=ImageResponseFormat.B64_JSON, description="The format in which the edited image is returned")
    seed: Optional[int] = Field(default=42, description="The seed for the image edit")
    size: Optional[ImageSize] = Field(None, description="The size of the edited image")
    steps: Optional[int] = Field(default=28, description="The number of inference steps for the image edit")

class ImageEditResponse(OpenAIBaseModel):
    """Response schema for OpenAI-compatible image edit API"""
    created: int = Field(..., description="The Unix timestamp (in seconds) when the image was edited")
    data: List[ImageData] = Field(..., description="List of edited images")

class TranscriptionRequest(OpenAIBaseModel):
    """Request schema for OpenAI-compatible transcription API"""
    file: UploadFile = Field(..., description="The audio file to transcribe")
    model: Optional[str] = Field(default=Config.TRANSCRIPTION_MODEL, description="The model to use for transcription")
    language: Optional[str] = Field(None, description="The language of the audio file")
    prompt: Optional[str] = Field(None, description="The prompt for the transcription")
    response_format: Optional[TranscriptionResponseFormat] = Field(default=TranscriptionResponseFormat.JSON, description="The format in which the transcription is returned")
    stream: Optional[bool] = Field(default=False, description="Whether to stream the transcription")
    temperature: Optional[float] = Field(default=0.0, description="The temperature for the transcription")
    top_p: Optional[float] = Field(default=None, description="The top-p for the transcription")
    top_k: Optional[int] = Field(default=None, description="The top-k for the transcription")
    min_p: Optional[float] = Field(default=None, description="The min-p for the transcription")
    seed: Optional[int] = Field(default=None, description="The seed for the transcription")
    frequency_penalty: Optional[float] = Field(default=None, description="The frequency penalty for the transcription")
    repetition_penalty: Optional[float] = Field(default=None, description="The repetition penalty for the transcription")
    presence_penalty: Optional[int] = Field(default=None, description="The repetition context size for the transcription")

# Transcription response objects
class TranscriptionUsageAudio(OpenAIBaseModel):
    type: Literal["duration"] = Field(..., description="The type of usage, always 'duration'")
    seconds: int = Field(..., description="The duration of the audio in seconds")

class TranscriptionResponse(OpenAIBaseModel):
    text: str = Field(..., description="The transcribed text.")
    usage: TranscriptionUsageAudio = Field(..., description="The usage of the transcription.")

class TranscriptionResponseStreamChoice(OpenAIBaseModel):
    delta: Delta = Field(..., description="The delta for this streaming choice.")
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = None

class TranscriptionResponseStream(OpenAIBaseModel):
    id: str = Field(..., description="The ID of the transcription.")
    object: Literal["transcription.chunk"] = Field(..., description="The object type, always 'transcription.chunk'.")
    created: int = Field(..., description="The creation timestamp of the chunk.")
    model: str = Field(..., description="The model used for the transcription.")
    choices: List[TranscriptionResponseStreamChoice] = Field(..., description="The choices for this streaming response.")
    usage: Optional[UsageInfo] = Field(default=None, description="The usage of the transcription.")