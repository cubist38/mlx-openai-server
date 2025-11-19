"""Model metadata schemas for model registry."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ModelMetadata(BaseModel):
    """
    Metadata for a registered model.

    Attributes:
        id: Unique identifier for the model
        type: Model type (lm, multimodal, embeddings, etc.)
        context_length: Maximum context length (if applicable)
        created_at: Timestamp when model was loaded
        capabilities: Optional dict of model capabilities
    """

    id: str = Field(..., description="Unique model identifier")
    type: str = Field(..., description="Model type (lm, multimodal, embeddings, whisper, image-generation, image-edit)")
    context_length: Optional[int] = Field(None, description="Maximum context length for language models")
    created_at: int = Field(..., description="Unix timestamp when model was loaded")
    object: str = Field(default="model", description="Object type, always 'model'")
    owned_by: str = Field(default="local", description="Model owner/organization")

    class Config:
        """Pydantic configuration."""
        frozen = False
