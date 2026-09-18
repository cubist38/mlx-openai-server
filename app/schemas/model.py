"""Model metadata schemas for model registry."""

from pydantic import BaseModel, ConfigDict, Field


class ModelMetadata(BaseModel):
    """
    Metadata for a registered model.

    Attributes
    ----------
        id: Unique identifier for the model
        type: Model type (lm, multimodal, embeddings, etc.)
        context_length: Maximum context length (if applicable)
        created_at: Timestamp when model was loaded
        version: Optional version tag for the served checkpoint
        aliases: Alternative names that route to this model
        capabilities: Optional dict of model capabilities
    """

    model_config = ConfigDict(frozen=False)

    id: str = Field(..., description="Unique model identifier")
    type: str = Field(
        ...,
        description="Model type (lm, multimodal, embeddings, whisper, image-generation, image-edit)",
    )
    context_length: int | None = Field(
        None, description="Maximum context length for language models"
    )
    created_at: int = Field(..., description="Unix timestamp when model was loaded")
    version: str | None = Field(
        None, description="Version tag of the served checkpoint, if declared"
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names that resolve to this model in API requests",
    )
    object: str = Field(default="model", description="Object type, always 'model'")
    owned_by: str = Field(default="local", description="Model owner/organization")
