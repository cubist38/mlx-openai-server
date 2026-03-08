from __future__ import annotations

from .abstract_converter import AbstractMessageConverter
from .glm4_moe import GLM4MoEMessageConverter

# Mapping from converter name strings to converter classes
MESSAGE_CONVERTER_MAP: dict[str, type[AbstractMessageConverter]] = {
    "glm4_moe": GLM4MoEMessageConverter,
    "minimax_m2": GLM4MoEMessageConverter,  # use the same converter as glm4_moe
    "minimax": GLM4MoEMessageConverter,  # use the same converter as glm4_moe
    "nemotron3_nano": GLM4MoEMessageConverter,  # use the same converter as glm4_moe
    "qwen3_coder": GLM4MoEMessageConverter,  # use the same converter as glm4_moe
    "longcat_flash_lite": GLM4MoEMessageConverter,  # use the same converter as glm4_moe
    "step_35": GLM4MoEMessageConverter,  # use the same converter as glm4_moe (Opencode compatibility)
}


def resolve_message_converter_name(
    converter_name: str | None = None,
    tool_parser_name: str | None = None,
    reasoning_parser_name: str | None = None,
) -> str | None:
    """Resolve the effective message converter name.

    Parameters
    ----------
    converter_name : str | None
        Explicit message converter name supplied by the caller.
    tool_parser_name : str | None
        Tool parser name used for automatic converter detection.
    reasoning_parser_name : str | None
        Reasoning parser name used as a fallback for automatic detection.

    Returns
    -------
    str | None
        Normalized converter name, or ``None`` when no compatible
        converter can be resolved.
    """
    if converter_name is not None:
        normalized_converter_name = converter_name.lower()
        return (
            normalized_converter_name
            if normalized_converter_name in MESSAGE_CONVERTER_MAP
            else None
        )

    for parser_name in (tool_parser_name, reasoning_parser_name):
        if parser_name is None:
            continue
        normalized_parser_name = parser_name.lower()
        if normalized_parser_name in MESSAGE_CONVERTER_MAP:
            return normalized_parser_name

    return None


def get_message_converter(converter_name: str | None) -> type[AbstractMessageConverter] | None:
    """Get a message converter class by name.

    Parameters
    ----------
    converter_name : str | None
        Name of the message converter (e.g., 'glm4_moe', 'minimax', 'nemotron3_nano').

    Returns
    -------
    type[AbstractMessageConverter] | None
        The message converter class, or None if not found.
    """
    if converter_name is None:
        return None
    return MESSAGE_CONVERTER_MAP.get(converter_name.lower())


class MessageConverterManager:
    """Factory for creating message converters.

    This manager provides a centralized way to instantiate message converters
    based on configuration, similar to how ParserManager handles parsers.

    Examples
    --------
    >>> converter = MessageConverterManager.create_converter("glm4_moe")
    >>> converted = converter.convert_messages(messages)
    """

    @staticmethod
    def create_converter(
        converter_name: str | None = None,
        tool_parser_name: str | None = None,
        reasoning_parser_name: str | None = None,
    ) -> AbstractMessageConverter | None:
        """Create a message converter instance based on configuration.

        Parameters
        ----------
        converter_name : str | None
            Explicit name of the message converter (e.g., 'glm4_moe', 'minimax').
        tool_parser_name : str | None
            Tool parser name used for automatic converter detection.
        reasoning_parser_name : str | None
            Reasoning parser name used as a fallback for automatic detection.

        Returns
        -------
        AbstractMessageConverter | None
            Message converter instance, or None if converter not found or not specified.
        """
        resolved_converter_name = resolve_message_converter_name(
            converter_name=converter_name,
            tool_parser_name=tool_parser_name,
            reasoning_parser_name=reasoning_parser_name,
        )
        if resolved_converter_name is None:
            return None

        converter_class = get_message_converter(resolved_converter_name)
        if converter_class is None:
            return None

        return converter_class()


__all__ = [
    # Base class
    "AbstractMessageConverter",
    # Converter implementations
    "GLM4MoEMessageConverter",
    # Mapping and helper functions
    "MESSAGE_CONVERTER_MAP",
    "resolve_message_converter_name",
    "get_message_converter",
    # Converter manager
    "MessageConverterManager",
]
