"""
Parser factory for creating thinking and tool parsers based on manual configuration.

This module provides a centralized way to create parsers through explicit
manual specification. Parsers are only created when explicitly requested.
"""

from collections.abc import Callable
from typing import Any, TypedDict

from loguru import logger

from .base import BaseMessageConverter, BaseThinkingParser, BaseToolParser
from .glm4_moe import Glm4MoEMessageConverter, Glm4MoEThinkingParser, Glm4MoEToolParser
from .harmony import HarmonyParser
from .hermes import HermesThinkingParser, HermesToolParser
from .minimax import MiniMaxMessageConverter, MinimaxThinkingParser, MinimaxToolParser
from .qwen3 import Qwen3ThinkingParser, Qwen3ToolParser
from .qwen3_moe import Qwen3MoEThinkingParser, Qwen3MoEToolParser
from .qwen3_next import Qwen3NextThinkingParser, Qwen3NextToolParser
from .qwen3_vl import Qwen3VLThinkingParser, Qwen3VLToolParser

# Type alias for parser constructor functions
ParserConstructor = Callable[[], BaseThinkingParser | BaseToolParser | HarmonyParser]


class ParserMetadata(TypedDict):
    """Metadata for parser configuration."""

    respects_enable_thinking: bool
    needs_redacted_reasoning_prefix: bool
    has_special_parsing: bool


class ParserRegistryEntry(TypedDict, total=False):
    """Registry entry for a parser with its available types."""

    thinking: ParserConstructor
    tool: ParserConstructor
    unified: ParserConstructor


# Registry mapping parser names to their classes
PARSER_REGISTRY: dict[str, ParserRegistryEntry] = {
    "qwen3": {
        "thinking": Qwen3ThinkingParser,
        "tool": Qwen3ToolParser,
    },
    "glm4_moe": {
        "thinking": Glm4MoEThinkingParser,
        "tool": Glm4MoEToolParser,
    },
    "qwen3_moe": {
        "thinking": Qwen3MoEThinkingParser,
        "tool": Qwen3MoEToolParser,
    },
    "qwen3_next": {
        "thinking": Qwen3NextThinkingParser,
        "tool": Qwen3NextToolParser,
    },
    "qwen3_vl": {
        "thinking": Qwen3VLThinkingParser,
        "tool": Qwen3VLToolParser,
    },
    "harmony": {
        # Harmony parser handles both thinking and tools
        "unified": HarmonyParser,
    },
    "minimax": {
        "thinking": MinimaxThinkingParser,
        "tool": MinimaxToolParser,
    },
    "hermes": {
        "thinking": HermesThinkingParser,
        "tool": HermesToolParser,
    },
    "llama4_pythonic": {
        "tool": Llama4PythonicToolParser,
    },
}

# Registry mapping model types to their converter classes
CONVERTER_REGISTRY: dict[str, type[BaseMessageConverter]] = {
    "glm4_moe": Glm4MoEMessageConverter,
    "minimax": MiniMaxMessageConverter,
}

# Registry mapping parser names to their metadata/properties
PARSER_METADATA: dict[str, ParserMetadata] = {
    "qwen3": {
        "respects_enable_thinking": True,  # Parser respects enable_thinking flag
        "needs_redacted_reasoning_prefix": False,  # Needs <think> prefix
        "has_special_parsing": False,  # Has special parsing logic (e.g., harmony returns dict)
    },
    "glm4_moe": {
        "respects_enable_thinking": True,
        "needs_redacted_reasoning_prefix": False,
        "has_special_parsing": False,
    },
    "qwen3_moe": {
        "respects_enable_thinking": False,
        "needs_redacted_reasoning_prefix": True,  # Needs prefix for both stream and response
        "has_special_parsing": False,
    },
    "qwen3_next": {
        "respects_enable_thinking": False,
        "needs_redacted_reasoning_prefix": True,  # Needs prefix for both stream and response
        "has_special_parsing": False,
    },
    "qwen3_vl": {
        "respects_enable_thinking": False,
        "needs_redacted_reasoning_prefix": True,  # Needs prefix for both stream and response
        "has_special_parsing": False,
    },
    "harmony": {
        "respects_enable_thinking": False,
        "needs_redacted_reasoning_prefix": False,
        "has_special_parsing": True,  # Harmony parser returns dict from parse() instead of tuple
    },
    "minimax": {
        "respects_enable_thinking": False,
        "needs_redacted_reasoning_prefix": True,  # Needs prefix for both stream and response
        "has_special_parsing": False,
    },
    "hermes": {
        "respects_enable_thinking": False,
        "needs_redacted_reasoning_prefix": False,  # Needs prefix for both stream and response
        "has_special_parsing": False,
    },
    "llama4_pythonic": {
        "respects_enable_thinking": False,
        "needs_redacted_reasoning_prefix": False,
        "has_special_parsing": False,
    },
}


class ParserFactory:
    """Factory for creating thinking and tool parsers."""

    @staticmethod
    def create_parser(
        parser_name: str, parser_type: str, **kwargs: Any
    ) -> BaseThinkingParser | BaseToolParser | HarmonyParser | None:
        """
        Create a parser instance from the registry.

        Args:
            parser_name: Name of the parser (e.g., "qwen3", "glm4_moe", "harmony")
            parser_type: Type of parser ("thinking", "tool", or "unified")

        Returns
        -------
            Parser instance or None if parser type not available
        """
        if parser_name not in PARSER_REGISTRY:
            logger.warning(f"Unknown parser name: {parser_name}")
            return None

        # Reference kwargs to silence ARG004/ANN003
        _kwargs = kwargs

        parser_config = PARSER_REGISTRY[parser_name]

        # Handle unified parsers (like Harmony)
        if parser_type == "unified" and "unified" in parser_config:
            return parser_config["unified"]()

        # Handle specific parser types
        parser_constructor: ParserConstructor | None = parser_config.get(parser_type)  # type: ignore[assignment]
        if parser_constructor:
            return parser_constructor()

        return None

    @staticmethod
    def create_parsers(
        model_type: str,
        manual_reasoning_parser: str | None = None,
        manual_tool_parser: str | None = None,
        **kwargs: Any,
    ) -> tuple[
        BaseThinkingParser | BaseToolParser | HarmonyParser | None,
        BaseThinkingParser | BaseToolParser | HarmonyParser | None,
    ]:
        """
        Create thinking and tool parsers based on manual configuration.

        Parsers are only created when explicitly specified. If no parsers are
        specified, both will be None.

        Args:
            model_type: The type of the model (for logging/debugging purposes)
            manual_reasoning_parser: Manually specified reasoning parser name
            manual_tool_parser: Manually specified tool parser name

        Returns
        -------
            Tuple of (thinking_parser, tool_parser). Both will be None if not specified.
        """
        # Reference kwargs to silence ARG004/ANN003
        _kwargs = kwargs

        # Handle unified parsers (harmony) - handles both thinking and tools
        if manual_reasoning_parser == "harmony" or manual_tool_parser == "harmony":
            harmony_parser = ParserFactory.create_parser("harmony", "unified")
            if harmony_parser:
                return harmony_parser, None
            logger.warning("Failed to create Harmony parser")

        # Create reasoning parser if explicitly specified
        thinking_parser = None
        if manual_reasoning_parser:
            parser_instance = ParserFactory.create_parser(manual_reasoning_parser, "thinking")
            if parser_instance is not None:
                thinking_parser = parser_instance
            else:
                logger.warning(
                    f"Failed to create thinking parser '{manual_reasoning_parser}' "
                    f"for model type '{model_type}'"
                )

        # Create tool parser if explicitly specified
        tool_parser = None
        if manual_tool_parser:
            parser_instance = ParserFactory.create_parser(manual_tool_parser, "tool")
            if parser_instance is not None:
                tool_parser = parser_instance
            else:
                logger.warning(
                    f"Failed to create tool parser '{manual_tool_parser}' "
                    f"for model type '{model_type}'"
                )

        return thinking_parser, tool_parser

    @staticmethod
    def create_converter(model_type: str, **kwargs: Any) -> BaseMessageConverter | None:
        """
        Create a message converter based on model type.

        Args:
            model_type: The type of the model (e.g., "glm4_moe", "minimax")

        Returns
        -------
            Message converter instance or None if no converter needed
        """
        # Reference kwargs to silence ARG004/ANN003
        _kwargs = kwargs

        if model_type not in CONVERTER_REGISTRY:
            return None

        converter_class = CONVERTER_REGISTRY[model_type]
        return converter_class()

    @staticmethod
    def respects_enable_thinking(parser_name: str | None) -> bool:
        """
        Check if a parser respects the enable_thinking flag.

        Args:
            parser_name: Name of the parser to check

        Returns
        -------
            True if parser respects enable_thinking, False otherwise
        """
        if not parser_name:
            return False
        metadata = PARSER_METADATA.get(parser_name)
        return metadata["respects_enable_thinking"] if metadata else False

    @staticmethod
    def needs_redacted_reasoning_prefix(parser_name: str | None) -> bool:
        """
        Check if a parser needs the <think> prefix added to responses.

        Args:
            parser_name: Name of the parser to check

        Returns
        -------
            True if parser needs redacted_reasoning prefix, False otherwise
        """
        if not parser_name:
            return False
        metadata = PARSER_METADATA.get(parser_name)
        return metadata["needs_redacted_reasoning_prefix"] if metadata else False

    @staticmethod
    def has_special_parsing(parser_name: str | None) -> bool:
        """
        Check if a parser has special parsing logic (e.g., harmony returns dict from parse()).

        Args:
            parser_name: Name of the parser to check

        Returns
        -------
            True if parser has special parsing, False otherwise
        """
        if not parser_name:
            return False
        metadata = PARSER_METADATA.get(parser_name)
        return metadata["has_special_parsing"] if metadata else False
