"""
Parser factory for creating thinking and tool parsers based on manual configuration.

This module provides a centralized way to create parsers through explicit
manual specification. Parsers are only created when explicitly requested.
"""

from typing import Any, Callable, Dict, Optional, Tuple

from loguru import logger

from . import (
    Glm4MoEThinkingParser,
    Glm4MoEToolParser,
    HarmonyParser,
    MinimaxThinkingParser,
    MinimaxToolParser,
    Qwen3MoEThinkingParser,
    Qwen3MoEToolParser,
    Qwen3NextThinkingParser,
    Qwen3NextToolParser,
    Qwen3ThinkingParser,
    Qwen3ToolParser,
    Qwen3VLThinkingParser,
    Qwen3VLToolParser,
    HermesThinkingParser,
    HermesToolParser,
    Llama4PythonicToolParser,
    Ministral3ThinkingParser,
    Ministral3ToolParser,
)
from .glm4_moe import Glm4MoEMessageConverter
from .minimax import MiniMaxMessageConverter

# Registry mapping parser names to their classes
PARSER_REGISTRY: Dict[str, Dict[str, Callable]] = {
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
    "ministral3": {
        "thinking": Ministral3ThinkingParser,
        "tool": Ministral3ToolParser,
    },
}

# Registry mapping model types to their converter classes
CONVERTER_REGISTRY: Dict[str, Callable] = {
    "glm4_moe": Glm4MoEMessageConverter,
    "minimax": MiniMaxMessageConverter,
}

# Registry mapping parser names to their metadata/properties
PARSER_METADATA: Dict[str, Dict[str, Any]] = {
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
    def create_parser(parser_name: str, parser_type: str, **kwargs) -> Optional[Any]:
        """
        Create a parser instance from the registry.

        Args:
            parser_name: Name of the parser (e.g., "qwen3", "glm4_moe", "harmony")
            parser_type: Type of parser ("thinking", "tool", or "unified")
            **kwargs: Additional arguments for parser initialization

        Returns:
            Parser instance or None if parser type not available
        """
        if parser_name not in PARSER_REGISTRY:
            logger.warning(f"Unknown parser name: {parser_name}")
            return None

        parser_config = PARSER_REGISTRY[parser_name]

        # Handle unified parsers (like Harmony)
        if parser_type == "unified" and "unified" in parser_config:
            return parser_config["unified"]()

        # Handle specific parser types
        if parser_type in parser_config:
            parser_class = parser_config[parser_type]
            return parser_class()

        return None

    @staticmethod
    def create_parsers(
        model_type: str,
        manual_reasoning_parser: Optional[str] = None,
        manual_tool_parser: Optional[str] = None,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Create thinking and tool parsers based on manual configuration.

        Parsers are only created when explicitly specified. If no parsers are
        specified, both will be None.

        Args:
            model_type: The type of the model (for logging/debugging purposes)
            tools: Whether tools are available (for logging/debugging purposes)
            enable_thinking: Whether thinking/reasoning is enabled (for logging/debugging purposes)
            manual_reasoning_parser: Manually specified reasoning parser name
            manual_tool_parser: Manually specified tool parser name

        Returns:
            Tuple of (thinking_parser, tool_parser). Both will be None if not specified.
        """
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
    def create_converter(model_type: str) -> Optional[Any]:
        """
        Create a message converter based on model type.

        Args:
            model_type: The type of the model (e.g., "glm4_moe", "minimax")

        Returns:
            Message converter instance or None if no converter needed
        """
        if model_type not in CONVERTER_REGISTRY:
            return None

        converter_class = CONVERTER_REGISTRY[model_type]
        return converter_class()

    @staticmethod
    def respects_enable_thinking(parser_name: Optional[str]) -> bool:
        """
        Check if a parser respects the enable_thinking flag.

        Args:
            parser_name: Name of the parser to check

        Returns:
            True if parser respects enable_thinking, False otherwise
        """
        if not parser_name:
            return False
        return PARSER_METADATA.get(parser_name, {}).get("respects_enable_thinking", False)

    @staticmethod
    def needs_redacted_reasoning_prefix(parser_name: Optional[str]) -> bool:
        """
        Check if a parser needs the <think> prefix added to responses.

        Args:
            parser_name: Name of the parser to check

        Returns:
            True if parser needs redacted_reasoning prefix, False otherwise
        """
        if not parser_name:
            return False
        return PARSER_METADATA.get(parser_name, {}).get("needs_redacted_reasoning_prefix", False)

    @staticmethod
    def has_special_parsing(parser_name: Optional[str]) -> bool:
        """
        Check if a parser has special parsing logic (e.g., harmony returns dict from parse()).

        Args:
            parser_name: Name of the parser to check

        Returns:
            True if parser has special parsing, False otherwise
        """
        if not parser_name:
            return False
        return PARSER_METADATA.get(parser_name, {}).get("has_special_parsing", False)
