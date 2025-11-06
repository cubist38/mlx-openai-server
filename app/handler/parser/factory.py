"""
Parser factory for creating thinking and tool parsers based on manual configuration.

This module provides a centralized way to create parsers through explicit
manual specification. Parsers are only created when explicitly requested.
"""
from typing import Optional, Dict, Any, Tuple, Callable
from loguru import logger
from app.handler.parser import (
    Qwen3ToolParser,
    Qwen3ThinkingParser,
    HarmonyParser,
    Glm4MoEToolParser,
    Glm4MoEThinkingParser,
)


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
    "harmony": {
        # Harmony parser handles both thinking and tools
        "unified": HarmonyParser,
    },
}

# Model type to parser name mapping for auto-detection
MODEL_TYPE_TO_PARSER: Dict[str, Dict[str, Optional[str]]] = {
    # Language models
    "qwen3": {
        "thinking": "qwen3",
        "tool": "qwen3",
    },
    "glm4_moe": {
        "thinking": "glm4_moe",
        "tool": "glm4_moe",
    },
    "gpt_oss": {
        # Harmony parser handles both
        "unified": "harmony",
    },
    # Vision language models
    "qwen3_vl": {
        "tool": "qwen3",
    },
    "qwen3_vl_moe": {
        "tool": "qwen3",
    },
    "glm4v_moe": {
        "thinking": "glm4_moe",
        "tool": "glm4_moe",
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
            logger.warning(f"Failed to create Harmony parser")
        
        # Create reasoning parser if explicitly specified
        thinking_parser = None
        if manual_reasoning_parser:
            parser_instance = ParserFactory.create_parser(
                manual_reasoning_parser, "thinking"
            )
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

