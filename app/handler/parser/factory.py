"""
Parser factory for creating thinking and tool parsers based on configuration.

This module provides a centralized way to create parsers either through
manual specification or auto-detection based on model type.
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
    def auto_detect_parsers(
        model_type: str,
        tools: Optional[Any] = None,
        enable_thinking: bool = True,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Auto-detect parser names based on model type.
        
        Args:
            model_type: The type of the model (e.g., "qwen3", "glm4_moe")
            tools: Whether tools are available (affects tool parser creation)
            enable_thinking: Whether thinking/reasoning is enabled
            
        Returns:
            Tuple of (reasoning_parser_name, tool_parser_name)
        """
        if model_type not in MODEL_TYPE_TO_PARSER:
            logger.debug(f"Unknown model type for auto-detection: {model_type}")
            return None, None
        
        model_config = MODEL_TYPE_TO_PARSER[model_type]
        
        # Handle unified parsers (like Harmony for gpt_oss)
        if "unified" in model_config:
            return model_config["unified"], None
        
        reasoning_parser = None
        tool_parser = None
        
        if "thinking" in model_config and enable_thinking:
            reasoning_parser = model_config["thinking"]
        
        if "tool" in model_config and tools:
            tool_parser = model_config["tool"]
        
        return reasoning_parser, tool_parser

    @staticmethod
    def create_parsers(
        model_type: str,
        tools: Optional[Any] = None,
        enable_thinking: bool = True,
        manual_reasoning_parser: Optional[str] = None,
        manual_tool_parser: Optional[str] = None,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Create thinking and tool parsers based on configuration.
        
        This method handles both manual parser specification and auto-detection.
        When manual parsers are specified, they override auto-detection for that
        specific parser type. If only one parser is manually specified, the other
        uses auto-detection.
        
        Args:
            model_type: The type of the model
            tools: Whether tools are available
            enable_thinking: Whether thinking/reasoning is enabled
            manual_reasoning_parser: Manually specified reasoning parser name
            manual_tool_parser: Manually specified tool parser name
            
        Returns:
            Tuple of (thinking_parser, tool_parser)
        """
        # Handle unified parsers manually specified
        if manual_reasoning_parser == "harmony" or manual_tool_parser == "harmony":
            harmony_parser = ParserFactory.create_parser("harmony", "unified")
            if harmony_parser:
                return harmony_parser, None
        
        # Determine which parsers to use (manual or auto-detected)
        reasoning_parser_name = manual_reasoning_parser
        tool_parser_name = manual_tool_parser
        
        # Auto-detect missing parsers
        auto_reasoning, auto_tool = ParserFactory.auto_detect_parsers(
            model_type, tools, enable_thinking
        )
        
        # Handle unified parser from auto-detection
        if auto_reasoning == "harmony":
            harmony_parser = ParserFactory.create_parser("harmony", "unified")
            if harmony_parser:
                return harmony_parser, None
        
        if reasoning_parser_name is None:
            reasoning_parser_name = auto_reasoning
        
        if tool_parser_name is None:
            tool_parser_name = auto_tool
        
        # Create parser instances
        thinking_parser = None
        if reasoning_parser_name:
            parser_instance = ParserFactory.create_parser(
                reasoning_parser_name, "thinking"
            )
            if parser_instance is not None:
                thinking_parser = parser_instance
            elif reasoning_parser_name:  # Only log if manual override was attempted
                logger.debug(
                    f"Thinking parser '{reasoning_parser_name}' not available or not supported "
                    f"for model type '{model_type}'"
                )
        
        tool_parser = None
        if tool_parser_name:
            parser_instance = ParserFactory.create_parser(tool_parser_name, "tool")
            if parser_instance is not None:
                tool_parser = parser_instance
            elif tool_parser_name:  # Only log if manual override was attempted
                logger.debug(
                    f"Tool parser '{tool_parser_name}' not available or not supported "
                    f"for model type '{model_type}'"
                )
        
        return thinking_parser, tool_parser

