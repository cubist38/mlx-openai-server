from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .abstract_parser import (
    AbstractReasoningParser,
    AbstractToolParser,
    ReasoningParserState,
    ToolParserState,
)
from .functiongemma import FunctionGemmaToolParser
from .glm4_moe import GLM4MoEReasoningParser, GLM4MoEToolParser
from .harmony import HarmonyParser
from .hermes import HermesReasoningParser, HermesToolParser
from .minimax_m2 import MiniMaxM2ReasoningParser, MiniMaxM2ToolParser
from .nemotron3_nano import Nemotron3NanoReasoningParser, Nemotron3NanoToolParser
from .qwen3 import Qwen3ReasoningParser, Qwen3ToolParser
from .qwen3_coder import Qwen3CoderToolParser
from .qwen3_moe import Qwen3MoEReasoningParser, Qwen3MoEToolParser
from .qwen3_vl import Qwen3VLReasoningParser, Qwen3VLToolParser
from .iquest_coder_v1 import IQuestCoderV1ToolParser
from .solar_open import SolarOpenReasoningParser, SolarOpenToolParser

# Mapping from parser name strings to reasoning parser classes
REASONING_PARSER_MAP: dict[str, type[AbstractReasoningParser]] = {
    "hermes": HermesReasoningParser,
    "qwen3": Qwen3ReasoningParser,
    "qwen3_moe": Qwen3MoEReasoningParser,
    "qwen3_vl": Qwen3VLReasoningParser,
    "glm4_moe": GLM4MoEReasoningParser,
    "minimax_m2": MiniMaxM2ReasoningParser,
    "nemotron3_nano": Nemotron3NanoReasoningParser,
    "solar_open": SolarOpenReasoningParser,
}

# Mapping from parser name strings to tool parser classes
TOOL_PARSER_MAP: dict[str, type[AbstractToolParser]] = {
    "hermes": HermesToolParser,
    "qwen3": Qwen3ToolParser,
    "qwen3_coder": Qwen3CoderToolParser,
    "qwen3_moe": Qwen3MoEToolParser,
    "qwen3_vl": Qwen3VLToolParser,
    "glm4_moe": GLM4MoEToolParser,
    "minimax_m2": MiniMaxM2ToolParser,
    "nemotron3_nano": Nemotron3NanoToolParser,
    "functiongemma": FunctionGemmaToolParser,
    "iquest_coder_v1": IQuestCoderV1ToolParser,
    "solar_open": SolarOpenToolParser,
}

# Unified parsers that handle BOTH reasoning and tool calls in one class
UNIFIED_PARSER_MAP: dict[str, type] = {
    "harmony": HarmonyParser,
}


def get_reasoning_parser(parser_name: str | None) -> type[AbstractReasoningParser] | None:
    """Get a reasoning parser class by name.
    
    Parameters
    ----------
    parser_name : str
        Name of the reasoning parser (e.g., 'qwen3', 'hermes', 'glm4-moe').
        
    Returns
    -------
    type[AbstractReasoningParser] | None
        The reasoning parser class, or None if not found.
    """
    if parser_name is None:
        return None
    return REASONING_PARSER_MAP.get(parser_name.lower())


def get_tool_parser(parser_name: str | None) -> type[AbstractToolParser] | None:
    """Get a tool parser class by name.
    
    Parameters
    ----------
    parser_name : str
        Name of the tool parser (e.g., 'qwen3', 'hermes', 'functiongemma').
        
    Returns
    -------
    type[AbstractToolParser] | None
        The tool parser class, or None if not found.
    """
    if parser_name is None:
        return None
    return TOOL_PARSER_MAP.get(parser_name.lower())


def get_unified_parser(parser_name: str | None) -> type | None:
    """Get a unified parser class by name.
    
    Parameters
    ----------
    parser_name : str
        Name of the unified parser (e.g., 'harmony').
        
    Returns
    -------
    type | None
        The unified parser class, or None if not found.
    """
    if parser_name is None:
        return None
    return UNIFIED_PARSER_MAP.get(parser_name.lower())


@dataclass
class ParsersResult:
    """Result container for created parsers.
    
    Attributes
    ----------
    reasoning_parser : AbstractReasoningParser | None
        Standalone reasoning parser instance.
    tool_parser : AbstractToolParser | None
        Standalone tool parser instance.
    unified_parser : Any | None
        Unified parser that handles both reasoning and tools.
    parser_name : str | None
        The primary parser name used (for metadata lookups).
    """
    reasoning_parser: AbstractReasoningParser | None = None
    tool_parser: AbstractToolParser | None = None
    unified_parser: Any | None = None
    parser_name: str | None = None
    
    @property
    def is_unified(self) -> bool:
        """Check if using a unified parser.
        
        Returns
        -------
        bool
            True if using a unified parser, False otherwise.
        """
        return self.unified_parser is not None
    
    @property
    def has_reasoning(self) -> bool:
        """Check if reasoning parsing is available.
        
        Returns
        -------
        bool
            True if reasoning parsing is available, False otherwise.
        """
        return self.reasoning_parser is not None or self.unified_parser is not None
    
    @property
    def has_tool_parsing(self) -> bool:
        """Check if tool parsing is available.
        
        Returns
        -------
        bool
            True if tool parsing is available, False otherwise.
        """
        return self.tool_parser is not None or self.unified_parser is not None


class ParserManager:
    """
    Factory for creating reasoning and tool parsers.
    
    Handles unified parsers (like Harmony) that combine both capabilities,
    ensuring only one instance is created when both --reasoning-parser 
    and --tool-call-parser point to the same unified parser.
    
    Examples
    --------
    >>> result = ParserManager.create_parsers("harmony", "harmony")
    >>> result.is_unified
    True
    >>> result.unified_parser  # Single HarmonyParser instance
    
    >>> result = ParserManager.create_parsers("qwen3", "hermes")
    >>> result.reasoning_parser  # Qwen3ReasoningParser
    >>> result.tool_parser       # HermesToolParser
    """

    @staticmethod
    def create_parsers(
        reasoning_parser_name: str | None = None,
        tool_parser_name: str | None = None,
    ) -> ParsersResult:
        """
        Create parser instances based on configuration.
        
        Parameters
        ----------
        reasoning_parser_name : str | None
            Name of the reasoning parser (e.g., 'qwen3', 'harmony').
        tool_parser_name : str | None
            Name of the tool parser (e.g., 'hermes', 'harmony').
            
        Returns
        -------
        ParsersResult
            Container with created parser instances.
        """
        result = ParsersResult()
        
        # Normalize names
        reasoning_name = reasoning_parser_name.lower() if reasoning_parser_name else None
        tool_name = tool_parser_name.lower() if tool_parser_name else None
        
        # Case 1: Check for unified parser
        unified_name = ParserManager._get_unified_parser_name(reasoning_name, tool_name)
        if unified_name:
            parser_class = UNIFIED_PARSER_MAP[unified_name]
            result.unified_parser = parser_class()
            result.parser_name = unified_name
            return result
        
        # Case 2: Create separate parsers
        if reasoning_name and reasoning_name in REASONING_PARSER_MAP:
            result.reasoning_parser = REASONING_PARSER_MAP[reasoning_name]()
            result.parser_name = reasoning_name
        
        if tool_name and tool_name in TOOL_PARSER_MAP:
            result.tool_parser = TOOL_PARSER_MAP[tool_name]()
            if not result.parser_name:
                result.parser_name = tool_name
        
        return result
    
    @staticmethod
    def _get_unified_parser_name(
        reasoning_name: str | None,
        tool_name: str | None,
    ) -> str | None:
        """
        Check if configuration should use a unified parser.
        
        Parameters
        ----------
        reasoning_name : str | None
            Normalized reasoning parser name.
        tool_name : str | None
            Normalized tool parser name.
            
        Returns
        -------
        str | None
            Parser name if unified, None otherwise.
        """
        # Both point to same unified parser
        if (reasoning_name and tool_name and 
            reasoning_name == tool_name and
            reasoning_name in UNIFIED_PARSER_MAP):
            return reasoning_name
        
        # Either one is a unified parser (takes precedence)
        if reasoning_name and reasoning_name in UNIFIED_PARSER_MAP:
            return reasoning_name
        if tool_name and tool_name in UNIFIED_PARSER_MAP:
            return tool_name
        
        return None
    
    @staticmethod
    def is_unified_parser(parser_name: str | None) -> bool:
        """
        Check if a parser name refers to a unified parser.
        
        Parameters
        ----------
        parser_name : str | None
            Parser name to check.
            
        Returns
        -------
        bool
            True if parser is unified, False otherwise.
        """
        if not parser_name:
            return False
        return parser_name.lower() in UNIFIED_PARSER_MAP


__all__ = [
    # Base classes
    "AbstractReasoningParser",
    "AbstractToolParser",
    "ReasoningParserState",
    "ToolParserState",
    # Reasoning parsers
    "HermesReasoningParser",
    "Qwen3ReasoningParser",
    "Qwen3MoEReasoningParser",
    "Qwen3VLReasoningParser",
    "GLM4MoEReasoningParser",
    "MiniMaxM2ReasoningParser",
    "Nemotron3NanoReasoningParser",
    # Tool parsers
    "HermesToolParser",
    "Qwen3ToolParser",
    "Qwen3CoderToolParser",
    "Qwen3MoEToolParser",
    "Qwen3VLToolParser",
    "GLM4MoEToolParser",
    "MiniMaxM2ToolParser",
    "Nemotron3NanoToolParser",
    "FunctionGemmaToolParser",
    # Unified parsers
    "HarmonyParser",
    # Mappings and helper functions
    "REASONING_PARSER_MAP",
    "TOOL_PARSER_MAP",
    "UNIFIED_PARSER_MAP",
    "get_reasoning_parser",
    "get_tool_parser",
    "get_unified_parser",
    # Parser manager
    "ParserManager",
    "ParsersResult",
]
