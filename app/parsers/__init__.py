from __future__ import annotations

from .abstract_parser import (
    AbstractReasoningParser,
    AbstractToolParser,
    ReasoningParserState,
    ToolParserState,
)
from .functiongemma import FunctionGemmaToolParser
from .glm4_moe import GLM4MoEReasoningParser, GLM4MoEToolParser
from .hermes import HermesReasoningParser, HermesToolParser
from .minimax_m2 import MiniMaxM2ReasoningParser, MiniMaxM2ToolParser
from .nemotron3_nano import Nemotron3NanoReasoningParser, Nemotron3NanoToolParser
from .qwen3 import Qwen3ReasoningParser, Qwen3ToolParser
from .qwen3_moe import Qwen3MoEReasoningParser, Qwen3MoEToolParser
from .qwen3_vl import Qwen3VLReasoningParser, Qwen3VLToolParser

# Mapping from parser name strings to reasoning parser classes
REASONING_PARSER_MAP: dict[str, type[AbstractReasoningParser]] = {
    "hermes": HermesReasoningParser,
    "qwen3": Qwen3ReasoningParser,
    "qwen3_moe": Qwen3MoEReasoningParser,
    "qwen3_vl": Qwen3VLReasoningParser,
    "glm4_moe": GLM4MoEReasoningParser,
    "minimax_m2": MiniMaxM2ReasoningParser,
    "nemotron3_nano": Nemotron3NanoReasoningParser,
}

# Mapping from parser name strings to tool parser classes
TOOL_PARSER_MAP: dict[str, type[AbstractToolParser]] = {
    "hermes": HermesToolParser,
    "qwen3": Qwen3ToolParser,
    "qwen3_moe": Qwen3MoEToolParser,
    "qwen3_vl": Qwen3VLToolParser,
    "glm4_moe": GLM4MoEToolParser,
    "minimax_m2": MiniMaxM2ToolParser,
    "nemotron3_nano": Nemotron3NanoToolParser,
    "functiongemma": FunctionGemmaToolParser,
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
    "Qwen3MoEToolParser",
    "Qwen3VLToolParser",
    "GLM4MoEToolParser",
    "MiniMaxM2ToolParser",
    "Nemotron3NanoToolParser",
    "FunctionGemmaToolParser",

    # Mappings and helper functions
    "REASONING_PARSER_MAP",
    "TOOL_PARSER_MAP",
    "get_reasoning_parser",
    "get_tool_parser",
]