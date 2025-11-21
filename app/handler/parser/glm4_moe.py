"""
Parsers for GLM4 MoE model response formats.

This module provides specialized parsers for GLM4 MoE model's tool calls and
thinking traces, handling GLM4-specific JSON parsing and message conversion.
"""

import ast
import json
import re
from typing import Any

from loguru import logger

from .base import BaseMessageConverter, BaseThinkingParser, BaseToolParser

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
THINKING_OPEN = "<think>"
THINKING_CLOSE = "</think>"


class Glm4MoEThinkingParser(BaseThinkingParser):
    """Parser for GLM4 model's thinking response format."""

    def __init__(self) -> None:
        super().__init__(thinking_open=THINKING_OPEN, thinking_close=THINKING_CLOSE)


class Glm4MoEToolParser(BaseToolParser):
    """Parser for GLM4 model's tool response format with XML-style arguments."""

    def __init__(self) -> None:
        super().__init__(tool_open=TOOL_OPEN, tool_close=TOOL_CLOSE)
        # Regex patterns for parsing GLM4 XML-style tool calls
        self.func_detail_regex = re.compile(r"([^\n]*)\n(.*)", re.DOTALL)
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
        )

    def _deserialize_value(self, value: str) -> Any:
        """Try to deserialize a value from string to appropriate Python type."""
        value = value.strip()

        # Try JSON parsing first
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try literal eval for Python literals
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass

        # Return as string if all else fails
        return value

    def _parse_tool_content(self, tool_content: str) -> dict[str, Any] | None:
        """Override the base method to parse GLM4's specific tool call format."""
        try:
            # Extract function name and arguments section
            detail_match = self.func_detail_regex.search(tool_content)
            if not detail_match:
                return None

            func_name = detail_match.group(1).strip()
            args_section = detail_match.group(2)

            # Extract all key-value pairs
            arg_pairs = self.func_arg_regex.findall(args_section)

            arguments = {}
            for key, value in arg_pairs:
                arg_key = key.strip()
                arg_value = self._deserialize_value(value)
                arguments[arg_key] = arg_value

            # Build tool call object

        except Exception as e:
            logger.warning("Error parsing GLM4 tool call content: {}, Error: {}", tool_content, e)
            return None
        else:
            return {"name": func_name, "arguments": arguments}


class Glm4MoEMessageConverter(BaseMessageConverter):
    """GLM4 MoE-specific message format converter."""

    def _parse_arguments_string(self, arguments_str: str) -> Any:
        """Parse GLM4 MoE-specific argument string format."""
        try:
            return json.loads(arguments_str)
        except json.JSONDecodeError:
            return arguments_str
