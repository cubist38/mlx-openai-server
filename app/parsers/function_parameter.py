"""Tool parser for function/parameter-style XML tool-call payloads."""

from __future__ import annotations

import json
import re
from typing import Any

from .abstract_parser import AbstractToolParser

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"


class FunctionParameterToolParser(AbstractToolParser):
    """Base tool parser for models using <function=...><parameter=...> format.

    Handles tool calls in the format:
    <tool_call>
    <function=function_name>
    <parameter=param_name>param_value</parameter>
    </function>
    </tool_call>

    Used by: Qwen3Coder, Nemotron3Nano, Step 3.5
    """

    def __init__(
        self,
        tool_open: str = TOOL_OPEN,
        tool_close: str = TOOL_CLOSE,
    ) -> None:
        """Initialize the function-parameter tool parser.

        Parameters
        ----------
        tool_open : str
            Opening tag for tool calls.
        tool_close : str
            Closing tag for tool calls.
        """
        super().__init__(tool_open=tool_open, tool_close=tool_close)
        # Regex pattern to extract function name and content
        self.tool_regex = re.compile(
            r"<function=([^>]+)>\s*(.*?)\s*</function>",
            re.DOTALL,
        )
        self.tool_call_block_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>",
            re.DOTALL,
        )
        self.permissive_function_open_regex = re.compile(
            r"<function\s*=\s*([^>]+)>",
            re.DOTALL,
        )
        # Regex pattern to extract parameter key-value pairs
        self.parameter_regex = re.compile(
            r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
            re.DOTALL,
        )
        self.permissive_parameter_regex = re.compile(
            r"<parameter\s*=\s*([^>]+)>\s*(.*?)\s*</parameter>",
            re.DOTALL,
        )

    @staticmethod
    def _coerce_parameter_value(param_value: str) -> str | int | float | bool | list[Any] | dict[str, Any]:
        """Parse tool argument values as JSON when possible."""
        try:
            loaded = json.loads(param_value)
        except (json.JSONDecodeError, ValueError):
            return param_value

        if isinstance(loaded, (str, int, float, bool, list, dict)):
            return loaded
        return param_value

    def _extract_tool_calls_strict(self, model_output: str) -> list[dict[str, str]]:
        """Extract tool calls using the strict function/parameter tag format."""
        matches = self.tool_regex.findall(model_output)
        if not matches:
            return []

        tool_calls: list[dict[str, str]] = []
        for function_name_raw, function_content_raw in matches:
            function_name = function_name_raw.strip()
            function_content = function_content_raw.strip()

            param_matches = self.parameter_regex.findall(function_content)
            if not param_matches and "<parameter" in function_content:
                param_matches = self.permissive_parameter_regex.findall(function_content)
            arguments: dict[str, str | int | float | bool | list[Any] | dict[str, Any]] = {}
            for param_name_raw, param_value_raw in param_matches:
                param_name = param_name_raw.strip()
                param_value = param_value_raw.strip()
                arguments[param_name] = self._coerce_parameter_value(param_value)

            tool_calls.append(
                {
                    "name": function_name,
                    "arguments": json.dumps(arguments),
                }
            )

        return tool_calls

    def _extract_tool_calls_permissive(self, model_output: str) -> list[dict[str, str]]:
        """Best-effort extraction for malformed function tags inside tool_call blocks."""
        tool_calls: list[dict[str, str]] = []
        for block in self.tool_call_block_regex.findall(model_output):
            function_match = self.permissive_function_open_regex.search(block)
            if function_match is None:
                continue

            function_name = function_match.group(1).strip()
            function_content_start = function_match.end()
            function_close_idx = block.find("</function>", function_content_start)
            if function_close_idx == -1:
                function_content = block[function_content_start:]
            else:
                function_content = block[function_content_start:function_close_idx]

            arguments: dict[str, str | int | float | bool | list[Any] | dict[str, Any]] = {}
            for param_name_raw, param_value_raw in self.permissive_parameter_regex.findall(function_content):
                param_name = param_name_raw.strip()
                param_value = param_value_raw.strip()
                arguments[param_name] = self._coerce_parameter_value(param_value)

            tool_calls.append(
                {
                    "name": function_name,
                    "arguments": json.dumps(arguments),
                }
            )

        return tool_calls

    def extract_tool_calls(self, model_output: str) -> dict[str, list] | None:
        """Extract tool calls from complete model output.

        Parameters
        ----------
        model_output : str
            Complete model output containing tool calls in XML-like format.

        Returns
        -------
        dict[str, list] | None
            Dictionary with 'tool_calls' key containing list of parsed tool calls,
            or None if no tool calls found. Each tool call has 'name' and 'arguments'.
        """
        tool_calls = self._extract_tool_calls_strict(model_output)
        if tool_calls:
            return {"tool_calls": tool_calls}

        tool_calls = self._extract_tool_calls_permissive(model_output)
        if tool_calls:
            return {"tool_calls": tool_calls}

        return {"content": model_output}
