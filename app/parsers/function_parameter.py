from __future__ import annotations

import json
import re

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

    Used by: Qwen3Coder, Nemotron3Nano
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
        # Regex pattern to extract parameter key-value pairs
        self.parameter_regex = re.compile(
            r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
            re.DOTALL,
        )

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
        matches = self.tool_regex.findall(model_output)
        if not matches:
            return {"content": model_output}

        tool_calls = []
        for match in matches:
            function_name = match[0].strip()
            function_content = match[1].strip()

            # Extract parameters from function content
            param_matches = self.parameter_regex.findall(function_content)
            arguments: dict[str, str | int | float | bool | list | dict] = {}
            for param_match in param_matches:
                param_name = param_match[0].strip()
                param_value = param_match[1].strip()

                # Try to parse the value as JSON (for numbers, booleans, objects, arrays)
                try:
                    arguments[param_name] = json.loads(param_value)
                except (json.JSONDecodeError, ValueError):
                    arguments[param_name] = param_value

            tool_calls.append({
                "name": function_name,
                "arguments": json.dumps(arguments),
            })

        return {"tool_calls": tool_calls}
