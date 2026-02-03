from __future__ import annotations

import re
import json
from enum import Enum
from .hermes import HermesToolParser

TOOL_CALL_SECTION_BEGIN = "<|tool_calls_section_begin|>"
TOOL_CALL_SECTION_END = "<|tool_calls_section_end|>"
TOOL_CALL_BEGIN = "<|tool_call_begin|>"
TOOL_CALL_END = "<|tool_call_end|>"
TOOL_CALL_ARGUMENTS_BEGIN = "<|tool_call_arguments_begin|>"


class KimiK2ToolState(Enum):
    """State constants for Kimi K2 tool parser streaming operations."""

    NORMAL = "normal"
    FOUND_TOOL_SECTION = "found_tool_section"

class KimiK2ToolParser(HermesToolParser):
    """Kimi K2 tool parser.
    
    Handles tool calls in format:
    <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "New York"}<|tool_call_end|><|tool_calls_section_end|>
    """

    def __init__(self, tool_open: str = TOOL_CALL_SECTION_BEGIN, tool_close: str = TOOL_CALL_SECTION_END) -> None:
        """Initialize Solar Open tool parser."""
        super().__init__(tool_open=tool_open, tool_close=tool_close)
        
        self.state = KimiK2ToolState.NORMAL
        self.tool_call_section_regex = re.compile(
            re.escape(self.tool_open) + r"(.*?)" + re.escape(self.tool_close),
            re.DOTALL,
        )
        self.tool_call_begin = TOOL_CALL_BEGIN
        self.tool_call_end = TOOL_CALL_END
        self.tool_call_arguments_begin = TOOL_CALL_ARGUMENTS_BEGIN
        self.tool_call_block_regex = re.compile(
            re.escape(self.tool_call_begin) + r"(.*?)" + re.escape(self.tool_call_end),
            re.DOTALL,
        )
        self.tool_name_regex = re.compile(r"\.([^.:]+)(?::\d*)?$")

    def extract_tool_calls(self, tool_output: str) -> dict[str, list] | None:
        """Parse tool output into a list of tool calls.

        Parses format:
        <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_arguments_begin|>{"city": "New York"}<|tool_call_end|><|tool_calls_section_end|>
        to {"name": "get_weather", "arguments": {"city": "New York"}}.

        Parameters
        ----------
        tool_output : str
            Raw tool section string (may include section begin/end tokens).

        Returns
        -------
        dict[str, list] | None
            {"tool_calls": [{"name": str, "arguments": dict}, ...]} or None if no valid calls.
        """
        matches = self.tool_call_section_regex.findall(tool_output)
        if not matches:
            return {"content": tool_output}
        tool_calls = []
        for match in matches:
            blocks = self.tool_call_block_regex.findall(match)
            if not blocks:
                continue
            for block in blocks:
                if self.tool_call_arguments_begin not in block:
                    continue
                header, _, args_str = block.partition(self.tool_call_arguments_begin)
                header = header.strip()
                args_str = args_str.strip()
                name_match = self.tool_name_regex.search(header)
                name = name_match.group(1)
                try:
                    arguments = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append({"name": name, "arguments": arguments})
        if not tool_calls:
            return None
        return {"tool_calls": tool_calls}