from __future__ import annotations

import json
from typing import Any

from .abstract_converter import AbstractMessageConverter

class GLM4MoEMessageConverter(AbstractMessageConverter):
    """GLM4 MoE-specific message format converter"""

    def convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert message format to be compatible with GLM4 MoE chat templates.
        
        Parameters
        ----------
        messages : list[dict[str, Any]]
            List of messages in OpenAI API format.
            
        Returns
        -------
        list[dict[str, Any]]
            List of messages converted to GLM4 MoE format.
        """
        converted_messages = []

        for message in messages:
            converted_message = self._convert_single_message(message)
            if converted_message:
                converted_messages.append(converted_message)

        return converted_messages

    def _convert_single_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Convert a single message.
        
        Parameters
        ----------
        message : dict[str, Any]
            Single message to convert.
            
        Returns
        -------
        dict[str, Any]
            Converted message.
        """

        # Convert function.arguments from string to object in tool_calls
        tool_calls = message.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            self._convert_tool_calls(tool_calls)

        return message

    def _convert_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        """Convert arguments format in tool calls.
        
        Parameters
        ----------
        tool_calls : list[dict[str, Any]]
            List of tool calls to convert.
        """
        for tool_call in tool_calls:
            if isinstance(tool_call, dict) and "function" in tool_call:
                function = tool_call["function"]
                if isinstance(function, dict) and "arguments" in function:
                    arguments = function["arguments"]
                    if isinstance(arguments, str):
                        function["arguments"] = self._parse_arguments_string(arguments)

    def _parse_arguments_string(self, arguments_str: str) -> Any:
        """Parse GLM4 MoE-specific argument string format.
        
        Parameters
        ----------
        arguments_str : str
            Arguments in string format.
            
        Returns
        -------
        Any
            Parsed arguments object or original string if parsing fails.
        """
        try:
            return json.loads(arguments_str)
        except json.JSONDecodeError:
            return arguments_str