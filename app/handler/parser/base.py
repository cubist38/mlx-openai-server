"""
Base parser classes for handling thinking and tool parsing in model outputs.

This module provides abstract base classes for parsing structured content from
language model outputs, including thinking traces and tool calls.
"""

import json
from typing import Any

from json_repair import repair_json
from loguru import logger


class BaseThinkingParser:
    """
    Base class for parsing thinking traces from model outputs.

    This class provides the foundation for extracting thinking content
    enclosed in opening and closing tags from language model responses.
    """

    def __init__(self, thinking_open: str, thinking_close: str):
        """
        Initialize the thinking parser with opening and closing tags.

        Parameters
        ----------
        thinking_open : str
            The opening tag for thinking content.
        thinking_close : str
            The closing tag for thinking content.
        """
        self.thinking_open = thinking_open
        self.thinking_close = thinking_close
        self.is_thinking = False

    def get_thinking_open(self):
        """
        Get the opening tag for thinking content.

        Returns
        -------
        str
            The opening tag string.
        """
        return self.thinking_open

    def get_thinking_close(self):
        """
        Get the closing tag for thinking content.

        Returns
        -------
        str
            The closing tag string.
        """
        return self.thinking_close

    def parse(self, content: str) -> tuple[str | None, str]:
        """
        Parse thinking content from the given text.

        Extracts thinking content enclosed between the opening and closing tags,
        returning the thinking content and remaining text.

        Parameters
        ----------
        content : str
            The text content to parse for thinking traces.

        Returns
        -------
        tuple[str | None, str]
            A tuple of (thinking_content, remaining_content). thinking_content is
            None if no thinking tags are found.
        """
        start_thinking = content.find(self.thinking_open)
        if start_thinking == -1:
            return None, content

        thinking_open_len = len(self.thinking_open)
        thinking_close_len = len(self.thinking_close)
        start_content = start_thinking + thinking_open_len
        end_thinking = content.find(self.thinking_close, start_content)

        if end_thinking == -1:
            return None, content

        thinking_content = content[start_content:end_thinking].strip()
        remaining_content = content[end_thinking + thinking_close_len :].strip()
        return thinking_content, remaining_content

    def parse_stream(self, chunk: str | None = None) -> tuple[Any | None, bool]:
        """
        Parse streaming chunks for thinking content.

        Processes incremental text chunks to extract thinking content as it streams,
        maintaining state across multiple calls.

        Parameters
        ----------
        chunk : str | None
            The text chunk to parse. If None, returns current state.

        Returns
        -------
        tuple[Any | None, bool]
            A tuple of (parsed_content, is_complete) where parsed_content is the
            extracted thinking content and is_complete indicates if the thinking
            section has finished.
        """
        if chunk is None:
            return None, False

        if not self.is_thinking:
            # Check if thinking_open is in the chunk
            if self.thinking_open in chunk:
                self.is_thinking = True
                start_idx = chunk.find(self.thinking_open)
                after_open = chunk[start_idx + len(self.thinking_open) :]
                before_open = chunk[:start_idx]

                # Check if thinking_close is also in this chunk (both tags in same chunk)
                if self.thinking_close in after_open:
                    close_idx = after_open.find(self.thinking_close)
                    self.is_thinking = False
                    # Return content before open tag + content after close tag
                    after_close = after_open[close_idx + len(self.thinking_close) :]
                    return (before_open + after_close) if (
                        before_open + after_close
                    ) else None, True

                # Only opening tag found, return content before it (if any) and reasoning content after
                # If there's content after the opening tag, return it as reasoning_content
                if after_open:
                    return {"reasoning_content": after_open}, False
                # Just the opening tag with nothing after it
                return before_open if before_open else None, False
            # No thinking tag, return chunk as is
            return chunk, False

        # Currently in thinking mode
        if self.thinking_close in chunk:
            close_idx = chunk.find(self.thinking_close)
            reasoning_part = chunk[:close_idx]
            after_close = chunk[close_idx + len(self.thinking_close) :]
            self.is_thinking = False

            # If there's reasoning content before the close tag, return it with completion signal
            if reasoning_part:
                result = {"reasoning_content": reasoning_part}
                # If there's also content after the close tag, include it as text
                if after_close:
                    result["content"] = after_close
                return result, True
            # Close tag found, thinking complete, return content after close tag (if any)
            return after_close if after_close else None, True

        # Still in thinking mode, return as reasoning content
        return {"reasoning_content": chunk}, False


class ParseToolState:
    """
    Enumeration of states for tool parsing.

    Used to track the current state during incremental parsing of tool calls.
    """

    NORMAL = 0
    FOUND_PREFIX = 1


class BaseToolParser:
    """
    Base class for parsing tool calls from model outputs.

    This class provides the foundation for extracting and parsing tool calls
    enclosed in opening and closing tags from language model responses.
    """

    def __init__(self, tool_open: str, tool_close: str):
        """
        Initialize the tool parser with opening and closing tags.

        Parameters
        ----------
        tool_open : str
            The opening tag for tool calls.
        tool_close : str
            The closing tag for tool calls.
        """
        self.tool_open = tool_open
        self.tool_close = tool_close
        self.buffer = ""
        self.state = ParseToolState.NORMAL

    def get_tool_open(self):
        """
        Get the opening tag for tool calls.

        Returns
        -------
        str
            The opening tag string.
        """
        return self.tool_open

    def get_tool_close(self):
        """
        Get the closing tag for tool calls.

        Returns
        -------
        str
            The closing tag string.
        """
        return self.tool_close

    def _parse_tool_content(self, tool_content: str) -> dict[str, Any] | None:
        """
        Parse the content of a tool call.

        Subclasses can override this method to support different content formats
        (e.g., XML, YAML).

        Parameters
        ----------
        tool_content : str
            The string content extracted from between the tool tags.

        Returns
        -------
        dict[str, Any] | None
            A dictionary representing the parsed tool call, or None if parsing fails.
        """
        repaired_json = repair_json(tool_content)
        return json.loads(repaired_json)

    def parse(self, content: str) -> tuple[list[dict[str, Any]] | None, str]:
        """
        Parse tool calls from the given content.

        Extracts and parses all tool calls enclosed in the opening and closing tags,
        returning the parsed tool calls and remaining content.

        Parameters
        ----------
        content : str
            The text content to parse for tool calls.

        Returns
        -------
        tuple[list[dict[str, Any]] | None, str]
            A tuple of (tool_calls, remaining_content) where tool_calls is a list
            of parsed tool call dictionaries.
        """
        tool_calls = []
        remaining_parts = []

        if self.tool_open not in content:
            return [], content

        tool_open_len = len(self.tool_open)
        tool_close_len = len(self.tool_close)
        pos = 0

        while True:
            start_tool = content.find(self.tool_open, pos)
            if start_tool == -1:
                # No more tool calls, add remaining content
                if pos < len(content):
                    remaining_parts.append(content[pos:].strip())
                break

            # Add content before tool call
            if start_tool > pos:
                remaining_parts.append(content[pos:start_tool].strip())

            # Find closing tag
            search_start = start_tool + tool_open_len
            end_tool = content.find(self.tool_close, search_start)
            if end_tool == -1:
                # Unclosed tool tag, add remaining content and break
                remaining_parts.append(content[pos:].strip())
                break

            # Extract and parse tool content
            tool_content = content[search_start:end_tool].strip()
            try:
                json_output = self._parse_tool_content(tool_content)
                tool_calls.append(json_output)
            except json.JSONDecodeError:
                logger.warning("Error parsing tool call: {}", tool_content)
                # Continue processing remaining content after error
                remaining_parts.append(content[pos:].strip())
                break

            # Move position past the closing tag
            pos = end_tool + tool_close_len

        remaining_content = " ".join(filter(None, remaining_parts))
        return tool_calls, remaining_content

    def parse_stream(self, chunk: str | None = None) -> tuple[Any | None, bool]:
        """
        Parse streaming chunks for tool calls.

        Processes incremental text chunks to extract tool calls as they stream,
        maintaining state across multiple calls.

        Parameters
        ----------
        chunk : str | None
            The text chunk to parse. If None, returns current state.

        Returns
        -------
        tuple[Any | None, bool]
            A tuple of (parsed_content, is_complete) where parsed_content is the
            extracted tool call data and is_complete indicates if the tool call
            has finished.
        """
        if chunk is None:
            return None, True

        if self.tool_open in chunk:
            self.state = ParseToolState.FOUND_PREFIX
            start_tool_index = chunk.find(self.tool_open)
            end_tool_index = chunk.find(self.tool_close)
            if end_tool_index != -1:
                self.buffer = chunk[start_tool_index + len(self.tool_open) : end_tool_index]
                self.state = ParseToolState.NORMAL
                try:
                    json_output = self._parse_tool_content(self.buffer)
                except json.JSONDecodeError:
                    logger.warning("Error parsing tool call: {}", self.buffer)
                    return None, True
                return {
                    "name": json_output["name"],
                    "arguments": json.dumps(json_output["arguments"]),
                }, True

            self.buffer += chunk[start_tool_index + len(self.tool_open) :]

            return chunk[:start_tool_index], False

        if self.state == ParseToolState.FOUND_PREFIX:
            end_tool_index = chunk.find(self.tool_close)
            if end_tool_index != -1:
                self.buffer += chunk[:end_tool_index]
                try:
                    json_output = self._parse_tool_content(self.buffer)
                except json.JSONDecodeError:
                    logger.warning("Error parsing tool call: {}", self.buffer)
                    return None, False
                return {
                    "name": json_output["name"],
                    "arguments": json.dumps(json_output["arguments"]),
                }, True
            self.buffer += chunk
            return None, False

        return chunk, False


"""
Base Message Converter
Provides generic conversion from OpenAI API message format to model-compatible format.
"""


class BaseMessageConverter:
    """Base message format converter class."""

    def convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert message format to be compatible with specific model chat templates."""
        converted_messages = []

        for message in messages:
            converted_message = self._convert_single_message(message)
            if converted_message:
                converted_messages.append(converted_message)

        return converted_messages

    def _convert_single_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Convert a single message."""
        if not isinstance(message, dict):
            return message

        # Convert function.arguments from string to object in tool_calls
        tool_calls = message.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            self._convert_tool_calls(tool_calls)

        return message

    def _convert_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        """Convert arguments format in tool calls."""
        for tool_call in tool_calls:
            if isinstance(tool_call, dict) and "function" in tool_call:
                function = tool_call["function"]
                if isinstance(function, dict) and "arguments" in function:
                    arguments = function["arguments"]
                    if isinstance(arguments, str):
                        function["arguments"] = self._parse_arguments_string(arguments)

    def _parse_arguments_string(self, arguments_str: str) -> Any:
        """Parse arguments string to object, can be overridden by subclasses."""
        try:
            return json.loads(arguments_str)
        except json.JSONDecodeError:
            return arguments_str
