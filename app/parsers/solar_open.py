from __future__ import annotations

import json
from enum import Enum

from loguru import logger

from .hermes import HermesReasoningParser
from .abstract_parser import AbstractToolParser

REASONING_OPEN = "<|think|>"
REASONING_CLOSE = "<|end|>"

CONTENT_RESPONSE_TOKEN = "<|content|>"

TOOL_OPEN = "<|tool_call:begin|>"
TOOL_CLOSE = "<|tool_call:end|>"

TOOL_NAME_PREFIX = "<|tool_call:name|>"
TOOL_ARGS_PREFIX = "<|tool_call:args|>"


class SolarOpenToolState(Enum):
    """State constants for Solar Open tool parser streaming operations."""

    NORMAL = "normal"
    FOUND_CONTENT = "found_content"
    FOUND_TOOL_CALL = "found_tool_call"


class SolarOpenReasoningParser(HermesReasoningParser):
    """Solar Open reasoning parser.
    
    Handles reasoning content in format: <|think|>reasoning_content<|end|>
    """

    def __init__(self) -> None:
        """Initialize Solar Open reasoning parser."""
        super().__init__(reasoning_open=REASONING_OPEN, reasoning_close=REASONING_CLOSE)


class SolarOpenToolParser(AbstractToolParser):
    """Solar Open tool parser.
    
    Handles tool calls in format:
    <|tool_call:begin|><tool-call-id><|tool_call:name|><tool-name><|tool_call:args|><args-json-object><|tool_call:end|>
    """

    def __init__(self) -> None:
        """Initialize Solar Open tool parser."""
        super().__init__(tool_open=TOOL_OPEN, tool_close=TOOL_CLOSE)
        self.state = SolarOpenToolState.NORMAL
        self.content_response_token = CONTENT_RESPONSE_TOKEN
        self.tool_name_prefix = TOOL_NAME_PREFIX
        self.tool_args_prefix = TOOL_ARGS_PREFIX

    def extract_tool_calls(self, model_output: str) -> dict[str, list] | None:
        """Extract tool calls from complete model output.
        
        Parameters
        ----------
        model_output : str
            Complete model output containing tool calls.
            
        Returns
        -------
        dict[str, list] | None
            Dictionary with 'content' or 'tool_calls' key, or None if parsing fails.
        """
        # Check for content response first
        content_idx = model_output.find(self.content_response_token)
        if content_idx != -1:
            content = model_output[content_idx + len(self.content_response_token) :]
            return {"content": content}

        # Parse tool calls
        tool_calls = []
        remaining_output = model_output

        while self.tool_open in remaining_output:
            tool_call_open_idx = remaining_output.find(self.tool_open)
            tool_call_name_idx = remaining_output.find(
                self.tool_name_prefix, tool_call_open_idx
            )
            tool_call_args_idx = remaining_output.find(
                self.tool_args_prefix, tool_call_name_idx
            )
            tool_call_close_idx = remaining_output.find(
                self.tool_close, tool_call_args_idx
            )

            # Validate all required tokens were found
            if (
                tool_call_name_idx == -1
                or tool_call_args_idx == -1
                or tool_call_close_idx == -1
            ):
                logger.warning(
                    f"Malformed tool call in output, missing required tokens: {remaining_output[:100]}"
                )
                break

            # Extract tool name and arguments
            tool_name = remaining_output[
                tool_call_name_idx + len(self.tool_name_prefix) : tool_call_args_idx
            ].strip()
            tool_args = remaining_output[
                tool_call_args_idx + len(self.tool_args_prefix) : tool_call_close_idx
            ].strip()

            # Validate JSON arguments
            try:
                json.loads(tool_args)  # Validate JSON format
                tool_calls.append({"name": tool_name, "arguments": tool_args})
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Invalid JSON in tool arguments for '{tool_name}': {e}"
                )
                # Skip this malformed tool call and continue

            # Move past this tool call
            remaining_output = remaining_output[
                tool_call_close_idx + len(self.tool_close) :
            ]

        return {
            "tool_calls": tool_calls,
            "content": remaining_output if remaining_output else None,
        }

    def extract_tool_calls_streaming(
        self, chunk: str
    ) -> tuple[dict[str, list] | str | None, bool]:
        """Extract tool calls from streaming chunks.
        
        Parameters
        ----------
        chunk : str
            Chunk of model output to process.
            
        Returns
        -------
        tuple[dict[str, list] | str | None, bool]
            Tuple of (extracted_content, is_complete) where:
            - extracted_content: Parsed content or None if buffering
            - is_complete: True if chunk should be sent, False if buffering
        """
        # Add chunk to buffer for processing
        self.buffer += chunk

        # Check for content response token
        if self.content_response_token in self.buffer:
            self.state = SolarOpenToolState.FOUND_CONTENT
            content_idx = self.buffer.find(self.content_response_token)
            content = self.buffer[content_idx + len(self.content_response_token) :]
            self.buffer = ""  # Clear buffer
            return {"content": content}, True

        # If already in content mode, stream content directly
        if self.state == SolarOpenToolState.FOUND_CONTENT:
            return {"content": chunk}, True

        # Check for tool call
        if self.tool_open in self.buffer:
            self.state = SolarOpenToolState.FOUND_TOOL_CALL

        # If in tool call state, buffer until we have complete tool call(s)
        if self.state == SolarOpenToolState.FOUND_TOOL_CALL:
            if self.tool_close in self.buffer:
                # We have at least one complete tool call
                result = self.extract_tool_calls(self.buffer)
                self.buffer = ""
                self.state = SolarOpenToolState.NORMAL
                return result, True
            # Still buffering
            return None, False

        # Normal state - keep buffering
        return None, False