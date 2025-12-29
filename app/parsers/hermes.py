from __future__ import annotations

import re
import json

from .abstract_parser import (
    AbstractReasoningParser,
    AbstractToolParser,
    ReasoningParserState,
    ToolParserState,
)

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
REASONING_OPEN = "<think>"
REASONING_CLOSE = "</think>"


class HermesReasoningParser(AbstractReasoningParser):
    """Reasoning parser for Hermes model's reasoning response format.

    Handles the Hermes model's reasoning response format:
    <think>reasoning_content</think>
    """

    def __init__(self, reasoning_open: str = REASONING_OPEN, reasoning_close: str = REASONING_CLOSE) -> None:
        """Initialize the Hermes reasoning parser with appropriate regex patterns."""
        super().__init__(reasoning_open=reasoning_open, reasoning_close=reasoning_close)
        self.reasoning_regex = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    def extract_reasoning(self, model_output: str) -> dict[str, str] | None:
        """Extract reasoning content from complete model output.
        
        Parameters
        ----------
        model_output : str
            Complete model output containing reasoning tags.
            
        Returns
        -------
        dict[str, str] | None
            Dictionary with 'reasoning' key containing extracted content,
            or None if no reasoning found.
        """
        matches = self.reasoning_regex.findall(model_output)
        after_reasoning_close_content = None
        if not matches:
            return {
                "content": model_output
            }
        reasoning_content_end_idx = model_output.rfind(self.reasoning_close)
        after_reasoning_close_content = model_output[reasoning_content_end_idx + len(self.reasoning_close):]
        return {
            "reasoning_content": matches[0],
            "after_reasoning_close_content": after_reasoning_close_content
        }

    def extract_reasoning_streaming(
        self, chunk: str
    ) -> tuple[dict[str, str] | str | None, bool]:
        """Extract reasoning content from streaming chunks.
        
        Parameters
        ----------
        chunk : str
            Chunk of model output to process.
            
        Returns
        -------
        tuple[dict[str, str] | str | None, bool]
            Tuple of (extracted_content, is_complete) where:
            - extracted_content: Reasoning dict if complete, chunk if passthrough, None if buffering
            - is_complete: True if chunk should be sent, False if still buffering
        """

        if self.reasoning_open in chunk:
            self.state = ReasoningParserState.FOUND_PREFIX
            reasoning_content_start_idx = chunk.find(self.reasoning_open)
            reasoning_content = chunk[reasoning_content_start_idx + len(self.reasoning_open):]
            return {
                "reasoning_content": reasoning_content
            }, False

        if self.state == ReasoningParserState.FOUND_PREFIX:
            if self.reasoning_close in chunk:
                reasoning_content_end_idx = chunk.find(self.reasoning_close)
                reasoning_content = chunk[:reasoning_content_end_idx]
                after_reasoning_close_content = chunk[reasoning_content_end_idx + len(self.reasoning_close):]
                return {
                    "reasoning_content": reasoning_content,
                    "after_reasoning_close_content": after_reasoning_close_content
                }, True
            else:
                reasoning_content = chunk
                return {
                    "reasoning_content": reasoning_content
                }, False
        
        return {
            "content": chunk
        }, True



class HermesToolParser(AbstractToolParser):
    """Tool parser for Hermes model's tool response format.

    Handles the Hermes model's tool response format:
    <tool_call>{"name": "tool_name", "arguments": {"argument_name": "argument_value"}}</tool_call>
    """

    def __init__(self, tool_open: str = TOOL_OPEN, tool_close: str = TOOL_CLOSE) -> None:
        """Initialize the Hermes4 tool parser with appropriate regex patterns."""
        super().__init__(tool_open=tool_open, tool_close=tool_close)
        self.tool_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    def extract_tool_calls(self, model_output: str) -> dict[str, list] | None:
        """Extract tool calls from complete model output.
        
        Parameters
        ----------
        model_output : str
            Complete model output containing tool calls in JSON format.
            
        Returns
        -------
        dict[str, list] | None
            Dictionary with 'tool_calls' key containing list of parsed tool calls,
            or None if no tool calls found. Each tool call has 'name' and 'arguments'.
        """
        matches = self.tool_regex.findall(model_output)
        if not matches:
            return {
                "content": model_output
            }
        tool_calls = []
        for match in matches:
            try:
                tool_data = json.loads(match.strip())
                tool_calls.append(
                    {
                        "name": tool_data.get("name", ""),
                        "arguments": json.dumps(tool_data.get("arguments", {})),
                    }
                )
            except json.JSONDecodeError:
                # Skip malformed tool calls
                continue
        return {"tool_calls": tool_calls}

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
            - extracted_content: Tool calls dict if complete, chunk if passthrough, None if buffering
            - is_complete: True if chunk should be sent, False if still buffering
        """
        if self.tool_open in chunk:
            self.state = ToolParserState.FOUND_PREFIX

            if self.tool_close in chunk:
                self.buffer += chunk
                result = self.extract_tool_calls(self.buffer)
                self.buffer = ""
                self.state = ToolParserState.NORMAL
                return result, True
            else:
                self.buffer += chunk
                return None, False

        if self.state == ToolParserState.FOUND_PREFIX:
            if self.tool_close in chunk:
                self.buffer += chunk
                result = self.extract_tool_calls(self.buffer)
                self.buffer = ""
                self.state = ToolParserState.NORMAL
                return result, True
            else:
                self.buffer += chunk
                return None, False

        return chunk, True

if __name__ == "__main__":
    reasoning_parser = HermesReasoningParser()
    tool_parser = HermesToolParser()

    chunks = [
        "<think>I am ",
        "thinking about the",
        "problem",
        ".</think><tool_call>",
        "{\"name\": \"tool_name\",",
        "\"arguments\": {\"argument_name\": \"argument_value\"}}",
        "</tool_call>",
    ]
    after_thinking_close_content = None
    for chunk in chunks:
        if chunk is None:
            continue
        if reasoning_parser:
            reasoning, is_complete = reasoning_parser.extract_reasoning_streaming(chunk)
            print("Reasoning: ", reasoning)
            if is_complete:
                reasoning_parser = None
                if reasoning.get("content"):
                    after_thinking_close_content = reasoning.get("content")
            continue
        if after_thinking_close_content:
            chunk = after_thinking_close_content + chunk
            after_thinking_close_content = None
        print("Chunk: ", chunk)
        if tool_parser:
            tool_calls, is_complete = tool_parser.extract_tool_calls_streaming(chunk)
            print("Tool calls: ", tool_calls)
            print("Is complete: ", is_complete)