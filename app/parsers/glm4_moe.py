from __future__ import annotations

import re
import json

from .abstract_parser import (
    AbstractToolParser,
    ToolParserState,
)
from .hermes import HermesReasoningParser

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
REASONING_OPEN = "<think>"
REASONING_CLOSE = "</think>"


class GLM4MoEReasoningParser(HermesReasoningParser):
    """Reasoning parser for GLM4 MoE model's reasoning response format.

    Handles the GLM4 MoE model's reasoning response format:
    <think>reasoning_content</think>
    """

    def __init__(self) -> None:
        """Initialize the Hermes4 reasoning parser with appropriate regex patterns."""
        super().__init__(reasoning_open=REASONING_OPEN, reasoning_close=REASONING_CLOSE)
    
    def respects_enable_thinking(self) -> bool:
        """Check if the reasoning parser respects the enable_thinking flag.
        
        Returns
        -------
        bool
            True if the reasoning parser respects the enable_thinking flag, False otherwise.
        """
        return True


class GLM4MoEToolParser(AbstractToolParser):
    """Tool parser for GLM4 MoE model's tool response format.

    Handles the GLM4 MoE model's tool response format:
    <tool_call>{function_name}
    <arg_key>{arg-key-1}</arg_key>
    <arg_value>{arg-value-1}</arg_value>
    <arg_key>{arg-key-2}</arg_key>
    <arg_value>{arg-value-2}</arg_value>
    ...
    <arg_key>{arg-key-n}</arg_key>
    <arg_value>{arg-value-n}</arg_value>
    </tool_call>
    """

    def __init__(self, tool_open: str = TOOL_OPEN, tool_close: str = TOOL_CLOSE) -> None:
        """Initialize the Qwen3 VL tool parser with appropriate regex patterns."""
        super().__init__(tool_open=tool_open, tool_close=tool_close)
        
        self.func_call_regex = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
        self.func_detail_regex = re.compile(
            r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )

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
        matches = self.func_call_regex.findall(model_output)
        if not matches:
            return {
                "content": model_output
            }
        tool_calls = []
        for match in matches:
            tc_detail = self.func_detail_regex.search(match)
            tc_name = tc_detail.group(1)
            tc_args = tc_detail.group(2)
            pairs = self.func_arg_regex.findall(tc_args)
            arg_dct = {}
            for key, value in pairs:
                arg_key = key.strip()
                arg_value = value.strip()
                arg_dct[arg_key] = arg_value
            tool_calls.append({
                "name": tc_name,
                "arguments": json.dumps(arg_dct, ensure_ascii=False)
            })
        return {
            "tool_calls": tool_calls
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

        return {
            "content": chunk
        }, True

if __name__ == "__main__":
    reasoning_parser = GLM4MoEReasoningParser()
    tool_parser = GLM4MoEToolParser()

    chunks = [
        "<think>The user is asking",
        "for the weather in Tokyo. I have a function available called \"get_weather\" that takes a city parameter and returns the weather information for that city. The user has provided \"Tokyo\" as the city they want weather information for, so I have all the required parameters to make the function call.",
        "call.</think>",
        "I'll get the current weather information for Tokyo for you.",
        "<tool_call>get_weather",
        "<arg_key>city</arg_key>",
        "<arg_value>Tokyo</arg_value>",
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
            parsed_content, is_complete = tool_parser.extract_tool_calls_streaming(chunk)
            if parsed_content:
                tool_calls = parsed_content.get("tool_calls")
                if tool_calls:
                    for tool_call in tool_calls:
                        print("Tool call: ", tool_call)