from __future__ import annotations

import re

from .hermes import HermesReasoningParser
from .glm4_moe import GLM4MoEToolParser

TOOL_OPEN = "<minimax:tool_call>"
TOOL_CLOSE = "</minimax:tool_call>"
THINKING_OPEN = "<think>"
THINKING_CLOSE = "</think>"

class MiniMaxM2ReasoningParser(HermesReasoningParser):
    """Reasoning parser for MiniMax M2 model's reasoning response format.

    Handles the MiniMax M2 model's reasoning response format:
    <think>reasoning_content</think>
    """

    def __init__(self) -> None:
        """Initialize the Hermes4 reasoning parser with appropriate regex patterns."""
        super().__init__(reasoning_open=THINKING_OPEN, reasoning_close=THINKING_CLOSE)
    
    def needs_redacted_reasoning_prefix(self) -> bool:
        return True


class MiniMaxM2ToolParser(GLM4MoEToolParser):
    """Tool parser for MiniMax M2 model's tool response format.

    Handles the MiniMax M2 model's tool response format:
    <minimax:tool_call>
    <invoke name="tool-name-1">
    <parameter name="param-key-1">param-value-1</parameter>
    <parameter name="param-key-2">param-value-2</parameter>
    ...
    </invoke>
    <minimax:tool_call>
    """

    def __init__(self, tool_open: str = TOOL_OPEN, tool_close: str = TOOL_CLOSE) -> None:
        """Initialize the MiniMax M2 tool parser with appropriate regex patterns."""
        super().__init__(tool_open=tool_open, tool_close=tool_close)
        
        self.func_call_regex = re.compile(
            r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL
        )
        
        # Regex patterns for parsing MiniMax tool calls
        self.func_detail_regex = re.compile(
            r'<invoke name="([^"]+)"\s*>(.*)', re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r'<parameter name="([^"]+)"\s*>([^<]*)</parameter>', re.DOTALL
        )

if __name__ == "__main__":
    reasoning_parser = MiniMaxM2ReasoningParser()
    tool_parser = MiniMaxM2ToolParser()

    chunks = [
        "I am ",
        "thinking about the",
        "problem",
        ".</think><minimax:tool_call>",
        "<invoke name=\"tool_name\">\n",
        "<parameter name=\"argument_name\">argument_value</parameter>\n",
        "<parameter name=\"argument_name\">argument_value</parameter>\n",
        "</invoke>\n",
        "</minimax:tool_call>",
    ]
    after_reasoning_close_content = None
    is_first_chunk = True
    for chunk in chunks:
        if chunk is None:
            continue
        if is_first_chunk:
            if reasoning_parser and reasoning_parser.needs_redacted_reasoning_prefix():
                chunk = reasoning_parser.get_reasoning_open() + chunk
                is_first_chunk = False
        if reasoning_parser:
            parsed_content, is_complete = reasoning_parser.extract_reasoning_streaming(chunk)
            if parsed_content:
                after_reasoning_close_content = parsed_content.get("after_reasoning_close_content")
                print("Parsed content: ", parsed_content)
            if is_complete:
                reasoning_parser = None
            if after_reasoning_close_content:
                chunk = after_reasoning_close_content
                after_reasoning_close_content = None
            else:
                continue
        if tool_parser:
            print("Chunk: ", chunk)
            parsed_content, is_complete = tool_parser.extract_tool_calls_streaming(chunk)
            if parsed_content:
                tool_calls = parsed_content.pop("tool_calls", None)
                for tool_call in tool_calls:
                    print(f"Tool call: {tool_call}")
            if is_complete:
                tool_parser = None
            continue
        print(f"Text: {chunk}")