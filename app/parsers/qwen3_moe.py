from __future__ import annotations

from .hermes import HermesReasoningParser, HermesToolParser

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
REASONING_OPEN = "<think>"
REASONING_CLOSE = "</think>"


class Qwen3MoEReasoningParser(HermesReasoningParser):
    """Reasoning parser for Qwen3 MoE model's reasoning response format.

    Handles the Qwen3 MoE model's reasoning response format:
    <think>reasoning_content</think>
    """

    def __init__(self, reasoning_open: str = REASONING_OPEN, reasoning_close: str = REASONING_CLOSE) -> None:
        """Initialize the Qwen3 MoE reasoning parser with appropriate regex patterns."""
        super().__init__(reasoning_open=reasoning_open, reasoning_close=reasoning_close)

    def needs_redacted_reasoning_prefix(self) -> bool:
        """Check if the reasoning parser needs a redacted reasoning prefix.
        
        Returns
        -------
        bool
            True if the reasoning parser needs a redacted reasoning prefix, False otherwise.
        """
        return True

Qwen3MoEToolParser = HermesToolParser

if __name__ == "__main__":
    reasoning_parser = Qwen3MoEReasoningParser()
    tool_parser = Qwen3MoEToolParser()

    chunks = [
        "I am ",
        "thinking about the",
        "problem",
        ".</think><tool_call>",
        "{\"name\": \"tool_name\",",
        "\"arguments\": {\"argument_name\": \"argument_value\"}}",
        "</tool_call>",
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