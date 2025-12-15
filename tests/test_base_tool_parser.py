"""Tests for the base tool parser."""

import unittest

from mlx_openai_server.handler.parser.base import BaseToolParser  # type: ignore[import-not-found]


class TestBaseToolParser(unittest.TestCase):
    """Test cases for BaseToolParser."""

    def setUp(self) -> None:
        """Set up test cases."""
        self.test_cases = [
            {
                "name": "simple function call",
                "chunks": [
                    "Some text before <tool_call>\n",
                    '{"name": "get_weather", "arguments": {"city": "Hue"}}\n',
                    "</tool_call>\nMore text after\n",
                    "<tool_call>\n",
                    '{"name": "get_weather", "arguments": {"city": "Sydney"}}\n',
                    "</tool_call>\nFinal text",
                ],
                "expected_outputs": [
                    "Some text before ",  # Text before tool call
                    {"name": "get_weather", "arguments": '{"city": "Hue"}'},  # Complete tool call
                    "",  # Empty string from chunk with opening tag
                    {
                        "name": "get_weather",
                        "arguments": '{"city": "Sydney"}',
                    },  # Complete tool call
                ],
            },
            {
                "name": "streaming function call",
                "chunks": [
                    "Start <tool_call>\n",
                    '{"name": "python", "arguments": ',
                    '{"code": "print(\'hello\')"}}\n',
                    "</tool_call>\nEnd",
                ],
                "expected_outputs": [
                    "Start ",  # Text before tool call
                    {
                        "name": "python",
                        "arguments": '{"code": "print(\'hello\')"}',
                    },  # Complete tool call
                ],
            },
        ]

    def test_parse_stream(self) -> None:
        """Test parsing stream."""
        for test_case in self.test_cases:
            with self.subTest(msg=test_case["name"]):
                parser = BaseToolParser("<tool_call>", "</tool_call>")
                outputs = []

                for chunk in test_case["chunks"]:
                    parsed, _complete = parser.parse_stream(chunk)
                    if parsed is not None:
                        if isinstance(parsed, list):
                            outputs.extend(parsed)
                        else:
                            outputs.append(parsed)

                # Get any remaining content
                remaining, _complete = parser.parse_stream(None)
                if remaining is not None:
                    outputs.append(remaining)

                assert len(outputs) == len(test_case["expected_outputs"]), (
                    f"Expected {len(test_case['expected_outputs'])} outputs, got {len(outputs)}"
                )

                for i, (output, expected) in enumerate(
                    zip(outputs, test_case["expected_outputs"], strict=True)
                ):
                    assert output == expected, f"Chunk {i}: Expected {expected}, got {output}"
