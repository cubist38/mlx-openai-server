from __future__ import annotations

import json
import re

from loguru import logger

from .abstract_parser import (
    AbstractReasoningParser,
    AbstractToolParser,
    ReasoningParserState,
    _suffix_prefix_overlap,
)

REASONING_OPEN = "<|channel>thought"
REASONING_CLOSE = "<channel|>"

TOOL_OPEN = "<|tool_call>"
TOOL_CLOSE = "<tool_call|>"

STRING_OPEN = '<|"|>'
STRING_CLOSE = '<|"|>'


# ---------------------------------------------------------------------------
# Gemma 4 value format parser
# ---------------------------------------------------------------------------
# The model serialises tool-call arguments in a custom format:
#   - strings:  <|"|>text<|"|>
#   - booleans: true / false
#   - null:     null
#   - numbers:  raw digits (int or float)
#   - objects:  {key:value,key:value}   (keys are bare identifiers)
#   - arrays:   [value,value]
# ---------------------------------------------------------------------------


def _parse_value(text: str, pos: int) -> tuple[object, int]:
    """Parse a single value starting at *pos* and return (value, new_pos)."""
    if pos >= len(text):
        return None, pos

    # String
    if text[pos : pos + len(STRING_OPEN)] == STRING_OPEN:
        start = pos + len(STRING_OPEN)
        end = text.index(STRING_CLOSE, start)
        return text[start:end], end + len(STRING_CLOSE)

    # Object
    if text[pos] == "{":
        return _parse_object(text, pos)

    # Array
    if text[pos] == "[":
        return _parse_array(text, pos)

    # Boolean / null
    if text[pos : pos + 4] == "true":
        return True, pos + 4
    if text[pos : pos + 5] == "false":
        return False, pos + 5
    if text[pos : pos + 4] == "null":
        return None, pos + 4

    # Number – consume until delimiter
    end = pos
    while end < len(text) and text[end] not in ",}]":
        end += 1
    num_str = text[pos:end]
    try:
        return (float(num_str) if "." in num_str else int(num_str)), end
    except ValueError:
        return num_str, end


def _parse_object(text: str, pos: int) -> tuple[dict, int]:
    """Parse ``{key:value, ...}`` starting at *pos*."""
    pos += 1  # skip '{'
    result: dict = {}
    while pos < len(text) and text[pos] != "}":
        # Key – bare identifier (letters, digits, underscores)
        key_end = pos
        while key_end < len(text) and text[key_end] not in ":}":
            key_end += 1
        key = text[pos:key_end]
        pos = key_end
        if pos < len(text) and text[pos] == ":":
            pos += 1  # skip ':'
        value, pos = _parse_value(text, pos)
        result[key] = value
        if pos < len(text) and text[pos] == ",":
            pos += 1  # skip ','
    if pos < len(text) and text[pos] == "}":
        pos += 1
    return result, pos


def _parse_array(text: str, pos: int) -> tuple[list, int]:
    """Parse ``[value, ...]`` starting at *pos*."""
    pos += 1  # skip '['
    result: list = []
    while pos < len(text) and text[pos] != "]":
        value, pos = _parse_value(text, pos)
        result.append(value)
        if pos < len(text) and text[pos] == ",":
            pos += 1
    if pos < len(text) and text[pos] == "]":
        pos += 1
    return result, pos


def _parse_tool_call_body(body: str) -> dict | None:
    """Parse ``call:func_name{args}`` into ``{name, arguments}``."""
    m = re.match(r"call:(\w+)", body.strip())
    if not m:
        return None
    name = m.group(1)
    brace_idx = body.find("{", m.end())
    if brace_idx == -1:
        return {"name": name, "arguments": "{}"}
    try:
        args, _ = _parse_object(body, brace_idx)
    except (ValueError, IndexError):
        logger.warning(f"Failed to parse Gemma4 tool call arguments: {body[:120]}")
        return None
    return {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}


# ---------------------------------------------------------------------------
# Reasoning parser
# ---------------------------------------------------------------------------


class Gemma4ReasoningParser(AbstractReasoningParser):
    r"""Reasoning parser for Gemma 4 models.

    Thinking content is wrapped in:
        <|channel>thought\n ... <channel|>
    """

    def __init__(
        self,
        reasoning_open: str = REASONING_OPEN,
        reasoning_close: str = REASONING_CLOSE,
    ) -> None:
        super().__init__(reasoning_open=reasoning_open, reasoning_close=reasoning_close)
        self.reasoning_regex = re.compile(
            re.escape(reasoning_open) + r"\n?(.*?)" + re.escape(reasoning_close),
            re.DOTALL,
        )

    def respects_enable_thinking(self) -> bool:
        return True

    def extract_reasoning(self, model_output: str) -> dict[str, str] | None:
        matches = self.reasoning_regex.findall(model_output)
        if not matches:
            return {"content": model_output}
        reasoning_content_end_idx = model_output.rfind(self.reasoning_close)
        after = model_output[reasoning_content_end_idx + len(self.reasoning_close) :]
        return {
            "reasoning_content": matches[0],
            "after_reasoning_close_content": after,
        }

    def extract_reasoning_streaming(self, chunk: str) -> tuple[dict[str, str] | str | None, bool]:
        if self.state == ReasoningParserState.NORMAL:
            combined = self.buffer + chunk

            if self.reasoning_open in combined:
                self.state = ReasoningParserState.FOUND_PREFIX
                start_idx = combined.find(self.reasoning_open)
                before = combined[:start_idx]
                reasoning_content = combined[start_idx + len(self.reasoning_open) :].lstrip("\n")

                if self.reasoning_close in reasoning_content:
                    end_idx = reasoning_content.find(self.reasoning_close)
                    after = reasoning_content[end_idx + len(self.reasoning_close) :]
                    self.state = ReasoningParserState.NORMAL
                    self.buffer = ""
                    result: dict[str, str] = {
                        "reasoning_content": reasoning_content[:end_idx],
                        "after_reasoning_close_content": after,
                    }
                    if before:
                        result["content"] = before
                    return result, True

                overlap = _suffix_prefix_overlap(reasoning_content, self.reasoning_close)
                if overlap > 0:
                    emitted = reasoning_content[:-overlap]
                    self.buffer = reasoning_content[-overlap:]
                else:
                    emitted = reasoning_content
                    self.buffer = ""

                result = {}
                if before:
                    result["content"] = before
                if emitted:
                    result["reasoning_content"] = emitted
                return result or None, False

            # No full opening tag — check for partial match at end of combined
            overlap = _suffix_prefix_overlap(combined, self.reasoning_open)
            if overlap > 0:
                passthrough = combined[:-overlap]
                self.buffer = combined[-overlap:]
            else:
                passthrough = combined
                self.buffer = ""

            if passthrough:
                return {"content": passthrough}, False
            return None, False

        # FOUND_PREFIX state — inside reasoning content
        combined = self.buffer + chunk
        if self.reasoning_close in combined:
            end_idx = combined.find(self.reasoning_close)
            reasoning_content = combined[:end_idx]
            after = combined[end_idx + len(self.reasoning_close) :]
            self.buffer = ""
            self.state = ReasoningParserState.NORMAL
            return {
                "reasoning_content": reasoning_content,
                "after_reasoning_close_content": after,
            }, True

        overlap = _suffix_prefix_overlap(combined, self.reasoning_close)
        if overlap > 0:
            reasoning_content = combined[:-overlap]
            self.buffer = combined[-overlap:]
        else:
            reasoning_content = combined
            self.buffer = ""

        if reasoning_content:
            return {"reasoning_content": reasoning_content}, False
        return None, False


# ---------------------------------------------------------------------------
# Tool call parser
# ---------------------------------------------------------------------------


class Gemma4ToolParser(AbstractToolParser):
    """Tool parser for Gemma 4 models.

    Tool calls use the format:
        <|tool_call>call:func_name{key:<|"|>value<|"|>,num:42}<tool_call|>
    """

    def __init__(self, tool_open: str = TOOL_OPEN, tool_close: str = TOOL_CLOSE) -> None:
        super().__init__(tool_open=tool_open, tool_close=tool_close)
        self.tool_call_regex = re.compile(
            re.escape(TOOL_OPEN) + r"(.*?)" + re.escape(TOOL_CLOSE),
            re.DOTALL,
        )

    def extract_tool_calls(self, model_output: str) -> dict[str, list] | None:
        matches = self.tool_call_regex.findall(model_output)
        if not matches:
            return {"content": model_output}
        tool_calls = []
        for match in matches:
            parsed = _parse_tool_call_body(match)
            if parsed:
                tool_calls.append(parsed)
        if not tool_calls:
            return {"content": model_output}
        return {"tool_calls": tool_calls}
