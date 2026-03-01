"""Regression tests for streaming tool-call ID behavior."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterable
import json
from typing import Any
import unittest

from app.api.endpoints import handle_stream_response


class StreamToolCallIdTests(unittest.TestCase):
    """Validate tool-call IDs and indices in chat-completions streaming mode."""

    @staticmethod
    async def _chunk_generator(
        chunks: Iterable[str | dict[str, Any]],
    ) -> AsyncGenerator[str | dict[str, Any], None]:
        """Yield chunks that emulate handler streaming output."""
        for chunk in chunks:
            yield chunk

    async def _collect_stream_payloads(
        self,
        chunks: Iterable[str | dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Collect JSON payloads emitted by the SSE stream wrapper."""
        payloads: list[dict[str, Any]] = []
        async for event in handle_stream_response(
            self._chunk_generator(chunks),
            model="test-model",
        ):
            if not event.startswith("data: "):
                continue
            serialized = event[len("data: ") :].strip()
            if serialized == "[DONE]":
                continue
            payloads.append(json.loads(serialized))
        return payloads

    @staticmethod
    def _extract_tool_delta_entries(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract ``delta.tool_calls[0]`` entries from streamed payloads."""
        tool_entries: list[dict[str, Any]] = []
        for payload in payloads:
            choices = payload.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            tool_calls = delta.get("tool_calls") or []
            if tool_calls:
                tool_entries.append(tool_calls[0])
        return tool_entries

    def test_tool_call_id_is_stable_for_same_streamed_tool_call(self) -> None:
        """Tool-call ID must remain constant across deltas for one tool index."""
        chunks: list[dict[str, str]] = [
            {"name": "get_weather"},
            {"arguments": '{"city":"'},
            {"arguments": 'Boston"}'},
        ]

        payloads = asyncio.run(self._collect_stream_payloads(chunks))
        tool_entries = self._extract_tool_delta_entries(payloads)

        assert len(tool_entries) == 3
        assert [entry["index"] for entry in tool_entries] == [0, 0, 0]

        tool_ids = [entry["id"] for entry in tool_entries]
        assert len(set(tool_ids)) == 1, (
            "Expected one stable tool_call id for all deltas of index 0."
        )

    def test_tool_call_index_and_id_advance_with_new_calls(self) -> None:
        """A new tool call should increment index and use a new stable ID."""
        chunks: list[dict[str, str]] = [
            {"name": "get_weather"},
            {"arguments": '{"city":"Boston"}'},
            {"name": "get_time"},
            {"arguments": '{"timezone":"UTC"}'},
        ]

        payloads = asyncio.run(self._collect_stream_payloads(chunks))
        tool_entries = self._extract_tool_delta_entries(payloads)

        assert len(tool_entries) == 4
        assert [entry["index"] for entry in tool_entries] == [0, 0, 1, 1]

        tool_ids = [entry["id"] for entry in tool_entries]
        assert tool_ids[0] == tool_ids[1], "Expected index 0 deltas to reuse the same tool_call id."
        assert tool_ids[2] == tool_ids[3], "Expected index 1 deltas to reuse the same tool_call id."
        assert tool_ids[0] != tool_ids[2], (
            "Expected different tool calls to have different tool_call ids."
        )


if __name__ == "__main__":
    unittest.main()
