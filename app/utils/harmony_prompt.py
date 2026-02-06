"""
Harmony prompt renderer for GPT-OSS models.

Uses the official openai_harmony library to properly format prompts with correct
Harmony tokens (<|call|>, <|return|>, etc.) instead of relying on the HF chat template.
"""

from __future__ import annotations

import json
import datetime
from typing import Any, List, Dict, Optional

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    TextContent,
    ToolDescription,
    load_harmony_encoding,
)

# Cache the encoding to avoid reloading
_encoding = None

def get_harmony_encoding():
    """Get cached Harmony encoding."""
    global _encoding
    if _encoding is None:
        _encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _encoding


def _convert_openai_tools_to_harmony(tools: List[Dict[str, Any]]) -> List[ToolDescription]:
    """Convert OpenAI-format tools to Harmony ToolDescription objects."""
    harmony_tools = []
    
    for tool in tools:
        if tool.get("type") != "function":
            continue
            
        func = tool.get("function", {})
        name = func.get("name", "")
        description = func.get("description", "")
        parameters = func.get("parameters", {})
        
        tool_desc = ToolDescription.new(
            name=name,
            description=description,
            parameters=parameters
        )
        harmony_tools.append(tool_desc)
    
    return harmony_tools


def _build_system_message(
    tools: Optional[List[Dict[str, Any]]] = None,
    reasoning_effort: str = "medium"
) -> Message:
    """Build the system message with optional tools."""
    effort_map = {
        "high": ReasoningEffort.HIGH,
        "medium": ReasoningEffort.MEDIUM,
        "low": ReasoningEffort.LOW,
    }
    
    content = (
        SystemContent.new()
        .with_reasoning_effort(effort_map.get(reasoning_effort, ReasoningEffort.MEDIUM))
        .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
    )
    
    return Message.from_role_and_content(Role.SYSTEM, content)


def _build_developer_message(
    tools: Optional[List[Dict[str, Any]]] = None,
    instructions: str = ""
) -> Optional[Message]:
    """Build the developer message with tool definitions."""
    if not tools and not instructions:
        return None
    
    content = DeveloperContent.new()
    
    if instructions:
        content = content.with_instructions(instructions)
    
    if tools:
        harmony_tools = _convert_openai_tools_to_harmony(tools)
        if harmony_tools:
            content = content.with_function_tools(harmony_tools)
    
    return Message.from_role_and_content(Role.DEVELOPER, content)


def render_harmony_prompt(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    reasoning_effort: str = "medium"
) -> str:
    """
    Render OpenAI-style messages to Harmony format prompt.
    
    This bypasses the HF chat template and uses the official openai_harmony
    library to ensure proper token formatting for tool calls and responses.
    
    Args:
        messages: List of OpenAI-format messages
        tools: Optional list of tool definitions
        reasoning_effort: "low", "medium", or "high"
    
    Returns:
        Properly formatted Harmony prompt string
    """
    encoding = get_harmony_encoding()
    harmony_messages = []
    
    # First pass: collect system message content from user messages
    system_instructions = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if content:
                system_instructions.append(content)
    
    # Build system message (with Harmony metadata)
    system_message = _build_system_message(tools, reasoning_effort)
    harmony_messages.append(system_message)
    
    # Build developer message with tools AND user's system instructions
    combined_instructions = "\n\n".join(system_instructions)
    developer_message = _build_developer_message(tools, combined_instructions)
    if developer_message:
        harmony_messages.append(developer_message)
    
    # Process each OpenAI message
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            # Already handled above - merged into developer message
            continue
            
        elif role == "developer":
            # Developer instructions
            if content:
                dev_content = DeveloperContent.new().with_instructions(content)
                harmony_messages.append(Message.from_role_and_content(Role.DEVELOPER, dev_content))
        
        elif role == "user":
            # User message
            if content:
                harmony_messages.append(Message.from_role_and_content(Role.USER, content))
        
        elif role == "assistant":
            # Assistant message - may have tool_calls and/or content
            tool_calls = msg.get("tool_calls", [])
            reasoning_content = msg.get("reasoning_content")
            
            if tool_calls:
                # This is a tool call message
                for tc in tool_calls:
                    func = tc.get("function", {})
                    func_name = func.get("name", "")
                    func_args = func.get("arguments", "{}")
                    
                    # Create tool call message
                    # The author sends TO the function (recipient)
                    tool_call_msg = (
                        Message(
                            author=Author.new(Role.ASSISTANT, "assistant"),
                            content=[TextContent(text=func_args)]
                        )
                        .with_recipient(f"functions.{func_name}")
                        .with_channel("commentary")
                    )
                    harmony_messages.append(tool_call_msg)
            elif content:
                # Regular assistant response
                harmony_messages.append(Message.from_role_and_content(Role.ASSISTANT, content))
        
        elif role == "tool":
            # Tool response message
            tool_call_id = msg.get("tool_call_id", "")
            tool_name = msg.get("name", "")
            
            # Try to find the function name from the tool_call_id
            # or use the name field directly
            if not tool_name and tool_call_id:
                # Look for matching tool call in previous messages
                for prev_msg in messages:
                    for tc in prev_msg.get("tool_calls", []):
                        if tc.get("id") == tool_call_id:
                            tool_name = tc.get("function", {}).get("name", "")
                            break
            
            if not tool_name:
                tool_name = "unknown_function"
            
            # Create tool response message
            # The author is the function, sending TO assistant
            tool_response_msg = (
                Message(
                    author=Author.new(Role.TOOL, f"functions.{tool_name}"),
                    content=[TextContent(text=content or "")]
                )
                .with_recipient("assistant")
                .with_channel("commentary")
            )
            harmony_messages.append(tool_response_msg)
    
    # Render the conversation for completion
    conversation = Conversation.from_messages(harmony_messages)
    tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
    
    # Decode tokens back to string
    return encoding.decode(tokens)
