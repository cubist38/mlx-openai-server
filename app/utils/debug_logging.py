"""Debug logging utilities for request and generation statistics."""

import time
from typing import Any
from loguru import logger


def log_debug_request(request_dict: dict[str, Any]) -> None:
    """Log request details in a beautiful format for debug mode.
    
    Parameters
    ----------
    request_dict : dict[str, Any]
        The request dictionary to log.
    """
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("🔍 DEBUG: Request Details")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Extract and format key information
    if "messages" in request_dict:
        logger.info(f"📨 Messages: {len(request_dict['messages'])} message(s)")
        for i, msg in enumerate(request_dict["messages"], 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            content_preview = str(content)[:100] + "..." if len(str(content)) > 100 else str(content)
            logger.info(f"   {i}. [{role}] {content_preview}")
    
    if request_dict.get("max_tokens"):
        logger.info(f"🎯 Max Tokens: {request_dict['max_tokens']:,}")
    
    if request_dict.get("temperature"):
        logger.info(f"🌡️  Temperature: {request_dict['temperature']}")
    
    if request_dict.get("top_p"):
        logger.info(f"🎲 Top P: {request_dict['top_p']}")
    
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def log_debug_stats(
    prompt_tokens: int,
    generation_tokens: int,
    total_tokens: int,
    generation_tps: float,
    peak_memory: float,
) -> None:
    """Log generation statistics in a beautiful format for debug mode.
    
    Parameters
    ----------
    prompt_tokens : int
        Number of tokens in the prompt.
    generation_tokens : int
        Number of tokens generated.
    total_tokens : int
        Total number of tokens.
    generation_tps : float
        Generation speed in tokens per second.
    peak_memory : float
        Peak memory usage in GB.
    """
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("📊 DEBUG: Generation Statistics")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"🎫 Prompt Tokens:     {prompt_tokens:,}")
    logger.info(f"✨ Generation Tokens: {generation_tokens:,}")
    logger.info(f"📈 Total Tokens:      {total_tokens:,}")
    logger.info(f"⚡ Generation Speed:  {generation_tps:.2f} tokens/sec")
    logger.info(f"💾 Peak Memory:       {peak_memory:.2f} GB")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def log_debug_prompt(prompt: str) -> None:
    """Log input prompt in a beautiful format for debug mode.
    
    Parameters
    ----------
    prompt : str
        The input prompt to log.
    """
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━INPUT PROMPT━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(prompt)
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def log_debug_raw_text_response(raw_text: str) -> None:
    """Log raw text response in a beautiful format for debug mode.
    
    Parameters
    ----------
    raw_text : str
        The raw text response to log.
    """
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("📝 DEBUG: Raw Text Response")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"Raw text: {raw_text}")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def log_debug_cache_stats(total_input_tokens: int, remaining_tokens: int) -> None:
    """Log prompt cache statistics in a beautiful format for debug mode.
    
    Parameters
    ----------
    total_input_tokens : int
        Total number of input tokens before cache lookup.
    remaining_tokens : int
        Number of tokens remaining after cache hit.
    """
    cached_tokens = total_input_tokens - remaining_tokens
    cache_hit_ratio = (cached_tokens / total_input_tokens * 100) if total_input_tokens > 0 else 0.0
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("💾 DEBUG: Prompt Cache Statistics")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"📊 Total Input Tokens:  {total_input_tokens:,}")
    logger.info(f"✅ Cached Tokens:       {cached_tokens:,}")
    logger.info(f"🔄 Remaining Tokens:    {remaining_tokens:,}")
    logger.info(f"📈 Cache Hit Ratio:     {cache_hit_ratio:.1f}%")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def log_debug_chat_template(
    chat_template_file: str | None = None,
    template_content: str | None = None,
    preview_lines: int = 15,
) -> None:
    """Log chat template source, size, and a preview of content.

    Parameters
    ----------
    chat_template_file : str | None
        Path to custom chat template file, or None if using model default.
    template_content : str | None
        Content of the template (for size and preview). Only used when chat_template_file is set.
    preview_lines : int
        Maximum number of template lines to show in the log (default 15).
    """
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("✦ DEBUG: Chat Template")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if chat_template_file:
        logger.info(f"✦ Loaded custom chat template from: {chat_template_file}")
        if template_content:
            logger.info(f"✦ Chat template size: {len(template_content)} characters")
            lines = template_content.strip().splitlines()
            show = lines[:preview_lines]
            logger.info("✦ Template preview:")
            for i, line in enumerate(show, 1):
                logger.info(f"   {i:2d} | {line}")
            if len(lines) > preview_lines:
                logger.info(f"   ... ({len(lines) - preview_lines} more lines)")
    else:
        logger.info("✦ Using default chat template from model")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

def make_prompt_progress_callback(start_time: float | None = None) -> callable:
    """Create a callback function for tracking prompt processing progress.
    
    Parameters
    ----------
    start_time : float | None
        The start time for calculating speed. If None, uses current time.
        
    Returns
    -------
    callable
        A callback function that logs processing progress.
    """
    if start_time is None:
        start_time = time.time()
    
    def callback(processed: int, total_tokens: int) -> None:
        """Log prompt processing progress with speed metrics."""
        elapsed = time.time() - start_time
        speed = processed / elapsed if elapsed > 0 else 0
        logger.info(f"⚡ Processed {processed:6d}/{total_tokens} tokens ({speed:6.2f} tok/s)")
    
    return callback


def _preview_text(value: str, max_chars: int = 220) -> str:
    """Return a single-line preview string with a hard size cap."""
    escaped = value.replace("\n", "\\n")
    if len(escaped) <= max_chars:
        return escaped
    return f"{escaped[:max_chars]}...(+{len(escaped) - max_chars} chars)"


def _summarize_for_debug(value: Any, max_depth: int = 2) -> Any:
    """Summarize nested values to keep debug logs readable."""
    if max_depth <= 0:
        return type(value).__name__

    if isinstance(value, str):
        return _preview_text(value)

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    if isinstance(value, list):
        preview_items = [_summarize_for_debug(item, max_depth - 1) for item in value[:5]]
        if len(value) > 5:
            preview_items.append(f"...(+{len(value) - 5} items)")
        return {"type": "list", "len": len(value), "preview": preview_items}

    if isinstance(value, tuple):
        preview_items = [_summarize_for_debug(item, max_depth - 1) for item in value[:5]]
        if len(value) > 5:
            preview_items.append(f"...(+{len(value) - 5} items)")
        return {"type": "tuple", "len": len(value), "preview": preview_items}

    if isinstance(value, dict):
        summary: dict[str, Any] = {}
        for key, item in list(value.items())[:20]:
            summary[str(key)] = _summarize_for_debug(item, max_depth - 1)
        if len(value) > 20:
            summary["..."] = f"+{len(value) - 20} more keys"
        return summary

    return f"<{type(value).__name__}>"


def log_debug_server_request(
    route: str,
    request_payload: dict[str, Any],
    request_id: str | None = None,
) -> None:
    """Log inbound API payload after request parsing on the server."""
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("🛰️  DEBUG: Inbound Server Request")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"Route: {route}")
    if request_id:
        logger.info(f"Request ID: {request_id}")
    logger.info(f"Payload: {_summarize_for_debug(request_payload, max_depth=3)}")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def log_debug_model_dispatch(
    op_name: str,
    payload: dict[str, Any],
) -> None:
    """Log payload forwarded from handler to inference worker/model."""
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("🚀 DEBUG: Model Dispatch Payload")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"Operation: {op_name}")
    logger.info(f"Payload: {_summarize_for_debug(payload, max_depth=2)}")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def log_debug_parser_event(
    component: str,
    chunk_index: int,
    phase: str,
    parser: Any,
    text: str | None = None,
    parsed_content: Any | None = None,
    is_complete: bool | None = None,
) -> None:
    """Log parser state transitions and parse outputs for a chunk."""
    parser_name = parser.__class__.__name__ if parser is not None else "None"
    parser_state = getattr(parser, "state", None)
    state_repr = getattr(parser_state, "value", parser_state)
    parser_buffer = getattr(parser, "buffer", None)
    buffer_len = len(parser_buffer) if isinstance(parser_buffer, str) else None

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("🧩 DEBUG: Parser Event")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"Component: {component}")
    logger.info(f"Chunk: {chunk_index}")
    logger.info(f"Phase: {phase}")
    logger.info(f"Parser: {parser_name}")
    logger.info(f"State: {state_repr}")
    if buffer_len is not None:
        logger.info(f"Buffer len: {buffer_len}")
    if text is not None:
        logger.info(f"Chunk text: {_preview_text(text)}")
    if parsed_content is not None:
        logger.info(f"Parsed: {_summarize_for_debug(parsed_content, max_depth=2)}")
    if is_complete is not None:
        logger.info(f"Is complete: {is_complete}")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def log_debug_tool_call_emission(
    component: str,
    chunk_index: int,
    tool_call: dict[str, Any],
) -> None:
    """Log each emitted tool call payload."""
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("🛠️  DEBUG: Tool Call Emission")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info(f"Component: {component}")
    logger.info(f"Chunk: {chunk_index}")
    logger.info(f"Tool call: {_summarize_for_debug(tool_call, max_depth=3)}")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
