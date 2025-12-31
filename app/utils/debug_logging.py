"""Debug logging utilities for request and generation statistics."""

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

