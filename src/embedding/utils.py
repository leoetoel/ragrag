"""
Utility functions for embedding module
"""

import logging

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def check_text_length(texts: list, max_length: int = 512, avg_chars_per_token: float = 2.0) -> dict:
    """
    Check if texts exceed max length.

    Args:
        texts: List of texts to check
        max_length: Maximum token length
        avg_chars_per_token: Average characters per token (default: 2.0 for Chinese)

    Returns:
        Dictionary with statistics
    """
    total_texts = len(texts)
    lengths = [len(t) for t in texts]
    estimated_tokens = [l / avg_chars_per_token for l in lengths]

    exceeded = sum(1 for t in estimated_tokens if t > max_length)

    return {
        "total_texts": total_texts,
        "max_text_length": max(lengths),
        "avg_text_length": sum(lengths) / total_texts if total_texts > 0 else 0,
        "exceeded_count": exceeded,
        "exceeded_ratio": exceeded / total_texts if total_texts > 0 else 0
    }


def truncate_text(text: str, max_length: int = 512, avg_chars_per_token: float = 2.0) -> str:
    """
    Truncate text to fit max length.

    Args:
        text: Text to truncate
        max_length: Maximum token length
        avg_chars_per_token: Average characters per token

    Returns:
        Truncated text
    """
    max_chars = int(max_length * avg_chars_per_token)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]
