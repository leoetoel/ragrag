"""
Utility functions for chunking module
"""

import re
import logging
from typing import Optional, List

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extract_page_numbers(content: str) -> Optional[List[int]]:
    """
    Extract page numbers from content.

    Args:
        content: Text content that may contain page markers like **第 X 页**

    Returns:
        [start_page, end_page] or None if no pages found
    """
    # Pattern for **第 X 页** format (with optional bold markers)
    pattern = re.compile(r'\*{0,2}第\s*(\d+)\s*页\*{0,2}', re.IGNORECASE)
    matches = list(pattern.finditer(content))

    if not matches:
        return None

    pages = [int(m.group(1)) for m in matches]

    if len(pages) == 1:
        return [pages[0], pages[0]]
    else:
        return [min(pages), max(pages)]


def validate_chunk_size(size: int) -> bool:
    """Validate chunk size parameter."""
    return 100 <= size <= 10000


def validate_overlap(overlap: int, chunk_size: int) -> bool:
    """Validate overlap parameter."""
    return 0 <= overlap < chunk_size
