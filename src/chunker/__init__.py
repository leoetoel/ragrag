"""
Document Chunking Module for RAG

This module provides functionality to chunk Markdown documents into
smaller pieces while preserving structure and semantic integrity.
"""

from .chunker import MarkdownChunker
from .models import Chunk, ChunkResult

__all__ = ["MarkdownChunker", "Chunk", "ChunkResult"]
__version__ = "0.1.0"
