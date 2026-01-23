"""
Document Chunking Module for RAG

This module provides functionality to chunk Markdown documents into
smaller pieces while preserving structure and semantic integrity.
"""

from .chunker import AnnualReportChunker

__all__ = ["AnnualReportChunker"]
__version__ = "0.1.0"
