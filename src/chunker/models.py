"""
Data models for document chunking
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single chunk of document content"""
    chunk_id: str = Field(description="Unique chunk identifier")
    content: str = Field(description="Chunk text content")
    metadata: Dict[str, Any] = Field(description="Metadata including source, page, section")
    char_count: int = Field(default=None, description="Character count of this chunk")

    # Internal fields (not exported in final JSON)
    title_path: List[str] = Field(default_factory=list, description="Hierarchy of titles")
    page_range: Optional[List[int]] = Field(default=None, description="[start_page, end_page]")

    class Config:
        json_encoders = {
            # Add any custom encoders if needed
        }


class HeaderNode(BaseModel):
    """A header in the Markdown document tree"""
    level: int = Field(description="Header level (1-6)")
    title: str = Field(description="Header text")
    line_number: int = Field(description="Line number in the document")
    start_index: Optional[int] = Field(default=None, description="Start character index in the document")
    children: List['HeaderNode'] = Field(default_factory=list, description="Child headers")
    content: str = Field(default="", description="Content under this header")

    # Allow forward reference
    class Config:
        arbitrary_types_allowed = True


# Resolve forward reference
HeaderNode.model_rebuild()


class ChunkResult(BaseModel):
    """Result of chunking a document"""
    source_file: str = Field(description="Source file path")
    total_chunks: int = Field(description="Total number of chunks")
    chunks: List[Chunk] = Field(description="All chunks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document-level metadata")
