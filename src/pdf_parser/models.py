"""
Data models for PDF parsing results
"""
# 这个文件定义了数据的存储格式（一些class），他自己不知道如何获取这些信息
# 如果想知道获取这些信息的逻辑，需要查看执行代码也就是
from typing import List, Optional, Any
from pydantic import BaseModel, Field


class ImageInfo(BaseModel):
    """Information about an extracted image"""
    page_num: int = Field(description="Page number where image is located")
    bbox: List[float] = Field(description="Bounding box coordinates [x0, y0, x1, y1]")
    width: float = Field(description="Image width in points")
    height: float = Field(description="Image height in points")


class TableCell(BaseModel):
    """Represents a single table cell"""
    row: int = Field(description="Row index")
    col: int = Field(description="Column index")
    text: str = Field(default="", description="Cell text content")


class TableInfo(BaseModel):
    """Information about an extracted table"""
    page_num: int = Field(description="Page number where table is located")
    rows: int = Field(description="Number of rows")
    cols: int = Field(description="Number of columns")
    headers: List[str] = Field(default_factory=list, description="Table headers")
    cells: List[TableCell] = Field(default_factory=list, description="Table cells")


class TextContent(BaseModel):
    """Text content with position information"""
    text: str = Field(description="Text content")
    page_num: int = Field(description="Page number")
    bbox: Optional[List[float]] = Field(default=None, description="Bounding box coordinates")


class DocumentStructure(BaseModel):
    """Document structure element (title, heading, etc.)"""
    type: str = Field(description="Element type (title, heading, paragraph, etc.)")
    level: Optional[int] = Field(default=None, description="Heading level (1-6)")
    text: str = Field(description="Text content")
    page_num: int = Field(description="Page number")


class ParseResult(BaseModel):
    """Complete parsing result for a PDF document"""
    file_path: str = Field(description="Path to the PDF file")
    file_name: str = Field(description="Name of the PDF file")
    total_pages: int = Field(default=0, description="Total number of pages")

    # Extracted content
    text_content: List[TextContent] = Field(default_factory=list, description="All text content")
    tables: List[TableInfo] = Field(default_factory=list, description="Extracted tables")
    images: List[ImageInfo] = Field(default_factory=list, description="Extracted images")
    structure: List[DocumentStructure] = Field(default_factory=list, description="Document structure")

    # Raw text
    full_text: str = Field(default="", description="Full text content of the document")

    # Metadata
    metadata: dict = Field(default_factory=dict, description="Document metadata")

    class Config:
        json_encoders = {
            # Add any custom encoders if needed
        }
