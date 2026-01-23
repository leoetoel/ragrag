"""
Data models for embedding and vector store
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Configuration for embedding model"""
    model_name: str = Field(default="BAAI/bge-m3", description="Model name or path")
    device: str = Field(default="cuda", description="Device: cuda, cpu, or mps")
    batch_size: int = Field(default=32, description="Batch size for encoding")
    max_length: int = Field(default=512, description="Maximum sequence length")
    normalize_embeddings: bool = Field(default=True, description="Whether to normalize embeddings")
    query_instruction: str = Field(
        default="为这个句子生成表示以用于检索相关文章：",
        description="Instruction for query encoding"
    )


class VectorSearchResult(BaseModel):
    """Result from vector search"""
    chunk_id: str = Field(description="Chunk ID")
    content: str = Field(description="Chunk text content")
    source: str = Field(description="Source file")
    page: Optional[int] = Field(description="Page number")
    section: Optional[str] = Field(description="Section title")
    score: float = Field(description="Similarity score")
    distance: float = Field(description="Distance value")


class MilvusConfig(BaseModel):
    """Configuration for Milvus connection"""
    host: str = Field(default="localhost", description="Milvus server host")
    port: int = Field(default=19530, description="Milvus server port")
    collection_name: str = Field(default="rag_chunks", description="Collection name")
    dimension: int = Field(default=1024, description="Vector dimension")
    index_type: str = Field(default="HNSW", description="Index type: FLAT, IVF_FLAT, HNSW")
    metric_type: str = Field(default="IP", description="Metric type: IP, COSINE, L2")
    index_params: Dict[str, Any] = Field(
        default={"M": 16, "efConstruction": 256},
        description="Index parameters"
    )
