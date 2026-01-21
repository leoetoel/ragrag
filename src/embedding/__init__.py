"""
Embedding and Vector Database Module for RAG

This module provides functionality to:
1. Generate embeddings using BGE models
2. Manage Milvus vector database collections
3. Insert and search vectors
"""

from .embedder import BGEEmbedder
from .vector_store import MilvusVectorStore
from .models import EmbeddingConfig, MilvusConfig, VectorSearchResult

__all__ = ["BGEEmbedder", "MilvusVectorStore", "EmbeddingConfig", "MilvusConfig", "VectorSearchResult"]
__version__ = "0.1.0"
