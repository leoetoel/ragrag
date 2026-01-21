"""
Embedding and Vector Database Module for RAG

This module provides functionality to:
1. Generate embeddings using BGE models
2. Manage Milvus vector database collections
3. Insert and search vectors
4. Hybrid search (BM25 + Vector + RRF)
"""

from .embedder import BGEEmbedder
from .vector_store import MilvusVectorStore
from .hybrid_search import HybridSearcher, rrf_fusion
from .models import EmbeddingConfig, MilvusConfig, VectorSearchResult

__all__ = [
    "BGEEmbedder",
    "MilvusVectorStore",
    "HybridSearcher",
    "rrf_fusion",
    "EmbeddingConfig",
    "MilvusConfig",
    "VectorSearchResult"
]
__version__ = "0.2.0"
