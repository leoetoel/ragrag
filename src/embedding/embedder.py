"""
BGE Embedding Model Wrapper
"""

import logging
from typing import List, Union, Optional
import numpy as np

try:
    from FlagEmbedding import FlagModel
except ImportError:
    raise ImportError(
        "FlagEmbedding is not installed. "
        "Install it with: pip install -U FlagEmbedding"
    )

from .models import EmbeddingConfig
from .utils import logger

# 添加 format 异常处理
try:
    from rich.logging import RichHandler
    from rich.console import Console
except ImportError:
    RichHandler = None
    Console = None


class BGEEmbedder:
    """
    Wrapper for BGE (BAAI General Embedding) models.

    Supports:
    - bge-large-zh-v1.5: Chinese optimized, 512 max length, 1024 dimensions
    - bge-m3: Multilingual, 8192 max length, 1024 dimensions
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize BGE embedder.

        Args:
            config: EmbeddingConfig with model settings
        """
        self.config = config or EmbeddingConfig()
        self.model = None
        self.dimension = None
        self._load_model()

        logger.info(
            f"BGEEmbedder initialized (model={self.config.model_name}, "
            f"device={self.config.device}, dim={self.dimension})"
        )

    def _load_model(self):
        """Load the BGE model."""
        logger.info(f"Loading BGE model: {self.config.model_name}")

        try:
            self.model = FlagModel(
                self.config.model_name,
                query_instruction_for_retrieval=self.config.query_instruction,
                device=self.config.device,
                normalize_embeddings=self.config.normalize_embeddings
            )

            # Get dimension by encoding a dummy text
            dummy_embedding = self.model.encode(["测试"])
            self.dimension = dummy_embedding.shape[1]

            logger.info(f"Model loaded successfully (dimension={self.dimension})")

        except Exception as e:
            logger.error(f"Failed to load BGE model: {str(e)}")
            raise

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding (default from config)
            show_progress: Whether to show progress bar (not supported by FlagEmbedding)

        Returns:
            numpy array of embeddings with shape (n_texts, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]

        if batch_size is None:
            batch_size = self.config.batch_size

        try:
            # FlagEmbedding doesn't support show_progress_bar parameter
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size
            )

            return embeddings

        except Exception as e:
            logger.error(f"Encoding failed: {str(e)}")
            raise

    def encode_queries(self, queries: Union[str, List[str]]) -> np.ndarray:
        """
        Encode query texts with instruction.

        Args:
            queries: Single query or list of queries

        Returns:
            numpy array of embeddings
        """
        embeddings = self.encode(queries)
        return embeddings.astype(np.float32)

    def encode_documents(self, documents: Union[str, List[str]]) -> np.ndarray:
        """
        Encode document texts (without instruction).

        Args:
            documents: Single document or list of documents

        Returns:
            numpy array of embeddings
        """
        if isinstance(documents, str):
            documents = [documents]

        # For documents, encode without the query instruction
        # Using encode method which doesn't add instruction for documents
        embeddings = self.model.encode(
            documents,
            batch_size=self.config.batch_size
        )

        # Ensure float32 for Milvus compatibility
        return embeddings.astype(np.float32)

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimension

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_name": self.config.model_name,
            "device": self.config.device,
            "dimension": self.dimension,
            "max_length": self.config.max_length,
            "normalize_embeddings": self.config.normalize_embeddings,
            "batch_size": self.config.batch_size
        }
