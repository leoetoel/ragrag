"""
Milvus Vector Store Manager
"""

import logging
from typing import List, Dict, Any, Optional
import json

try:
    from pymilvus import (
        connections, Collection, CollectionSchema, FieldSchema, DataType,
        utility
    )
except ImportError:
    raise ImportError(
        "pymilvus is not installed. "
        "Install it with: pip install pymilvus"
    )

from .models import MilvusConfig, VectorSearchResult
from .utils import logger


class MilvusVectorStore:
    """
    Milvus vector database manager.

    Handles:
    - Connection to Milvus server
    - Collection creation and management
    - Insert, search, and delete operations
    """

    def __init__(self, config: Optional[MilvusConfig] = None):
        """
        Initialize Milvus vector store.

        Args:
            config: MilvusConfig with connection settings
        """
        self.config = config or MilvusConfig()
        self.collection = None
        self._connected = False

    def connect(self):
        """Connect to Milvus server."""
        if self._connected:
            logger.info("Already connected to Milvus")
            return

        try:
            connections.connect(
                alias="default",
                host=self.config.host,
                port=self.config.port
            )
            self._connected = True
            logger.info(f"Connected to Milvus at {self.config.host}:{self.config.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise

    def disconnect(self):
        """Disconnect from Milvus server."""
        if self._connected:
            connections.disconnect("default")
            self._connected = False
            logger.info("Disconnected from Milvus")

    def create_collection(self, overwrite: bool = False):
        """
        Create a new collection.

        Args:
            overwrite: Whether to drop existing collection
        """
        self._ensure_connected()

        # Check if collection exists
        if utility.has_collection(self.config.collection_name):
            if overwrite:
                logger.info(f"Dropping existing collection: {self.config.collection_name}")
                utility.drop_collection(self.config.collection_name)
            else:
                logger.info(f"Collection already exists: {self.config.collection_name}")
                self.collection = Collection(self.config.collection_name)
                return

        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.config.dimension),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=512),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="RAG document chunks with BGE embeddings"
        )

        # Create collection
        self.collection = Collection(
            name=self.config.collection_name,
            schema=schema
        )

        logger.info(f"Created collection: {self.config.collection_name}")

        # Create index
        self._create_index()

    def _create_index(self):
        """Create index on vector field."""
        index_params = {
            "index_type": self.config.index_type,
            "metric_type": self.config.metric_type,
            "params": self.config.index_params
        }

        self.collection.create_index(
            field_name="vector",
            index_params=index_params
        )

        logger.info(f"Created index: {self.config.index_type} ({self.config.metric_type})")

    def load_collection(self):
        """Load collection into memory for search."""
        self._ensure_connected()

        if not utility.has_collection(self.config.collection_name):
            raise ValueError(f"Collection does not exist: {self.config.collection_name}")

        self.collection = Collection(self.config.collection_name)
        self.collection.load()

        logger.info(f"Loaded collection: {self.config.collection_name}")

    def insert_chunks(
        self,
        chunks: List[dict],
        embeddings: List[List[float]]
    ):
        """
        Insert chunks with embeddings into collection.

        Args:
            chunks: List of chunk dictionaries with metadata
            embeddings: List of embedding vectors
        """
        self._ensure_connected()

        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have same length")

        # Prepare data - convert None values to empty strings
        data = [
            [chunk['chunk_id'] for chunk in chunks],
            embeddings,
            [chunk['content'] for chunk in chunks],
            [chunk['metadata'].get('source', '') or '' for chunk in chunks],
            [int(chunk['metadata'].get('page', 0)) if chunk['metadata'].get('page') else 0 for chunk in chunks],
            [chunk['metadata'].get('section', '') or '' for chunk in chunks],
        ]

        # Insert
        insert_result = self.collection.insert(data)

        # Flush to ensure data is persisted
        self.collection.flush()

        logger.info(f"Inserted {len(chunks)} chunks (insert_count={insert_result.insert_count})")

        return insert_result

    def insert_from_json(self, json_file: str, embedder):
        """
        Load chunks from JSON file, generate embeddings, and insert.

        Args:
            json_file: Path to JSON file with chunks
            embedder: BGEEmbedder instance
        """
        # Load chunks
        with open(json_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        logger.info(f"Loaded {len(chunks)} chunks from {json_file}")

        # Generate embeddings
        logger.info("Generating embeddings...")
        texts = [chunk['content'] for chunk in chunks]
        embeddings = embedder.encode_documents(texts)

        # Insert
        self.insert_chunks(chunks, embeddings)

        return len(chunks)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        output_fields: Optional[List[str]] = None
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            output_fields: Fields to return in results

        Returns:
            List of VectorSearchResult
        """
        self._ensure_collection_loaded()

        if output_fields is None:
            output_fields = ["content", "source", "page", "section"]

        # Search parameters
        search_params = {
            "metric_type": self.config.metric_type,
            "params": {"ef": 64}  # HNSW specific
        }

        results = self.collection.search(
            data=[query_embedding],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=output_fields
        )

        # Convert to VectorSearchResult
        search_results = []
        for hit in results[0]:
            result = VectorSearchResult(
                chunk_id=hit.id,
                content=hit.entity.get('content'),
                source=hit.entity.get('source'),
                page=hit.entity.get('page'),
                section=hit.entity.get('section'),
                score=hit.score,
                distance=hit.distance
            )
            search_results.append(result)

        return search_results

    def get_collection_info(self) -> dict:
        """Get collection statistics."""
        self._ensure_collection_loaded()

        stats = {
            "name": self.collection.name,
            "num_entities": self.collection.num_entities,
            "schema": {
                "fields": [
                    {
                        "name": f.name,
                        "type": str(f.dtype),
                        "primary_key": f.is_primary
                    }
                    for f in self.collection.schema.fields
                ]
            }
        }

        return stats

    def _ensure_connected(self):
        """Ensure connected to Milvus."""
        if not self._connected:
            self.connect()

    def _ensure_collection_loaded(self):
        """Ensure collection is loaded."""
        self._ensure_connected()

        if self.collection is None:
            self.load_collection()
