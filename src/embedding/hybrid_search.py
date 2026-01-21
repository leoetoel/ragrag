"""
Hybrid Search Module - BM25 + Vector + RRF

Combines sparse (BM25) and dense (vector) retrieval for better results.
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    from fastbm25 import fastbm25
except ImportError:
    raise ImportError(
        "fastbm25 is not installed. "
        "Install it with: pip install fastbm25"
    )

from .vector_store import MilvusVectorStore
from .embedder import BGEEmbedder
from .models import VectorSearchResult
from .utils import logger


def rrf_fusion(
    results_list1: List[VectorSearchResult],
    results_list2: List[VectorSearchResult],
    k: int = 60
) -> List[Tuple[VectorSearchResult, float]]:
    """
    Reciprocal Rank Fusion (RRF) algorithm.

    Combines two ranked lists by summing reciprocal rank scores.
    Formula: score = 1 / (k + rank)

    Args:
        results_list1: First ranked list (e.g., BM25 results)
        results_list2: Second ranked list (e.g., Vector results)
        k: Constant to prevent rank differences from being too large (default: 60)

    Returns:
        List of (result, fusion_score) tuples, sorted by fusion_score
    """
    fusion_scores = {}

    # Process first list (e.g., BM25)
    for rank, result in enumerate(results_list1):
        chunk_id = result.chunk_id
        if chunk_id not in fusion_scores:
            fusion_scores[chunk_id] = {"result": result, "score": 0.0}
        fusion_scores[chunk_id]["score"] += 1 / (k + rank)

    # Process second list (e.g., Vector)
    for rank, result in enumerate(results_list2):
        chunk_id = result.chunk_id
        if chunk_id not in fusion_scores:
            fusion_scores[chunk_id] = {"result": result, "score": 0.0}
        fusion_scores[chunk_id]["score"] += 1 / (k + rank)

    # Sort by fusion score (descending)
    sorted_items = sorted(
        fusion_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return [(item["result"], item["score"]) for item in sorted_items]


class HybridSearcher:
    """
    Hybrid search combining BM25 and vector retrieval.

    Features:
    - BM25 sparse retrieval (keyword matching)
    - Vector dense retrieval (semantic similarity)
    - RRF fusion (combine rankings)
    """

    def __init__(
        self,
        vector_store: MilvusVectorStore,
        embedder: BGEEmbedder,
        bm25_corpus: Optional[List[str]] = None
    ):
        """
        Initialize hybrid searcher.

        Args:
            vector_store: MilvusVectorStore instance
            embedder: BGEEmbedder instance
            bm25_corpus: List of documents for BM25 indexing (optional)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_model = None
        self.bm25_corpus = bm25_corpus

        # Initialize BM25 if corpus is provided
        if bm25_corpus:
            self._init_bm25(bm25_corpus)

    def _init_bm25(self, corpus: List[str]):
        """Initialize BM25 model with corpus."""
        logger.info(f"Initializing BM25 with {len(corpus)} documents...")
        self.bm25_model = fastbm25(corpus)
        self.bm25_corpus = corpus
        logger.info("BM25 model initialized")

    def load_chunks_for_bm25(self, chunks: List[Dict]):
        """
        Load chunks and initialize BM25.

        Args:
            chunks: List of chunk dictionaries with 'content' field
        """
        corpus = [chunk['content'] for chunk in chunks]
        self._init_bm25(corpus)

        # Store chunk metadata for retrieval
        self.chunk_metadata = {chunk['content']: chunk for chunk in chunks}

    def search_bm25(
        self,
        query: str,
        top_k: int = 50,
        return_metadata: bool = False
    ) -> List[VectorSearchResult]:
        """
        Search using BM25 (sparse retrieval).

        Args:
            query: Query text
            top_k: Number of results to return
            return_metadata: Whether to include metadata in results

        Returns:
            List of VectorSearchResult
        """
        if self.bm25_model is None:
            raise ValueError("BM25 model not initialized. Call load_chunks_for_bm25() first.")

        # BM25 search
        results_raw = self.bm25_model.top_k_sentence(query, k=top_k)

        # Convert to VectorSearchResult
        search_results = []
        for idx, (text, corpus_idx, score) in enumerate(results_raw):
            # Try to get metadata from stored chunks
            if hasattr(self, 'chunk_metadata') and text in self.chunk_metadata:
                chunk = self.chunk_metadata[text]
                result = VectorSearchResult(
                    chunk_id=chunk.get('chunk_id', f'bm25_{idx}'),
                    content=text,
                    source=chunk.get('metadata', {}).get('source', ''),
                    page=chunk.get('metadata', {}).get('page'),
                    section=chunk.get('metadata', {}).get('section', ''),
                    score=score,
                    distance=1 - score  # Convert similarity to distance
                )
            else:
                # Fallback without metadata
                result = VectorSearchResult(
                    chunk_id=f'bm25_{idx}',
                    content=text,
                    source='',
                    page=None,
                    section='',
                    score=score,
                    distance=1 - score
                )

            search_results.append(result)

        return search_results

    def search_vector(
        self,
        query: str,
        top_k: int = 50
    ) -> List[VectorSearchResult]:
        """
        Search using vector (dense retrieval).

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of VectorSearchResult
        """
        # Encode query
        query_embedding = self.embedder.encode_queries(query)

        # Search in Milvus
        results = self.vector_store.search(
            query_embedding=query_embedding[0],
            top_k=top_k
        )

        return results

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        bm25_k: int = 50,
        vector_k: int = 50,
        rrf_k: int = 60,
        alpha: float = 0.5
    ) -> List[Dict]:
        """
        Hybrid search combining BM25 and Vector with RRF fusion.

        Args:
            query: Query text
            top_k: Final number of results to return
            bm25_k: Number of results from BM25
            vector_k: Number of results from Vector
            rrf_k: RRF constant (default: 60)
            alpha: Weight for RRF (not used currently, for future)

        Returns:
            List of dicts with result info and fusion scores
        """
        logger.info(f"Hybrid search for query: {query}")

        # Step 1: BM25 retrieval (sparse)
        logger.info(f"[1/3] BM25 retrieval (top {bm25_k})...")
        bm25_results = self.search_bm25(query, top_k=bm25_k)

        # Step 2: Vector retrieval (dense)
        logger.info(f"[2/3] Vector retrieval (top {vector_k})...")
        vector_results = self.search_vector(query, top_k=vector_k)

        # Step 3: RRF fusion
        logger.info(f"[3/3] RRF fusion...")
        fusion_results = rrf_fusion(bm25_results, vector_results, k=rrf_k)

        # Take top-k results
        final_results = fusion_results[:top_k]

        # Format output
        formatted_results = []
        for result, fusion_score in final_results:
            formatted_results.append({
                'chunk_id': result.chunk_id,
                'content': result.content,
                'source': result.source,
                'page': result.page,
                'section': result.section,
                'vector_score': result.score,
                'fusion_score': fusion_score
            })

        logger.info(f"Hybrid search completed: {len(formatted_results)} results")

        return formatted_results

    def compare_methods(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict:
        """
        Compare BM25, Vector, and Hybrid search results.

        Args:
            query: Query text
            top_k: Number of results to compare

        Returns:
            Dict with comparison results
        """
        # BM25 only
        bm25_results = self.search_bm25(query, top_k=top_k)

        # Vector only
        vector_results = self.search_vector(query, top_k=top_k)

        # Hybrid
        hybrid_results = self.search_hybrid(query, top_k=top_k)

        return {
            'query': query,
            'bm25_results': bm25_results,
            'vector_results': vector_results,
            'hybrid_results': hybrid_results
        }
