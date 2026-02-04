# Copyright (c) Microsoft. All rights reserved.

"""Similarity matcher for cross-domain strategy matching."""

import logging
from typing import Dict, List, Optional, Tuple

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    logging.warning("sentence-transformers not installed. Cross-domain matching will not work.")

logger = logging.getLogger(__name__)


class SimilarityMatcher:
    """Matcher for finding similar strategies using embedding-based similarity.
    
    This class uses sentence transformers to compute embeddings and find
    the most similar strategies for a given problem.
    """
    
    def __init__(self, embedding_model_path: str = "BAAI/bge-large-en-v1.5"):
        """Initialize the similarity matcher.
        
        Args:
            embedding_model_path: Path to the embedding model.
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for cross-domain matching. "
                "Install it with: pip install sentence-transformers"
            )
        
        self.embedding_model_path = embedding_model_path
        logger.info(f"Loading embedding model: {embedding_model_path}")
        self.model = SentenceTransformer(embedding_model_path)
        logger.info("Embedding model loaded successfully")
        
        # Cache for pre-computed strategy embeddings
        self._strategy_cache: Dict[str, List[float]] = {}
    
    def compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for a text string.
        
        Args:
            text: Input text.
            
        Returns:
            Embedding vector as a list of floats.
        """
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.
            
        Returns:
            Cosine similarity score (between -1 and 1, typically 0-1 for normalized embeddings).
        """
        import numpy as np
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def find_best_strategy(
        self,
        problem: str,
        candidate_strategies: List[str],
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Find the best matching strategies for a problem.
        
        Args:
            problem: The problem text to match against.
            candidate_strategies: List of candidate strategy strings.
            top_k: Number of top strategies to return.
            
        Returns:
            List of tuples (strategy, similarity_score), sorted by similarity (descending).
        """
        if not candidate_strategies:
            return []
        
        # Compute embedding for the problem
        problem_embedding = self.compute_embedding(problem)
        
        # Compute similarities
        similarities = []
        for strategy in candidate_strategies:
            # Check cache first
            if strategy in self._strategy_cache:
                strategy_embedding = self._strategy_cache[strategy]
            else:
                strategy_embedding = self.compute_embedding(strategy)
                self._strategy_cache[strategy] = strategy_embedding
            
            similarity = self.compute_similarity(problem_embedding, strategy_embedding)
            similarities.append((strategy, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return similarities[:top_k]
    
    def precompute_strategy_embeddings(self, strategies: List[str]) -> None:
        """Pre-compute and cache embeddings for a list of strategies.
        
        This can be used to speed up matching when the same strategies
        are used repeatedly.
        
        Args:
            strategies: List of strategy strings to pre-compute.
        """
        logger.info(f"Pre-computing embeddings for {len(strategies)} strategies")
        for strategy in strategies:
            if strategy not in self._strategy_cache:
                self._strategy_cache[strategy] = self.compute_embedding(strategy)
        logger.info(f"Cached embeddings for {len(self._strategy_cache)} strategies")

