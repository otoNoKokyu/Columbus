"""Embedding-based relevance scorer using sentence-transformers."""

import asyncio
import logging
from typing import List, Dict, Any

from .base import BaseScorer

logger = logging.getLogger(__name__)


class EmbeddingScorer(BaseScorer):
    """Semantic similarity scorer using sentence-transformers.

    Encodes the query and each candidate's (anchor_text + context) into
    dense vectors, then ranks by cosine similarity.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self._model_name = model_name
        self._model = None
        logger.info("EmbeddingScorer initialized locally for model: %s", model_name)

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    async def score(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 20,
    ) -> List[Dict[str, Any]]:
        """Score candidates against query using cosine similarity.

        Args:
            query: The original research query.
            candidates: List of dicts with 'anchor_text' and 'context' keys.
            top_n: Number of top-scoring candidates to return.

        Returns:
            Top N candidates sorted by embedding_score descending.
        """
        if not candidates:
            logger.warning("EmbeddingScorer: No candidates to score")
            return []

        def _compute():
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            # Build candidate text representations
            texts = [
                f"{c.get('anchor_text', '')} {c.get('context', '')}"
                for c in candidates
            ]

            # Load the local model and compute embeddings locally in a batch
            model = self._get_model()
            query_emb = model.encode(query)
            candidate_embs = model.encode(texts)

            # Convert to numpy arrays for sklearn
            query_emb = np.array([query_emb])
            candidate_embs = np.array(candidate_embs)

            # Compute cosine similarity
            scores = cosine_similarity(query_emb, candidate_embs)[0]

            # Attach scores to candidates
            for i, c in enumerate(candidates):
                c["embedding_score"] = float(scores[i])

            # Sort and truncate
            ranked = sorted(
                candidates,
                key=lambda x: x["embedding_score"],
                reverse=True,
            )

            logger.info(
                "Embedding scoring: %d candidates → top %d (best=%.4f)",
                len(candidates),
                min(top_n, len(ranked)),
                ranked[0]["embedding_score"] if ranked else 0,
            )
            return ranked[:top_n]

        return await asyncio.to_thread(_compute)
