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

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        logger.info("EmbeddingScorer initialized: %s", model_name)

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

            # Build candidate text representations
            texts = [
                f"{c.get('anchor_text', '')} {c.get('context', '')}"
                for c in candidates
            ]

            # Batch encode query and candidates
            query_emb = self._model.encode([query])
            candidate_embs = self._model.encode(texts)

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
