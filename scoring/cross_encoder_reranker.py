"""Cross-encoder reranker using sentence-transformers.

Local model — no API cost, fast inference.
"""

import logging
from typing import List, Dict, Any, Optional

from .base import BaseReranker

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """Reranker using a local cross-encoder model.

    Analogous to refactor's PineconeReranker but uses a local
    cross-encoder model instead of an API.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
    ):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        self._top_n = top_n
        logger.info("CrossEncoderReranker initialized: %s (top_n=%d)", model_name, top_n)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None,
        callbacks: Optional[list] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using cross-encoder scoring.

        Args:
            query: The research query.
            candidates: List of dicts with 'anchor_text' and 'context'.
            top_n: Override for number of results to return.
            callbacks: Unused (no LLM calls), kept for interface compat.

        Returns:
            Top N candidates sorted by rerank_score descending.
        """
        top_n = top_n or self._top_n
        if not candidates:
            logger.warning("CrossEncoderReranker: No candidates to rerank")
            return []

        # Build (query, candidate_text) pairs for the cross-encoder
        pairs = [
            (query, f"{c.get('anchor_text', '')} {c.get('context', '')}")
            for c in candidates
        ]

        # Score all pairs
        scores = self._model.predict(pairs)

        # Attach scores
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])

        # Sort and truncate
        ranked = sorted(
            candidates,
            key=lambda x: x.get("rerank_score", 0),
            reverse=True,
        )

        logger.info(
            "Cross-encoder reranking: %d candidates → top %d (best=%.4f)",
            len(candidates),
            min(top_n, len(ranked)),
            ranked[0]["rerank_score"] if ranked else 0,
        )
        return ranked[:top_n]
