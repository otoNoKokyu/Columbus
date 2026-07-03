"""Local Reranker using SentenceTransformers CrossEncoder."""

import logging
from typing import List, Dict, Any, Optional

from .base import BaseReranker

logger = logging.getLogger(__name__)

class LocalReranker(BaseReranker):
    """Reranker using a local HuggingFace CrossEncoder model."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        top_n: int = 5,
    ):
        """Initialize LocalReranker.

        Args:
            model_name: HuggingFace model name. Defaults to 'BAAI/bge-reranker-v2-m3'.
            top_n: Default number of top results to return.
        """
        self._model_name = model_name
        self._top_n = top_n
        
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading local reranker model: {self._model_name}")
            self.model = CrossEncoder(self._model_name)
        except ImportError:
            raise ImportError("Please install sentence-transformers to use LocalReranker")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model {self._model_name}: {e}")
            raise e

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None,
        callbacks: Optional[list] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using the local CrossEncoder.

        Args:
            query: The research query.
            candidates: List of dicts with 'anchor_text' and 'context' or 'text'.
            top_n: Override for number of results to return.
            callbacks: Unused.

        Returns:
            Top N candidates sorted by rerank_score descending.
        """
        top_n = top_n or self._top_n
        if not candidates:
            logger.warning("LocalReranker: No candidates to rerank")
            return []

        # Prepare pairs for scoring
        pairs = []
        for c in candidates:
            anchor = c.get("anchor_text") or c.get("title") or ""
            context = c.get("context") or c.get("snippet") or c.get("text") or ""
            text = f"{anchor} {context}".strip()
            pairs.append([query, text])

        try:
            logger.debug(f"Scoring {len(pairs)} candidates locally using {self._model_name}")
            scores = self.model.predict(pairs)
            
            for i, score in enumerate(scores):
                candidates[i]["rerank_score"] = float(score)

        except Exception as e:
            logger.error("Local reranker prediction failed: %s", e)
            raise e

        # Sort and return top candidates by score descending
        ranked = sorted(
            candidates,
            key=lambda x: x.get("rerank_score", -999.0),
            reverse=True,
        )

        for i, res in enumerate(ranked[:top_n]):
            logger.debug(f"Local Reranked Rank {i+1} | Score: {res.get('rerank_score', -999.0):.4f} | URL: {res.get('url')}")

        return ranked[:top_n]
