"""Pinecone Reranker using the Pinecone Inference API.

Fast and managed API-based reranking.
"""

import os
import logging
from typing import List, Dict, Any, Optional

from .base import BaseReranker

logger = logging.getLogger(__name__)


class PineconeReranker(BaseReranker):
    """Reranker using the Pinecone Inference API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "bge-reranker-v2-m3",
        top_n: int = 5,
    ):
        """Initialize PineconeReranker.

        Args:
            api_key: Pinecone API key. Defaults to PINECONE_API_KEY env var.
            model_name: Pinecone inference model name. Defaults to 'bge-reranker-v2-m3'.
            top_n: Default number of top results to return.
        """
        self._api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self._model_name = model_name
        self._top_n = top_n

        if not self._api_key:
            logger.warning(
                "PINECONE_API_KEY is not set in environment or constructor."
            )

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None,
        callbacks: Optional[list] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using Pinecone Rerank API.

        Args:
            query: The research query.
            candidates: List of dicts with 'anchor_text' and 'context'.
            top_n: Override for number of results to return.
            callbacks: Unused.

        Returns:
            Top N candidates sorted by rerank_score descending.
        """
        from pinecone import Pinecone

        top_n = top_n or self._top_n
        if not candidates:
            logger.warning("PineconeReranker: No candidates to rerank")
            return []

        # Load API key dynamically if it wasn't set at init
        api_key = self._api_key or os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError(
                "PINECONE_API_KEY is missing. Please set it in your environment or pass it to the constructor."
            )

        pc = Pinecone(api_key=api_key)

        # Prepare documents for Pinecone Rerank
        documents = []
        for c in candidates:
            anchor = c.get("anchor_text") or c.get("title") or ""
            context = c.get("context") or c.get("snippet") or ""
            # Combine anchor text and context to form text
            text = f"{anchor} {context}".strip()
            documents.append({"text": text})

        # Initialize all candidates with a default low score
        for c in candidates:
            c["rerank_score"] = -999.0

        batch_size = 100
        try:
            logger.debug(
                "Calling Pinecone Rerank API for %d candidates in batches of %d using model '%s'",
                len(candidates),
                batch_size,
                self._model_name,
            )
            
            for batch_start in range(0, len(candidates), batch_size):
                batch_end = batch_start + batch_size
                batch_candidates = candidates[batch_start:batch_end]
                batch_documents = documents[batch_start:batch_end]

                # Call the rerank API for the current batch with truncation enabled
                response = pc.inference.rerank(
                    model=self._model_name,
                    query=query,
                    documents=batch_documents,
                    top_n=len(batch_candidates),
                    parameters={"truncate": "END"}
                )

                # Map responses back to candidate indices using global offset
                for doc in response.data:
                    local_idx = doc.index
                    score = doc.score
                    global_idx = batch_start + local_idx
                    if 0 <= global_idx < len(candidates):
                        candidates[global_idx]["rerank_score"] = score

        except Exception as e:
            logger.error("Pinecone Rerank API execution failed: %s", e)
            raise e

        # Sort and return top candidates by score descending
        ranked = sorted(
            candidates,
            key=lambda x: x.get("rerank_score", -999.0),
            reverse=True,
        )

        logger.debug(
            "Pinecone Rerank success: %d candidates -> top %d (highest score=%.4f)",
            len(candidates),
            min(top_n, len(ranked)),
            ranked[0].get("rerank_score", 0.0) if ranked else 0.0,
        )

        # Log the first 5 to 10 words of each of the top reranked candidates
        for i, res in enumerate(ranked[:top_n]):
            anchor = res.get("anchor_text") or res.get("title") or ""
            context = res.get("context") or res.get("snippet") or ""
            text = f"{anchor} {context}".strip()
            words = text.split()
            short_text = " ".join(words[:10]) + ("..." if len(words) > 10 else "")
            logger.debug(f"Reranked Rank {i+1} | Score: {res.get('rerank_score', -999.0):.4f} | URL: {res.get('url')} | Context: '{short_text}'")

        return ranked[:top_n]
