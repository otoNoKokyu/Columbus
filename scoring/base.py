"""Base interfaces for the scoring/reranking module.

Copied pattern from refactor/rerank/base.py.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseScorer(ABC):
    """Abstract scorer for embedding-based relevance scoring."""

    @abstractmethod
    async def score(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 20,
    ) -> List[Dict[str, Any]]:
        """Score candidates against query. Return sorted by score descending."""
        ...


class BaseReranker(ABC):
    """Abstract reranker for second-pass reranking."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 5,
        callbacks: Optional[list] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates. Return top N sorted by relevance."""
        ...
