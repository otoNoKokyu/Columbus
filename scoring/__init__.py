"""Scoring module — plug-in any reranker or scorer.

Copied pattern from refactor/rerank/__init__.py.
"""

from typing import Dict, Optional, Type

from .base import BaseScorer, BaseReranker
from .embedding_scorer import EmbeddingScorer
from .llm_scorer import LLMScorer

__all__ = [
    # Core interfaces
    "BaseScorer",
    "BaseReranker",
    # Implementations
    "EmbeddingScorer",
    "LLMScorer",
    # Factory
    "get_reranker",
    # Registry
    "RERANKERS",
]

# ---------------------------------------------------------------------------
# Reranker registry
# ---------------------------------------------------------------------------
RERANKERS: Dict[str, str] = {
    "cross-encoder": "CrossEncoderReranker",
    "llm": "LLMRelevanceReranker",
}

DEFAULT_RERANKER = "cross-encoder"


def get_reranker(name: Optional[str] = None, **kwargs) -> BaseReranker:
    """Instantiate and return a reranker by name.

    Args:
        name: Key in the RERANKERS dict. Defaults to 'cross-encoder'.
        **kwargs: Arguments passed to the reranker constructor.

            For 'cross-encoder': model_name, top_n.
            For 'llm': llm (required), top_n, max_concurrency.

    Returns:
        A BaseReranker instance.

    Raises:
        ValueError: If the name is not registered.
    """
    name = name or DEFAULT_RERANKER

    if name == "cross-encoder":
        from .cross_encoder_reranker import CrossEncoderReranker
        return CrossEncoderReranker(**kwargs)
    elif name == "llm":
        from .llm_relevance_ranker import LLMRelevanceReranker
        return LLMRelevanceReranker(**kwargs)
    else:
        available = ", ".join(sorted(RERANKERS.keys()))
        raise ValueError(
            f"Unknown reranker: '{name}'. Available: [{available}]"
        )
