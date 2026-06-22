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
    "pinecone": "PineconeReranker",
}

DEFAULT_RERANKER = "pinecone"


def get_reranker(name: Optional[str] = None, **kwargs) -> BaseReranker:
    """Instantiate and return a reranker by name.

    Args:
        name: Key in the RERANKERS dict. Defaults to 'pinecone'.
        **kwargs: Arguments passed to the reranker constructor.

            For 'pinecone': model_name, api_key, top_n.

    Returns:
        A BaseReranker instance.

    Raises:
        ValueError: If the name is not registered.
    """
    name = name or DEFAULT_RERANKER

    if name == "pinecone":
        from .pinecone_reranker import PineconeReranker
        return PineconeReranker(**kwargs)
    else:
        available = ", ".join(sorted(RERANKERS.keys()))
        raise ValueError(
            f"Unknown reranker: '{name}'. Available: [{available}]"
        )
