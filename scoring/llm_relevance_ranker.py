"""LLM-backed Relevance Ranker as a reranker.

Copied pattern from refactor/rerank/llm_relevance_ranker.py.

The scoring chain is assembled as a pure LCEL pipeline::

    RunnableLambda(prepare_inputs)
        | ChatPromptTemplate
        | LLM.with_structured_output(DocumentScore)

Each (query, link_context) pair is scored independently via batch.
LangSmith trace name: "ResearchAgent:LLMRerank".
"""

import logging
from typing import List, Dict, Any, Optional, Union

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_core.runnables import Runnable, RunnableLambda

from .base import BaseReranker
from .prompts import get_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class DocumentScore(BaseModel):
    """Structured output expected from the LLM."""

    score: float = Field(
        description=(
            "A relevance score between 0 and 10 based on the given query. "
            "10 is highly relevant."
        )
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Short reasoning for the assigned score.",
    )


# ---------------------------------------------------------------------------
# Runnable helper — normalises caller args into prompt-ready dicts
# ---------------------------------------------------------------------------

def _prepare_inputs(raw: Dict[str, Any]) -> Dict[str, str]:
    """Normalise caller inputs into the template variable dict.

    The prompt template uses {query}, {url}, {anchor_text}, {context}
    as placeholders.
    """
    return {
        "query": raw.get("query", ""),
        "url": raw.get("url", ""),
        "anchor_text": raw.get("anchor_text", ""),
        "context": raw.get("context", ""),
    }


# ---------------------------------------------------------------------------
# LLM Relevance Ranker
# ---------------------------------------------------------------------------

class LLMRelevanceReranker(BaseReranker):
    """Relevance ranker powered by a LangChain LLM using pure LCEL.

    Scores each (query, link) pair independently via batch inference,
    then sorts and returns top N.

    The LCEL chain::

        RunnableLambda(prepare) → ChatPromptTemplate → LLM.with_structured_output(DocumentScore)
    """

    def __init__(
        self,
        llm: Union[BaseLanguageModel, BaseChatModel],
        top_n: int = 5,
        max_concurrency: int = 5,
    ):
        self._llm = llm
        self._top_n = top_n
        self._max_concurrency = max_concurrency
        self._chain: Runnable = (
            RunnableLambda(_prepare_inputs)
            | get_prompt()
            | llm.with_structured_output(DocumentScore)
        )
        logger.info(
            "LLMRelevanceReranker initialized (top_n=%d, max_concurrency=%d)",
            top_n,
            max_concurrency,
        )

    def _score_result(self, result: Any) -> float:
        """Extract score from a batch result, guarding against exceptions."""
        if isinstance(result, Exception) or result is None:
            logger.warning(
                "LLM scoring failed or returned None: %s. Assigning default score.",
                result,
            )
            return 0.0
        return result.score

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None,
        callbacks: Optional[list] = None,
    ) -> List[Dict[str, Any]]:
        """Score and rerank candidates against the query.

        Args:
            query: The research query.
            candidates: List of dicts with 'url', 'anchor_text', 'context'.
            top_n: Override for number of results.
            callbacks: Optional LangChain callbacks for LangSmith tracing.

        Returns:
            Top N candidates sorted by rerank_score descending,
            with rerank_score and rerank_reasoning added.
        """
        top_n = top_n or self._top_n
        if not candidates:
            logger.warning("LLMRelevanceReranker: No candidates to rerank")
            return []

        # Build batch inputs
        batch_inputs = [
            {
                "query": query,
                "url": c.get("url", ""),
                "anchor_text": c.get("anchor_text", ""),
                "context": c.get("context", ""),
            }
            for c in candidates
        ]

        # Execute parallel batch inference
        try:
            results = self._chain.batch(
                batch_inputs,
                config={
                    "max_concurrency": self._max_concurrency,
                    "callbacks": callbacks,
                    "run_name": "ResearchAgent:LLMRerank",
                },
                return_exceptions=True,
            )
        except Exception as exc:
            logger.error("Batch execution for LLM reranker failed: %s", exc)
            results = [exc] * len(candidates)

        # Merge scores into candidates
        for c, result in zip(candidates, results):
            score = self._score_result(result)
            c["rerank_score"] = score
            if isinstance(result, DocumentScore) and result.reasoning:
                c["rerank_reasoning"] = result.reasoning

        # Sort by score descending and slice top_n
        ranked = sorted(
            candidates,
            key=lambda x: x.get("rerank_score", 0),
            reverse=True,
        )

        logger.info(
            "LLM reranking: %d candidates → top %d (best=%.1f)",
            len(candidates),
            min(top_n, len(ranked)),
            ranked[0].get("rerank_score", 0) if ranked else 0,
        )
        return ranked[:top_n]
