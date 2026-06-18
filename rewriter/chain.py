"""LLM-powered query rewriter as an LCEL Runnable.

Generates up to N semantically distinct search queries optimized for
web search engines. Copied pattern from refactor/rewriter/chain.py.
"""

import logging
from typing import List, Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda, Runnable

from .prompts import QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)


def create_rewrite_chain(
    llm: BaseChatModel,
    max_queries: int = 3,
) -> Runnable:
    """Build an LCEL chain that rewrites a query into multiple search queries.

    Input:  {"question": str}
    Output: List[str]  (rewritten queries, or [question] as fallback)

    The inner chain:
        prompt | llm | JsonOutputParser

    Wrapped in RunnableLambda for parsing + fallback handling.
    """
    # Inner LCEL chain: prompt -> LLM -> parsed JSON
    _inner = QUERY_REWRITE_PROMPT | llm | JsonOutputParser()

    def _rewrite(inputs: Dict[str, Any]) -> List[str]:
        question = inputs["question"]
        fallback = [question]
        try:
            parsed = _inner.invoke(
                {"question": question, "max_queries": max_queries},
                config={"run_name": "ResearchAgent:QueryRewrite"},
            )
            if isinstance(parsed, dict):
                queries = parsed.get("queries", [])
            elif isinstance(parsed, list):
                queries = parsed
            else:
                queries = []
            queries = [q.strip() for q in queries if q and q.strip()][:max_queries]
            if not queries:
                logger.warning("Rewriter returned empty — using fallback")
                return fallback
            logger.info("Rewrote into %d queries: %s", len(queries), queries)
            return queries
        except Exception as e:
            logger.error("Query rewrite failed: %s", e)
            return fallback

    return RunnableLambda(_rewrite)
