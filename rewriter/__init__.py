"""Query rewriter module — LLM-powered multi-query generation as an LCEL Runnable.

Usage::

    from ResearchAgent.rewriter import create_rewrite_chain
    from ResearchAgent.llm import get_langchain_llm

    chain = create_rewrite_chain(llm=get_langchain_llm())
    queries = chain.invoke({"question": "how does RAG work?"})
    # → ["RAG retrieval augmented generation overview", ...]
"""

from .chain import create_rewrite_chain
from .prompts import QUERY_REWRITE_PROMPT

__all__ = [
    "create_rewrite_chain",
    "QUERY_REWRITE_PROMPT",
]
