"""Data types for the research pipeline.

Copied pattern from refactor/retrieve/types.py.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field


# ── Pipeline internal state (flows through RunnableLambda stages) ─────

class PipelineState(TypedDict, total=False):
    """Mutable state dict threaded through every stage.

    Mirrors refactor/retrieve/types.py PipelineState.
    """
    query: str
    rewritten_queries: List[str]
    search_results: List[Dict[str, Any]]
    top_urls: List[str]
    crawled_pages: List[Dict[str, Any]]
    extracted_links: List[Dict[str, Any]]
    embedding_scored_links: List[Dict[str, Any]]
    top_scored_links: List[Dict[str, Any]]
    reranked_links: List[Dict[str, Any]]
    recursive_crawl_output: List[Dict[str, Any]]
    callbacks: Optional[list]  # LangChain callbacks for LangSmith tracing


# ── Data models ───────────────────────────────────────────────────────

class ScoredLink(BaseModel):
    """A link with relevance score from embedding or reranker."""
    url: str
    anchor_text: str = ""
    context: str = ""
    score: float = 0.0
    source_url: str = ""


class CrawledPage(BaseModel):
    """Output of Firecrawl markdown extraction."""
    url: str
    markdown: str = ""
    links_extracted: int = 0
    error: Optional[str] = None
