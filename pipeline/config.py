"""Research pipeline configuration.

Copied pattern from refactor/retrieve/config.py.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchPipelineConfig:
    """All tunable parameters for the research pipeline.

    Every field has a sensible default. Override at construction time
    or via per-request parameters.
    """

    # ── Query Rewriting ─────────────────────────────────
    max_rewritten_queries: int = 3

    # ── Search ──────────────────────────────────────────
    search_provider: str = "ddg"
    search_results_per_query: int = 5
    top_urls_after_search: int = 10

    # ── Firecrawl ───────────────────────────────────────
    firecrawl_api_key: str | None = None

    # ── Embedding Scoring ───────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    top_links_after_embedding: int = 20

    # ── Reranking ───────────────────────────────────────
    reranker_strategy: str = "cross-encoder"  # "cross-encoder" | "llm"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_links_after_rerank: int = 5

    # ── Recursive Crawl ─────────────────────────────────
    recursive_crawl_depth: int = 2
    recursive_max_pages_per_seed: int = 5

    # ── LLM ─────────────────────────────────────────────
    llm_temperature: float = 0.0
