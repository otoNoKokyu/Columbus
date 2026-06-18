"""LCEL-based research pipeline — composes stages into a Runnable chain.

Same architecture as refactor/retrieve/pipeline.py.
"""

import logging

from langchain_core.runnables import RunnableLambda, Runnable

from .config import ResearchPipelineConfig
from .stages import (
    build_rewrite_stage,
    build_search_stage,
    build_firecrawl_stage,
    build_link_extraction_stage,
    build_embedding_score_stage,
    build_rerank_stage,
    build_recursive_crawl_stage,
)

logger = logging.getLogger(__name__)


def build_research_chain(
    rewrite_chain: Runnable,
    embedding_scorer,           # EmbeddingScorer instance
    reranker,                   # BaseReranker instance
    config: ResearchPipelineConfig = ResearchPipelineConfig(),
) -> Runnable:
    """Build the full research pipeline as a single LCEL Runnable.

    Input:  {"query": str}
    Output: PipelineState dict with all stage outputs.

    The chain structure::

        rewrite → search → firecrawl → link_extract
        → embedding_score → rerank → recursive_crawl

    Args:
        rewrite_chain: LCEL Runnable from create_rewrite_chain().
        embedding_scorer: EmbeddingScorer instance for stages 6-7 and 9.
        reranker: BaseReranker instance for stage 8.
        config: Pipeline configuration with all tunable parameters.

    Returns:
        A single LCEL Runnable that executes the full 9-stage pipeline.
    """
    # ── Build stage callables ──────────────────────────────────────
    rewrite       = RunnableLambda(build_rewrite_stage(rewrite_chain))
    search        = RunnableLambda(build_search_stage(config))
    firecrawl     = RunnableLambda(build_firecrawl_stage(config))
    link_extract  = RunnableLambda(build_link_extraction_stage())
    emb_score     = RunnableLambda(build_embedding_score_stage(embedding_scorer, config))
    rerank        = RunnableLambda(build_rerank_stage(reranker, config))
    recursive     = RunnableLambda(build_recursive_crawl_stage(embedding_scorer, config))

    logger.info("Research pipeline chain assembled with %d stages", 7)

    # ── Full chain ─────────────────────────────────────────────────
    return (
        rewrite
        | search
        | firecrawl
        | link_extract
        | emb_score
        | rerank
        | recursive
    )
