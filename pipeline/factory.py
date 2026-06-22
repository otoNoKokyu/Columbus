"""Factory — constructs a fully-wired research chain.

Same pattern as refactor/retrieve/factory.py.
"""

import logging

from langchain_core.runnables import Runnable

from .config import ResearchPipelineConfig
from .pipeline import build_research_chain

logger = logging.getLogger(__name__)


def create_research_chain(
    config: ResearchPipelineConfig = ResearchPipelineConfig(),
) -> Runnable:
    """Build the research pipeline with all deps wired up.

    Returns a Runnable:
        Input:  {"query": str}
        Output: PipelineState dict

    All dependencies are instantiated here and injected into the
    pipeline stages. This is the only place that imports concrete
    implementations.
    """
    from ..llm import get_langchain_llm
    from ..rewriter import create_rewrite_chain
    from ..scoring import get_reranker
    from ..scoring.embedding_scorer import EmbeddingScorer

    logger.info("Creating research chain with config: %s", config)

    # ── Build LLM ──────────────────────────────────────────────────
    llm = get_langchain_llm(temperature=config.llm_temperature)

    # ── Build rewrite chain ────────────────────────────────────────
    rewrite_chain = create_rewrite_chain(
        llm, max_queries=config.max_rewritten_queries
    )

    # ── Build embedding scorer ─────────────────────────────────────
    embedding_scorer = EmbeddingScorer(model_name=config.embedding_model)

    # ── Build reranker ─────────────────────────────────────────────
    reranker = get_reranker(
        "pinecone",
        model_name=config.pinecone_rerank_model,
        top_n=config.top_links_after_rerank,
    )

    logger.info("Research chain created successfully")

    return build_research_chain(
        rewrite_chain=rewrite_chain,
        embedding_scorer=embedding_scorer,
        reranker=reranker,
        config=config,
    )
