"""Pipeline stages — each is a PipelineState → PipelineState function.

These are assembled into a Runnable chain in pipeline.py.
Dependencies (scorer, rerankers, etc.) are injected via closure at build time.

Same architecture as refactor/retrieve/stages.py.
"""

import logging
from typing import Callable, Dict, Any, Set, List

from langchain_core.runnables import Runnable

from .config import ResearchPipelineConfig

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# Stage builders — each returns a Callable[[PipelineState], PipelineState]
# that closes over the injected dependencies.
# ────────────────────────────────────────────────────────────────────────


def build_rewrite_stage(
    rewrite_chain: Runnable,
) -> Callable:
    """Stage 1: Rewrite the user query into multiple search variants."""

    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(
            "Stage 1: Rewrite - Input query: '%s'", state.get("query")
        )
        queries = rewrite_chain.invoke(
            {"question": state["query"]},
            config={
                "run_name": "Columbus:QueryRewrite",
                "callbacks": state.get("callbacks"),
            },
        )
        state["rewritten_queries"] = queries
        logger.info(
            "Stage 1: Rewrite - Generated %d variants: %s",
            len(queries), queries,
        )
        return state

    return _run


def build_search_stage(
    config: ResearchPipelineConfig,
) -> Callable:
    """Stage 2-3: Fan-out DuckDuckGo search across rewritten queries,
    deduplicate, return top N URLs."""

    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        import asyncio
        from ..search import fan_out_search

        queries = state.get("rewritten_queries", [])
        logger.info(
            "Stage 2-3: Search - Executing fan-out for %d queries "
            "(max %d results/query, top %d URLs)",
            len(queries),
            config.search_results_per_query,
            config.top_urls_after_search,
        )

        results = asyncio.run(fan_out_search(
            queries=queries,
            provider=config.search_provider,
            max_results_per_query=config.search_results_per_query,
            top_n=config.top_urls_after_search,
        ))

        state["search_results"] = results
        state["top_urls"] = [r["url"] for r in results]
        logger.info(
            "Stage 2-3: Search - Deduplicated to %d unique URLs: %s",
            len(state["top_urls"]),
            [u[:60] for u in state["top_urls"]],
        )
        return state

    return _run


def build_firecrawl_stage(
    config: ResearchPipelineConfig,
) -> Callable:
    """Stage 4: Scrape top URLs via Firecrawl /search → markdown."""

    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        import asyncio
        from ..crawl.firecrawl_engine import scrape_urls_for_markdown

        urls = state.get("top_urls", [])
        logger.info(
            "Stage 4: Firecrawl - Scraping %d URLs for markdown", len(urls)
        )

        pages = asyncio.run(scrape_urls_for_markdown(
            urls=urls,
            api_key=config.firecrawl_api_key,
        ))

        state["crawled_pages"] = pages
        success_count = sum(1 for p in pages if not p.get("error"))
        logger.info(
            "Stage 4: Firecrawl - Completed. %d/%d pages scraped successfully",
            success_count, len(pages),
        )
        return state

    return _run


def build_link_extraction_stage() -> Callable:
    """Stage 5: Extract hyperlinks from crawled markdown pages."""

    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        from ..crawl.link_extractor import extract_links_from_markdown

        pages = state.get("crawled_pages", [])
        logger.info(
            "Stage 5: Link Extraction - Processing %d crawled pages",
            len(pages),
        )

        all_links: List[Dict[str, Any]] = []
        seen_urls: Set[str] = set()

        for page in pages:
            if page.get("error"):
                continue
            links = extract_links_from_markdown(
                markdown=page.get("markdown", ""),
                base_url=page.get("url", ""),
            )
            for link in links:
                if link["url"] not in seen_urls:
                    seen_urls.add(link["url"])
                    all_links.append(link)

        state["extracted_links"] = all_links
        logger.info(
            "Stage 5: Link Extraction - Extracted %d unique links from %d pages",
            len(all_links), len(pages),
        )
        return state

    return _run


def build_embedding_score_stage(
    scorer,  # EmbeddingScorer — injected, not imported
    config: ResearchPipelineConfig,
) -> Callable:
    """Stage 6-7: Score extracted links via embedding similarity, keep top N."""

    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        import asyncio

        links = state.get("extracted_links", [])
        query = state.get("query", "")
        logger.info(
            "Stage 6-7: Embedding Score - Scoring %d links against query '%s'",
            len(links), query,
        )

        if not links:
            state["embedding_scored_links"] = []
            state["top_scored_links"] = []
            logger.warning("Stage 6-7: Embedding Score - No links to score")
            return state

        scored = asyncio.run(scorer.score(
            query=query,
            candidates=links,
            top_n=config.top_links_after_embedding,
        ))

        state["embedding_scored_links"] = scored
        state["top_scored_links"] = scored
        logger.info(
            "Stage 6-7: Embedding Score - Kept top %d links (best score: %.4f)",
            len(scored),
            scored[0].get("embedding_score", 0) if scored else 0,
        )
        return state

    return _run


def build_rerank_stage(
    reranker,  # BaseReranker — injected
    config: ResearchPipelineConfig,
) -> Callable:
    """Stage 8: Rerank top scored links, keep top N."""

    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        candidates = state.get("top_scored_links", [])
        query = state.get("query", "")
        logger.info(
            "Stage 8: Rerank - Reranking %d candidates against query '%s'",
            len(candidates), query,
        )

        if not candidates:
            state["reranked_links"] = []
            logger.warning("Stage 8: Rerank - No candidates to rerank")
            return state

        reranked = reranker.rerank(
            query=query,
            candidates=candidates,
            top_n=config.top_links_after_rerank,
            callbacks=state.get("callbacks"),
        )

        state["reranked_links"] = reranked
        logger.info(
            "Stage 8: Rerank - Kept top %d: %s",
            len(reranked),
            [r.get("url", "")[:60] for r in reranked],
        )
        return state

    return _run


def build_recursive_crawl_stage(
    scorer,  # EmbeddingScorer — reused for child link scoring
    config: ResearchPipelineConfig,
) -> Callable:
    """Stage 9: Recursive depth-2 crawl on the top-ranked URLs."""

    def _run(state: Dict[str, Any]) -> Dict[str, Any]:
        import asyncio
        from ..crawl.recursive_crawler import recursive_crawl

        seed_urls = [r.get("url", "") for r in state.get("reranked_links", [])]
        logger.info(
            "Stage 9: Recursive Crawl - Starting depth-%d crawl on %d seed URLs",
            config.recursive_crawl_depth, len(seed_urls),
        )

        if not seed_urls:
            state["recursive_crawl_output"] = []
            logger.warning("Stage 9: Recursive Crawl - No seed URLs to crawl")
            return state

        tree = asyncio.run(recursive_crawl(
            seed_urls=seed_urls,
            query=state["query"],
            embedding_scorer=scorer,
            max_depth=config.recursive_crawl_depth,
            max_pages_per_seed=config.recursive_max_pages_per_seed,
            api_key=config.firecrawl_api_key,
        ))

        state["recursive_crawl_output"] = tree

        # Count total pages in the tree
        total_pages = 0
        for node in tree:
            total_pages += 1
            total_pages += len(node.get("children", []))

        logger.info(
            "Stage 9: Recursive Crawl - Completed. %d total pages in output tree",
            total_pages,
        )
        return state

    return _run
