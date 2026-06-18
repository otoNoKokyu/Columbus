"""Recursive depth-2 crawler with configurable backend.

CRAWL_STRATEGY env var selects the engine:
  - "firecrawl" → Firecrawl /crawl API or /search fallback
  - "crawl4ai"  → Crawl4AI BestFirstCrawlingStrategy

Dependencies (embedding_scorer, rate_limiter) are injected via args
for independent testability.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


async def recursive_crawl(
    seed_urls: List[str],
    query: str,
    embedding_scorer,               # EmbeddingScorer — injected
    max_depth: int = 2,
    max_pages_per_seed: int = 5,
    api_key: Optional[str] = None,
    rate_limiter=None,              # RateLimiter — injected (optional)
) -> List[Dict[str, Any]]:
    """BFS recursive crawl on seed URLs. Backend chosen by CRAWL_STRATEGY env.

    For each seed URL:
      1. Crawl the page → markdown
      2. Extract links from the markdown
      3. Score extracted links via embedding_scorer
      4. Recurse into top-scoring child links (up to max_depth)

    Args:
        seed_urls: Top-ranked URLs from reranking stage.
        query: The original research query (for link scoring).
        embedding_scorer: EmbeddingScorer instance for child link scoring.
        max_depth: Maximum recursion depth (default 2).
        max_pages_per_seed: Max child pages to crawl per seed.
        api_key: Firecrawl API key (used if strategy is "firecrawl").
        rate_limiter: Optional RateLimiter instance.

    Returns:
        List of tree nodes: [{url, depth, markdown, children: [...]}]
        Global visited set prevents re-crawling across all seeds.
    """
    strategy = os.environ.get("CRAWL_STRATEGY", "firecrawl").lower()
    logger.info(
        "Recursive crawl: strategy=%s, seeds=%d, max_depth=%d, max_pages_per_seed=%d",
        strategy, len(seed_urls), max_depth, max_pages_per_seed,
    )

    if strategy == "firecrawl":
        return await _firecrawl_recursive(
            seed_urls, query, embedding_scorer,
            max_depth, max_pages_per_seed, api_key, rate_limiter,
        )
    elif strategy == "crawl4ai":
        return await _crawl4ai_recursive(
            seed_urls, query, embedding_scorer,
            max_depth, max_pages_per_seed,
        )
    else:
        raise ValueError(f"Unknown CRAWL_STRATEGY: {strategy}. Use 'firecrawl' or 'crawl4ai'.")


async def _firecrawl_recursive(
    seed_urls: List[str],
    query: str,
    embedding_scorer,
    max_depth: int,
    max_pages_per_seed: int,
    api_key: Optional[str],
    rate_limiter,
) -> List[Dict[str, Any]]:
    """Firecrawl-backed recursive crawl using scrape_urls_for_markdown."""
    from ResearchAgent.crawl.firecrawl_engine import scrape_urls_for_markdown
    from ResearchAgent.crawl.link_extractor import extract_links_from_markdown

    visited: Set[str] = set()
    results: List[Dict[str, Any]] = []

    async def _crawl_node(url: str, depth: int) -> Dict[str, Any]:
        """Crawl a single URL and optionally recurse into child links."""
        if url in visited or depth > max_depth:
            return {"url": url, "depth": depth, "markdown": "", "children": [], "skipped": True}

        visited.add(url)
        logger.info(
            "Recursive crawl [depth=%d]: Fetching %s", depth, url[:80]
        )

        # Scrape the page
        pages = await scrape_urls_for_markdown(
            urls=[url],
            api_key=api_key,
            rate_limiter=rate_limiter,
            only_main_content=True,
        )

        page = pages[0] if pages else {"url": url, "error": "No result"}
        markdown = page.get("markdown", "")
        html = page.get("html", "")
        
        main_content = ""
        if html:
            import trafilatura
            # Extract pure article text from the rendered HTML
            extracted = trafilatura.extract(html)
            if extracted:
                main_content = extracted
        
        if not main_content and markdown:
            # Fallback to markdown if trafilatura fails (e.g. on purely structural pages)
            main_content = markdown

        node: Dict[str, Any] = {
            "url": url,
            "depth": depth,
            "main_content": main_content,
            "markdown": markdown,
            "error": page.get("error"),
            "children": [],
        }

        # If we haven't hit max depth, extract and score child links
        if depth < max_depth and markdown:
            links = extract_links_from_markdown(markdown, url)
            logger.info(
                "Recursive crawl [depth=%d]: Extracted %d links from %s",
                depth, len(links), url[:60],
            )

            if links:
                # Score links for relevance
                scored_links = await embedding_scorer.score(
                    query=query,
                    candidates=links,
                    top_n=max_pages_per_seed,
                )

                # Filter out already-visited URLs
                unvisited = [
                    link for link in scored_links
                    if link["url"] not in visited
                ]

                chosen_links = unvisited[:max_pages_per_seed]
                if chosen_links:
                    logger.info("Crawler selected %d links to crawl next from %s:", len(chosen_links), url[:80])
                    for i, c in enumerate(chosen_links):
                        logger.info("  -> %d. %s", i+1, c["url"])
                else:
                    logger.info("Crawler found no new unvisited links to follow from %s", url[:80])

                # Recurse into top child links
                for child_link in chosen_links:
                    child_node = await _crawl_node(child_link["url"], depth + 1)
                    node["children"].append(child_node)

        return node

    # Process all seed URLs
    for seed_url in seed_urls:
        tree_node = await _crawl_node(seed_url, depth=0)
        results.append(tree_node)

    logger.info(
        "Firecrawl recursive crawl complete: %d seed trees, %d total pages visited",
        len(results), len(visited),
    )
    return results


async def _crawl4ai_recursive(
    seed_urls: List[str],
    query: str,
    embedding_scorer,
    max_depth: int,
    max_pages_per_seed: int,
) -> List[Dict[str, Any]]:
    """Crawl4AI-backed recursive crawl using BestFirstCrawlingStrategy."""
    from ResearchAgent.crawl.crawl4ai_factory import initialize_dynamic_crawl
    from ResearchAgent.crawl.config import Crawl4AIConfiguration

    results: List[Dict[str, Any]] = []

    for seed_url in seed_urls:
        logger.info("Crawl4AI recursive crawl: Starting from %s", seed_url[:80])

        # Build Crawl4AI config for this seed
        crawl_config = Crawl4AIConfiguration(
            seed_url=seed_url,
            intent_keywords=query.split()[:5],  # Use first 5 words as keywords
            max_depth=max_depth,
            max_pages=max_pages_per_seed,
        )

        try:
            from crawl4ai import AsyncWebCrawler
            from crawl4ai.async_configs import BrowserConfig

            browser_config = BrowserConfig(headless=True, enable_stealth=True)
            run_config = initialize_dynamic_crawl(crawl_config)

            async with AsyncWebCrawler(config=browser_config) as crawler:
                stream = await crawler.arun(url=seed_url, config=run_config)

                node: Dict[str, Any] = {
                    "url": seed_url,
                    "depth": 0,
                    "markdown": "",
                    "children": [],
                }

                async for result in stream:
                    if result.success:
                        clean_md = ""
                        if result.markdown and result.markdown.fit_markdown:
                            clean_md = result.markdown.fit_markdown

                        if result.url == seed_url:
                            node["markdown"] = clean_md
                        else:
                            node["children"].append({
                                "url": result.url,
                                "depth": 1,
                                "markdown": clean_md,
                                "children": [],
                            })
                    else:
                        logger.warning(
                            "Crawl4AI fetch failed: %s - %s",
                            result.url, result.error_message,
                        )

                results.append(node)

        except Exception as e:
            logger.error("Crawl4AI recursive crawl failed for %s: %s", seed_url, e)
            results.append({
                "url": seed_url,
                "depth": 0,
                "markdown": "",
                "children": [],
                "error": str(e),
            })

    logger.info("Crawl4AI recursive crawl complete: %d seed trees", len(results))
    return results
