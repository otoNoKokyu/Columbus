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
    from .firecrawl_engine import scrape_urls_for_markdown, is_unwanted_url_fast, is_unwanted_url_async
    from .link_extractor import extract_links_from_markdown

    visited: Set[str] = set()
    results: List[Dict[str, Any]] = []

    async def _crawl_node(url: str, depth: int) -> Dict[str, Any]:
        """Crawl a single URL and optionally recurse into child links."""
        if url in visited or depth > max_depth:
            return {"url": url, "depth": depth, "markdown": "", "children": [], "skipped": True}

        if await is_unwanted_url_async(url):
            logger.info(f"Skipping unwanted URL in recursive crawl: '{url}'")
            return {"url": url, "depth": depth, "markdown": "", "children": [], "skipped": True, "error": "Filtered out: Unwanted URL/format"}

        visited.add(url)
        logger.info(f"Crawling URL: '{url}' at depth {depth}")

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

                # Filter out already-visited and unwanted URLs
                unvisited = [
                    link for link in scored_links
                    if link["url"] not in visited and not is_unwanted_url_fast(link["url"])
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
    from .crawl4ai_factory import initialize_dynamic_crawl
    from .config import Crawl4AIConfiguration

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


async def frontier_balanced_crawl(
    seed_candidates: List[Dict[str, Any]],
    query: str,
    reranker,
    max_depth: int = 2,
    pages_per_level: List[int] = [5, 4, 3],
    score_threshold: float = 0.7,
    api_key: Optional[str] = None,
    rate_limiter = None,
) -> List[Dict[str, Any]]:
    """Crawls URLs using a level-by-level BFS strategy.

    For each level from 0 to max_depth:
      - Crawls/scrapes up to max_pages_per_iteration highest scoring candidates.
      - Extracts outbound links from successful crawls.
      - Deduplicates, reranks and sorts discovered links to form the candidate pool for the next level.
    """
    from .firecrawl_engine import scrape_urls_for_markdown, is_unwanted_url_fast, is_unwanted_url_async
    from .link_extractor import extract_links_from_markdown
    from urllib.parse import urlparse
    import trafilatura

    visited: Set[str] = set()
    scraped_pages: List[Dict[str, Any]] = []

    # Initialize current level candidates with seed candidates
    current_level_candidates: List[Dict[str, Any]] = []
    for c in seed_candidates:
        url = c.get("url")
        if url:
            parsed = urlparse(url)
            parsed = parsed._replace(fragment="")
            url_clean = parsed.geturl()
            
            if await is_unwanted_url_async(url_clean):
                logger.info(f"[Level BFS Crawl] Skipping unwanted seed candidate URL: '{url_clean}'")
                continue
            
            current_level_candidates.append({
                "url": url_clean,
                "title": c.get("title") or c.get("anchor_text") or "",
                "snippet": c.get("snippet") or c.get("context") or "",
                "depth": 0,
                "rerank_score": (c.get("rerank_score") if c.get("rerank_score") is not None 
                                 else (c.get("score") if c.get("score") is not None else 1.0)),
            })

    # Sort level 0 candidates by initial score
    current_level_candidates = sorted(
        current_level_candidates,
        key=lambda x: x.get("rerank_score") or 0.0,
        reverse=True
    )

    logger.info(
        "[Level BFS Crawl] Initialized with %d seed candidates for query '%s'",
        len(current_level_candidates), query
    )

    for depth in range(max_depth + 1):
        if not current_level_candidates:
            logger.info(f"[Level BFS Crawl] No candidates left at depth {depth}. Terminating early.")
            break

        limit = pages_per_level[depth] if depth < len(pages_per_level) else 1

        # Filter out already visited links and those below threshold or unwanted
        unvisited_candidates = []
        for cand in current_level_candidates:
            if cand["url"] not in visited and not await is_unwanted_url_async(cand["url"]):
                score = cand.get("rerank_score") or cand.get("score") or 0.0
                if score >= score_threshold:
                    unvisited_candidates.append(cand)

        if not unvisited_candidates:
            logger.warning(f"[Level BFS Crawl] No unvisited candidates at depth {depth} met the score threshold >= {score_threshold}. Skipping level.")
            break

        # Scrape up to 'limit' pages at this level
        to_scrape = unvisited_candidates[:limit]
        urls_to_scrape = [cand["url"] for cand in to_scrape]

        logger.info(
            "[Level BFS Crawl] Depth %d | Scraping %d URLs: %s",
            depth, len(urls_to_scrape), urls_to_scrape
        )
        for u in urls_to_scrape:
            logger.info(f"Crawling URL: '{u}' at depth {depth}")

        pages = await scrape_urls_for_markdown(
            urls=urls_to_scrape,
            api_key=api_key,
            rate_limiter=rate_limiter,
            only_main_content=True,
        )

        next_level_raw_links: List[Dict[str, Any]] = []

        for cand, page in zip(to_scrape, pages):
            url = cand["url"]
            visited.add(url)

            if page.get("error"):
                logger.warning("[Level BFS Crawl] Failed to scrape %s: %s", url, page.get("error"))
                continue

            markdown = page.get("markdown", "")
            html = page.get("html", "")
            
            main_content = ""
            if html:
                extracted = trafilatura.extract(html)
                if extracted:
                    main_content = extracted
            if not main_content and markdown:
                main_content = markdown

            scraped_pages.append({
                "url": url,
                "title": cand.get("title") or page.get("title") or cand.get("anchor_text") or "",
                "content": main_content,
                "score": cand.get("rerank_score") or 0.0,
                "depth": depth,
            })

            # If we haven't hit the max depth limit, extract links for the next level
            if depth < max_depth and markdown:
                links = extract_links_from_markdown(markdown, url)
                for link in links:
                    link_url = link["url"]
                    parsed = urlparse(link_url)
                    parsed = parsed._replace(fragment="")
                    link_url_clean = parsed.geturl()

                    if link_url_clean in visited:
                        continue

                    if is_unwanted_url_fast(link_url_clean):
                        continue

                    next_level_raw_links.append({
                        "url": link_url_clean,
                        "url_path": parsed.path,
                        "anchor_text": link.get("anchor_text") or "",
                        "context": link.get("context") or "",
                        "depth": depth + 1,
                    })

        logger.info(
            "[Level BFS Crawl] Depth %d scraped. Discovered %d total links for the next level.",
            depth, len(next_level_raw_links)
        )

        # Deduplicate discovered links by URL before reranking
        seen_next_urls = set()
        next_level_candidates_dedup = []
        for link in next_level_raw_links:
            if link["url"] not in seen_next_urls and link["url"] not in visited:
                seen_next_urls.add(link["url"])
                next_level_candidates_dedup.append(link)

        # Rerank and sort next level candidates
        if depth < max_depth and next_level_candidates_dedup:
            logger.info(
                "[Level BFS Crawl] Reranking %d unique unvisited links for depth %d against query: '%s'...",
                len(next_level_candidates_dedup), depth + 1, query
            )
            
            scored_candidates = []
            for cand in next_level_candidates_dedup:
                url_path = cand.get("url_path") or ""
                anchor = cand.get("anchor_text") or ""
                context = cand.get("context") or ""
                
                doc_text = f"URL Path: {url_path} | Anchor Text: {anchor} | Context: {context}"
                scored_candidates.append({
                    "url": cand["url"],
                    "anchor_text": doc_text,
                    "context": "",
                })

            try:
                reranked_scores = reranker.rerank(
                    query=query,
                    candidates=scored_candidates,
                    top_n=len(scored_candidates),
                )
                
                score_map = {res["url"]: res.get("rerank_score", 0.0) for res in reranked_scores}
                for cand in next_level_candidates_dedup:
                    cand["rerank_score"] = score_map.get(cand["url"], -999.0)

            except Exception as e:
                logger.error("[Level BFS Crawl] Pinecone reranking failed: %s", e)
                for cand in next_level_candidates_dedup:
                    cand["rerank_score"] = 0.0

            # Sort next level candidates by score
            current_level_candidates = sorted(
                next_level_candidates_dedup,
                key=lambda x: x.get("rerank_score") or 0.0,
                reverse=True,
            )
        else:
            current_level_candidates = []

    # Sort final scraped pages by score before returning
    scraped_pages = sorted(
        scraped_pages,
        key=lambda x: x.get("score", 0.0),
        reverse=True,
    )

    logger.info(
        "[Level BFS Crawl] Crawl complete for query '%s'. Visited %d pages total.",
        query, len(scraped_pages)
    )
    
    return scraped_pages


