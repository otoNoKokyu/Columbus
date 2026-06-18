import asyncio
import os
import random
from typing import Any, Dict
import logging
from firecrawl import FirecrawlApp
from ResearchAgent.search import async_search

from .config import FirecrawlConfiguration
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


async def _scrape_single_url(
    url: str,
    config: FirecrawlConfiguration,
    app: FirecrawlApp,
    rate_limiter: RateLimiter
) -> Dict[str, Any]:
    """
    Scrapes a single URL via Firecrawl with rate limiting and exponential backoff.
    """
    max_retries = 5
    base_delay = 2.0

    for attempt in range(max_retries):
        await rate_limiter.acquire()
        try:
            logger.info("Scraping URL: %s", url)
            
            # Use Firecrawl's native LLM extraction feature
            scrape_params = {
                "formats": ["extract"],
                "extract": {
                    "schema": config.extraction_schema or {"type": "object", "properties": {"summary": {"type": "string"}}}
                }
            }
            
            result = await asyncio.to_thread(
                app.scrape_url,
                url,
                params=scrape_params
            )
            
            logger.info("Scraping complete for: %s", url)
            return {"url": url, "extracted_data": result}
            
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "too many requests" in error_msg:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning("429 Too Many Requests for '%s'. Retrying in %.2fs (Attempt %d/%d)", url, delay, attempt + 1, max_retries)
                await asyncio.sleep(delay)
            else:
                logger.error("Failed to scrape '%s': %s", url, e)
                return {"url": url, "error": str(e)}
        finally:
            rate_limiter.release()
            
    logger.warning("Max retries reached for URL: %s", url)
    return {"url": url, "error": "Max retries reached due to rate limits."}


async def _process_single_query(
    query: str, 
    config: FirecrawlConfiguration, 
    app: FirecrawlApp, 
    rate_limiter: RateLimiter
) -> Dict[str, Any]:
    """
    Process a single search query using DuckDuckGo, then scrape the discovered URLs using Firecrawl.
    """
    logger.info("Executing DuckDuckGo Search for: '%s'", query)
    
    urls_to_scrape = []
    try:
        # Use the existing search functionality from ResearchAgent.search
        results = await async_search(query, max_results=config.duckduckgo_max_results)
        
        for res in results:
            if "url" in res:
                urls_to_scrape.append(res["url"])
                
    except Exception as e:
        logger.error("DuckDuckGo search failed for '%s': %s", query, e)
        return {"query": query, "error": f"DDG search failed: {e}"}

    logger.info("Found %d URLs for query '%s'. Beginning scrape phase...", len(urls_to_scrape), query)

    # Create scraping tasks for the discovered URLs
    scrape_tasks = [
        _scrape_single_url(url, config, app, rate_limiter) 
        for url in urls_to_scrape
    ]
    
    # Wait for all scraping tasks for this query to complete
    scraped_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)
    
    return {
        "query": query,
        "search_results": urls_to_scrape,
        "scraped_data": scraped_results
    }


async def execute_crawl(config: FirecrawlConfiguration):
    """
    The main execution loop for the DDG + Firecrawl pipeline.
    """
    api_key = config.api_key or os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        logger.warning("No Firecrawl API key provided. Scrape requests may fail.")
        
    app = FirecrawlApp(api_key=api_key)
    
    # Firecrawl Free Tier: max 10 req/min, 2 concurrent
    rate_limiter = RateLimiter(max_concurrent=2, max_requests_per_window=10, window_seconds=60.0)
    
    logger.info("Starting pipeline for %d queries. Rate limit enforced at 10req/60s.", len(config.queries))
    
    tasks = [
        _process_single_query(query, config, app, rate_limiter) 
        for query in config.queries
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Pipeline execution complete.")
    return results


async def scrape_urls_for_markdown(
    urls: list[str],
    api_key: str | None = None,
    rate_limiter: RateLimiter | None = None,
    only_main_content: bool = True,
) -> list[dict]:
    """Scrape a list of URLs and extract markdown content.

    This is a pipeline-friendly function used by the research pipeline
    stages (Stage 4 and Stage 9).

    Args:
        urls: List of URLs to scrape.
        api_key: Firecrawl API key. Falls back to FIRECRAWL_API_KEY env var.
        rate_limiter: Optional RateLimiter. Creates a default if None.
        only_main_content: If True, Firecrawl removes navs, sidebars, and footers.

    Returns:
        List of dicts: [{url, markdown, error?}]
    """
    api_key = api_key or os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        logger.error("scrape_urls_for_markdown: No Firecrawl API key available")
        return [{"url": u, "markdown": "", "error": "No API key"} for u in urls]

    app = FirecrawlApp(api_key=api_key)
    rl = rate_limiter or RateLimiter(
        max_concurrent=2, max_requests_per_window=10, window_seconds=60.0
    )

    async def _scrape_one(url: str) -> dict:
        max_retries = 3
        base_delay = 2.0
        for attempt in range(max_retries):
            await rl.acquire()
            try:
                logger.info("scrape_urls_for_markdown: Fetching %s", url[:80])
                result = await asyncio.to_thread(
                    app.scrape_url,
                    url,
                    params={
                        "formats": ["markdown", "html"],
                        "pageOptions": {"includeHtml": True},
                        "onlyMainContent": only_main_content,
                    },
                )
                markdown = ""
                html = ""
                if isinstance(result, dict):
                    markdown = result.get("markdown", "")
                    html = result.get("html", "")
                logger.info(
                    "scrape_urls_for_markdown: Got %d markdown chars and %d html chars from %s",
                    len(markdown), len(html), url[:60],
                )
                return {"url": url, "markdown": markdown, "html": html}
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "too many" in error_msg:
                    import random
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        "scrape_urls_for_markdown: 429 for %s, retry in %.1fs (attempt %d/%d)",
                        url[:60], delay, attempt + 1, max_retries,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("scrape_urls_for_markdown: Failed %s: %s", url[:60], e)
                    return {"url": url, "markdown": "", "error": str(e)}
            finally:
                rl.release()
        return {"url": url, "markdown": "", "error": "Max retries reached"}

    tasks = [_scrape_one(url) for url in urls]
    return await asyncio.gather(*tasks)
