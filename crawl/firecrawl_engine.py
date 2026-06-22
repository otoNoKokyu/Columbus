import asyncio
import os
import random
from typing import Any, Dict
import logging
from firecrawl import FirecrawlApp
from ..search import async_search

from .config import FirecrawlConfiguration
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


def is_unwanted_url_fast(url: str) -> bool:
    """
    Synchronous fast check based on URL path/string patterns.
    """
    if not url:
        return True
    
    url_lower = url.lower()
    
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
            
        social_domains = {
            "facebook.com", "twitter.com", "x.com", "instagram.com", "tiktok.com",
            "reddit.com", "linkedin.com", "youtube.com", "pinterest.com", "tumblr.com",
            "quora.com", "twitch.tv", "medium.com"
        }
        if any(netloc == domain or netloc.endswith("." + domain) for domain in social_domains):
            return True
            
        path_lower = parsed.path.lower()
    except Exception:
        path_lower = url_lower
        
    unwanted_extensions = {
        # PDF
        '.pdf',
        # Archives
        '.zip', '.tar', '.gz', '.tgz', '.rar', '.7z', '.bz2', '.xz', '.zipx',
        # Media / Audio / Video
        '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.ico', '.tiff', '.bmp',
        '.mp3', '.mp4', '.wav', '.avi', '.mov', '.ogg', '.m4a', '.webm', '.flv', '.mkv',
        # Documents / Office files
        '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.odt', '.ods', '.odp', '.rtf', '.csv',
        # Executables / Packages
        '.dmg', '.exe', '.pkg', '.deb', '.rpm', '.iso', '.bin'
    }
    
    if any(path_lower.endswith(ext) for ext in unwanted_extensions):
        return True
        
    if '.pdf' in url_lower or '/pdf/' in url_lower or '/pdf?' in url_lower or 'format=pdf' in url_lower:
        return True
        
    return False


def _check_url_content_type_sync(url: str) -> str:
    import urllib.request
    try:
        req = urllib.request.Request(
            url, 
            method="HEAD",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        )
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            return resp.headers.get("Content-Type", "").lower()
    except Exception:
        # Fallback to GET with range header to only fetch headers/first bytes
        try:
            req = urllib.request.Request(
                url, 
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Range": "bytes=0-1024"
                }
            )
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                return resp.headers.get("Content-Type", "").lower()
        except Exception:
            return ""


async def is_unwanted_url_async(url: str) -> bool:
    """
    Asynchronous check combining fast path check and live Content-Type check.
    """
    if is_unwanted_url_fast(url):
        return True
        
    # Check live Content-Type to catch PDF files masked as clean URLs (e.g. Contentful, S3 buckets)
    try:
        content_type = await asyncio.to_thread(_check_url_content_type_sync, url)
        if not content_type:
            return False
            
        if "application/pdf" in content_type:
            return True
            
        binary_types = {
            "application/zip", "application/x-tar", "application/gzip", 
            "application/x-gzip", "application/octet-stream", "application/x-rar",
            "image/", "audio/", "video/"
        }
        if any(bt in content_type for bt in binary_types) and "text/html" not in content_type and "application/json" not in content_type:
            return True
    except Exception as e:
        logger.warning("Error checking Content-Type for %s: %s", url, e)
        
    return False


async def _scrape_single_url(
    url: str,
    config: FirecrawlConfiguration,
    app: FirecrawlApp,
    rate_limiter: RateLimiter
) -> Dict[str, Any]:
    """
    Scrapes a single URL via Firecrawl with rate limiting and exponential backoff.
    """
    if await is_unwanted_url_async(url):
        logger.info("Filtering out unwanted URL (PDF/binary/social/etc.) from scraping: %s", url)
        return {"url": url, "error": "Filtered out: Unwanted URL/format (PDF/binary/social/etc.)"}

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
        # Use the existing search functionality from Columbus.search
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
        if not is_unwanted_url_fast(url)
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


async def scrape_urls_for_markdown_crawl4ai(urls: list[str]) -> list[dict]:
    """Scrapes a list of URLs concurrently using Crawl4AI."""
    from crawl4ai import AsyncWebCrawler, CacheMode
    from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
    
    browser_config = BrowserConfig(headless=True, enable_stealth=True)
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    
    results = []
    urls_to_scrape = []
    filtered_results = {}
    
    for url in urls:
        if await is_unwanted_url_async(url):
            logger.info("crawl4ai: Filtering out unwanted URL: %s", url)
            filtered_results[url] = {"url": url, "markdown": "", "html": "", "error": "Filtered out: Unwanted URL/format"}
        else:
            urls_to_scrape.append(url)
            
    if not urls_to_scrape:
        return [filtered_results[url] for url in urls]
        
    try:
        logger.info("crawl4ai: Scrape initiated for %d URLs", len(urls_to_scrape))
        async with AsyncWebCrawler(config=browser_config) as crawler:
            crawl_results = await crawler.arun_many(urls_to_scrape, config=run_config)
            
            def process_res(res):
                url = res.url
                if res.success:
                    markdown = ""
                    if res.markdown:
                        markdown = res.markdown.fit_markdown or res.markdown.raw_markdown or ""
                    html = res.html or ""
                    filtered_results[url] = {"url": url, "markdown": markdown, "html": html}
                    logger.info("crawl4ai: Successfully scraped %s (%d chars)", url, len(markdown))
                else:
                    filtered_results[url] = {"url": url, "markdown": "", "html": "", "error": res.error_message}
                    logger.error("crawl4ai: Failed to scrape %s: %s", url, res.error_message)

            if hasattr(crawl_results, "__aiter__"):
                async for res in crawl_results:
                    process_res(res)
            else:
                for res in crawl_results:
                    process_res(res)

    except Exception as e:
        logger.error("crawl4ai: Concurrency scrape failed: %s. Falling back to sequential scrape.", e)
        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                for url in urls_to_scrape:
                    try:
                        res = await crawler.arun(url=url, config=run_config)
                        if res.success:
                            markdown = ""
                            if res.markdown:
                                markdown = res.markdown.fit_markdown or res.markdown.raw_markdown or ""
                            html = res.html or ""
                            filtered_results[url] = {"url": url, "markdown": markdown, "html": html}
                        else:
                            filtered_results[url] = {"url": url, "markdown": "", "html": "", "error": res.error_message}
                    except Exception as ex:
                        filtered_results[url] = {"url": url, "markdown": "", "html": "", "error": str(ex)}
        except Exception as fallback_e:
            logger.error("crawl4ai: Fallback scrape failed: %s", fallback_e)
            for url in urls_to_scrape:
                filtered_results[url] = {"url": url, "markdown": "", "html": "", "error": str(fallback_e)}
            
    return [filtered_results.get(url, {"url": url, "markdown": "", "html": "", "error": "Unknown error"}) for url in urls]


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
    strategy = os.environ.get("CRAWL_STRATEGY", "firecrawl").lower()
    if strategy == "crawl4ai":
        return await scrape_urls_for_markdown_crawl4ai(urls)
    api_key = api_key or os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        logger.error("scrape_urls_for_markdown: No Firecrawl API key available")
        return [{"url": u, "markdown": "", "error": "No API key"} for u in urls]

    app = FirecrawlApp(api_key=api_key)
    rl = rate_limiter or RateLimiter(
        max_concurrent=2, max_requests_per_window=10, window_seconds=60.0
    )

    async def _scrape_one(url: str) -> dict:
        if await is_unwanted_url_async(url):
            logger.info("scrape_urls_for_markdown: Filtering out unwanted URL: %s", url)
            return {"url": url, "markdown": "", "error": "Filtered out: Unwanted URL/format"}

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
