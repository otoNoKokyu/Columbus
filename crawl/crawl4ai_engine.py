import asyncio
import logging
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig

from .config import Crawl4AIConfiguration
from .crawl4ai_factory import initialize_dynamic_crawl

logger = logging.getLogger(__name__)

async def execute_crawl4ai(config: Crawl4AIConfiguration):
    """
    The main execution loop for the streaming crawler.
    """
    # Separate browser environment config from runtime execution config
    browser_config = BrowserConfig(headless=True, enable_stealth=True)
    run_config = initialize_dynamic_crawl(config)
    
    logger.info("Starting async deep crawl for: %s", config.seed_url)
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            # In v0.8.x+, stream=True returns an async generator from arun()
            stream = await crawler.arun(url=config.seed_url, config=run_config)
            
            async for result in stream:
                if result.success:
                    logger.info("Fetch success: %s", result.url)
                    
                    # Ensure markdown extraction succeeded and get the length
                    clean_markdown = ""
                    if result.markdown and result.markdown.fit_markdown:
                        clean_markdown = result.markdown.fit_markdown
                        
                    logger.info("Clean content length: %d chars", len(clean_markdown))
                    
                    # Illustrative Intent Termination Engine
                    if config.intent_keywords and clean_markdown:
                        content_lower = clean_markdown.lower()
                        primary_keyword = config.intent_keywords[0].lower()
                        if content_lower.count(primary_keyword) >= 3:
                            logger.info("Intent met: Found primary target '%s' at %s", primary_keyword, result.url)
                            logger.info("Terminating crawl engine and cleaning up contexts")
                            if hasattr(run_config, "deep_crawl_strategy") and run_config.deep_crawl_strategy:
                                run_config.deep_crawl_strategy.cancel()
                            
                            # Gracefully exhaust the remaining stream
                            async for _ in stream:
                                pass
                            break
                else:
                    logger.warning("Fetch failed: %s - Reason: %s", result.url, result.error_message)
                    
            return [{"url": config.seed_url, "status": "crawl_completed_or_terminated"}]
                    
        except Exception as e:
            logger.error("Pipeline error: %s", e)
            return [{"url": config.seed_url, "error": str(e)}]
