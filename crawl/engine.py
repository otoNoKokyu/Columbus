import logging
import os

from .config import AnyCrawlConfiguration, FirecrawlConfiguration, Crawl4AIConfiguration

logger = logging.getLogger(__name__)

async def execute_crawl(config: AnyCrawlConfiguration):
    """
    Main entry point for crawling. Inspects CRAWL_STRATEGY to route the request
    to either Firecrawl or Crawl4AI.
    """
    strategy = os.environ.get("CRAWL_STRATEGY", "firecrawl").lower()
    
    if strategy == "firecrawl":
        if not isinstance(config, FirecrawlConfiguration):
            raise ValueError(f"Expected FirecrawlConfiguration for strategy 'firecrawl', got {type(config).__name__}")
            
        from .firecrawl_engine import execute_crawl as firecrawl_execute
        logger.info("Delegating to Firecrawl Engine")
        return await firecrawl_execute(config)
        
    elif strategy == "crawl4ai":
        if not isinstance(config, Crawl4AIConfiguration):
            raise ValueError(f"Expected Crawl4AIConfiguration for strategy 'crawl4ai', got {type(config).__name__}")
            
        from .crawl4ai_engine import execute_crawl4ai
        logger.info("Delegating to Crawl4AI Engine")
        return await execute_crawl4ai(config)
        
    else:
        raise ValueError(f"Unknown CRAWL_STRATEGY: {strategy}. Use 'firecrawl' or 'crawl4ai'.")
