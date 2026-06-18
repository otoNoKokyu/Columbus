from crawl4ai.async_configs import CrawlerRunConfig
from crawl4ai import CacheMode
from crawl4ai.deep_crawling import (
    BestFirstCrawlingStrategy, 
    FilterChain, 
    DomainFilter, 
    URLPatternFilter
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from .config import Crawl4AIConfiguration

def initialize_dynamic_crawl(config: Crawl4AIConfiguration) -> CrawlerRunConfig:
    """
    Dynamically generates the runtime execution matrix based on the configuration payload.
    """
    # 1. Construct Link Filters
    filter_chain = FilterChain()
    
    if config.allowed_domains:
        filter_chain.add_filter(DomainFilter(allowed_domains=config.allowed_domains))
        
    if config.url_include_patterns:
        filter_chain.add_filter(URLPatternFilter(patterns=config.url_include_patterns))
        
    # 2. Construct Prioritization Scorer
    scorer = KeywordRelevanceScorer(keywords=config.intent_keywords)
    
    # 3. Combine into Strategy
    strategy = BestFirstCrawlingStrategy(
        max_depth=config.max_depth,
        max_pages=config.max_pages,
        filter_chain=filter_chain,
        url_scorer=scorer
    )
    
    # 4. Construct Noise-Cancellation Content Filter
    content_filter = BM25ContentFilter(
        user_query=" ".join(config.intent_keywords)
    )
    
    # 5. Wrap in Markdown Generator
    markdown_generator = DefaultMarkdownGenerator(content_filter=content_filter)
    
    # 6. Return CrawlerRunConfig
    return CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=True,
        deep_crawl_strategy=strategy,
        markdown_generator=markdown_generator
    )
