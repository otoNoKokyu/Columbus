from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class FirecrawlConfiguration(BaseModel):
    """Data contract for configuring the DuckDuckGo + Firecrawl pipeline."""
    queries: List[str] = Field(..., description="List of search queries to execute via DuckDuckGo")
    api_key: Optional[str] = Field(default=None, description="Firecrawl API key (can also be set via FIRECRAWL_API_KEY env var)")
    extraction_schema: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="A JSON schema defining the structured data you want Firecrawl to extract natively."
    )
    duckduckgo_max_results: int = Field(default=2, description="Maximum number of search results to scrape per query to conserve rate limits")

class Crawl4AIConfiguration(BaseModel):
    """Data contract for configuring the deep crawler."""
    seed_url: str = Field(..., description="The initial URL to start crawling from")
    allowed_domains: Optional[List[str]] = Field(default=None, description="List of domains allowed to be crawled")
    url_include_patterns: Optional[List[str]] = Field(default=None, description="List of regex patterns for URLs to include")
    intent_keywords: List[str] = Field(..., description="Keywords applied to BOTH link prioritization scoring and text noise-cancellation")
    max_depth: int = Field(default=2, description="Maximum depth for the crawl")
    max_pages: int = Field(default=50, description="Maximum number of pages to crawl")

AnyCrawlConfiguration = Union[FirecrawlConfiguration, Crawl4AIConfiguration]
