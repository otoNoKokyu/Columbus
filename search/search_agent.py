from exa_py.api import ContentsOptions
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import argparse
import sys
import time
from Columbus.utils.logger import logger

SOCIAL_MEDIA_DOMAINS = [
    "facebook.com", "twitter.com", "x.com", "instagram.com", "tiktok.com",
    "reddit.com", "linkedin.com", "youtube.com", "pinterest.com", "tumblr.com",
    "quora.com", "twitch.tv", "medium.com"
]


class BaseSearchClient(ABC):
    """Abstract Base Class representing a client-agnostic search engine."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5, **kwargs: Any) -> List[Dict[str, Any]]:
        """Perform a synchronous search.

        Args:
            query: The search query string.
            max_results: The maximum number of search results to return.
            **kwargs: Provider-specific search arguments.

        Returns:
            A list of dicts, each with standard keys: "title", "url", "snippet".
        """
        pass

    @abstractmethod
    async def async_search(self, query: str, max_results: int = 5, **kwargs: Any) -> List[Dict[str, Any]]:
        """Perform an asynchronous search.

        Args:
            query: The search query string.
            max_results: The maximum number of search results to return.
            **kwargs: Provider-specific search arguments.

        Returns:
            A list of dicts, each with standard keys: "title", "url", "snippet".
        """
        pass

class ExaSearchClient(BaseSearchClient):
    """Concrete implementation of BaseSearchClient for Exa.ai."""

    def __init__(self, **default_kwargs: Any):
        """Initialize the Exa client."""
        self.default_kwargs = default_kwargs
        import os
        api_key = os.environ.get("EXA_API_KEY")
        if not api_key:
            raise ValueError("EXA_API_KEY environment variable is missing.")
        try:
            from exa_py import Exa
            self.exa = Exa(api_key)
        except ImportError:
            raise ImportError("exa_py is not installed. Please run `pip install exa_py` first.")

    def search(self, query: str, max_results: int = 5, **kwargs: Any) -> List[Dict[str, Any]]:
        """Perform a synchronous Exa search."""
        merged_kwargs = {**self.default_kwargs, **kwargs}
        
        # Pop standard kwargs that exa doesn't expect or needs translation
        merged_kwargs.pop("region", None)
        merged_kwargs.pop("safesearch", None)
        merged_kwargs.pop("timelimit", None)

        # Merge user-supplied exclude_domains with default social media domains
        exclude_domains = merged_kwargs.pop("exclude_domains", []) or []
        if isinstance(exclude_domains, str):
            exclude_domains = [exclude_domains]
        all_excludes = list(set(list(exclude_domains) + SOCIAL_MEDIA_DOMAINS))
        merged_kwargs["exclude_domains"] = all_excludes

        try:
            contents_config = ContentsOptions(
                highlights=True,
                subpages=5,
            )

            # Make the API call passing the configuration object
            res = self.exa.search(
                query,
                type="neural",
                num_results=max_results,
                contents=contents_config,
                **merged_kwargs
            )
            
            standardized_results = []
            
            # Iterate through results and parse parameters safely
            for item in res.results:
                url = getattr(item, "url", "") or ""
                if url:
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    path_lower = parsed.path.lower()
                    if any(path_lower.endswith(ext) for ext in {'.pdf', '.zip', '.tar', '.gz', '.png', '.jpg', '.jpeg', '.gif', '.svg'}) or '/pdf/' in path_lower or '.pdf' in url.lower():
                        logger.info("ExaSearchClient: Filtering out PDF/unwanted URL from search results: %s", url)
                        continue

                # Safeguard text extracts if highlights arrays are empty or unavailable
                if getattr(item, "highlights", None) and len(item.highlights) > 0:
                    snippet = item.highlights[0]
                else:
                    snippet = item.text[:500] if getattr(item, "text", None) else ""
                    
                # Extract subpages safely (the SDK populates them inside item.subpages)
                subpages_list = getattr(item, "subpages", []) or []
                
                standardized_results.append({
                    "title": getattr(item, "title", "") or "",
                    "url": url,
                    "publishedDate": getattr(item, "published_date", "") or "",
                    "snippet": snippet,
                    "highlights": getattr(item, "highlights", []) or [],
                    "dss": subpages_list, # Extracted nested subpages array mapped to your custom key
                    "score": getattr(item, "score", None),
                    "highlightScore": getattr(item, "highlight_scores", None)
                })
            
            return standardized_results
        except Exception as e:
            logger.error("Error during Exa search execution: %s", e)
            return []

    async def async_search(self, query: str, max_results: int = 5, **kwargs: Any) -> List[Dict[str, Any]]:
        """Perform an asynchronous Exa search utilizing asyncio.to_thread."""
        return await asyncio.to_thread(self.search, query, max_results, **kwargs)


def get_search_client(provider: str = "exa", **kwargs: Any) -> BaseSearchClient:
    """Factory function to retrieve a concrete search client.

    Args:
        provider: The search provider key (defaults to 'exa').
        **kwargs: Optional configuration parameter dict for the provider.

    Returns:
        An instance of BaseSearchClient.
    """
    prov_lower = provider.lower()
    if prov_lower == "exa":
        return ExaSearchClient(**kwargs)
    else:
        raise ValueError(
            f"Unsupported search provider: '{provider}'. "
            f"Currently supported provider: 'exa'."
        )


async def async_search(query: str, provider: str = "exa", max_results: int = 5, **kwargs: Any) -> List[Dict[str, Any]]:
    """Convenience asynchronous function to execute a search."""
    client = get_search_client(provider)
    return await client.async_search(query, max_results=max_results, **kwargs)


async def fan_out_search(
    queries: List[str],
    provider: str = "exa",
    max_results_per_query: int = 5,
    deduplicate: bool = True,
    top_n: int = 10,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Fan-out search across multiple queries, deduplicate by URL, return top N.

    Designed for pipeline Stage 2-3: executes all queries concurrently,
    merges results, and deduplicates by URL.

    Args:
        queries: List of search queries (e.g., from the rewriter stage).
        provider: Search provider key (default: 'exa').
        max_results_per_query: Max results per individual query.
        deduplicate: Whether to deduplicate results by URL.
        top_n: Maximum number of unique results to return.
        **kwargs: Additional provider-specific kwargs.

    Returns:
        List of dicts with keys: title, url, snippet, source_query.
    """
    # Deduplicate queries while preserving order
    unique_queries = []
    seen_queries = set()
    for q in queries:
        q_clean = q.strip()
        if q_clean and q_clean not in seen_queries:
            seen_queries.add(q_clean)
            unique_queries.append(q_clean)

    if not unique_queries:
        logger.warning("fan_out_search: No valid queries provided after cleaning")
        return []

    logger.info("fan_out_search: Executing search for queries: %s", unique_queries)
    client = get_search_client(provider)

    # Execute all queries concurrently
    tasks = [
        client.async_search(query, max_results=max_results_per_query, **kwargs)
        for query in unique_queries
    ]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Print raw search results for each query for visibility
    logger.info("=====================================")
    logger.info("RAW SEARCH RESULTS FOR EACH QUERY:")
    logger.info("=====================================")
    for query, results in zip(unique_queries, all_results):
        if isinstance(results, BaseException):
            logger.error("  Query '%s' failed: %s", query, results)
            continue
        logger.info("  Query: '%s' (Found %d results)", query, len(results))
        for idx, res in enumerate(results):
            logger.info("    [%d] Score: %s | Title: %s", idx + 1, res.get("score"), res.get("title"))
            logger.info("        URL: %s", res.get("url"))
            snippet = res.get("snippet", "")
            logger.info("        Snippet: %s...", snippet[:120].replace('\n', ' '))
    logger.info("=====================================")

    # Merge and deduplicate, tracking all source queries per URL
    merged: List[Dict[str, Any]] = []
    url_index: Dict[str, int] = {}  # url → index in merged list

    for query, results in zip(unique_queries, all_results):
        if isinstance(results, BaseException):
            logger.error("fan_out_search: Query '%s' failed: %s", query, results)
            continue
        for result in results:
            url = result.get("url", "")
            if deduplicate and url in url_index:
                # Append this query to the existing entry's source_queries
                merged[url_index[url]].setdefault("source_queries", []).append(query)
                continue
            result["source_query"] = query
            result["source_queries"] = [query]
            if deduplicate:
                url_index[url] = len(merged)
            merged.append(result)

    logger.info(
        "fan_out_search: %d queries → %d total results → %d unique",
        len(queries), sum(
            len(r) for r in all_results if not isinstance(r, BaseException)
        ), len(merged),
    )

    return merged[:top_n]
