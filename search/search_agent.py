from exa_py.api import ContentsOptions
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import argparse
import sys
import time
import logging

# Attempt to import ddgs library
try:
    from ddgs import DDGS
except ImportError:
    # Fallback to old name if necessary, although ddgs should be preferred
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

logger = logging.getLogger(__name__)

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


class DuckDuckGoSearchClient(BaseSearchClient):
    """Concrete implementation of BaseSearchClient for DuckDuckGo."""

    def __init__(self, **default_kwargs: Any):
        """Initialize the client with default parameters."""
        self.default_kwargs = default_kwargs
        if DDGS is None:
            raise ImportError(
                "Neither 'ddgs' nor 'duckduckgo_search' package is installed. "
                "Please run `pip install ddgs` first."
            )

    def search(self, query: str, max_results: int = 5, **kwargs: Any) -> List[Dict[str, Any]]:
        """Perform a synchronous DuckDuckGo search."""
        merged_kwargs = {**self.default_kwargs, **kwargs}
        
        # Pop standard kwargs that might conflict or need translation
        region = merged_kwargs.pop("region", "us-en")
        safesearch = merged_kwargs.pop("safesearch", "moderate")
        timelimit = merged_kwargs.pop("timelimit", None)

        try:
            with DDGS() as ddgs:
                results = ddgs.text(
                    query,
                    max_results=max_results,
                    region=region,
                    safesearch=safesearch,
                    timelimit=timelimit,
                    **merged_kwargs
                )
                
                # Standardize output schema: title, url, snippet
                standardized_results = []
                for item in results:
                    url = item.get("href", "")
                    if url:
                        from urllib.parse import urlparse
                        parsed = urlparse(url)
                        path_lower = parsed.path.lower()
                        if any(path_lower.endswith(ext) for ext in {'.pdf', '.zip', '.tar', '.gz', '.png', '.jpg', '.jpeg', '.gif', '.svg'}) or '/pdf/' in path_lower or '.pdf' in url.lower():
                            logger.info("DuckDuckGoSearchClient: Filtering out PDF/unwanted URL from search results: %s", url)
                            continue
                    standardized_results.append({
                        "title": item.get("title", ""),
                        "url": url,
                        "snippet": item.get("body", "")
                    })
                return standardized_results
        except Exception as e:
            # Wrap client-specific errors or log them
            # For robustness in agentic flows, we return an empty list or bubble up a clean error
            # Depending on context, returning empty list is safer but logging it is important
            logger.error("Error during DuckDuckGo search execution: %s", e)
            return []

    async def async_search(self, query: str, max_results: int = 5, **kwargs: Any) -> List[Dict[str, Any]]:
        """Perform an asynchronous DuckDuckGo search utilizing asyncio.to_thread to keep it non-blocking."""
        return await asyncio.to_thread(self.search, query, max_results, **kwargs)

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


def get_search_client(provider: str = "ddg", **kwargs: Any) -> BaseSearchClient:
    """Factory function to retrieve a concrete search client.

    Args:
        provider: The search provider key (e.g. 'ddg', 'duckduckgo', 'exa').
        **kwargs: Optional configuration parameter dict for the provider.

    Returns:
        An instance of BaseSearchClient.
    """
    prov_lower = provider.lower()
    if prov_lower in ("ddg", "duckduckgo"):
        return DuckDuckGoSearchClient(**kwargs)
    elif prov_lower == "exa":
        return ExaSearchClient(**kwargs)
    else:
        raise ValueError(
            f"Unsupported search provider: '{provider}'. "
            f"Currently supported providers: 'ddg', 'exa'."
        )


def search(query: str, provider: str = "ddg", max_results: int = 5, **kwargs: Any) -> List[Dict[str, Any]]:
    """Convenience synchronous function to execute a search."""
    client = get_search_client(provider)
    return client.search(query, max_results=max_results, **kwargs)


async def async_search(query: str, provider: str = "exa", max_results: int = 5, **kwargs: Any) -> List[Dict[str, Any]]:
    """Convenience asynchronous function to execute a search."""
    client = get_search_client(provider)
    return await client.async_search(query, max_results=max_results, **kwargs)


async def fan_out_search(
    queries: List[str],
    provider: str = "ddg",
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
        provider: Search provider key (default: 'ddg').
        max_results_per_query: Max results per individual query.
        deduplicate: Whether to deduplicate results by URL.
        top_n: Maximum number of unique results to return.
        **kwargs: Additional provider-specific kwargs.

    Returns:
        List of dicts with keys: title, url, snippet, source_query.
    """
    if not queries:
        logger.warning("fan_out_search: No queries provided")
        return []

    client = get_search_client(provider)

    # Execute all queries concurrently
    tasks = [
        client.async_search(query, max_results=max_results_per_query, **kwargs)
        for query in queries
    ]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge and deduplicate
    merged: List[Dict[str, Any]] = []
    seen_urls: set = set()

    for query, results in zip(queries, all_results):
        if isinstance(results, BaseException):
            logger.error("fan_out_search: Query '%s' failed: %s", query, results)
            continue
        for result in results:
            url = result.get("url", "")
            if deduplicate and url in seen_urls:
                continue
            seen_urls.add(url)
            result["source_query"] = query
            merged.append(result)

    logger.info(
        "fan_out_search: %d queries → %d total results → %d unique",
        len(queries), sum(
            len(r) for r in all_results if not isinstance(r, BaseException)
        ), len(merged),
    )

    return merged[:top_n]


# Interactive CLI demonstration using 'rich'
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query various search engines using a client-agnostic interface.")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("-p", "--provider", type=str, default="ddg", help="Search provider (default: ddg)")
    parser.add_argument("-n", "--max-results", type=int, default=5, help="Max results to fetch (default: 5)")
    parser.add_argument("-a", "--async-mode", action="store_true", help="Execute search in async mode")
    parser.add_argument("-r", "--region", type=str, default="us-en", help="Region query parameter (default: us-en)")
    parser.add_argument("-s", "--safesearch", type=str, default="moderate", choices=["on", "moderate", "off"], help="SafeSearch level (default: moderate)")
    parser.add_argument("-t", "--timelimit", type=str, default=None, help="Time limit filter, e.g. d (day), w (week), m (month), y (year)")

    args = parser.parse_args()

    # Try importing rich for beautiful console output
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
        HAS_RICH = True
    except ImportError:
        HAS_RICH = False

    def print_results_classic(results: List[Dict[str, Any]], elapsed: float):
        print(f"\nFetched {len(results)} results in {elapsed:.3f}s using provider: {args.provider}\n" + "="*80)
        for i, res in enumerate(results, 1):
            print(f"{i}. {res['title']}")
            print(f"   URL: {res['url']}")
            print(f"   Snippet: {res['snippet']}")
            print("-" * 80)

    def print_results_rich(results: List[Dict[str, Any]], elapsed: float):
        console = Console()
        
        title_str = f"🔍 Search Results for: [bold yellow]{args.query}[/bold yellow] (Provider: [bold cyan]{args.provider}[/bold cyan])"
        subtitle_str = f"Found {len(results)} results in [green]{elapsed:.3f} seconds[/green] (Mode: {'Async' if args.async_mode else 'Sync'})"
        
        console.print(Panel(subtitle_str, title=title_str, border_style="blue", box=box.ROUNDED))
        
        if not results:
            console.print("[bold red]No results found or an error occurred during search.[/bold red]")
            return

        table = Table(box=box.DOUBLE_EDGE, show_lines=True, expand=True)
        table.add_column("#", justify="center", style="dim", width=4)
        table.add_column("Title / Link", style="bold magenta", ratio=2)
        table.add_column("Snippet / Summary", style="white", ratio=3)

        for idx, res in enumerate(results, 1):
            title_link = f"[link={res['url']}]{res['title']}[/link]\n[blue][underline]{res['url']}[/underline][/blue]"
            table.add_row(str(idx), title_link, res['snippet'])
            
        console.print(table)

    # Perform search
    start_time = time.time()
    if args.async_mode:
        results = asyncio.run(async_search(
            args.query,
            provider=args.provider,
            max_results=args.max_results,
            region=args.region,
            safesearch=args.safesearch,
            timelimit=args.timelimit
        ))
    else:
        results = search(
            args.query,
            provider=args.provider,
            max_results=args.max_results,
            region=args.region,
            safesearch=args.safesearch,
            timelimit=args.timelimit
        )
    elapsed = time.time() - start_time

    # Display results
    if HAS_RICH:
        print_results_rich(results, elapsed)
    else:
        print_results_classic(results, elapsed)
