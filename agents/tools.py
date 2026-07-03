import asyncio
from typing import Any, List, Dict, TYPE_CHECKING
from ..search import async_search
from ..crawl.recursive_crawler import frontier_balanced_crawl
from ..types.types import Query, RewriterInput, CrawlResult, PerspectiveResult, ResearchOutput

if TYPE_CHECKING:
    from ..scoring.pinecone_reranker import PineconeReranker

from ..llm import get_langchain_llm
from ..rewriter.chain import create_balanced_rewrite_chain

def decompose_query(query: str) -> Query:
    llm = get_langchain_llm(temperature=0.0)
    rewrite_chain = create_balanced_rewrite_chain(llm)
    rewritten_queries = rewrite_chain.invoke(RewriterInput(query=query))
    return rewritten_queries

async def search_crawl_rerank_single_query(
    bias_type: str,
    q_text: str,
    reranker: "PineconeReranker",
    api_key: str | None,
    rate_limiter: Any = None,
    max_search: int = 30,
    max_rerank: int = 5,
    recursive_crawl: bool = False,
    max_depth: int | None = None,
    exa_highlight: bool = False,
    return_markdown: bool = False,
    academic_citations: bool = False,
    skip_links: bool = True,
    visited_urls: Optional[List[str]] = None,
) -> List[CrawlResult]:
    import logging
    logger = logging.getLogger(__name__)

    if recursive_crawl:
        if max_depth is None or max_depth < 1:
            raise ValueError("max_depth must be at least 1 when recursive_crawl is enabled")
        actual_depth = max_depth
    else:
        actual_depth = 0

    # 1. Search
    logger.info(f"Phase: Exa Search for {bias_type} with query: {q_text}")
    results = await async_search(q_text, provider="exa", max_results=max_search)
    if not results:
        logger.warning(f"No search results for {bias_type}")
        return []

    # Filter out already visited URLs
    if visited_urls:
        visited_set = set(visited_urls)
        results = [res for res in results if res.get("url") not in visited_set]
        if not results:
            logger.info(f"All search results for {bias_type} were already visited.")
            return []

    # 2. Rerank candidates
    logger.debug(f"Phase: Pinecone Reranking for {bias_type}")
    reranked_results = reranker.rerank(query=q_text, candidates=results, top_n=max_rerank)



    reranked_urls = [res.get("url") for res in reranked_results if res.get("url")]
    logger.debug(f"Top reranked seed URLs for '{bias_type}': {reranked_urls}")

    # 3. Frontier Balanced Crawl
    logger.info(f"Phase: Frontier Crawl for {bias_type}")
    scraped_pages = await frontier_balanced_crawl(
        seed_candidates=reranked_results,
        query=q_text,
        reranker=reranker,
        max_depth=actual_depth,
        pages_per_level=[5, 4, 3],
        score_threshold=0.7,
        api_key=api_key,
        rate_limiter=rate_limiter,
        return_markdown=return_markdown,
        academic_citations=academic_citations,
        skip_links=skip_links,
        visited=set(visited_urls) if visited_urls else None,
    )

    # 4. Serialize to CrawlResult
    return [
        CrawlResult(
            url=page.get("url") or "",
            title=page.get("title") or "",
            depth=page.get("depth") or 0,
            score=page.get("score") or 0.0,
            content=page.get("content") or "",
            token_count=page.get("token_count")
        )
        for page in scraped_pages
    ]

async def search_crawl_rerank_queries(
    query: Query,
    original_user_input: str,
    max_search: int = 30,
    max_rerank: int = 5,
    recursive_crawl: bool = False,
    max_depth: int | None = None,
    chunking_strategy: str = "semantic",
    min_chunk_size: int = 500,
    chunk_overlap: int = 50,
    exa_highlight: bool = False,
    embedding_source: str = "local",
    return_markdown: bool = False,
    academic_citations: bool = False,
    skip_links: bool = True,
    cache_file: str | None = None,
    top_k: int = 5,
    visited_urls: Optional[List[str]] = None,
) -> ResearchOutput:
    import os
    import json
    import logging
    from ..scoring.local_reranker import LocalReranker
    from ..chunk_and_retrieve.main import chunk_store_and_retrieve

    logger = logging.getLogger(__name__)

    if recursive_crawl:
        if max_depth is None or max_depth < 1:
            raise ValueError("max_depth must be at least 1 when recursive_crawl is enabled")

    from Columbus.crawl.rate_limiter import RateLimiter
    
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    reranker = LocalReranker(model_name="BAAI/bge-reranker-v2-m3", top_n=max_rerank)
    
    # Use a single global rate limiter across all 5 branches to avoid Firecrawl 429 / queue timeouts
    global_rate_limiter = RateLimiter(max_concurrent=2, max_requests_per_window=10, window_seconds=60.0)

    logger.info("Decomposed Queries JSON:\n%s", json.dumps(query.model_dump(), indent=2))

    # Build tasks for branches
    tasks = []
    bias_types = []

    for bias_type, q_text in query.model_dump().items():
        if isinstance(q_text, str) and q_text.strip():
            bias_types.append((bias_type, q_text))
            
    # Check if cache exists and load it to bypass network calls
    pages_results = None
    if cache_file and os.path.exists(cache_file):
        logger.info(f"Loading raw crawl results from cache: {cache_file}")
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
                if len(cached_data) == len(bias_types):
                    pages_results = [[CrawlResult(**page) for page in branch] for branch in cached_data]
                else:
                    logger.warning(f"Cache mismatch: expected {len(bias_types)} branches, got {len(cached_data)}. Ignoring cache.")
        except Exception as e:
            logger.error(f"Failed to load cache from {cache_file}: {e}")
            
    if pages_results is None:
        logger.info("No cache found. Executing full Exa Search and Firecrawl Scraping...")
        for bias_type, q_text in bias_types:
            tasks.append(
                search_crawl_rerank_single_query(
                    bias_type,
                    q_text,
                    reranker,
                    api_key,
                    global_rate_limiter,
                    max_search=max_search,
                    max_rerank=max_rerank,
                    recursive_crawl=recursive_crawl,
                    max_depth=max_depth,
                    exa_highlight=exa_highlight,
                    return_markdown=return_markdown,
                    academic_citations=academic_citations,
                    skip_links=skip_links,
                    visited_urls=visited_urls,
                )
            )

        # Concurrently run the single query search/crawl pipeline
        pages_results = await asyncio.gather(*tasks)
        
        # Save to cache for next run
        if cache_file:
            logger.info(f"Saving raw crawl results to cache: {cache_file}")
            try:
                with open(cache_file, "w") as f:
                    serialized_data = [[page.model_dump() for page in branch] for branch in pages_results]
                    json.dump(serialized_data, f, indent=4)
            except Exception as e:
                logger.error(f"Failed to write cache to {cache_file}: {e}")

    # Prepare for chunking
    contents_by_bias = {}
    queries_by_bias = {}
    perspectives = []

    for i, (bias_type, q_text) in enumerate(bias_types):
        pages = pages_results[i]
        contents = [{"url": p.url, "content": p.content} for p in pages if p.content]

        contents_by_bias[bias_type] = contents
        queries_by_bias[bias_type] = q_text

        perspectives.append(
            PerspectiveResult(
                bias_type=bias_type,
                rewritten_query=q_text,
                pages_crawled=pages
            )
        )

    # Perform semantic chunking, deduplication, and retrieval
    retrieved_chunks = await chunk_store_and_retrieve(
        original_query=original_user_input,
        queries_by_bias=queries_by_bias,
        contents_by_bias=contents_by_bias,
        index_name="columbus-research",
        pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
        chunking_strategy=chunking_strategy,
        min_chunk_size=min_chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_source=embedding_source,
        top_k=top_k,
    )

    return ResearchOutput(
        original_query=original_user_input,
        perspectives=perspectives,
        retrieved_chunks=retrieved_chunks
    )