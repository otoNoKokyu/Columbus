import asyncio
from typing import Any, List, Dict, TYPE_CHECKING, Optional
from langchain_core.runnables import RunnableLambda
from ..search import async_search
from ..crawl.recursive_crawler import frontier_balanced_crawl
from ..types.types import Query, RewriterInput, CrawlResult, PerspectiveResult, ResearchOutput

if TYPE_CHECKING:
    from ..scoring.pinecone_reranker import PineconeReranker

from ..llm import get_langchain_llm
from ..rewriter.chain import create_adaptive_decomposition_chain
from ..types.types import DecomposedQuery

def decompose_query(query: str, query_budget: int = 6, callbacks: Optional[List[Any]] = None) -> DecomposedQuery:
    llm = get_langchain_llm(temperature=0.0)
    rewrite_chain = create_adaptive_decomposition_chain(llm, query_budget=query_budget)
    rewritten_queries = rewrite_chain.invoke(
        {"query": query},
        config={"callbacks": callbacks} if callbacks else {}
    )
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
    decomposed: DecomposedQuery,
    original_user_input: str,
    max_search: int = 10,
    max_rerank: int = 10,
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
    from Columbus.crawl.rate_limiter import RateLimiter
    from ..search import fan_out_search
    from ..crawl.recursive_crawler import frontier_balanced_crawl

    logger = logging.getLogger(__name__)

    if recursive_crawl:
        if max_depth is None or max_depth < 1:
            raise ValueError("max_depth must be at least 1 when recursive_crawl is enabled")
        actual_depth = max_depth
    else:
        actual_depth = 0

    api_key = os.environ.get("FIRECRAWL_API_KEY")
    reranker = LocalReranker(model_name="BAAI/bge-reranker-v2-m3", top_n=max_rerank)
    global_rate_limiter = RateLimiter(max_concurrent=2, max_requests_per_window=10, window_seconds=60.0)

    # 1. Fan-out Search
    queries = decomposed.get_flat_query_strings()
    logger.info("Executing global fan-out search for %d queries", len(queries))
    
    # We pass deduplicate=True to fan_out_search so it returns unique URLs
    # and keeps the source_query for the first query that found it
    search_results = await fan_out_search(
        queries=queries,
        provider="exa",
        max_results_per_query=max_search,
        deduplicate=True,
        top_n=100  # Pull a large pool before global rerank
    )
    
    # Filter out already visited URLs
    if visited_urls:
        visited_set = set(visited_urls)
        search_results = [res for res in search_results if res.get("url") not in visited_set]

    # 2. Global Rerank
    logger.info("Reranking %d unique search results against original user input", len(search_results))
    reranked_results = reranker.rerank(
        query=original_user_input, 
        candidates=search_results, 
        top_n=max_rerank
    )
    
    # 3. Frontier Crawl (Global)
    logger.info("Executing frontier crawl on top %d global URLs", len(reranked_results))
    scraped_pages = await frontier_balanced_crawl(
        seed_candidates=reranked_results,
        query=original_user_input,
        reranker=reranker,
        max_depth=actual_depth,
        pages_per_level=[5, 4, 3],
        score_threshold=0.7,
        api_key=api_key,
        rate_limiter=global_rate_limiter,
        return_markdown=return_markdown,
        academic_citations=academic_citations,
        skip_links=skip_links,
        visited=set(visited_urls) if visited_urls else None,
    )
    
    # Create mapping from url to its source query (from fan_out_search)
    url_to_source_query = {r.get("url"): r.get("source_query") for r in search_results}
    
    # Create mapping from query to objective_id
    query_to_obj_id = {sq.query: sq.objective_id for sq in decomposed.search_queries}
    query_to_obj_text = {sq.query: sq.objective_text for sq in decomposed.search_queries}
    
    perspectives_map = {} # objective_id -> PerspectiveResult
    contents_by_objective = {}
    queries_by_objective = {}
    
    # Initialize for all objectives
    for sq in decomposed.search_queries:
        if sq.objective_id not in perspectives_map:
            perspectives_map[sq.objective_id] = PerspectiveResult(
                objective_id=sq.objective_id,
                objective_text=sq.objective_text,
                rewritten_query=sq.query,
                pages_crawled=[]
            )
            contents_by_objective[sq.objective_id] = []
            queries_by_objective[sq.objective_id] = sq.query
            
    for page in scraped_pages:
        url = page.get("url")
        content = page.get("content")
        if url and content:
            source_q = url_to_source_query.get(url)
            obj_id = query_to_obj_id.get(source_q)
            if not obj_id:
                # Fallback to the first objective if something went wrong
                obj_id = decomposed.search_queries[0].objective_id
                
            crawl_res = CrawlResult(
                url=url,
                title=page.get("title") or "",
                depth=page.get("depth") or 0,
                score=page.get("score") or 0.0,
                content=content,
                token_count=page.get("token_count")
            )
            
            perspectives_map[obj_id].pages_crawled.append(crawl_res)
            contents_by_objective[obj_id].append({"url": url, "content": content})

    perspectives = list(perspectives_map.values())

    # 4. Chunk and Retrieve
    retrieved_chunks = await chunk_store_and_retrieve(
        original_query=original_user_input,
        queries_by_objective=queries_by_objective,
        contents_by_objective=contents_by_objective,
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
        decomposed=decomposed,
        perspectives=perspectives,
        retrieved_chunks=retrieved_chunks
    )