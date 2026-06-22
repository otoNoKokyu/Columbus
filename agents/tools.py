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
    rate_limiter: Any = None
) -> List[CrawlResult]:
    import logging
    logger = logging.getLogger(__name__)

    # 1. Search
    logger.info(f"Phase: Exa Search for {bias_type} with query: {q_text}")
    results = await async_search(q_text, provider="exa", max_results=30)
    if not results:
        logger.warning(f"No search results for {bias_type}")
        return []

    # 2. Rerank candidates
    logger.info(f"Phase: Pinecone Reranking for {bias_type}")
    reranked_results = reranker.rerank(query=q_text, candidates=results, top_n=5)
    reranked_urls = [res.get("url") for res in reranked_results if res.get("url")]
    logger.info(f"Top reranked seed URLs for '{bias_type}': {reranked_urls}")

    # 3. Frontier Balanced Crawl
    logger.info(f"Phase: Frontier Crawl for {bias_type}")
    scraped_pages = await frontier_balanced_crawl(
        seed_candidates=reranked_results,
        query=q_text,
        reranker=reranker,
        max_depth=1,
        pages_per_level=[5, 4, 3],
        score_threshold=0.7,
        api_key=api_key,
        rate_limiter=rate_limiter,
    )

    # 4. Serialize to CrawlResult
    return [
        CrawlResult(
            url=page.get("url") or "",
            title=page.get("title") or "",
            depth=page.get("depth") or 0,
            score=page.get("score") or 0.0,
            content=page.get("content") or ""
        )
        for page in scraped_pages
    ]

async def search_crawl_rerank_queries(query: Query, original_user_input: str) -> ResearchOutput:
    import os
    import logging
    from ..scoring.pinecone_reranker import PineconeReranker
    from ..chunk_and_retrieve.main import chunk_store_and_retrieve

    logger = logging.getLogger(__name__)

    from Columbus.crawl.rate_limiter import RateLimiter
    
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    reranker = PineconeReranker(model_name="bge-reranker-v2-m3", top_n=5)
    
    # Use a single global rate limiter across all 5 branches to avoid Firecrawl 429 / queue timeouts
    global_rate_limiter = RateLimiter(max_concurrent=2, max_requests_per_window=10, window_seconds=60.0)

    logger.info(f"Phase: Query Generation complete. Processing 5 branches: {list(query.model_dump().keys())}")
    for b_type, q_val in query.model_dump().items():
        logger.info(f"Decomposed query for branch '{b_type}': '{q_val}'")

    # Build tasks for 5 branches
    tasks = []
    bias_types = []

    for bias_type, q_text in query.model_dump().items():
        if isinstance(q_text, str) and q_text.strip():
            bias_types.append((bias_type, q_text))
            tasks.append(search_crawl_rerank_single_query(bias_type, q_text, reranker, api_key, global_rate_limiter))

    # Concurrently run the single query search/crawl pipeline
    pages_results = await asyncio.gather(*tasks)

    # Prepare for chunking
    contents_by_bias = {}
    queries_by_bias = {}
    perspectives = []

    for i, (bias_type, q_text) in enumerate(bias_types):
        pages = pages_results[i]
        contents = [p.content for p in pages if p.content]

        contents_by_bias[bias_type] = contents
        queries_by_bias[bias_type] = q_text

        perspectives.append(
            PerspectiveResult(
                bias_type=bias_type,
                rewritten_query=q_text,
                pages_crawled=pages
            )
        )

    # Perform semantic chunking and retrieval from Pinecone
    retrieved_chunks = None
    if any(contents_by_bias.values()):
        try:
            retrieved_chunks = await chunk_store_and_retrieve(
                queries_by_bias=queries_by_bias,
                contents_by_bias=contents_by_bias,
                index_name="columbus-research"
            )
        except Exception as e:
            logger.error("chunk_store_and_retrieve failed in search_crawl_rerank_queries: %s", e)

    return ResearchOutput(
        original_query=original_user_input,
        perspectives=perspectives,
        retrieved_chunks=retrieved_chunks
    )