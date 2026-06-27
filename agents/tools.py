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

    # 2. Rerank candidates
    logger.debug(f"Phase: Pinecone Reranking for {bias_type}")
    reranked_results = reranker.rerank(query=q_text, candidates=results, top_n=max_rerank)

    if exa_highlight:
        logger.debug(f"exa_highlight is True: retrieving highlights directly and skipping crawl stage.")
        return [
            CrawlResult(
                url=res.get("url") or "",
                title=res.get("title") or "",
                depth=0,
                score=res.get("score") or 0.0,
                content="\n".join(res.get("highlights", [])) if res.get("highlights") else res.get("snippet", "")
            )
            for res in reranked_results
        ]

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
) -> ResearchOutput:
    import os
    import json
    import logging
    from ..scoring.pinecone_reranker import PineconeReranker
    from ..chunk_and_retrieve.main import chunk_store_and_retrieve

    logger = logging.getLogger(__name__)

    if recursive_crawl:
        if max_depth is None or max_depth < 1:
            raise ValueError("max_depth must be at least 1 when recursive_crawl is enabled")

    from Columbus.crawl.rate_limiter import RateLimiter
    
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    reranker = PineconeReranker(model_name="bge-reranker-v2-m3", top_n=max_rerank)
    
    # Use a single global rate limiter across all 5 branches to avoid Firecrawl 429 / queue timeouts
    global_rate_limiter = RateLimiter(max_concurrent=2, max_requests_per_window=10, window_seconds=60.0)

    logger.info("Decomposed Queries JSON:\n%s", json.dumps(query.model_dump(), indent=2))

    # Build tasks for 5 branches
    tasks = []
    bias_types = []

    for bias_type, q_text in query.model_dump().items():
        if isinstance(q_text, str) and q_text.strip():
            bias_types.append((bias_type, q_text))
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
                )
            )

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

    # Perform semantic chunking and retrieval from Pinecone bypassed to reduce logging and overhead
    retrieved_chunks = None
    if exa_highlight:
        # Populate retrieved_chunks with Exa highlights directly
        retrieved_chunks = {}
        for i, (bias_type, q_text) in enumerate(bias_types):
            pages = pages_results[i]
            chunks = []
            for p in pages:
                if p.content:
                    for line in p.content.split("\n"):
                        if line.strip():
                            chunks.append({
                                "text": line.strip(),
                                "score": p.score
                            })
            retrieved_chunks[bias_type] = chunks

    return ResearchOutput(
        original_query=original_user_input,
        perspectives=perspectives,
        retrieved_chunks=retrieved_chunks
    )