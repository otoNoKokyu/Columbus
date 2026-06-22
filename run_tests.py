"""
Columbus — Modular Pipeline Test Runner
=======================================

Each function below tests ONE stage of the pipeline independently.
Every function:
  - Takes explicit inputs (no hidden state)
  - Returns explicit outputs (can be fed to the next stage)
  - Prints results in a clear, readable format

HOW TO USE:
  1. Run the full pipeline:     python run_tests.py
  2. Test a single stage:       Comment out lines in main() you don't need.
                                Hardcode inputs for the stage you want to test.
  3. Change the query:          Edit TEST_QUERY at the top.
  4. Override config:           Edit get_config().

Example — test ONLY the rewriter:

    if __name__ == "__main__":
        setup_logging()
        config = get_config()
        llm = build_llm(config)
        queries = test_1_query_rewrite(llm, "how does async Python work?", config)
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# ── Ensure Columbus is importable ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env from Columbus root
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION — Edit these to change test behavior
# ═══════════════════════════════════════════════════════════════════════

TEST_QUERY = "best practices for async Python web scraping"


def get_config():
    """Return the pipeline config. Edit values here to tune the test."""
    from Columbus.pipeline.config import ResearchPipelineConfig
    return ResearchPipelineConfig(
        max_rewritten_queries=3,
        search_results_per_query=5,
        top_urls_after_search=10,
        embedding_model="all-MiniLM-L6-v2",
        top_links_after_embedding=20,
        reranker_strategy="pinecone",
        top_links_after_rerank=5,
        recursive_crawl_depth=2,
        recursive_max_pages_per_seed=3,
        llm_temperature=0.0,
    )


# ═══════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════

def setup_logging(level=logging.INFO):
    """Configure logging for the test run."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _header(title: str):
    """Print a stage header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def _print_json(label: str, data: Any, max_items: int = 10):
    """Pretty-print a list/dict with a label, truncating if needed."""
    if isinstance(data, list):
        shown = data[:max_items]
        print(f"  {label}: ({len(data)} items, showing first {len(shown)})")
        for i, item in enumerate(shown):
            if isinstance(item, dict):
                # Show compact version of dict
                compact = {k: (v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v)
                           for k, v in item.items() if k != "markdown"}
                if "markdown" in item:
                    compact["markdown"] = f"[{len(item['markdown'])} chars]"
                print(f"    [{i}] {json.dumps(compact, indent=None, default=str)}")
            else:
                print(f"    [{i}] {item}")
    elif isinstance(data, dict):
        print(f"  {label}:")
        print(f"    {json.dumps(data, indent=2, default=str)}")
    else:
        print(f"  {label}: {data}")


def _print_summary(key_values: dict):
    """Print a key-value summary block."""
    for k, v in key_values.items():
        print(f"  {k}: {v}")


# ═══════════════════════════════════════════════════════════════════════
# SHARED COMPONENT BUILDERS — call these to get reusable instances
# ═══════════════════════════════════════════════════════════════════════

def build_llm(config):
    """Build the LLM instance. Needed by: rewriter, LLM reranker."""
    _header("Building LLM")
    from Columbus.llm import get_langchain_llm
    llm = get_langchain_llm(temperature=config.llm_temperature)
    print(f"  LLM ready: {type(llm).__name__}")
    return llm


def build_embedding_scorer(config):
    """Build the EmbeddingScorer. Needed by: Stage 6-7, Stage 9."""
    _header("Building EmbeddingScorer")
    from Columbus.scoring.embedding_scorer import EmbeddingScorer
    scorer = EmbeddingScorer(model_name=config.embedding_model)
    print(f"  Scorer ready: {config.embedding_model}")
    return scorer


def build_reranker(config, llm=None):
    """Build the reranker. Needed by: Stage 8."""
    _header("Building Reranker")
    from Columbus.scoring import get_reranker
    reranker = get_reranker(
        "pinecone",
        model_name=config.pinecone_rerank_model,
        top_n=config.top_links_after_rerank,
    )
    print(f"  Reranker ready: {config.reranker_strategy} ({type(reranker).__name__})")
    return reranker


# ═══════════════════════════════════════════════════════════════════════
# STAGE 1: Query Rewrite
# ═══════════════════════════════════════════════════════════════════════

def test_1_query_rewrite(llm, query: str, config) -> list[str]:
    """
    Stage 1: Rewrite a user query into N search variants using the LLM.

    Input:  query string, LLM instance
    Output: list of rewritten query strings

    Uses: Columbus.rewriter.chain.create_balanced_rewrite_chain
    """
    _header("STAGE 1: Query Rewrite")
    from Columbus.rewriter.chain import create_balanced_rewrite_chain
    from Columbus.types.types import RewriterInput

    chain = create_balanced_rewrite_chain(llm)
    query_obj = chain.invoke(RewriterInput(query=query))
    
    # Extract the 5 string queries from the Pydantic object
    queries = [v for k, v in query_obj.model_dump().items() if isinstance(v, str) and v.strip()]

    _print_summary({
        "Original query": query,
        "Rewritten count": len(queries),
    })
    _print_json("Rewritten queries", queries)
    return queries


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2-3: Fan-Out Search + Dedup
# ═══════════════════════════════════════════════════════════════════════

def test_2_3_search(queries: list[str], config) -> list[dict]:
    """
    Stage 2-3: Fan-out DuckDuckGo search across all query variants,
    deduplicate by URL, return top N results.

    Input:  list of search queries
    Output: list of search result dicts [{title, url, snippet, source_query}]

    Uses: Columbus.search.fan_out_search
    """
    _header("STAGE 2-3: Fan-Out Search + Dedup")
    from Columbus.search import fan_out_search

    results = asyncio.run(fan_out_search(
        queries=queries,
        provider=config.search_provider,
        max_results_per_query=config.search_results_per_query,
        top_n=config.top_urls_after_search,
    ))

    _print_summary({
        "Queries searched": len(queries),
        "Unique results": len(results),
    })
    _print_json("Search results", results)
    return results


# ═══════════════════════════════════════════════════════════════════════
# STAGE 4: Firecrawl Scrape → Markdown
# ═══════════════════════════════════════════════════════════════════════

def test_4_firecrawl(urls: list[str], config) -> list[dict]:
    """
    Stage 4: Scrape each URL via Firecrawl and extract markdown content.

    Input:  list of URL strings
    Output: list of dicts [{url, markdown, error?}]

    Uses: Columbus.crawl.firecrawl_engine.scrape_urls_for_markdown
    """
    _header("STAGE 4: Firecrawl Scrape → Markdown")
    from Columbus.crawl.firecrawl_engine import scrape_urls_for_markdown

    pages = asyncio.run(scrape_urls_for_markdown(
        urls=urls,
        api_key=config.firecrawl_api_key,
    ))

    success = sum(1 for p in pages if not p.get("error"))
    _print_summary({
        "URLs scraped": len(urls),
        "Successful": success,
        "Failed": len(urls) - success,
    })
    _print_json("Crawled pages", pages)
    return pages


# ═══════════════════════════════════════════════════════════════════════
# STAGE 5: Link Extraction
# ═══════════════════════════════════════════════════════════════════════

def test_5_link_extraction(pages: list[dict]) -> list[dict]:
    """
    Stage 5: Extract all hyperlinks from crawled markdown pages.
    Deduplicates by URL across all pages.

    Input:  list of crawled page dicts [{url, markdown, error?}]
    Output: list of link dicts [{url, anchor_text, context, source_url}]

    Uses: Columbus.crawl.link_extractor.extract_links_from_markdown
    """
    _header("STAGE 5: Link Extraction")
    from Columbus.crawl.link_extractor import extract_links_from_markdown

    all_links = []
    seen_urls = set()

    for page in pages:
        if page.get("error"):
            continue
        links = extract_links_from_markdown(
            markdown=page.get("markdown", ""),
            base_url=page.get("url", ""),
        )
        for link in links:
            if link["url"] not in seen_urls:
                seen_urls.add(link["url"])
                all_links.append(link)

    _print_summary({
        "Pages processed": len([p for p in pages if not p.get("error")]),
        "Unique links extracted": len(all_links),
    })
    _print_json("Extracted links", all_links, max_items=15)
    return all_links


# ═══════════════════════════════════════════════════════════════════════
# STAGE 6-7: Embedding Scoring → Top N
# ═══════════════════════════════════════════════════════════════════════

def test_6_7_embedding_score(
    scorer, query: str, links: list[dict], config
) -> list[dict]:
    """
    Stage 6-7: Score all extracted links by semantic similarity to the
    query using sentence-transformers, return top N.

    Input:  EmbeddingScorer, query string, list of link dicts
    Output: top N link dicts sorted by embedding_score descending

    Uses: Columbus.scoring.embedding_scorer.EmbeddingScorer
    """
    _header("STAGE 6-7: Embedding Scoring → Top N")

    if not links:
        print("  ⚠ No links to score. Returning empty.")
        return []

    scored = asyncio.run(scorer.score(
        query=query,
        candidates=links,
        top_n=config.top_links_after_embedding,
    ))

    _print_summary({
        "Input links": len(links),
        "Kept (top N)": len(scored),
        "Best score": f"{scored[0].get('embedding_score', 0):.4f}" if scored else "N/A",
        "Worst kept": f"{scored[-1].get('embedding_score', 0):.4f}" if scored else "N/A",
    })
    _print_json("Top scored links", scored, max_items=10)
    return scored


# ═══════════════════════════════════════════════════════════════════════
# STAGE 8: Rerank → Top K
# ═══════════════════════════════════════════════════════════════════════

def test_8_rerank(reranker, query: str, candidates: list[dict], config) -> list[dict]:
    """
    Stage 8: Rerank the embedding-scored links using either a
    cross-encoder or LLM-based reranker. Returns top K.

    Input:  BaseReranker, query string, list of scored link dicts
    Output: top K link dicts sorted by rerank_score descending

    Uses: Columbus.scoring.{CrossEncoderReranker | LLMRelevanceReranker}
    """
    _header("STAGE 8: Rerank → Top K")

    if not candidates:
        print("  ⚠ No candidates to rerank. Returning empty.")
        return []

    reranked = reranker.rerank(
        query=query,
        candidates=candidates,
        top_n=config.top_links_after_rerank,
    )

    _print_summary({
        "Input candidates": len(candidates),
        "Kept (top K)": len(reranked),
        "Reranker": config.reranker_strategy,
        "Best score": f"{reranked[0].get('rerank_score', 0):.4f}" if reranked else "N/A",
    })
    _print_json("Reranked links", reranked)
    return reranked


# ═══════════════════════════════════════════════════════════════════════
# STAGE 9: Recursive Crawl (depth=2)
# ═══════════════════════════════════════════════════════════════════════

def test_9_recursive_crawl(
    scorer, query: str, seed_links: list[dict], config
) -> list[dict]:
    """
    Stage 9: For each top-ranked URL, recursively crawl child links
    up to depth=2. Uses embedding_scorer to pick the most relevant
    child links at each level.

    Input:  EmbeddingScorer, query string, list of top-ranked link dicts
    Output: list of tree nodes [{url, depth, markdown, children: [...]}]

    Uses: Columbus.crawl.recursive_crawler.recursive_crawl
    """
    _header("STAGE 9: Recursive Crawl (depth=2)")
    from Columbus.crawl.recursive_crawler import recursive_crawl

    seed_urls = [link.get("url", "") for link in seed_links]

    if not seed_urls:
        print("  ⚠ No seed URLs to crawl. Returning empty.")
        return []

    tree = asyncio.run(recursive_crawl(
        seed_urls=seed_urls,
        query=query,
        embedding_scorer=scorer,
        max_depth=config.recursive_crawl_depth,
        max_pages_per_seed=config.recursive_max_pages_per_seed,
        api_key=config.firecrawl_api_key,
    ))

    # Count total pages
    total = 0
    for node in tree:
        total += 1
        total += len(node.get("children", []))

    _print_summary({
        "Seed URLs": len(seed_urls),
        "Total pages crawled": total,
        "Crawl depth": config.recursive_crawl_depth,
    })

    # Print tree structure
    for i, node in enumerate(tree):
        md_len = len(node.get("markdown", ""))
        err = node.get("error", "")
        status = f"error={err}" if err else f"{md_len} chars"
        print(f"  [{i}] {node['url'][:70]}  ({status})")
        for j, child in enumerate(node.get("children", [])):
            c_md = len(child.get("markdown", ""))
            c_err = child.get("error", "")
            c_status = f"error={c_err}" if c_err else f"{c_md} chars"
            print(f"       └─[{j}] {child['url'][:60]}  ({c_status})")

    return tree


# ═══════════════════════════════════════════════════════════════════════
# FULL PIPELINE — Run everything end-to-end
# ═══════════════════════════════════════════════════════════════════════

def test_full_pipeline(query: str, config):
    """
    Run the FULL pipeline as a single LCEL chain via the factory.
    This is the "production" invocation path.

    Uses: Columbus.pipeline.factory.create_research_chain
    """
    _header("FULL PIPELINE (via factory)")
    from Columbus.pipeline.factory import create_research_chain
    from Columbus.utils.callbacks import TokenAccumulatorCallbackHandler

    chain = create_research_chain(config)
    token_cb = TokenAccumulatorCallbackHandler()

    state = chain.invoke(
        {"query": query},
        config={
            "run_name": "ResearchPipeline",
            "callbacks": [token_cb],
            "tags": ["test"],
        },
    )

    _print_summary({
        "Query": query,
        "Rewritten queries": len(state.get("rewritten_queries", [])),
        "Search results": len(state.get("search_results", [])),
        "Top URLs": len(state.get("top_urls", [])),
        "Crawled pages": len(state.get("crawled_pages", [])),
        "Extracted links": len(state.get("extracted_links", [])),
        "Embedding scored": len(state.get("embedding_scored_links", [])),
        "Reranked links": len(state.get("reranked_links", [])),
        "Recursive crawl nodes": len(state.get("recursive_crawl_output", [])),
        "Tokens (input)": token_cb.input_tokens,
        "Tokens (output)": token_cb.output_tokens,
    })

    return state


# ═══════════════════════════════════════════════════════════════════════
# MAIN — Comment/uncomment stages to test what you need
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    setup_logging()
    config = get_config()

    # ── Build shared components ────────────────────────────────────
    llm = build_llm(config)
    scorer = build_embedding_scorer(config)
    reranker = build_reranker(config, llm=llm)

    # ── Stage 1: Query Rewrite ─────────────────────────────────────
    queries = test_1_query_rewrite(llm, TEST_QUERY, config)

    # ── Stage 2-3: Search ──────────────────────────────────────────
    search_results = test_2_3_search(queries, config)
    urls = [r["url"] for r in search_results]

    # ── Stage 4: Firecrawl ─────────────────────────────────────────
    pages = test_4_firecrawl(urls, config)

    # ── Stage 5: Link Extraction ───────────────────────────────────
    links = test_5_link_extraction(pages)

    # ── Stage 6-7: Embedding Scoring ───────────────────────────────
    scored = test_6_7_embedding_score(scorer, TEST_QUERY, links, config)

    # ── Stage 8: Rerank ────────────────────────────────────────────
    reranked = test_8_rerank(reranker, TEST_QUERY, scored, config)

    # ── Stage 9: Recursive Crawl ───────────────────────────────────
    tree = test_9_recursive_crawl(scorer, TEST_QUERY, reranked, config)

    # ── Final Summary ──────────────────────────────────────────────
    _header("PIPELINE COMPLETE")
    _print_summary({
        "Query": TEST_QUERY,
        "Rewritten queries": len(queries),
        "Search results": len(search_results),
        "Crawled pages": len(pages),
        "Extracted links": len(links),
        "After embedding score": len(scored),
        "After rerank": len(reranked),
        "Recursive crawl trees": len(tree),
    })

    # ──────────────────────────────────────────────────────────────
    # ALTERNATIVE: Uncomment below to test the full LCEL pipeline
    #              (factory-wired, single invoke call)
    # ──────────────────────────────────────────────────────────────
    # state = test_full_pipeline(TEST_QUERY, config)
