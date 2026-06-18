"""Functional tests for ResearchAgent modules — NO LLM calls.

Tests link_extractor, embedding_scorer, config, and callbacks.
"""

import logging
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test")


def test_link_extractor():
    """Test pure markdown link extraction."""
    from ResearchAgent.crawl.link_extractor import extract_links_from_markdown

    md = """
# Python Resources
Check out [Python docs](https://docs.python.org/3/) for the official documentation.
Also see [Real Python tutorials](https://realpython.com) for practical guides.
Here is an [image](/logo.png) and a [stylesheet](/style.css) to skip.
A [relative link](./subpage) should resolve against base.
A [mailto](mailto:test@example.com) should be skipped.
A [duplicate](https://docs.python.org/3/) should be deduplicated.
"""

    links = extract_links_from_markdown(md, "https://example.com/page")
    logger.info("Extracted %d links", len(links))
    for link in links:
        logger.info("  %s -> %s", link["anchor_text"], link["url"])

    assert len(links) == 3, f"Expected 3, got {len(links)}"
    assert links[0]["url"] == "https://docs.python.org/3/"
    assert links[1]["url"] == "https://realpython.com"
    assert links[2]["url"] == "https://example.com/subpage"
    assert all(link["source_url"] == "https://example.com/page" for link in links)
    logger.info("Test link_extractor PASSED")


def test_embedding_scorer():
    """Test embedding-based scoring — downloads model on first run (~80MB)."""
    from ResearchAgent.scoring.embedding_scorer import EmbeddingScorer

    scorer = EmbeddingScorer(model_name="all-MiniLM-L6-v2")

    candidates = [
        {"anchor_text": "Python async tutorial", "context": "Learn async await in Python 3.12"},
        {"anchor_text": "JavaScript promises", "context": "Understanding JS promises and callbacks"},
        {"anchor_text": "Python web scraping guide", "context": "Using aiohttp and asyncio for scraping"},
        {"anchor_text": "Recipe for chocolate cake", "context": "Baking instructions for dessert"},
    ]

    scored = asyncio.run(
        scorer.score(query="async Python web scraping", candidates=candidates, top_n=2)
    )

    logger.info("Scored %d -> kept top 2", len(candidates))
    for s in scored:
        logger.info("  %.4f | %s", s["embedding_score"], s["anchor_text"])

    assert len(scored) == 2
    assert scored[0]["embedding_score"] > scored[1]["embedding_score"]
    assert "cake" not in scored[0]["anchor_text"].lower()
    logger.info("Test embedding_scorer PASSED")


def test_pipeline_config():
    """Test frozen dataclass config."""
    from ResearchAgent.pipeline.config import ResearchPipelineConfig

    config = ResearchPipelineConfig(reranker_strategy="llm", top_links_after_rerank=3)
    assert config.reranker_strategy == "llm"
    assert config.top_links_after_rerank == 3
    try:
        config.reranker_strategy = "cross-encoder"
        assert False, "Should be frozen"
    except Exception:
        pass
    logger.info("Test pipeline_config PASSED")


def test_token_callback():
    """Test callback handler initializes correctly."""
    from ResearchAgent.utils.callbacks import TokenAccumulatorCallbackHandler

    cb = TokenAccumulatorCallbackHandler()
    assert cb.input_tokens == 0
    assert cb.output_tokens == 0
    logger.info("Test token_callback PASSED")


if __name__ == "__main__":
    test_link_extractor()
    test_pipeline_config()
    test_token_callback()
    test_embedding_scorer()
    logger.info("=== ALL TESTS PASSED ===")
