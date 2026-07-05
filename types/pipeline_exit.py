from enum import Enum
from typing import Any


class EarlyExitReason(Enum):
    """Enum of all the reasons the Columbus research pipeline can exit early.

    Each member name maps to a specific filtering checkpoint in the pipeline.
    The associated value is a human-readable label used in logs and error messages.
    """

    # ── Search stage ────────────────────────────────────────────────────────
    NO_SEARCH_RESULTS = "no_search_results"
    """Exa fan-out search returned zero results across all sub-queries."""

    NO_URLS_AFTER_VISITED_FILTER = "no_urls_after_visited_filter"
    """All search-result URLs were already in the visited set."""

    # ── Rerank stage ────────────────────────────────────────────────────────
    NO_URLS_AFTER_RERANK = "no_urls_after_rerank"
    """Global reranker returned zero candidates (input pool was non-empty)."""

    NO_URLS_AFTER_SCORE_THRESHOLD = "no_urls_after_score_threshold"
    """Reranker returned candidates but all had rerank_score below the score_threshold.
    Filtered out before entering the crawl phase."""

    # ── Crawl stage ─────────────────────────────────────────────────────────
    NO_PAGES_CRAWLED = "no_pages_crawled"
    """Frontier crawl scraped zero pages (all seed URLs failed or were unwanted)."""

    NO_CONTENT_AFTER_CRAWL = "no_content_after_crawl"
    """Every scraped page had an empty URL or empty content; nothing to chunk."""

    # ── Chunk stage ─────────────────────────────────────────────────────────
    NO_CHUNKS_GENERATED = "no_chunks_generated"
    """Chunker produced zero chunks from all page content (all content too short or empty)."""

    NO_CHUNKS_AFTER_DEDUP = "no_chunks_after_dedup"
    """All chunks were filtered out as near-duplicates (cosine >= 0.95); nothing unique remains."""

    # ── Retrieval stage ─────────────────────────────────────────────────────
    NO_CHUNKS_RETRIEVED = "no_chunks_retrieved"
    """MMR retrieval returned zero chunks for every objective."""


class PipelineEarlyExit(Exception):
    """Raised when the Columbus pipeline cannot proceed past a filtering checkpoint.

    Attributes:
        reason  (EarlyExitReason): The enum member identifying *why* we exited.
        value   (Any):             The triggering value at the checkpoint
                                   (e.g. the count of URLs that survived, the threshold used).
        message (str):             A human-readable description automatically derived from
                                   ``reason`` and ``value`` if not supplied explicitly.

    Example::

        raise PipelineEarlyExit(
            reason=EarlyExitReason.NO_URLS_AFTER_RERANK,
            value=len(search_results),     # how many went IN to the reranker
            message="Reranker returned 0 results from a pool of 47 candidates."
        )
    """

    def __init__(self, reason: EarlyExitReason, value: Any, message: str = "") -> None:
        self.reason = reason
        self.value = value
        self.message = message or (
            f"[PipelineEarlyExit] Reason={reason.value!r}  TriggeringValue={value!r}"
        )
        super().__init__(self.message)

    def __repr__(self) -> str:
        return (
            f"PipelineEarlyExit(reason={self.reason!r}, value={self.value!r})"
        )
