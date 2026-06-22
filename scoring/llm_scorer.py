"""LLM-based relevance scorer for intelligent link re-ranking."""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field

from .base import BaseScorer
from ..llm.llm import get_langchain_llm

logger = logging.getLogger(__name__)

from urllib.parse import urlparse

class ScoredLink(BaseModel):
    id: int = Field(description="The ID of the candidate link")
    score: float = Field(description="Relevance score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief reasoning for why this link is relevant")

class ScoredLinkList(BaseModel):
    links: List[ScoredLink] = Field(description="List of scored links")

class LLMScorer(BaseScorer):
    """Semantic relevance scorer using an LLM."""

    def __init__(self, provider: Optional[str] = None):
        self._provider = provider
        self._llm = get_langchain_llm(temperature=0.0, provider=provider)
        self._structured_llm = self._llm.with_structured_output(ScoredLinkList)
        logger.info("LLMScorer initialized with provider: %s", provider)

    async def score(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 20,
    ) -> List[Dict[str, Any]]:
        """Score candidates against query using an LLM."""
        if not candidates:
            return []

        candidates = candidates[:100]

        # Optimize token usage by using IDs and stripping domains for internal links
        candidates_for_llm = []
        for i, c in enumerate(candidates):
            parsed = urlparse(c["url"])
            source_parsed = urlparse(c.get("source_url", ""))
            
            # If it's an internal link (same domain), just send the path
            if source_parsed.netloc and parsed.netloc == source_parsed.netloc:
                display_url = parsed.path
                if parsed.query:
                    display_url += "?" + parsed.query
                if not display_url:
                    display_url = "/"
            else:
                display_url = c["url"]
                
            candidates_for_llm.append({"id": i, "link": display_url})

        candidates_json = json.dumps(candidates_for_llm, indent=2)

        prompt = f"""You are an expert research assistant.
Your task is to evaluate a list of candidate links found on a webpage and determine how relevant they are to the user's research query.

User's Research Query: {query}

Candidate Links:
{candidates_json}

Evaluate each link and assign a relevance score from 0.0 to 1.0, where 1.0 is highly relevant and 0.0 is completely irrelevant.
Only return links that have a score greater than 0.0.
"""
        
        def _compute():
            try:
                response = self._structured_llm.invoke(prompt)
                
                # Map scores back to candidates using the returned IDs
                score_map = {link.id: link.score for link in response.links}
                reason_map = {link.id: link.reasoning for link in response.links}
                
                for i, c in enumerate(candidates):
                    # We use "embedding_score" key for compatibility with existing code
                    c["embedding_score"] = score_map.get(i, 0.0)
                    c["reasoning"] = reason_map.get(i, "No reasoning provided.")
                    
                ranked = sorted(
                    candidates,
                    key=lambda x: x["embedding_score"],
                    reverse=True,
                )
                
                top_ranked = ranked[:top_n]
                logger.info(
                    "LLM scoring complete: evaluated %d candidates. Returning top %d links:",
                    len(candidates),
                    len(top_ranked)
                )
                for i, link in enumerate(top_ranked):
                    logger.info("  #%d [%.2f] %s\n      Reason: %s", i+1, link["embedding_score"], link["url"], link["reasoning"])
                
                return top_ranked
            except Exception as e:
                logger.error("LLMScorer failed: %s", e)
                # Fallback: return everything with 0 score
                for c in candidates:
                    c["embedding_score"] = 0.0
                return candidates[:top_n]

        return await asyncio.to_thread(_compute)
