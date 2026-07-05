from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Query(BaseModel):
    # Deprecated: Kept for backward compatibility
    supporting: str
    opposing: str
    neutral_analysis: str
    expert_criticism: str
    real_world_evidence: str

class SearchQuery(BaseModel):
    """A single search query with metadata."""
    query: str
    objective_id: str
    objective_text: str
    intent: Optional[str] = None

class DecomposedQuery(BaseModel):
    """Result of adaptive query decomposition."""
    original_query: str
    query_type: str            # "single" or "multi"
    objectives: List[str]      # The research objectives
    search_queries: List[SearchQuery]
    
    def get_queries_by_objective(self) -> Dict[str, List[SearchQuery]]:
        from collections import defaultdict
        groups = defaultdict(list)
        for sq in self.search_queries:
            groups[sq.objective_id].append(sq)
        return dict(groups)
    
    def get_flat_query_strings(self) -> List[str]:
        return [sq.query for sq in self.search_queries]

class SearchResult(BaseModel):
    url: str
    title:str
    snippet:str
    score:float

class CrawlResult(BaseModel):
    url: str
    title: str
    depth: int
    score: float
    content: str
    token_count: Optional[int] = None
    
class PerspectiveResult(BaseModel):
    objective_id: str
    objective_text: str
    rewritten_query: str
    pages_crawled: List[CrawlResult]

class ResearchOutput(BaseModel):
    original_query: str
    decomposed: Optional[DecomposedQuery] = None
    perspectives: List[PerspectiveResult]
    retrieved_chunks: Optional[Dict[str, List[Dict[str, Any]]]] = None


class RewriterInput(BaseModel):
    query: str = Field(..., description="The query to be rewritten")
