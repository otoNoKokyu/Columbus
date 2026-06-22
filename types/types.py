from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Query(BaseModel):
    supporting: str
    opposing: str
    neutral_analysis: str
    expert_criticism: str
    real_world_evidence: str

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
    
class PerspectiveResult(BaseModel):
    bias_type: str
    rewritten_query: str
    pages_crawled: List[CrawlResult]

class ResearchOutput(BaseModel):
    original_query: str
    perspectives: List[PerspectiveResult]
    retrieved_chunks: Optional[Dict[str, List[Dict[str, Any]]]] = None


class RewriterInput(BaseModel):
    query: str = Field(..., description="The query to be rewritten")
    

