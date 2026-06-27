import json
from .tools import decompose_query, search_crawl_rerank_queries
from langgraph.graph import StateGraph, END
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, model_validator
from ..types.types import Query

class QueryState(BaseModel):
    user_input: str
    decmoposed_queries: Optional[Query] = None
    visited_urls: List[str] = []
    evidence: Optional[str] = None
    retrieved_chunks: Optional[Dict[str, List[Dict[str, Any]]]] = None
    max_rerank: int = 10
    max_search: int = 10
    recursive_crawl: bool = False
    max_depth: Optional[int] = None
    chunking_strategy: str = "semantic"
    embedding_source: str = "local"
    min_chunk_size: int = 500
    chunk_overlap: int = 50
    exa_highlight: bool = False

    @model_validator(mode="after")
    def validate_depth(self):
        if self.recursive_crawl and (self.max_depth is None or self.max_depth < 1):
            raise ValueError("max_depth must be at least 1 when recursive_crawl is True")
        return self

def DECOMPOSE(state: QueryState):
    queries = decompose_query(state.user_input)
    return {"decmoposed_queries": queries}

async def RESEARCH_AGENT(state: QueryState):
    if not state.decmoposed_queries:
        return {"evidence": "Error: Queries were not decomposed."}
    results = await search_crawl_rerank_queries(
        query=state.decmoposed_queries,
        original_user_input=state.user_input,
        max_search=state.max_search,
        max_rerank=state.max_rerank,
        recursive_crawl=state.recursive_crawl,
        max_depth=state.max_depth,
        chunking_strategy=state.chunking_strategy,
        min_chunk_size=state.min_chunk_size,
        chunk_overlap=state.chunk_overlap,
        exa_highlight=state.exa_highlight,
        embedding_source=state.embedding_source,
    )
    visited = []
    evidence_data = {
        "supporting": [],
        "opposing": [],
        "retrieved_chunks": results.retrieved_chunks
    }

    for perspective in results.perspectives:
        for page in perspective.pages_crawled:
            visited.append(page.url)
            if perspective.bias_type == "supporting":
                evidence_data["supporting"].append(page.content)
            elif perspective.bias_type == "opposing":
                evidence_data["opposing"].append(page.content)

    return {
        "visited_urls": visited,
        "evidence": json.dumps(evidence_data, indent=4),
        "retrieved_chunks": results.retrieved_chunks
    }
        
    

builder = StateGraph(QueryState)
builder.add_node("DECOMPOSE", DECOMPOSE)
builder.add_node("RESEARCH", RESEARCH_AGENT)

builder.set_entry_point("DECOMPOSE")
builder.add_edge("DECOMPOSE", "RESEARCH")
builder.add_edge("RESEARCH", END)

graph = builder.compile()
