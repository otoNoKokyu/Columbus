import json
from .tools import decompose_query, search_crawl_rerank_queries
from langgraph.graph import StateGraph, END
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from ..types.types import Query

class QueryState(BaseModel):
    user_input: str
    decmoposed_queries: Optional[Query] = None
    visited_urls: List[str] = []
    evidence: Optional[str] = None
    retrieved_chunks: Optional[Dict[str, List[Dict[str, Any]]]] = None
    max_rerank: int = 10
    max_depth: int = 2
    max_search: int = 10

def DECOMPOSE(state: QueryState):
    queries = decompose_query(state.user_input)
    return {"decmoposed_queries": queries}

async def RESEARCH_AGENT(state: QueryState):
    if not state.decmoposed_queries:
        return {"evidence": "Error: Queries were not decomposed."}
    results = await search_crawl_rerank_queries(state.decmoposed_queries, state.user_input)
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
