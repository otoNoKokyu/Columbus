from Columbus.agents.tools import gather_evidence
import json
from .tools import decompose_query, search_crawl_rerank_queries
from Columbus.utils.logger import logger
from langgraph.graph import StateGraph, END
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, model_validator
from ..types.types import DecomposedQuery
from Columbus.report_synthesizer.main import synthesize_report

class QueryState(BaseModel):
    user_input: str
    decmoposed_queries: Optional[DecomposedQuery] = None
    query_budget: int = 6
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
    current_turn: int = 1
    max_turns: int = 2
    critic_evaluation: Optional[Dict[str, Any]] = None
    report: Optional[str] = None

    @model_validator(mode="after")
    def validate_depth(self):
        if self.recursive_crawl and (self.max_depth is None or self.max_depth < 1):
            raise ValueError("max_depth must be at least 1 when recursive_crawl is True")
        return self

def DECOMPOSE(state: QueryState):
    logger.info(f"--- [NODE] DECOMPOSE (Turn {state.current_turn}) ---")
    logger.info(f"Decomposing query: '{state.user_input}'")
    feedback_str = None
    if state.critic_evaluation:
        eval_data = state.critic_evaluation
        
        # 1. covered_topics (with coverage scores as weightages)
        by_topic = eval_data.get("coverage", {}).get("by_topic", [])
        covered_topics = [
            {
                "topic": item.get("topic"),
                "coverage_percentage": item.get("coverage", 0)
            }
            for item in by_topic
        ]
        
        # 2. missing_information (with priority weightages)
        missing_info = [
            {
                "topic": item.get("topic"),
                "reason": item.get("reason"),
                "priority": item.get("priority", 3)  # default to middle priority if not set
            }
            for item in eval_data.get("missing_information", [])
        ]
        
        # 3. conflicting_evidence (with priority weightages)
        conflicting_ev = [
            {
                "topic": item.get("topic"),
                "description": item.get("description"),
                "priority": item.get("priority", 3)
            }
            for item in eval_data.get("conflicting_evidence", [])
        ]
        
        formatted_feedback = {
            "covered_topics": covered_topics,
            "missing_information": missing_info,
            "conflicting_evidence": conflicting_ev
        }
        feedback_str = json.dumps(formatted_feedback, indent=2, ensure_ascii=False)
        
    queries = decompose_query(
        state.user_input, 
        query_budget=state.query_budget,
        critic_feedback=feedback_str
    )
    
    # Increment turn count if this is a follow-up decomposition pass
    new_turn = state.current_turn + 1 if state.decmoposed_queries is not None else state.current_turn
    
    return {
        "decmoposed_queries": queries,
        "current_turn": new_turn
    }

async def RESEARCH_AGENT(state: QueryState):
    logger.info(f"--- [NODE] RESEARCH_AGENT (Turn {state.current_turn}) ---")
    if not state.decmoposed_queries:
        logger.error("Queries were not decomposed.")
        return {"evidence": "Error: Queries were not decomposed."}
        
    results = await search_crawl_rerank_queries(
        decomposed=state.decmoposed_queries,
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
        visited_urls=state.visited_urls, # Filter out already visited URLs
    )
    
    # 1. Accumulate visited URLs
    visited = list(state.visited_urls)
    for perspective in results.perspectives:
        for page in perspective.pages_crawled:
            if page.url and page.url not in visited:
                visited.append(page.url)

    # 2. Merge retrieved chunks across turns
    new_chunks = dict(state.retrieved_chunks or {})
    if results.retrieved_chunks:
        for obj_id, chunk_list in results.retrieved_chunks.items():
            new_chunks[obj_id] = chunk_list

    # 3. Parse and merge evidence structure
    existing_evidence = {}
    if state.evidence:
        try:
            existing_evidence = json.loads(state.evidence)
        except Exception:
            pass

    # Merge new perspective page content
    for perspective in results.perspectives:
        obj_id = perspective.objective_id
        if obj_id not in existing_evidence:
            existing_evidence[obj_id] = []
        for page in perspective.pages_crawled:
            if page.content and page.content not in existing_evidence[obj_id]:
                existing_evidence[obj_id].append(page.content)

    # Store aggregated chunks inside the evidence block
    existing_evidence["retrieved_chunks"] = new_chunks
    
    logger.info(f"Research turn {state.current_turn} complete. Total visited URLs: {len(visited)}")
    return {
        "visited_urls": visited,
        "evidence": json.dumps(existing_evidence, indent=4, ensure_ascii=False),
        "retrieved_chunks": new_chunks
    }

async def SYNTHESIZER_AGENT(state: QueryState):
    logger.info("--- [NODE] SYNTHESIZER_AGENT ---")
    logger.info("Gathering evidence and synthesizing final report.")
    # Extract query strings from DecomposedQuery object
    decomposed_list = []
    if state.decmoposed_queries:
        if isinstance(state.decmoposed_queries, list):
            decomposed_list = state.decmoposed_queries
        elif hasattr(state.decmoposed_queries, "objectives"):
            decomposed_list = state.decmoposed_queries.objectives
        elif hasattr(state.decmoposed_queries, "get_flat_query_strings"):
            decomposed_list = state.decmoposed_queries.get_flat_query_strings()
            
    # Format evidence chunks using gather_evidence
    evidence_str = ""
    chunks = state.retrieved_chunks
    if not chunks and state.evidence:
        try:
            ev_data = json.loads(state.evidence)
            chunks = ev_data.get("retrieved_chunks")
        except Exception:
            pass
            
    if chunks:
        evidence_str = gather_evidence(chunks)
    
    # If formatting chunks didn't yield anything but state.evidence is a raw string, fallback to it
    if not evidence_str and state.evidence:
        evidence_str = state.evidence

    report = await synthesize_report(
        original_query=state.user_input,
        decomposed_queries=decomposed_list,
        evidence=evidence_str
    )
    
    # Save the report to disk
    import os, time
    os.makedirs("reports", exist_ok=True)
    filename = f"reports/report_{int(time.time())}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved to {filename}")
    
    return {
        "report": report
    }

def CRITIC(state: QueryState):
    logger.info(f"--- [NODE] CRITIC (Turn {state.current_turn}) ---")
    from Columbus.critic.main import critic_gathered_eveidence
    
    all_chunks = []
    if state.retrieved_chunks:
        for chunk_list in state.retrieved_chunks.values():
            all_chunks.extend(chunk_list)
            
    critic_evaluation = critic_gathered_eveidence(
        chunks=all_chunks,
        original_query=state.user_input,
        decomposed_queries=state.decmoposed_queries
    )
    
    return {
        "critic_evaluation": critic_evaluation
    }

def should_continue(state: QueryState) -> str:
    if state.current_turn >= state.max_turns:
        logger.info(f"Max turns ({state.max_turns}) reached. Proceeding to SYNTHESIZER.")
        return "SYNTHESIZER"
        
    eval_data = state.critic_evaluation or {}
    coverage = eval_data.get("coverage", {})
    overall_coverage = coverage.get("overall", 0)
    missing_info = eval_data.get("missing_information", [])
    
    # Sufficiency check: exit if overall coverage >= 80 and no gaps remain
    if overall_coverage >= 80 and not missing_info:
        logger.info(f"Sufficiency check passed (Coverage: {overall_coverage}%). Proceeding to SYNTHESIZER.")
        return "SYNTHESIZER"
        
    logger.info(f"Sufficiency check failed (Coverage: {overall_coverage}%, Gaps: {len(missing_info)}). Routing back to DECOMPOSE.")
    return "DECOMPOSE"

builder = StateGraph(QueryState)
builder.add_node("DECOMPOSE", DECOMPOSE)
builder.add_node("RESEARCH", RESEARCH_AGENT)
builder.add_node("CRITIC", CRITIC)
builder.add_node("SYNTHESIZER", SYNTHESIZER_AGENT)

builder.set_entry_point("DECOMPOSE")
builder.add_edge("DECOMPOSE", "RESEARCH")
builder.add_edge("RESEARCH", "CRITIC")
builder.add_conditional_edges(
    "CRITIC",
    should_continue,
    {
        "DECOMPOSE": "DECOMPOSE",
        "SYNTHESIZER": "SYNTHESIZER"
    }
)
builder.add_edge("SYNTHESIZER", END)

graph = builder.compile()
