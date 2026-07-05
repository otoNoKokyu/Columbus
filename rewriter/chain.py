"""LLM-powered query rewriter as an LCEL Runnable.

Generates up to N semantically distinct search queries optimized for
web search engines. Copied pattern from refactor/rewriter/chain.py.
"""

from ..types.types import RewriterInput, Query
import logging
from typing import List, Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda, Runnable

from .prompts import QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)


def create_rewrite_chain(
    llm: BaseChatModel,
    max_queries: int = 3,
) -> Runnable:
    """Build an LCEL chain that rewrites a query into multiple search queries.

    Input:  {"question": str}
    Output: List[str]  (rewritten queries, or [question] as fallback)

    The inner chain:
        prompt | llm | JsonOutputParser

    Wrapped in RunnableLambda for parsing + fallback handling.
    """
    # Inner LCEL chain: prompt -> LLM -> parsed JSON
    _inner = QUERY_REWRITE_PROMPT | llm | JsonOutputParser()

    def _rewrite(inputs: Dict[str, Any]) -> List[str]:
        question = inputs.get("question") or inputs.get("original_query") or ""
        fallback = [question]
        try:
            parsed = _inner.invoke(
                {"question": question, "max_queries": max_queries},
                config={"run_name": "Columbus:QueryRewrite"},
            )
            if isinstance(parsed, dict):
                queries = parsed.get("queries", [])
            elif isinstance(parsed, list):
                queries = parsed
            else:
                queries = []
            queries = [q.strip() for q in queries if q and q.strip()][:max_queries]
            if not queries:
                logger.warning("Rewriter returned empty — using fallback")
                return fallback
            logger.info("Rewrote into %d queries: %s", len(queries), queries)
            return queries
        except Exception as e:
            logger.error("Query rewrite failed: %s", e)
            return fallback

    return RunnableLambda(_rewrite).with_config({"run_name": "QueryRewriterNode"})


def create_balanced_rewrite_chain(
    llm: BaseChatModel,
) -> Runnable[RewriterInput, Query]:
    """Build an LCEL chain that rewrites a query into exactly two balanced queries.

    One supporting/mimicking the query premise, and one opposing.
    """
    from .prompts import BALANCED_REWRITE_PROMPT
    from typing import cast

    structured_llm = llm.with_structured_output(Query)
    return cast(
        Runnable[RewriterInput, Query],
        (
            RunnableLambda(lambda x: {"query": x.query})
            | BALANCED_REWRITE_PROMPT
            | structured_llm
        ).with_config({"run_name": "BalancedQueryRewriterNode"})
    )

def create_adaptive_decomposition_chain(
    llm: BaseChatModel,
    query_budget: int = 6,
) -> Runnable:
    from .prompts import (
        QUERY_CLASSIFIER_PROMPT,
        RESEARCH_PLANNER_PROMPT,
        BUDGET_QUERY_GENERATOR_PROMPT,
    )
    from ..types.types import DecomposedQuery, SearchQuery

    parser = JsonOutputParser()

    def _adaptive_decompose(inputs: Dict[str, Any]) -> DecomposedQuery:
        query = inputs["query"]
        critic_feedback = inputs.get("critic_feedback")
        callbacks = inputs.get("callbacks")
        config = {"callbacks": callbacks} if callbacks else {}

        if critic_feedback:
            from .prompts import CRITIC_DECOMPOSITION_PROMPT
            critic_chain = CRITIC_DECOMPOSITION_PROMPT | llm | parser
            try:
                if not isinstance(critic_feedback, str):
                    import json
                    critic_feedback_str = json.dumps(critic_feedback, indent=2, ensure_ascii=False)
                else:
                    critic_feedback_str = critic_feedback

                res = critic_chain.invoke({
                    "query": query,
                    "critic_feedback": critic_feedback_str,
                    "budget": query_budget
                }, config=config)

                objectives_raw = res.get("objectives", ["Address gaps in retrieved evidence"])
                raw_queries = res.get("search_queries", [])

                objectives = []
                for obj in objectives_raw:
                    if isinstance(obj, dict):
                        objectives.append(obj.get("objective") or obj.get("text") or "Address gaps in retrieved evidence")
                    else:
                        objectives.append(str(obj))

                search_queries = []
                for q_idx, q in enumerate(raw_queries):
                    idx = int(q.get("objective_index", 0))
                    if idx >= len(objectives):
                        idx = 0
                    search_queries.append(SearchQuery(
                        query=q.get("query", ""),
                        objective_id=f"critic_obj_{idx}_q_{q_idx}",
                        objective_text=f"{objectives[idx]}",
                        intent=q.get("intent", "critical")
                    ))

                if not search_queries:
                    search_queries = [
                        SearchQuery(
                            query=query,
                            objective_id="critic_obj_0_q_0",
                            objective_text=objectives[0]
                        )
                    ]

                return DecomposedQuery(
                    original_query=query,
                    query_type="multi",
                    objectives=objectives,
                    search_queries=search_queries
                )
            except Exception as e:
                logger.error("Critic-aware decomposition failed: %s", e)
                # Fallback to normal flow below

        # 1. Classify
        classifier_chain = QUERY_CLASSIFIER_PROMPT | llm | parser
        try:
            classification = classifier_chain.invoke({"query": query}, config=config)
            q_type = classification.get("type", "single")
        except Exception as e:
            logger.error("Classification failed: %s", e)
            q_type = "single"

        # 2. Handle based on type
        if q_type == "single":
            # Just generate simple queries
            rewrite_chain = create_rewrite_chain(llm, max_queries=max(3, min(5, query_budget)))
            queries = rewrite_chain.invoke({"question": query}, config=config)
            
            search_queries = [
                SearchQuery(
                    query=q, 
                    objective_id=f"single_{i}", 
                    objective_text=f"Direct query search - Angle {i+1}",
                    intent="foundational"
                ) for i, q in enumerate(queries)
            ]
            
            return DecomposedQuery(
                original_query=query,
                query_type="single",
                objectives=[f"Direct query search - Angle {i+1}" for i in range(len(queries))],
                search_queries=search_queries
            )
        else:
            # Multi-hop: Plan objectives
            planner_chain = RESEARCH_PLANNER_PROMPT | llm | parser
            try:
                plan = planner_chain.invoke({"query": query}, config=config)
                objectives = plan.get("objectives", [query])
            except Exception as e:
                logger.error("Planning failed: %s", e)
                objectives = [query]
                
            # Generate budget-aware queries
            generator_chain = BUDGET_QUERY_GENERATOR_PROMPT | llm | parser
            obj_list_str = "\\n".join(f"{i}. {obj}" for i, obj in enumerate(objectives))
            
            try:
                allocations_res = generator_chain.invoke({
                    "original_query": query,
                    "objectives_list": obj_list_str,
                    "budget": query_budget
                }, config=config)
                
                allocs = allocations_res.get("allocations", [])
                
                search_queries = []
                for q_idx, a in enumerate(allocs):
                    idx = int(a.get("objective_index", 0))
                    if idx >= len(objectives):
                        idx = 0
                    
                    search_queries.append(SearchQuery(
                        query=a.get("query", ""),
                        objective_id=f"obj_{idx}_q_{q_idx}",
                        objective_text=f"{objectives[idx]} - Variant {q_idx + 1}",
                        intent=a.get("intent", "")
                    ))
                    
                # Fallback if empty
                if not search_queries:
                    search_queries = [SearchQuery(query=query, objective_id="obj_0_q_0", objective_text=objectives[0])]
                    
                return DecomposedQuery(
                    original_query=query,
                    query_type="multi",
                    objectives=objectives,
                    search_queries=search_queries
                )
            except Exception as e:
                logger.error("Budget query generation failed: %s", e)
                # Fallback
                return DecomposedQuery(
                    original_query=query,
                    query_type="multi",
                    objectives=objectives,
                    search_queries=[SearchQuery(query=query, objective_id="obj_0_q_0", objective_text=objectives[0])]
                )

    return RunnableLambda(_adaptive_decompose).with_config({"run_name": "AdaptiveDecompositionNode"})
