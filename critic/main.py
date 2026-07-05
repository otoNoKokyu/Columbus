from typing import Dict, Any, List
from langchain_core.output_parsers import JsonOutputParser
from Columbus.llm import get_langchain_llm
from .prompts import RETRIEVAL_CRITIC_PROMPT

def critic_gathered_eveidence(
    chunks: List[Dict[str, Any]], original_query: str, decomposed_queries: Any, callbacks: Any = None
) -> Dict[str, Any]:
    llm = get_langchain_llm(temperature=0.0, provider='bedrock', region='us-east-1', model='us.meta.llama4-maverick-17b-instruct-v1:0')
    chain = RETRIEVAL_CRITIC_PROMPT | llm | JsonOutputParser()

    candidate_chunks_str = "\n\n".join(
        f"chunk_id: {i+1}\n"
        f"source: {c.get('url') or c.get('source') or 'Unknown'}"
        + (f"\ntitle: {c['title']}" if c.get('title') else "")
        + f"\ncontent: {c.get('text') or c.get('content') or ''}"
        for i, c in enumerate(chunks)
    )

    if not isinstance(decomposed_queries, str):
        objs = getattr(decomposed_queries, "objectives", decomposed_queries)
        decomposed_queries = "\n".join(f"- {o}" for o in (objs if isinstance(objs, list) else [objs]))

    return chain.invoke({
        "original_query": original_query,
        "decomposed_queries": decomposed_queries,
        "candidate_chunks": candidate_chunks_str
    }, config={"callbacks": callbacks} if callbacks else {})