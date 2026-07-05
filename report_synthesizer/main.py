from Columbus.llm import get_langchain_llm
from Columbus.report_synthesizer.prompts import REPORT_WRITER_PROMPT
from typing import List

async def synthesize_report(original_query: str,decomposed_queries: List[str],evidence: str) -> str:
    llm = get_langchain_llm()
    chain = REPORT_WRITER_PROMPT | llm
    report = chain.invoke(
        {
            "original_query": original_query,
            "decomposed_queries": decomposed_queries,
            "candidate_chunks": evidence
        }
    )

    return report.content