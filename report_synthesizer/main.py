from Columbus.llm.llm import get_langchain_llm
from Columbus.report_synthesizer.prompts import REPORT_WRITER_PROMPT
from langchain_core.output_parsers import StrOutputParser
from typing import List

async def synthesize_report(original_query: str,decomposed_queries: List[str],evidence: str) -> str:
    llm = get_langchain_llm()
    chain = REPORT_WRITER_PROMPT | llm | StrOutputParser()
    report_text = await chain.ainvoke(
        {
            "original_query": original_query,
            "decomposed_queries": decomposed_queries,
            "candidate_chunks": evidence
        }
    )

    return report_text