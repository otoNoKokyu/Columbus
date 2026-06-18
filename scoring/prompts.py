"""Relevance scoring prompt for web research links.

Copied pattern from refactor/rerank/prompts.py, adapted for web links.
"""

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# RELEVANCE SCORING — evaluate a link's relevance to a query (0-10)
# ---------------------------------------------------------------------------

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a **relevance judge** inside a web research pipeline.\n\n"
            "Your task is to evaluate how relevant a discovered web link and "
            "its surrounding context are to the user's research query and "
            "assign a numerical score.\n\n"
            "## Scoring Guidelines\n"
            "- **9–10**: The link directly and completely addresses the query "
            "or points to the exact information requested.\n"
            "- **7–8**: The link is highly relevant — it addresses the core "
            "topic and provides substantial supporting detail.\n"
            "- **5–6**: The link is moderately relevant — it touches on the "
            "topic but lacks specificity or only partially answers the query.\n"
            "- **3–4**: The link is tangentially relevant — related domain "
            "but different focus.\n"
            "- **1–2**: The link is barely relevant — only shares surface-"
            "level keywords with the query.\n"
            "- **0**: The link is completely irrelevant to the query.\n\n"
            "## Rules\n"
            "1. Score based on **semantic relevance**, not just keyword "
            "overlap.\n"
            "2. Consider URL authority (official docs, academic sources > "
            "random blogs).\n"
            "3. Consider information density suggested by the context "
            "snippet.\n"
            "4. Give higher scores to links containing exact entities, "
            "technical terms, or project names mentioned in the query.\n"
            "5. Provide a brief reasoning (1–2 sentences) justifying the "
            "score.\n"
            "6. Be consistent — similar content should receive similar "
            "scores across evaluations.",
        ),
        (
            "human",
            "## Query\n{query}\n\n"
            "## Link\nURL: {url}\nAnchor Text: {anchor_text}\n"
            "Context: {context}\n\n"
            "---\n"
            "Evaluate the relevance of this link to the query. "
            "Provide your score (0–10) and a brief reasoning.",
        ),
    ]
)


def get_prompt() -> ChatPromptTemplate:
    """Return the relevance scoring ChatPromptTemplate.

    Returns:
        A ready-to-use ChatPromptTemplate with input vars
        {query, url, anchor_text, context}.
    """
    return RELEVANCE_PROMPT
