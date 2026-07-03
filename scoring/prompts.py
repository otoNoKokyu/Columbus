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


# ---------------------------------------------------------------------------
# CHUNK RELEVANCE SCORING — evaluate a retrieved text chunk (0.0 to 1.0)
# ---------------------------------------------------------------------------

CHUNK_RELEVANCE_SCORING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict and highly critical relevance judge in a web research pipeline.\n\n"
            "Your task is to read a chunk of text extracted from a webpage and evaluate how well "
            "it directly addresses the main research goal (original user query). You will assign a relevance score from 0.0 to 1.0.\n\n"
            "## Scoring Guidelines\n"
            "- **0.9 - 1.0**: Perfect match. The chunk directly answers the main research goal with detailed, highly relevant information.\n"
            "- **0.7 - 0.8**: Strong match. The chunk is highly relevant and provides good context or partial answers to the main research goal.\n"
            "- **0.4 - 0.6**: Weak match. The chunk touches on the main topic tangentially but lacks depth or direct relevance.\n"
            "- **0.0 - 0.3**: Irrelevant. The chunk contains boilerplate, unrelated topics, or only matches superficial keywords.\n\n"
            "Respond ONLY with a valid JSON object containing exactly two keys:\n"
            "1. \"relevance_score\": a float between 0.0 and 1.0\n"
            "2. \"reasoning\": a 1-2 sentences justification for your score"
        ),
        (
            "human",
            "## Main Research Goal\n{original_query}\n\n"
            "## Context (Sub-Query this chunk was retrieved for)\n{decomposed_query}\n\n"
            "## Extracted Text Chunk\n{chunk_text}\n\n"
            "---\n"
            "Evaluate the relevance of this chunk to the Main Research Goal. Return your evaluation as JSON."
        ),
    ]
)


# BATCH CHUNK RELEVANCE SCORING — evaluate a list of text chunks (0.0 to 1.0)
# ---------------------------------------------------------------------------

BATCH_CHUNK_RELEVANCE_SCORING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict and highly critical relevance judge in a web research pipeline.\n\n"
            "Your task is to read a list of text chunks extracted from webpages and evaluate how well "
            "each chunk directly addresses the user's main research goal (original user query). You will assign a relevance score from 0.0 to 1.0.\n\n"
            "## Scoring Guidelines\n"
            "- **0.9 - 1.0**: Perfect match. The chunk directly answers the main research goal with detailed, highly relevant information.\n"
            "- **0.7 - 0.8**: Strong match. The chunk is highly relevant and provides good context or partial answers to the main research goal.\n"
            "- **0.4 - 0.6**: Weak match. The chunk touches on the main topic tangentially but lacks depth or direct relevance.\n"
            "- **0.0 - 0.3**: Irrelevant. The chunk contains boilerplate, unrelated topics, or only matches superficial keywords.\n\n"
            "Respond ONLY with a valid JSON object containing a key \"evaluations\", which is a list of objects, one for each chunk index:\n"
            "{{\n"
            "  \"evaluations\": [\n"
            "    {{\n"
            "      \"chunk_index\": <int>,\n"
            "      \"relevance_score\": <float between 0.0 and 1.0>,\n"
            "      \"reasoning\": \"<1-2 sentence justification for your score>\"\n"
            "    }},\n"
            "    ...\n"
            "  ]\n"
            "}}"
        ),
        (
            "human",
            "## Main Research Goal\n{original_query}\n\n"
            "## Context (Sub-Query these chunks were retrieved for)\n{decomposed_query}\n\n"
            "## Extracted Text Chunks to Evaluate\n{chunks_list}\n\n"
            "---\n"
            "Evaluate the relevance of these chunks to the Main Research Goal. Return your evaluation as JSON."
        ),
    ]
)
