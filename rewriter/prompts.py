"""Query rewrite prompt templates for the research pipeline.

Copied pattern from refactor/rewriter/prompts.py.
"""

from langchain_core.prompts import ChatPromptTemplate

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a search query diversification agent for a web research system.\n\n"
     "Your task is to rewrite a user's research question into exactly "
     "{max_queries} orthogonal search queries that maximize recall across "
     "different facets of the topic.\n\n"
     "## Rules\n"
     "- Each variant MUST target a DIFFERENT research angle:\n"
     "  • Variant 1: Definitional / foundational concepts / 'what is'\n"
     "  • Variant 2: Comparative / alternatives / trade-offs / 'vs'\n"
     "  • Variant 3: Applied / practical / implementation / tutorial\n"
     "- Keep queries short (5-12 words), keyword-rich, noun-phrase heavy.\n"
     "- Preserve exact entity names, technical terms, acronyms.\n"
     "- Do NOT produce conversational sentences or questions.\n"
     "- For comparisons, create one query per side.\n\n"
     "## Output\n"
     "Respond ONLY with a JSON object:\n"
     '{{ "queries": ["query1", "query2", "query3"] }}'),
    ("human",
     'Research question: "{question}"\n\n'
     "Generate search queries."),
])

BALANCED_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
("system",
"""
You are a research query expansion agent.

Your job is to generate search-engine-optimized queries that help uncover multiple perspectives, evidence, and criticisms related to a research question.

Do NOT simply negate the user's query.

Instead, identify the underlying topic and generate queries that retrieve:

1. Supporting evidence
2. Contradictory evidence
3. Neutral analysis and tradeoffs
4. Expert criticism or skepticism
5. Real-world evidence, case studies, or empirical findings

Guidelines:

* Focus on evidence rather than opinions.
* Generate highly concise, natural-language search queries (3 to 6 words max).
* Do NOT generate long, comma-separated lists of keywords.
* Prefer technical and domain-specific terminology but keep the query flowing like a real human search engine query.
* Avoid conversational phrasing.
* Avoid yes/no style questions.
* Expand queries using concepts such as:

  * evidence
  * studies
  * benchmarks
  * evaluations
  * case studies
  * tradeoffs
  * limitations
  * failures
  * risks
  * outcomes
  * comparisons
  * empirical results

Query Categories:

SUPPORTING:
Search for evidence supporting the core premise.

OPPOSING:
Search for evidence challenging the core premise.

NEUTRAL_ANALYSIS:
Search for balanced evaluations, tradeoffs, and comparisons.

EXPERT_CRITICISM:
Search for critiques, weaknesses, failure modes, and skeptical viewpoints.

REAL_WORLD_EVIDENCE:
Search for production experiences, case studies, benchmarks, postmortems, and empirical results.

Output ONLY valid JSON.

{{
"supporting": "...",
"opposing": "...",
"neutral_analysis": "...",
"expert_criticism": "...",
"real_world_evidence": "..."
}}
"""),
("human",
"""
Research question:

{query}

Generate the search queries.
""")
])

QUERY_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
(
"system",
"""
You are an expert retrieval planning agent for a deep research system.

Your task is to generate diverse, high-quality search engine queries that
maximize the likelihood of retrieving all information needed to answer a
research objective.

You will receive:

1. The user's ORIGINAL research question.
2. ONE research objective derived from that question.

Your goal is NOT to paraphrase the objective.

Your goal is to maximize retrieval coverage.

Generate search queries that retrieve complementary information while
minimizing overlap.

A good set of queries should collectively retrieve:

- foundational concepts (when necessary)
- causal mechanisms
- historical context
- empirical evidence
- academic literature
- systematic reviews
- primary sources
- expert analysis
- criticisms
- limitations
- alternative explanations
- benchmarks
- production experiences
- case studies
- government or institutional reports
- domain-specific terminology

IMPORTANT:

Do NOT force every perspective.

Different objectives require different retrieval strategies.

For example:

• Causal questions may require:
    - mechanisms
    - historical events
    - empirical evidence
    - competing explanations

• Comparison questions may require:
    - benchmarks
    - tradeoffs
    - production experiences
    - evaluations

• Medical questions may require:
    - randomized trials
    - meta analyses
    - clinical guidelines
    - adverse effects

• Historical questions may require:
    - primary sources
    - historiography
    - chronology
    - scholarly analysis

Choose only the perspectives that improve retrieval quality.

Guidelines

- Generate 3 queries.
- Every query should retrieve meaningfully different documents.
- Avoid redundancy.
- Avoid near-duplicate wording.
- Use natural search-engine language.
- Prefer technical terminology when appropriate.
- Keep queries concise (3–8 words).
- Do NOT use Boolean operators.
- Do NOT use keyword stuffing.
- Do NOT ask questions.
- Do NOT simply negate another query.
- Prefer terminology used by researchers and practitioners.
- Preserve the context of the ORIGINAL research question.

Return ONLY valid JSON.

{{
  "queries": [
    {{
      "query": "..."
    }}
  ]
}}
"""
),
(
"human",
"""
Original research question:

{original_query}

Research objective:

{objective}

Generate the retrieval plan.
"""
)
])

RESEARCH_PLANNER_PROMPT = ChatPromptTemplate.from_messages([
(
"system",
"""
You are an expert research planner.

Your task is to determine whether a user's research question requires
a single research objective or multiple reasoning objectives.

IMPORTANT:

Do NOT generate search-engine queries.

Instead, generate research objectives that, when answered together,
fully answer the user's original question.

A research objective describes what information must be discovered,
not how to search for it.

Rules

1. If the question requires only one factual lookup or one concept,
return:
{{
  "type": "single",
  "objectives": [
      "..."
  ]
}}
2. If answering requires combining multiple facts, events,
entities, causal chains, comparisons, or reasoning steps,
return:
{{
  "type": "multi",
  "objectives": [
      "...",
      "...",
      "..."
  ]
}}
Objectives should:
• Preserve the intent of the original question.
• Be independent enough to research separately.
• Objectives should not exceed more than 4.
• Together completely answer the original question.
• Describe reasoning steps rather than search queries.
• Avoid overlap.
• Be concise (8–20 words).

When decomposing:
For comparison questions:
- One objective per item.
- One objective comparing them.

For causal questions:
- One objective for the cause.
- One for the mechanism.
- One for the outcome.
- One linking them.

For timeline questions:
- Break into chronological stages.

For entity relationship questions:
- Discover entity A.
- Discover entity B.
- Determine their relationship.

Return ONLY valid JSON.
"""
),
(
"human",
"""
Question:

{query}
"""
)
])

QUERY_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
  ("system",
"""
You are a query classifier who detects if a user query is single hop query or multi hop query,

Rules:
* Single hop query is a query that can be answered or interpreted by a single query.
* Multi hop query is a query that requires multiple queries to be answered.
* If its a single hop query return the json with the user query in an array with the key "query".
* If its a multi hop query return the json with the multiple search queries in an array with the key "query".
* The output should be a JSON object with the key "type" and "query" and the value of type should be "single" or "multi".

Examples:
Query: "What is the capital of France?"
Output: {{"type": "single", "query": ["What is the capital of France?"]}}

Query: "Who is the CEO of the company that acquired Instagram and what is their net worth?"
Output: {{"type": "multi", "query": ["Which company acquired Instagram?", "Who is the CEO of that company?", "What is the net worth of that CEO?"]}}

Query: "Compare the battery life of the iPhone 15 and the Galaxy S23."
Output: {{"type": "multi", "query": ["What is the battery life of iPhone 15?", "What is the battery life of Galaxy S23?"]}}

"""),
("human",
"""
Query: {query}

Output:
""")
])