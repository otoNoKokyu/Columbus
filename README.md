# Columbus — Autonomous Web Research & Crawling Pipeline

Columbus is a modular, high-performance web research and crawling pipeline built using **LangChain**. It is designed to autonomously search the web, scrape relevant sites, extract child links, prioritize them semantically using embedding models, and rerank them to extract deep information on any given topic.

### How it works:
1. **Query Rewrite**: Rewrites an initial query into multiple search variants using an LLM.
2. **Fan-Out Search**: Concurrent search queries utilizing DuckDuckGo or Exa APIs, followed by automatic URL deduplication.
3. **Markdown Crawling**: Converts web pages into clean markdown formats using Firecrawl or Crawl4AI.
4. **Link Extraction**: Parses hyperlinks out of the scraped page markdown.
5. **Semantic Scoring**: Scores link relevance against the original query using `sentence-transformers`.
6. **Reranking**: Filters candidates down to the top results using a Cross-Encoder or LLM reranker.
7. **Recursive Deep Crawling**: Continues to crawl child pages recursively for deeper analysis.
