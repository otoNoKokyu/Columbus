import asyncio
import logging
import sys
import os

# Add the parent directory of Columbus (python-practice) to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)


from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

from Columbus.scoring.pinecone_reranker import PineconeReranker
from Columbus.search.search_agent import get_search_client
from Columbus.crawl.firecrawl_engine import scrape_urls_for_markdown
from Columbus.crawl.link_extractor import extract_links_from_markdown

# Configure logging to output to command line
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

async def test_pinecone_reranker_with_firecrawl():
    print("Initializing PineconeReranker...")
    try:
        reranker = PineconeReranker(model_name="bge-reranker-v2-m3", top_n=5)
    except Exception as e:
        print(f"Failed to initialize PineconeReranker: {e}")
        return

    # Check for Pinecone API Key
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("\n[WARNING] PINECONE_API_KEY is not set in environment or .env file.")
        print("Please set PINECONE_API_KEY to run Pinecone Inference API call successfully.")

    query = "jaoon kahan bata ae dil movie review"
    print(f"\n1. Executing search for query: '{query}'...")
    
    # 1. Search using Exa or fallback to DuckDuckGo search client
    try:
        search_client = get_search_client("ddg")
        search_results = await search_client.async_search(query, max_results=3)
    except Exception as e:
        print(f"Search failed: {e}")
        return

    if not search_results:
        print("No search results found.")
        return

    urls = [res["url"] for res in search_results if "url" in res]
    print(f"✔ Search complete. Found {len(urls)} URLs to scrape: {urls}")

    # 2. Scrape the URLs with Firecrawl
    print("\n2. Scraping URLs using Firecrawl to extract markdown...")
    firecrawl_api_key = os.environ.get("FIRECRAWL_API_KEY")
    try:
        pages = await scrape_urls_for_markdown(urls, api_key=firecrawl_api_key)
        success_pages = [p for p in pages if not p.get("error")]
        print(f"✔ Scrape complete. Successfully scraped {len(success_pages)}/{len(pages)} pages.")
    except Exception as e:
        print(f"Scraping failed: {e}")
        return

    # 3. Extract child links from scraped markdown
    print("\n3. Extracting child links from scraped markdown pages...")
    candidates = []
    seen_urls = set()
    for page in success_pages:
        links = extract_links_from_markdown(page.get("markdown", ""), page.get("url", ""))
        for link in links:
            if link["url"] not in seen_urls:
                seen_urls.add(link["url"])
                candidates.append(link)

    print(f"✔ Link extraction complete. Found {len(candidates)} unique candidates to rerank.")

    if not candidates:
        print("No candidate links extracted to rerank.")
        return

    # 4. Rerank the candidates using Pinecone Reranker
    print(f"\n4. Reranking {len(candidates)} candidates against query: '{query}'...")
    if not api_key:
        print("Skipping Pinecone API call as API Key is not set.")
        print("First 3 candidate links extracted from Firecrawl markdown:")
        for idx, c in enumerate(candidates[:3]):
            print(f"Candidate {idx+1}:")
            print(f"  URL: {c.get('url')}")
            print(f"  Anchor Text: {c.get('anchor_text')}")
            print(f"  Context: {c.get('context')[:120]}...")
        return

    try:
        results = reranker.rerank(query=query, candidates=candidates, top_n=5)

        print("\n--- Reranking Results ---")
        for i, res in enumerate(results):
            print(f"Rank {i+1}:")
            print(f"  URL: {res.get('url', 'N/A')}")
            print(f"  Anchor Text: {res.get('anchor_text', 'N/A')}")
            print(f"  Context: {res.get('context', 'N/A')[:120]}...")
            print(f"  Pinecone Score: {res.get('rerank_score', 0):.4f}")
            print()
    except Exception as e:
        print(f"Pinecone Rerank call failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_pinecone_reranker_with_firecrawl())
