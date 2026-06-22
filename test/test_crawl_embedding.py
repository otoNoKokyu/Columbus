import os
import sys
import json
import asyncio
import hashlib
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from pinecone import Pinecone

# Add parent directory of Columbus (python-practice) to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

from Columbus.chunk_and_retrieve.main import HFInferenceEmbeddings, SemanticChunker, get_or_create_index

async def embed_and_store_balanced_results(json_path: str, index_name: str = "columbus-crawl-index"):
    # 1. Check keys
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is not set.")
    
    # 2. Initialize Pinecone
    print(f"Initializing Pinecone and connecting to index: '{index_name}'...")
    pc = Pinecone(api_key=pinecone_api_key)
    index = get_or_create_index(pc, index_name, dimension=768)
    
    # Delete existing entries in the index to ensure a clean test
    print("Clearing existing vectors in the index to ensure a clean start...")
    try:
        index.delete(delete_all=True)
    except Exception as e:
        print(f"Note: Could not clear index (might be empty or does not support delete_all): {e}")

    # 3. Load crawl results JSON
    print(f"Loading balanced crawl results from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    original_query = data.get("original_query", "Unknown query")
    perspectives = data.get("perspectives", [])
    print(f"Original Query: '{original_query}'")
    print(f"Found {len(perspectives)} perspectives to process.")
    
    chunker = SemanticChunker(hf_token=hf_token)
    embeddings_provider = HFInferenceEmbeddings(hf_token=hf_token)
    
    all_vectors = []
    
    # 4. Chunk and prepare vectors
    page_count = 0
    for p_idx, perspective in enumerate(perspectives):
        bias_type = perspective.get("bias_type", "unknown")
        rewritten_query = perspective.get("rewritten_query", "")
        pages = perspective.get("pages_crawled", [])
        
        print(f"\nPerspective {p_idx + 1}/{len(perspectives)}: Bias={bias_type}, Query='{rewritten_query}'")
        print(f"Contains {len(pages)} pages.")
        
        for doc_idx, page in enumerate(pages):
            url = page.get("url", f"unknown_doc_{doc_idx}")
            title = page.get("title", "No Title")
            content = page.get("content", "").strip()
            
            if not content:
                print(f"  ⚠️ Page {doc_idx} ({url}) has empty 'content'. Skipping.")
                continue
                
            page_count += 1
            print(f"  [{bias_type.upper()}] Chunking page {doc_idx + 1}/{len(pages)}: {url[:50]}... ({len(content)} chars)")
            
            # Split text into semantic chunks
            chunks = await chunker.chunk(content, min_chunk_size=150, percentile_threshold=20.0)
            print(f"    -> Generated {len(chunks)} semantic chunks.")
            
            if not chunks:
                continue
                
            # Embed chunks
            print(f"    -> Embedding {len(chunks)} chunks...")
            embeddings = embeddings_provider.embed_documents(chunks)
            
            # Prepare vector records
            for chunk_idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                # Create a unique ID using md5 hash of the chunk content to avoid duplication
                chunk_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()
                vector_id = f"doc_{page_count}_chunk_{chunk_idx}_{chunk_hash[:12]}"
                
                # Ensure embedding values are python floats to avoid Pinecone orjson serialization errors
                clean_emb = [float(x) for x in emb]
                
                all_vectors.append({
                    "id": vector_id,
                    "values": clean_emb,
                    "metadata": {
                        "text": chunk,
                        "url": url,
                        "title": title,
                        "bias_type": bias_type,
                        "chunk_idx": chunk_idx
                    }
                })
            
    # 5. Upsert to Pinecone
    print(f"\nUpserting {len(all_vectors)} total vector chunks to Pinecone...")
    batch_size = 100
    for j in range(0, len(all_vectors), batch_size):
        batch = all_vectors[j : j + batch_size]
        index.upsert(vectors=batch)
        print(f"  Uploaded chunks {j} to {min(j + batch_size, len(all_vectors))}")
        
    print("✔ Successfully embedded and stored all crawl content!")
    return index, embeddings_provider

async def run_query(index, embeddings_provider, query: str, bias_type: Optional[str] = None, top_k: int = 3):
    bias_desc = f" [{bias_type}]" if bias_type else ""
    print(f"\nSearching for: \"{query}\"{bias_desc}")
    # Embed the search query
    query_vector = embeddings_provider.embed_query(query)
    clean_query_vector = [float(x) for x in query_vector]
    
    # Query Pinecone
    query_params: Dict[str, Any] = {
        "vector": clean_query_vector,
        "top_k": top_k,
        "include_metadata": True
    }
    if bias_type:
        query_params["filter"] = {"bias_type": bias_type}
        
    res = index.query(**query_params)
    
    matches = res.get("matches", [])
    print(f"Found {len(matches)} results:")
    for idx, match in enumerate(matches):
        metadata = match.get("metadata", {})
        url = metadata.get("url", "Unknown URL")
        title = metadata.get("title", "No Title")
        text = metadata.get("text", "No text content available")
        match_bias = metadata.get("bias_type", "unknown")
        score = match.get("score", 0.0)
        
        print(f"\n  Rank {idx + 1} [Similarity Score: {score:.4f}] [Bias: {match_bias}]")
        print(f"  Title: {title}")
        print(f"  Source URL: {url}")
        print(f"  Excerpt: {text.strip()[:350]}...")
        print("-" * 60)

async def main():
    json_path = os.path.join(parent_dir, "balanced_crawl_results.json")
    index_name = "columbus-crawl-index"
    
    # 1. Embed and store
    index, embeddings_provider = await embed_and_store_balanced_results(json_path, index_name)
    
    # 2. Run Pre-defined Test Queries
    print("\n" + "="*80)
    print("RUNNING AUTOMATED TEST QUERIES")
    print("="*80)
    
    print("\n>>> 1. General Queries (No bias filtering)")
    await run_query(index, embeddings_provider, "What are the benefits of minoxidil for hair growth and density?", top_k=2)
    await run_query(index, embeddings_provider, "What are the common side effects, scalp irritation and cardiac risks of minoxidil?", top_k=2)
    
    print("\n>>> 2. Segregated Queries (With bias filtering)")
    # Query for supporting perspective
    await run_query(index, embeddings_provider, "Clinical efficacy, hair follicles, and hair count results", bias_type="supporting", top_k=2)
    # Query for opposing perspective
    await run_query(index, embeddings_provider, "Safety warnings, skin irritation, cardiovascular problems, and drawbacks", bias_type="opposing", top_k=2)
        
    # 3. Interactive Loop for user testing
    if not sys.stdin.isatty():
        print("\n" + "="*80)
        print("Non-interactive terminal detected. Skipping interactive retrieval loop.")
        print("="*80)
        return

    print("\n" + "="*80)
    print("INTERACTIVE RETRIEVAL TESTING")
    print("="*80)
    print("You can now enter your own search queries to test the retriever.")
    print("Type your query and press Enter. Leave empty or type 'exit' to quit.")
    print("Format: 'query_text' OR 'query_text | supporting' OR 'query_text | opposing'\n")
    
    while True:
        try:
            loop = asyncio.get_running_loop()
            user_input = await loop.run_in_executor(None, lambda: input("Enter search query: ").strip())
            
            if not user_input or user_input.lower() in ["exit", "quit"]:
                print("Exiting interactive test loop.")
                break
            
            # Parse filter if provided (e.g., "scalp irritation | opposing")
            filter_bias = None
            if "|" in user_input:
                parts = user_input.split("|")
                user_query = parts[0].strip()
                bias_part = parts[1].strip().lower()
                if bias_part in ["supporting", "opposing"]:
                    filter_bias = bias_part
                else:
                    user_query = user_input # treat entire thing as query if invalid filter
            else:
                user_query = user_input
                
            await run_query(index, embeddings_provider, user_query, bias_type=filter_bias, top_k=3)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive test loop.")
            break

if __name__ == "__main__":
    asyncio.run(main())
