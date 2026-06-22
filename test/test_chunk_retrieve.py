import asyncio
import logging
import os
import sys

# Add parent directory of Columbus (python-practice) to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)


from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

from Columbus.chunk_and_retrieve.main import SemanticChunker, chunk_store_and_retrieve

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

async def test_chunk_and_retrieve_flow():
    print("=== Testing Semantic Chunking & Pinecone Retrieval Flow ===")
    
    # 1. Check for API keys
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")
    
    if not pinecone_api_key:
        print("\n[WARNING] PINECONE_API_KEY is not set.")
        print("Please configure PINECONE_API_KEY to test index storage and query retrieval.")
        return
        
    # 2. Setup mock data
    supporting_content = [
        "Minoxidil is an extremely effective treatment for hair loss. It works as a vasodilator, dilating blood vessels and improving blood flow to hair follicles. This increased circulation delivers essential oxygen and nutrients to follicles, revitalizing them and promoting new growth. Clinical studies have shown that a majority of users see significant hair regrowth within 3 to 6 months of daily application. Dermatologists globally recommend minoxidil as a first-line defense against androgenetic alopecia. Regular usage helps maintain hair density and prevents further thinning."
    ]
    
    opposing_content = [
        "Minoxidil can lead to several undesirable side effects and medical risks. Many users experience localized scalp irritation, dryness, itching, and redness upon initiating treatment. In some cases, it can cause hypertrichosis, which is the unwanted growth of hair on other parts of the body such as the face. More seriously, because it is a vasodilator, system absorption can lead to cardiovascular changes. Users have reported rapid heartbeat, chest pain, dizziness, and sudden weight gain due to water retention. Medical consultation is strongly advised before starting treatment."
    ]
    
    supporting_query = "evidence that minoxidil regrows hair and boosts density"
    opposing_query = "side effects cardiovascular risks scalp irritation of minoxidil"
    
    # 3. Test Chunker directly first
    print("\n1. Testing SemanticChunker directly on supporting content...")
    chunker = SemanticChunker(hf_token=hf_token)
    chunks = await chunker.chunk(supporting_content[0], min_chunk_size=100)
    print(f"✔ Successfully split text into {len(chunks)} semantic chunks:")
    for idx, c in enumerate(chunks):
        print(f"  Chunk {idx + 1} ({len(c)} chars): {c[:120]}...")
        
    # 4. Test end-to-end storage and retrieval
    print("\n2. Executing chunk, store, and query retrieval from Pinecone...")
    try:
        results = await chunk_store_and_retrieve(
            supporting_contents=supporting_content,
            opposing_contents=opposing_content,
            supporting_query=supporting_query,
            opposing_query=opposing_query,
            index_name="columbus-test-index",
            top_k=2,
            pinecone_api_key=pinecone_api_key,
            hf_token=hf_token
        )
        
        print("\n--- Semantic Retrieval Results ---")
        print("\n✔ Supporting Matches:")
        for idx, match in enumerate(results.get("supporting", [])):
            print(f"  Rank {idx+1} [Score: {match['score']:.4f}]:")
            print(f"    {match['text']}")
            
        print("\n✔ Opposing Matches:")
        for idx, match in enumerate(results.get("opposing", [])):
            print(f"  Rank {idx+1} [Score: {match['score']:.4f}]:")
            print(f"    {match['text']}")
            
        print("\n✔ End-to-end execution complete!")
    except Exception as e:
        import traceback
        print("End-to-end execution failed:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_chunk_and_retrieve_flow())
