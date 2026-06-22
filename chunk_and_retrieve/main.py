import os
import re
import asyncio
import hashlib
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, ServerlessSpec
from langchain_core.embeddings import Embeddings

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        from langchain.embeddings import HuggingFaceEmbeddings

class HFInferenceEmbeddings(HuggingFaceEmbeddings):
    """LangChain-compatible local HuggingFaceEmbeddings wrapper."""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", hf_token: str | None = None):
        super().__init__(model_name=model_name)



class SemanticChunker:
    """Wrapper around LangChain's SemanticChunker using HFInferenceEmbeddings."""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", hf_token: str | None = None):
        self._model_name = model_name
        self._hf_token = hf_token
        # Cache the local embeddings instance once to avoid repeatedly reloading model weights
        self._embeddings = HFInferenceEmbeddings(self._model_name, self._hf_token)

    async def embed_sentences(self, sentences: List[str]) -> List[List[float]]:
        """Concurrency wrapper for embedding list of texts asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._embeddings.embed_documents(sentences))

    async def chunk(self, text: str, min_chunk_size: int = 150, percentile_threshold: float = 20.0) -> List[str]:
        """Chunks document text semantically using LangChain's SemanticChunker."""
        from langchain_experimental.text_splitter import SemanticChunker as LcSemanticChunker
        
        # breakpoint threshold percentile amount (e.g. 20th percentile similarity corresponds to 80th percentile distance drop)
        breakpoint_amount = 100.0 - percentile_threshold
        
        text_splitter = LcSemanticChunker(
            embeddings=self._embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=breakpoint_amount,
            min_chunk_size=min_chunk_size
        )
        
        loop = asyncio.get_running_loop()
        chunks = await loop.run_in_executor(None, lambda: text_splitter.split_text(text))
        return chunks


def get_or_create_index(pc: Pinecone, index_name: str, dimension: int = 768) -> Any:
    """Retrieves or creates a Serverless AWS Pinecone index."""
    # Fetch list of existing index names
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    return pc.Index(index_name)


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: List[List[float]],
    lambda_mult: float = 0.5,
    k: int = 4
) -> List[int]:
    """Calculate maximal marginal relevance to optimize for similarity to query and diversity among selected documents."""
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_recent = np.argmax(similarity_to_query)
    idxs = [int(most_recent)]
    selected = np.array([embedding_list[most_recent]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            item_score = lambda_mult * query_score - (1 - lambda_mult) * np.max(similarity_to_selected[i])
            if item_score > best_score:
                best_score = item_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs


async def chunk_store_and_retrieve(
    queries_by_bias: Dict[str, str],
    contents_by_bias: Dict[str, List[str]],
    index_name: str = "columbus-research",
    top_k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    pinecone_api_key: str | None = None,
    hf_token: str | None = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Chunks documents semantically, uploads to Pinecone DB with metadata, and retrieves top-k chunks segregated by query."""
    import logging
    logger = logging.getLogger(__name__)

    # 1. Initialize Clients
    pc_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
    if not pc_key:
        raise ValueError("PINECONE_API_KEY is not set in environment or constructor parameters.")
        
    pc = Pinecone(api_key=pc_key)
    index = get_or_create_index(pc, index_name, dimension=768)
    
    chunker = SemanticChunker(hf_token=hf_token)
    
    logger.info("Phase 1: Semantic Chunking")
    # 2. Concurrently chunk all page content
    chunk_tasks = []
    chunk_task_metadata = [] # To map results back to bias_type
    
    for bias_type, contents in contents_by_bias.items():
        for content in contents:
            chunk_tasks.append(chunker.chunk(content))
            chunk_task_metadata.append(bias_type)
            
    if not chunk_tasks:
        logger.warning("No content to chunk. Returning empty matches.")
        return {b: [] for b in queries_by_bias.keys()}
            
    chunk_results = await asyncio.gather(*chunk_tasks)
    
    # Flatten and map chunks to bias type
    all_chunks = []
    chunk_bias_types = []
    
    for i, chunks in enumerate(chunk_results):
        bias_type = chunk_task_metadata[i]
        all_chunks.extend(chunks)
        chunk_bias_types.extend([bias_type] * len(chunks))
        
    logger.info(f"Generated {len(all_chunks)} total chunks.")
    
    if not all_chunks:
        return {b: [] for b in queries_by_bias.keys()}
        
    logger.info("Phase 2: Embedding chunks")
    # 3. Generate embeddings in parallel for all chunks
    for idx, chunk in enumerate(all_chunks):
        first_sentence = chunk.split(".")[0].strip() if "." in chunk else chunk.strip()
        words = first_sentence.split()
        short_text = " ".join(words[:10]) + ("..." if len(words) > 10 else "")
        logger.info(f"Embedding chunk {idx+1}/{len(all_chunks)}: '{short_text}'")
    chunk_embeddings = await chunker.embed_sentences(all_chunks)
    
    logger.info("Phase 3: Pinecone Upsert")
    # 4. Formulate records and upsert to Pinecone
    vectors = []
    
    for i, chunk in enumerate(all_chunks):
        emb = chunk_embeddings[i]
        bias_type = chunk_bias_types[i]
        chunk_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()
        chunk_id = f"{bias_type}_chunk_{i}_{chunk_hash}"
        
        vectors.append({
            "id": chunk_id,
            "values": emb,
            "metadata": {
                "text": chunk,
                "bias_type": bias_type
            }
        })
        
    # Upsert in batches of 100 to Pinecone
    batch_size = 100
    for j in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[j : j + batch_size])
        
    logger.info("Phase 4: Retrieval")
    # 5. Embed queries
    bias_types_for_queries = list(queries_by_bias.keys())
    queries_text = [queries_by_bias[b] for b in bias_types_for_queries]
    for idx, q_text in enumerate(queries_text):
        logger.info(f"Embedding query {idx+1}/{len(queries_text)}: '{q_text}'")
    
    query_embs = await chunker.embed_sentences(queries_text)
    
    # 6. Query Pinecone separately using metadata filter
    query_tasks = []
    for i, bias_type in enumerate(bias_types_for_queries):
        q_emb = query_embs[i]
        task = index.query(
            vector=q_emb,
            top_k=max(top_k, fetch_k),
            filter={"bias_type": bias_type},
            include_metadata=True,
            include_values=True
        )
        query_tasks.append(task)
        
    loop = asyncio.get_running_loop()
    query_results = await asyncio.gather(*(
        loop.run_in_executor(None, lambda t=task: t) for task in query_tasks
    ))
    
    # 7. Extract matched texts, apply MMR to re-rank and filter
    retrieval_matches = {}
    for i, bias_type in enumerate(bias_types_for_queries):
        res = query_results[i]
        q_emb = query_embs[i]
        
        matches = res.get("matches", [])
        if not matches:
            retrieval_matches[bias_type] = []
            continue
            
        embeddings = [match["values"] for match in matches]
        
        # Apply MMR
        mmr_selected_indices = maximal_marginal_relevance(
            query_embedding=np.array(q_emb),
            embedding_list=embeddings,
            lambda_mult=lambda_mult,
            k=top_k
        )
        
        mmr_matches = [
            {"text": matches[idx].get("metadata", {}).get("text", ""), "score": matches[idx].get("score", 0.0)}
            for idx in mmr_selected_indices
        ]
        retrieval_matches[bias_type] = mmr_matches
        
    return retrieval_matches
