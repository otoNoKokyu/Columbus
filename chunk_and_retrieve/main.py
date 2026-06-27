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



class PineconeHostedEmbeddings:
    """Wrapper around Pinecone's Hosted Inference Embeddings."""
    
    def __init__(self, model_name: str = "multilingual-e5-large", pinecone_api_key: str | None = None):
        self.model_name = model_name
        self.api_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY is required for PineconeHostedEmbeddings.")
        self.pc = Pinecone(api_key=self.api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        batch_size = 96
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            res = self.pc.inference.embed(
                model=self.model_name,
                inputs=batch,
                parameters={"input_type": "passage", "truncate": "END"}
            )
            for item in res.data:
                embeddings.append(item.values)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        res = self.pc.inference.embed(
            model=self.model_name,
            inputs=[text],
            parameters={"input_type": "query", "truncate": "END"}
        )
        return res.data[0].values


def get_index_embed_config(pc: Pinecone, index_name: str) -> Tuple[str | None, int | None]:
    """Retrieves the integrated embedding model and dimension for a Pinecone index if configured."""
    try:
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if index_name in existing_indexes:
            desc = pc.describe_index(index_name)
            if hasattr(desc, "embed") and desc.embed is not None:
                model = getattr(desc.embed, "model", None)
                dim = getattr(desc.embed, "dimension", None)
                return model, dim
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to fetch index embed config for '{index_name}': {e}")
    return None, None


class BaseChunker:
    """Abstract base class for document chunkers."""
    
    def __init__(
        self, 
        model_name: str = "BAAI/bge-base-en-v1.5", 
        hf_token: str | None = None,
        embedding_source: str = "local",
        pinecone_api_key: str | None = None,
        index_name: str | None = None
    ):
        self._model_name = model_name
        self._hf_token = hf_token
        self.embedding_source = embedding_source
        self.pinecone_api_key = pinecone_api_key
        self.index_name = index_name
        
        if embedding_source == "integrated":
            # If integrated, try to retrieve the model name from the index description dynamically
            resolved_model = None
            resolved_dim = 1024
            
            pc_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
            if pc_key and index_name:
                try:
                    pc = Pinecone(api_key=pc_key)
                    resolved_model, resolved_dim = get_index_embed_config(pc, index_name)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Could not retrieve embed config from index '{index_name}': {e}")
            
            if not resolved_model:
                resolved_model = "llama-text-embed-v2" if model_name == "BAAI/bge-base-en-v1.5" else model_name
                resolved_dim = 1024
                
            self._model_name = resolved_model
            self._dimension = resolved_dim
            self._embeddings = PineconeHostedEmbeddings(model_name=resolved_model, pinecone_api_key=pinecone_api_key)
            
        elif embedding_source == "pinecone":
            model = "multilingual-e5-large" if model_name == "BAAI/bge-base-en-v1.5" else model_name
            self._model_name = model
            self._dimension = 1024
            self._embeddings = PineconeHostedEmbeddings(model_name=model, pinecone_api_key=pinecone_api_key)
        else:
            self._model_name = model_name
            self._dimension = 768
            self._embeddings = HFInferenceEmbeddings(self._model_name, self._hf_token)

    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors produced by this chunker."""
        if hasattr(self, "_dimension") and self._dimension is not None:
            return self._dimension
        if self.embedding_source in ("pinecone", "integrated"):
            return 1024
        else:
            return 768

    async def embed_sentences(self, sentences: List[str]) -> List[List[float]]:
        """Concurrency wrapper for embedding list of texts asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._embeddings.embed_documents(sentences))

    async def chunk(self, text: str) -> List[str]:
        """Abstract chunk method to be overridden by subclasses."""
        raise NotImplementedError


class SemanticChunker(BaseChunker):
    """Wrapper around LangChain's SemanticChunker using HFInferenceEmbeddings."""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", hf_token: str | None = None, min_chunk_size: int = 150, percentile_threshold: float = 20.0, embedding_source: str = "local", pinecone_api_key: str | None = None, index_name: str | None = None):
        super().__init__(model_name=model_name, hf_token=hf_token, embedding_source=embedding_source, pinecone_api_key=pinecone_api_key, index_name=index_name)
        self.min_chunk_size = min_chunk_size
        self.percentile_threshold = percentile_threshold

    async def chunk(self, text: str, min_chunk_size: int | None = None, percentile_threshold: float | None = None) -> List[str]:
        """Chunks document text semantically using LangChain's SemanticChunker."""
        from langchain_experimental.text_splitter import SemanticChunker as LcSemanticChunker
        
        min_sz = min_chunk_size if min_chunk_size is not None else self.min_chunk_size
        pct = percentile_threshold if percentile_threshold is not None else self.percentile_threshold
        
        # breakpoint threshold percentile amount (e.g. 20th percentile similarity corresponds to 80th percentile distance drop)
        breakpoint_amount = 100.0 - pct
        
        text_splitter = LcSemanticChunker(
            embeddings=self._embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=breakpoint_amount,
            min_chunk_size=min_sz
        )
        
        loop = asyncio.get_running_loop()
        chunks = await loop.run_in_executor(None, lambda: text_splitter.split_text(text))
        return chunks


class RecursiveChunker(BaseChunker):
    """Chunks document text recursively based on word count, ensuring each chunk is at least min_chunk_size words."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", hf_token: str | None = None, min_chunk_size: int = 500, chunk_overlap: int = 50, embedding_source: str = "local", pinecone_api_key: str | None = None, index_name: str | None = None):
        super().__init__(model_name=model_name, hf_token=hf_token, embedding_source=embedding_source, pinecone_api_key=pinecone_api_key, index_name=index_name)
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap

    async def chunk(self, text: str) -> List[str]:
        """Chunks document text recursively using Word Count metrics, ensuring chunks are at least min_chunk_size words."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        # 1. Split into small pieces first (sentences/clauses)
        # Using a small chunk size of 50 words to get fine-grained fragments
        raw_splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=0,
            length_function=lambda x: len(x.split()),
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        loop = asyncio.get_running_loop()
        raw_chunks = await loop.run_in_executor(None, lambda: raw_splitter.split_text(text))
        
        # 2. Merge pieces to ensure each chunk is at least self.min_chunk_size words
        merged_chunks = []
        current_chunk = []
        current_words = 0
        
        for piece in raw_chunks:
            piece = piece.strip()
            if not piece:
                continue
            piece_words = len(piece.split())
            current_chunk.append(piece)
            current_words += piece_words
            
            if current_words >= self.min_chunk_size:
                merged_chunks.append(" ".join(current_chunk))
                # Implement overlap
                overlap_pieces = []
                overlap_words = 0
                for p in reversed(current_chunk):
                    p_w = len(p.split())
                    if overlap_words + p_w <= self.chunk_overlap:
                        overlap_pieces.insert(0, p)
                        overlap_words += p_w
                    else:
                        break
                current_chunk = overlap_pieces
                current_words = overlap_words
                
        # 3. Handle remaining pieces
        if current_chunk and current_words > 0:
            remaining_text = " ".join(current_chunk)
            if merged_chunks:
                merged_chunks[-1] += " " + remaining_text
            else:
                merged_chunks.append(remaining_text)
                
        return merged_chunks


def get_chunker(
    strategy: str = "semantic",
    model_name: str = "BAAI/bge-base-en-v1.5",
    hf_token: str | None = None,
    embedding_source: str = "local",
    pinecone_api_key: str | None = None,
    index_name: str | None = None,
    **kwargs
) -> BaseChunker:
    """Factory function to instantiate the requested chunker strategy."""
    if strategy == "recursive":
        min_sz = kwargs.get("min_chunk_size")
        if min_sz is None:
            min_sz = kwargs.get("chunk_size", 500)
        return RecursiveChunker(
            model_name=model_name,
            hf_token=hf_token,
            min_chunk_size=min_sz,
            chunk_overlap=kwargs.get("chunk_overlap", 50),
            embedding_source=embedding_source,
            pinecone_api_key=pinecone_api_key,
            index_name=index_name
        )
    elif strategy == "semantic":
        return SemanticChunker(
            model_name=model_name,
            hf_token=hf_token,
            min_chunk_size=kwargs.get("min_chunk_size", 150),
            percentile_threshold=kwargs.get("percentile_threshold", 20.0),
            embedding_source=embedding_source,
            pinecone_api_key=pinecone_api_key,
            index_name=index_name
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def clean_chunk_text(text: str) -> str:
    """Cleans chunk text by stripping out markdown links, naked URLs, and unicode escape sequences/characters."""
    import re
    # 1. Clean markdown links [anchor](url) -> anchor
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\s\)]+)\)", r"\1", text)
    # 2. Clean naked URLs
    text = re.sub(r"https?://[^\s]+", "", text)
    # 3. Clean specific unicode representations like u2013 (en-dash), u2014 (em-dash), etc.
    # Replace them with standard dashes
    text = re.sub(r"\\?u2013\d*", "-", text)
    text = re.sub(r"\\?u2014\d*", "-", text)
    # Clean actual unicode en-dash and em-dash if they are characters
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    # Clean generic unicode escapes if any remain (e.g., \uXXXX)
    text = re.sub(r"\\u[0-9a-fA-F]{4}", " ", text)
    # Clean multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_or_create_index(
    pc: Pinecone, 
    index_name: str, 
    dimension: int = 768, 
    embed_config: Dict[str, Any] | None = None
) -> Any:
    """Retrieves or creates a Serverless AWS Pinecone index."""
    # Fetch list of existing index names
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        if embed_config:
            pc.create_index_for_model(
                name=index_name,
                cloud="aws",
                region="us-east-1",
                embed=embed_config
            )
        else:
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
    queries_by_bias: Dict[str, str] | None = None,
    contents_by_bias: Dict[str, List[str]] | None = None,
    index_name: str = "columbus-research",
    top_k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    pinecone_api_key: str | None = None,
    hf_token: str | None = None,
    chunking_strategy: str = "semantic",
    min_chunk_size: int | None = None,
    chunk_overlap: int = 50,
    percentile_threshold: float = 20.0,
    embedding_source: str = "local",
    **kwargs
) -> Dict[str, List[Dict[str, Any]]]:
    """Chunks documents using the pluggable strategy, uploads to Pinecone DB, and retrieves top-k chunks segregated by query."""
    import logging
    logger = logging.getLogger(__name__)

    # Support backward compatibility parameters
    if queries_by_bias is None:
        queries_by_bias = {}
    if contents_by_bias is None:
        contents_by_bias = {}

    supporting_query = kwargs.get("supporting_query")
    opposing_query = kwargs.get("opposing_query")
    supporting_contents = kwargs.get("supporting_contents")
    opposing_contents = kwargs.get("opposing_contents")

    if supporting_query and "supporting" not in queries_by_bias:
        queries_by_bias["supporting"] = supporting_query
    if opposing_query and "opposing" not in queries_by_bias:
        queries_by_bias["opposing"] = opposing_query
    if supporting_contents and "supporting" not in contents_by_bias:
        contents_by_bias["supporting"] = supporting_contents
    if opposing_contents and "opposing" not in contents_by_bias:
        contents_by_bias["opposing"] = opposing_contents

    # Map legacy chunk_size / min_chunk_size defaults
    legacy_chunk_size = kwargs.get("chunk_size")
    legacy_min_chunk_size = kwargs.get("min_chunk_size")
    
    resolved_min_chunk_size = min_chunk_size
    if resolved_min_chunk_size is None:
        if legacy_chunk_size is not None:
            resolved_min_chunk_size = legacy_chunk_size
        elif legacy_min_chunk_size is not None:
            resolved_min_chunk_size = legacy_min_chunk_size
        else:
            resolved_min_chunk_size = 500 if chunking_strategy == "recursive" else 150

    # 1. Initialize Clients
    pc_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
    if not pc_key:
        raise ValueError("PINECONE_API_KEY is not set in environment or constructor parameters.")
        
    pc = Pinecone(api_key=pc_key)
    
    chunker = get_chunker(
        strategy=chunking_strategy,
        hf_token=hf_token,
        min_chunk_size=resolved_min_chunk_size,
        chunk_overlap=chunk_overlap,
        percentile_threshold=percentile_threshold,
        embedding_source=embedding_source,
        pinecone_api_key=pc_key,
        index_name=index_name
    )

    # Resolve embed_config for integrated index creation if it does not exist
    embed_config = None
    if embedding_source == "integrated":
        embed_config = {
            "model": chunker._model_name,
            "field_map": {"text": "text"}
        }

    index = get_or_create_index(pc, index_name, dimension=chunker.dimension, embed_config=embed_config)
    
    # 1.5 Special check for Pinecone Integrated Embeddings
    if embedding_source == "integrated":
        logger.info("Using Pinecone Integrated Embeddings (no local embedding model client-side)")
        
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
            cleaned = [clean_chunk_text(c) for c in chunks]
            cleaned = [c for c in cleaned if c.strip()]
            all_chunks.extend(cleaned)
            chunk_bias_types.extend([bias_type] * len(cleaned))
            
        logger.info(f"Generated {len(all_chunks)} total chunks for integrated upsert.")
        
        if not all_chunks:
            return {b: [] for b in queries_by_bias.keys()}
            
        # 3. Formulate records for integrated upsert
        records = []
        for i, chunk in enumerate(all_chunks):
            bias_type = chunk_bias_types[i]
            chunk_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()
            chunk_id = f"{bias_type}_chunk_{i}_{chunk_hash}"
            records.append({
                "_id": chunk_id,
                "text": chunk,
                "bias_type": bias_type
            })
            
        # Upsert records in batches of 100 to Pinecone default namespace
        batch_size = 100
        for j in range(0, len(records), batch_size):
            index.upsert_records(
                namespace="default",
                records=records[j : j + batch_size]
            )
            
        # 4. Search Pinecone integrated index
        retrieval_matches = {}
        for bias_type, q_text in queries_by_bias.items():
            if not q_text.strip():
                retrieval_matches[bias_type] = []
                continue
                
            logger.info(f"Searching integrated index for {bias_type}: '{q_text}'")
            res = index.search(
                namespace="default",
                query={
                    "inputs": {
                        "text": q_text
                    },
                    "top_k": top_k,
                    "filter": {"bias_type": bias_type}
                }
            )
            
            hits = res.result.hits if (res.result and res.result.hits) else []
            matches = []
            for hit in hits:
                matches.append({
                    "text": hit.fields.get("text", ""),
                    "score": hit.score
                })
            retrieval_matches[bias_type] = matches
            
        return retrieval_matches

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
    
    # Flatten, clean and map chunks to bias type
    all_chunks = []
    chunk_bias_types = []
    
    for i, chunks in enumerate(chunk_results):
        bias_type = chunk_task_metadata[i]
        cleaned = [clean_chunk_text(c) for c in chunks]
        # Filter out empty chunks
        cleaned = [c for c in cleaned if c.strip()]
        
        all_chunks.extend(cleaned)
        chunk_bias_types.extend([bias_type] * len(cleaned))
        
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
