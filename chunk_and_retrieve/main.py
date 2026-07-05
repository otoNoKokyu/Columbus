from langchain_huggingface import HuggingFaceEmbeddings
import os
import re
import asyncio
import hashlib
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.embeddings import Embeddings
from ..types.pipeline_exit import EarlyExitReason, PipelineEarlyExit


def get_embeddings(
    source: str = "local",
    model_name: str = "BAAI/bge-base-en-v1.5"
) -> Embeddings:
    """Factory function to retrieve LangChain-compatible embeddings.
    
    Supports: 'local' (HuggingFaceEmbeddings).
    """
    if source == "local":
        return HuggingFaceEmbeddings(model_name=model_name)
    else:
        raise ValueError(f"Unknown embedding source: {source}")


class BaseChunker:
    """Abstract base class for document chunkers."""
    
    def __init__(
        self, 
        model_name: str = "BAAI/bge-base-en-v1.5", 
        hf_token: str | None = None,
        min_words: int = 100,
        embeddings: Embeddings | None = None,
        **kwargs
    ):
        self._model_name = model_name
        self._hf_token = hf_token
        self.min_words = min_words
        self._dimension = 768
        
        if embeddings is not None:
            self._embeddings = embeddings
        else:
            self._embeddings = get_embeddings(source="local", model_name=self._model_name)

    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors produced by this chunker."""
        return self._dimension

    async def embed_sentences(self, sentences: List[str]) -> List[List[float]]:
        """Concurrency wrapper for embedding list of texts asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._embeddings.embed_documents(sentences))

    async def chunk(self, text: str) -> List[str]:
        """Abstract chunk method to be overridden by subclasses."""
        raise NotImplementedError


class SemanticChunker(BaseChunker):
    """Wrapper around LangChain's SemanticChunker using HFInferenceEmbeddings."""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", hf_token: str | None = None, min_chunk_size: int = 150, percentile_threshold: float = 20.0, min_words: int = 100, embeddings: Embeddings | None = None, **kwargs):
        super().__init__(model_name=model_name, hf_token=hf_token, min_words=min_words, embeddings=embeddings)
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
        if self.min_words > 0:
            chunks = [c for c in chunks if len(c.split()) >= self.min_words]
        return chunks


class RecursiveChunker(BaseChunker):
    """Chunks document text recursively based on word count, ensuring each chunk is at least min_chunk_size words."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", hf_token: str | None = None, min_chunk_size: int = 500, chunk_overlap: int = 50, min_words: int = 100, embeddings: Embeddings | None = None, **kwargs):
        super().__init__(model_name=model_name, hf_token=hf_token, min_words=min_words, embeddings=embeddings)
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
                
        if self.min_words > 0:
            merged_chunks = [c for c in merged_chunks if len(c.split()) >= self.min_words]
        return merged_chunks


class MarkdownStructuralChunker(BaseChunker):
    """Chunks document text using markdown headers and falls back to recursive splitting."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", hf_token: str | None = None, min_chunk_size: int = 350, chunk_overlap: int = 50, min_words: int = 100, embeddings: Embeddings | None = None, **kwargs):
        super().__init__(model_name=model_name, hf_token=hf_token, min_words=min_words, embeddings=embeddings)
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap

    async def chunk(self, text: str) -> List[str]:
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
        
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        
        unwanted_headers = {
            "references", "bibliography", "sources", "further reading", "related articles", 
            "see also", "external links", "navigation", "table of contents", "footer", 
            "header", "copyright", "privacy policy", "terms of service", "cookie policy", 
            "comments", "advertisements", "newsletter signup", "share buttons", "author bio", 
            "social links", "tags", "categories", "breadcrumbs", "previous article", 
            "previous/next article", "next article", "recent posts", "popular posts", "recommended reading"
        }
        
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        md_header_splits = markdown_splitter.split_text(text)
        
        filtered_splits = []
        for doc in md_header_splits:
            skip = False
            for v in doc.metadata.values():
                if v.lower().strip() in unwanted_headers:
                    skip = True
                    break
            if not skip:
                filtered_splits.append(doc.page_content)

        # Recursive fallback for large chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.min_chunk_size, 
            chunk_overlap=self.chunk_overlap,
            length_function=lambda x: len(x.split()),
        )
        
        loop = asyncio.get_running_loop()
        final_chunks = []
        
        def process_splits(splits):
            res = []
            for split_text in splits:
                word_count = len(split_text.split())
                if word_count > self.min_chunk_size:
                    sub_chunks = text_splitter.split_text(split_text)
                    res.extend([c for c in sub_chunks if len(c.split()) >= 100])
                elif word_count >= 100:
                    res.append(split_text)
            return res
            
        final_chunks = await loop.run_in_executor(None, process_splits, filtered_splits)
        if self.min_words > 0:
            final_chunks = [c for c in final_chunks if len(c.split()) >= self.min_words]
        return final_chunks


def get_chunker(
    strategy: str = "semantic",
    model_name: str = "BAAI/bge-base-en-v1.5",
    hf_token: str | None = None,
    embeddings: Embeddings | None = None,
    **kwargs
) -> BaseChunker:
    """Factory function to instantiate the requested chunker strategy."""
    min_words = kwargs.get("min_words", 100)
    kw = kwargs.copy()
    kw.pop("min_words", None)
    kw.pop("embeddings", None)
    kw.pop("model_name", None)
    kw.pop("hf_token", None)
    
    if strategy == "recursive":
        min_sz = kw.pop("min_chunk_size", None)
        if min_sz is None:
            min_sz = kw.pop("chunk_size", 500)
        overlap = kw.pop("chunk_overlap", 50)
        return RecursiveChunker(
            model_name=model_name,
            hf_token=hf_token,
            min_chunk_size=min_sz,
            chunk_overlap=overlap,
            min_words=min_words,
            embeddings=embeddings,
            **kw
        )
    elif strategy == "semantic":
        min_sz = kw.pop("min_chunk_size", 150)
        pct = kw.pop("percentile_threshold", 20.0)
        return SemanticChunker(
            model_name=model_name,
            hf_token=hf_token,
            min_chunk_size=min_sz,
            percentile_threshold=pct,
            min_words=min_words,
            embeddings=embeddings,
            **kw
        )
    elif strategy == "structural":
        min_sz = kw.pop("min_chunk_size", 350)
        overlap = kw.pop("chunk_overlap", 50)
        return MarkdownStructuralChunker(
            model_name=model_name,
            hf_token=hf_token,
            min_chunk_size=min_sz,
            chunk_overlap=overlap,
            min_words=min_words,
            embeddings=embeddings,
            **kw
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


async def chunk_store_and_retrieve(
    original_query: str = "",
    queries_by_objective: Dict[str, str] | None = None,
    contents_by_objective: Dict[str, List[Any]] | None = None,
    index_name: str = "columbus-research",
    model_name: str = "BAAI/bge-base-en-v1.5",
    hf_token: str | None = None,
    top_k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    chunking_strategy: str = "structural",
    min_chunk_size: int | None = None,
    chunk_overlap: int = 50,
    percentile_threshold: float = 20.0,
    **kwargs
) -> Dict[str, List[Dict[str, Any]]]:
    """Chunks documents using the pluggable strategy and retrieves top-k chunks segregated by query (local in-memory)."""
    import logging
    logger = logging.getLogger(__name__)

    # Support backward compatibility parameters
    if queries_by_objective is None:
        queries_by_objective = {}
    if contents_by_objective is None:
        contents_by_objective = {}

    supporting_query = kwargs.get("supporting_query")
    opposing_query = kwargs.get("opposing_query")
    supporting_contents = kwargs.get("supporting_contents")
    opposing_contents = kwargs.get("opposing_contents")

    if supporting_query and "supporting" not in queries_by_objective:
        queries_by_objective["supporting"] = supporting_query
    if opposing_query and "opposing" not in queries_by_objective:
        queries_by_objective["opposing"] = opposing_query
    if supporting_contents and "supporting" not in contents_by_objective:
        contents_by_objective["supporting"] = supporting_contents
    if opposing_contents and "opposing" not in contents_by_objective:
        contents_by_objective["opposing"] = opposing_contents

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

    chunker = get_chunker(
        strategy=chunking_strategy,
        hf_token=hf_token,
        min_chunk_size=resolved_min_chunk_size,
        chunk_overlap=chunk_overlap,
        percentile_threshold=percentile_threshold,
        **kwargs
    )

    logger.info("Phase 1: Semantic Chunking")
    # 2. Concurrently chunk all page content
    chunk_tasks = []
    chunk_task_metadata = [] # To map results back to objective_id
    
    for objective_id, contents in contents_by_objective.items():
        for item in contents:
            if isinstance(item, dict):
                content = item.get("content", "")
                url = item.get("url", "")
            else:
                content = item
                url = ""
            chunk_tasks.append(chunker.chunk(content))
            chunk_task_metadata.append({"objective_id": objective_id, "url": url})
            
    if not chunk_tasks:
        raise PipelineEarlyExit(
            reason=EarlyExitReason.NO_CHUNKS_GENERATED,
            value=0,
            message="No content was available to chunk — all objectives had empty page content."
        )
            
    chunk_results = await asyncio.gather(*chunk_tasks)
    
    # Flatten, clean and map chunks to objective_id and url
    all_chunks = []
    chunk_objective_ids = []
    chunk_urls = []
    
    for i, chunks in enumerate(chunk_results):
        meta = chunk_task_metadata[i]
        objective_id = meta["objective_id"]
        url = meta["url"]
        
        cleaned = [clean_chunk_text(c) for c in chunks]
        # Filter out chunks less than 100 words
        cleaned = [c for c in cleaned if len(c.split()) >= 100]
        
        all_chunks.extend(cleaned)
        chunk_objective_ids.extend([objective_id] * len(cleaned))
        chunk_urls.extend([url] * len(cleaned))
        
    logger.info(f"Generated {len(all_chunks)} total chunks.")

    if not all_chunks:
        raise PipelineEarlyExit(
            reason=EarlyExitReason.NO_CHUNKS_GENERATED,
            value=0,
            message=(
                "All chunks were discarded after cleaning (< 100 words each). "
                "Page content may be too short or consist entirely of boilerplate."
            )
        )
        
    # 3. Generate embeddings in parallel for all chunks
    chunk_embeddings = []
    # Implementation 1 of Stability Fix: Batch the local embedding generation to avoid massive memory spikes
    batch_size = 100
    for idx in range(0, len(all_chunks), batch_size):
        batch = all_chunks[idx:idx + batch_size]
        batch_embs = await chunker.embed_sentences(batch)
        chunk_embeddings.extend(batch_embs)
    
    # Apply >= 0.95 Deduplication
    unique_chunks = []
    unique_embeddings = []
    unique_objective_ids = []
    unique_urls = []
    duplicate_chunks = []
    duplicate_urls = []
    
    if chunk_embeddings:
        embeddings_np = np.array(chunk_embeddings)
        sim_matrix = cosine_similarity(embeddings_np)
        
        keep_indices = []
        for i in range(len(all_chunks)):
            duplicate = False
            for j in keep_indices:
                if sim_matrix[i, j] >= 0.95:
                    duplicate = True
                    break
            if not duplicate:
                keep_indices.append(i)
                unique_chunks.append(all_chunks[i])
                unique_embeddings.append(chunk_embeddings[i])
                unique_objective_ids.append(chunk_objective_ids[i])
                unique_urls.append(chunk_urls[i])
            else:
                duplicate_chunks.append(all_chunks[i])
                duplicate_urls.append(chunk_urls[i])
                
        logger.info(f"Deduplicated chunks: {len(all_chunks)} -> {len(unique_chunks)}")
        all_chunks = unique_chunks
        chunk_embeddings = unique_embeddings
        chunk_objective_ids = unique_objective_ids
        chunk_urls = unique_urls

        if not all_chunks:
            raise PipelineEarlyExit(
                reason=EarlyExitReason.NO_CHUNKS_AFTER_DEDUP,
                value=len(duplicate_chunks),
                message=(
                    f"All {len(duplicate_chunks)} chunks were near-duplicates (cosine >= 0.95). "
                    "No unique chunks remain after deduplication."
                )
            )

    # Save to evidence folder grouped by URL
    from collections import defaultdict
    
    os.makedirs("evidence/chunks", exist_ok=True)
    chunks_by_url = defaultdict(list)
    for i, chunk in enumerate(all_chunks):
        url = chunk_urls[i] or "unknown_url"
        chunks_by_url[url].append(chunk)
        
    for url, chunks in chunks_by_url.items():
        url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
        with open(f"evidence/chunks/{url_hash}_chunk.md", "w") as f:
            f.write("\n\n---\n\n".join(chunks))
            
    # Save duplicates to evidence folder grouped by URL
    if duplicate_chunks:
        os.makedirs("evidence/duplicate_chunks", exist_ok=True)
        dup_chunks_by_url = defaultdict(list)
        for i, chunk in enumerate(duplicate_chunks):
            url = duplicate_urls[i] or "unknown_url"
            dup_chunks_by_url[url].append(chunk)
            
        for url, chunks in dup_chunks_by_url.items():
            url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
            with open(f"evidence/duplicate_chunks/{url_hash}_chunk.md", "w") as f:
                f.write("\n\n---\n\n".join(chunks))
    
    logger.info("Phase 3: Initializing In-Memory Vector Store")
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_core.documents import Document

    # Wrap the unique deduplicated chunks into LangChain Documents
    unique_documents = [
        Document(
            page_content=chunk,
            metadata={"objective_id": chunk_objective_ids[idx], "url": chunk_urls[idx]}
        )
        for idx, chunk in enumerate(all_chunks)
    ]

    vectorstore = InMemoryVectorStore.from_documents(
        unique_documents,
        embedding=chunker._embeddings
    )
        
    logger.info("Phase 4: Diverse Local Retrieval (MMR)")
    resolved_fetch_k = max(fetch_k, top_k * 2)
    retrieval_matches = {}
    for objective_id, q_text in queries_by_objective.items():
        if not q_text.strip():
            retrieval_matches[objective_id] = []
            continue
            
        logger.info(f"Performing LangChain MMR search for {objective_id}: '{q_text}'")
        retrieved_docs = vectorstore.max_marginal_relevance_search(
            query=q_text,
            k=top_k,
            fetch_k=resolved_fetch_k,
            lambda_mult=lambda_mult,
            filter=lambda doc: doc.metadata.get("objective_id") == objective_id
        )
        
        # Calculate exact similarity scores for output compatibility
        matched_texts = [doc.page_content for doc in retrieved_docs]
        q_emb = await chunker.embed_sentences([q_text])
        matched_embs = await chunker.embed_sentences(matched_texts) if matched_texts else []
        
        mmr_matches = []
        if q_emb and matched_embs:
            q_emb_np = np.array(q_emb[0]).reshape(1, -1)
            for doc, emb in zip(retrieved_docs, matched_embs):
                score = float(cosine_similarity(q_emb_np, np.array(emb).reshape(1, -1))[0][0])
                mmr_matches.append({
                    "id": hashlib.md5(doc.page_content.encode("utf-8")).hexdigest(),
                    "text": doc.page_content,
                    "score": score,
                    "url": doc.metadata.get("url", "")
                })
                
        retrieval_matches[objective_id] = mmr_matches
        
    return retrieval_matches
