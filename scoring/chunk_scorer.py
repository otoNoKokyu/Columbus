from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import re
import logging
import json

from Columbus.scoring.local_reranker import LocalReranker

logger = logging.getLogger(__name__)

class BaseChunkScorer(ABC):
    """Abstract base class for scoring the relevance of retrieved chunks."""
    
    @abstractmethod
    def score_chunks(self, original_query: str, decomposed_query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score a list of chunks based on their relevance to the queries.
        
        Args:
            original_query: The main research question from the user.
            decomposed_query: The specific sub-query/perspective these chunks were retrieved for.
            chunks: List of chunk dictionaries containing at least a 'text' key.
            
        Returns:
            List of chunk dictionaries with updated 'score' or 'rerank_score'.
        """
        pass
        
    @abstractmethod
    async def ascore_chunks(self, original_query: str, decomposed_query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Asynchronously score a list of chunks based on their relevance to the queries.
        """
        pass

class RerankerChunkScorer(BaseChunkScorer):
    """Chunk scorer using the local SentenceTransformers Cross-Encoder API."""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.reranker = LocalReranker(model_name=model_name)
        
    def score_chunks(self, original_query: str, decomposed_query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return []
            
        # Format for LocalReranker which expects 'context'
        candidates = []
        for c in chunks:
            candidates.append({
                "context": c.get("text", ""),
                "original_chunk": c
            })
            
        # The user requested to score against the original query
        logger.info(f"Reranking {len(chunks)} chunks against original query: '{original_query}'")
        try:
            ranked_candidates = self.reranker.rerank(
                query=original_query,
                candidates=candidates,
                top_n=len(chunks)
            )
            
            # Reconstruct the original chunk dictionaries with the new scores
            scored_chunks = []
            for c in ranked_candidates:
                orig_chunk = c["original_chunk"]
                orig_chunk["score"] = c.get("rerank_score", orig_chunk.get("score", 0.0))
                scored_chunks.append(orig_chunk)
                
            return scored_chunks
            
        except Exception as e:
            logger.error(f"Failed to rerank chunks using PineconeReranker: {e}")
            return chunks

    async def ascore_chunks(self, original_query: str, decomposed_query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        import asyncio
        loop = asyncio.get_running_loop()
        # SentenceTransformers is CPU/GPU bound, run in a thread pool to avoid blocking the event loop
        return await loop.run_in_executor(None, self.score_chunks, original_query, decomposed_query, chunks)

from Columbus.scoring.prompts import BATCH_CHUNK_RELEVANCE_SCORING_PROMPT
from langchain_core.output_parsers import JsonOutputParser


class LLMChunkScorer(BaseChunkScorer):
    """Chunk scorer using a generative LLM as a relevance judge with batching and native JsonOutputParser support."""
    
    def __init__(self, llm):
        self.llm = llm
        self.parser = JsonOutputParser()
        
    def _score_single_chunk_sync(self, original_query: str, decomposed_query: str, chunk: Dict[str, Any]):
        chunk_text = chunk.get("text", "")
        if not chunk_text:
            return
        from Columbus.scoring.prompts import CHUNK_RELEVANCE_SCORING_PROMPT
        try:
            chain = CHUNK_RELEVANCE_SCORING_PROMPT | self.llm | self.parser
            result = chain.invoke({
                "original_query": original_query,
                "decomposed_query": decomposed_query,
                "chunk_text": chunk_text
            })
            if result:
                chunk["score"] = float(result.get("relevance_score", 0.0))
                chunk["llm_reasoning"] = result.get("reasoning", "")
        except Exception as e:
            logger.error(f"Single chunk sync fallback failed: {e}")

    async def _ascore_single_chunk(self, original_query: str, decomposed_query: str, chunk: Dict[str, Any]):
        chunk_text = chunk.get("text", "")
        if not chunk_text:
            return
        from Columbus.scoring.prompts import CHUNK_RELEVANCE_SCORING_PROMPT
        try:
            chain = CHUNK_RELEVANCE_SCORING_PROMPT | self.llm | self.parser
            result = await chain.ainvoke({
                "original_query": original_query,
                "decomposed_query": decomposed_query,
                "chunk_text": chunk_text
            })
            if result:
                chunk["score"] = float(result.get("relevance_score", 0.0))
                chunk["llm_reasoning"] = result.get("reasoning", "")
        except Exception as e:
            logger.error(f"Single chunk async fallback failed: {e}")

    async def _ascore_batch(self, original_query: str, decomposed_query: str, batch_chunks: List[Dict[str, Any]], batch_idx: int):
        chunks_list_str = ""
        for idx, chunk in enumerate(batch_chunks):
            chunks_list_str += f"--- Chunk Index {idx} ---\n{chunk.get('text', '')}\n\n"
            
        try:
            chain = BATCH_CHUNK_RELEVANCE_SCORING_PROMPT | self.llm | self.parser
            result = await chain.ainvoke({
                "original_query": original_query,
                "decomposed_query": decomposed_query,
                "chunks_list": chunks_list_str
            })
            if result and "evaluations" in result:
                evaluations = result["evaluations"]
                for eval_item in evaluations:
                    try:
                        idx = int(eval_item.get("chunk_index", -1))
                        if 0 <= idx < len(batch_chunks):
                            score = float(eval_item.get("relevance_score", 0.0))
                            batch_chunks[idx]["score"] = score
                            batch_chunks[idx]["llm_reasoning"] = eval_item.get("reasoning", "")
                    except Exception as e:
                        logger.error(f"Error mapping async eval item: {e}")
            else:
                logger.warning(f"Failed to parse async batch JSON. Falling back to individual scoring for batch {batch_idx}.")
                for chunk in batch_chunks:
                    await self._ascore_single_chunk(original_query, decomposed_query, chunk)
        except Exception as e:
            logger.error(f"Failed to execute async batch relevance scoring for batch {batch_idx}: {e}. Falling back to individual.")
            for chunk in batch_chunks:
                await self._ascore_single_chunk(original_query, decomposed_query, chunk)

    def score_chunks(self, original_query: str, decomposed_query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return []
            
        logger.info(f"LLM batch-scoring {len(chunks)} chunks against original query: '{original_query}'")
        
        batch_size = 10
        scored_chunks = list(chunks)
        
        # Divide into batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            
            # Format chunks list string
            chunks_list_str = ""
            for idx, chunk in enumerate(batch_chunks):
                chunks_list_str += f"--- Chunk Index {idx} ---\n{chunk.get('text', '')}\n\n"
                
            try:
                chain = BATCH_CHUNK_RELEVANCE_SCORING_PROMPT | self.llm | self.parser
                result = chain.invoke({
                    "original_query": original_query,
                    "decomposed_query": decomposed_query,
                    "chunks_list": chunks_list_str
                })
                if result and "evaluations" in result:
                    evaluations = result["evaluations"]
                    # Map evaluations back
                    for eval_item in evaluations:
                        try:
                            idx = int(eval_item.get("chunk_index", -1))
                            if 0 <= idx < len(batch_chunks):
                                score = float(eval_item.get("relevance_score", 0.0))
                                batch_chunks[idx]["score"] = score
                                batch_chunks[idx]["llm_reasoning"] = eval_item.get("reasoning", "")
                        except Exception as e:
                            logger.error(f"Error mapping eval item: {e}")
                else:
                    logger.warning(f"Failed to parse batch JSON. Falling back to individual scoring for batch {i//batch_size}.")
                    for chunk in batch_chunks:
                        self._score_single_chunk_sync(original_query, decomposed_query, chunk)
            except Exception as e:
                logger.error(f"Failed to execute batch relevance scoring: {e}. Falling back to individual.")
                for chunk in batch_chunks:
                    self._score_single_chunk_sync(original_query, decomposed_query, chunk)
                    
        # Sort by the new LLM score descending
        scored_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return scored_chunks

    async def ascore_chunks(self, original_query: str, decomposed_query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return []
            
        logger.info(f"LLM async batch-scoring {len(chunks)} chunks against original query: '{original_query}'")
        
        batch_size = 10
        tasks = []
        
        # Divide into batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            tasks.append(self._ascore_batch(original_query, decomposed_query, batch_chunks, i//batch_size))
            
        await asyncio.gather(*tasks)
        
        # Sort by the new LLM score descending
        scored_chunks = list(chunks)
        scored_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return scored_chunks
