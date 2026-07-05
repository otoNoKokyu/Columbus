from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import re
import logging
import json
import asyncio

from Columbus.scoring.local_reranker import LocalReranker

logger = logging.getLogger(__name__)

class BaseChunkScorer(ABC):
    """Abstract base class for scoring the relevance of retrieved chunks."""
    
    @abstractmethod
    def score_chunks(self, original_query: str, decomposed_query: str, chunks: List[Dict[str, Any]], callbacks: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """
        Score a list of chunks based on their relevance to the queries.
        
        Args:
            original_query: The main research question from the user.
            decomposed_query: The specific sub-query/perspective these chunks were retrieved for.
            chunks: List of chunk dictionaries containing at least a 'text' key.
            callbacks: Optional LangChain callbacks.
            
        Returns:
            List of chunk dictionaries with updated 'score' or 'rerank_score'.
        """
        pass
        
    @abstractmethod
    async def ascore_chunks(self, original_query: str, decomposed_query: str, chunks: List[Dict[str, Any]], callbacks: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """
        Asynchronously score a list of chunks based on their relevance to the queries.
        """
        pass

class RerankerChunkScorer(BaseChunkScorer):
    """Chunk scorer using the local SentenceTransformers Cross-Encoder API."""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.reranker = LocalReranker(model_name=model_name)
        
    def score_chunks(self, original_query: str, decomposed_query: str, chunks: List[Dict[str, Any]], callbacks: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
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
                top_n=len(chunks),
                callbacks=callbacks
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
 
    async def ascore_chunks(self, original_query: str, decomposed_query: str, chunks: List[Dict[str, Any]], callbacks: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        # SentenceTransformers is CPU/GPU bound, run in a thread pool to avoid blocking the event loop
        return await loop.run_in_executor(None, self.score_chunks, original_query, decomposed_query, chunks, callbacks)

from Columbus.scoring.prompts import BATCH_CHUNK_RELEVANCE_SCORING_PROMPT, CHUNK_RELEVANCE_SCORING_PROMPT
from langchain_core.output_parsers import JsonOutputParser


class LLMChunkScorer(BaseChunkScorer):
    """Chunk scorer using a generative LLM as a relevance judge with batching and native JsonOutputParser support."""
    
    def __init__(self, llm):
        self.llm = llm
        self.parser = JsonOutputParser()

    def score_chunks(self, original_query: str, decomposed_query: str, chunks: List[Dict[str, Any]], callbacks: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
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
                batch_chain = BATCH_CHUNK_RELEVANCE_SCORING_PROMPT | self.llm | self.parser
                result = batch_chain.invoke({
                    "original_query": original_query,
                    "decomposed_query": decomposed_query,
                    "chunks_list": chunks_list_str
                }, config={"callbacks": callbacks} if callbacks else {})
                
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
                    logger.warning(f"Failed to parse batch JSON. Falling back to native LangChain batch scoring for batch {i//batch_size}.")
                    self._fallback_batch_sync(original_query, decomposed_query, batch_chunks, callbacks)
            except Exception as e:
                logger.error(f"Failed to execute batch relevance scoring: {e}. Falling back to native LangChain batch scoring.")
                self._fallback_batch_sync(original_query, decomposed_query, batch_chunks, callbacks)
                    
        # Sort by the new LLM score descending
        scored_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return scored_chunks

    def _fallback_batch_sync(self, original_query: str, decomposed_query: str, batch_chunks: List[Dict[str, Any]], callbacks: Optional[List[Any]]):
        """Uses LangChain's native .batch() to score chunks concurrently using a ThreadPoolExecutor internally."""
        single_chain = CHUNK_RELEVANCE_SCORING_PROMPT | self.llm | self.parser
        
        batch_inputs = [{
            "original_query": original_query,
            "decomposed_query": decomposed_query,
            "chunk_text": chunk.get("text", "")
        } for chunk in batch_chunks]
        
        try:
            results = single_chain.batch(batch_inputs, config={"callbacks": callbacks} if callbacks else {}, return_exceptions=True)
            for chunk, result in zip(batch_chunks, results):
                if isinstance(result, Exception):
                    logger.error(f"Single chunk sync fallback failed: {result}")
                elif result:
                    chunk["score"] = float(result.get("relevance_score", 0.0))
                    chunk["llm_reasoning"] = result.get("reasoning", "")
        except Exception as e:
            logger.error(f"Fallback batch failed completely: {e}")

    async def ascore_chunks(self, original_query: str, decomposed_query: str, chunks: List[Dict[str, Any]], callbacks: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        if not chunks:
            return []
            
        logger.info(f"LLM async batch-scoring {len(chunks)} chunks against original query: '{original_query}'")
        
        batch_size = 10
        batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
        
        # Prepare batch payloads
        batch_inputs = []
        for batch in batches:
            chunks_list_str = ""
            for idx, chunk in enumerate(batch):
                chunks_list_str += f"--- Chunk Index {idx} ---\n{chunk.get('text', '')}\n\n"
            
            batch_inputs.append({
                "original_query": original_query,
                "decomposed_query": decomposed_query,
                "chunks_list": chunks_list_str
            })
            
        # Concurrently score batches using abatch
        try:
            batch_chain = BATCH_CHUNK_RELEVANCE_SCORING_PROMPT | self.llm | self.parser
            results = await batch_chain.abatch(batch_inputs, config={"callbacks": callbacks} if callbacks else {}, return_exceptions=True)
        except Exception as e:
            logger.error(f"abatch failed completely: {e}")
            results = [e] * len(batch_inputs)
        
        # Map evaluations back to chunks
        for i, result in enumerate(results):
            batch_chunks = batches[i]
            if isinstance(result, Exception):
                logger.warning(f"Batch {i} failed during abatch: {result}. Falling back to native LangChain abatch scoring.")
                await self._fallback_batch_async(original_query, decomposed_query, batch_chunks, callbacks)
            elif result and "evaluations" in result:
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
                logger.warning(f"Failed to parse async batch JSON. Falling back to native LangChain abatch scoring for batch {i}.")
                await self._fallback_batch_async(original_query, decomposed_query, batch_chunks, callbacks)
                    
        # Sort by the new LLM score descending
        scored_chunks = list(chunks)
        scored_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return scored_chunks

    async def _fallback_batch_async(self, original_query: str, decomposed_query: str, batch_chunks: List[Dict[str, Any]], callbacks: Optional[List[Any]]):
        """Uses LangChain's native .abatch() to score chunks concurrently via asyncio."""
        single_chain = CHUNK_RELEVANCE_SCORING_PROMPT | self.llm | self.parser
        
        batch_inputs = [{
            "original_query": original_query,
            "decomposed_query": decomposed_query,
            "chunk_text": chunk.get("text", "")
        } for chunk in batch_chunks]
        
        try:
            results = await single_chain.abatch(batch_inputs, config={"callbacks": callbacks} if callbacks else {}, return_exceptions=True)
            for chunk, result in zip(batch_chunks, results):
                if isinstance(result, Exception):
                    logger.error(f"Single chunk async fallback failed: {result}")
                elif result:
                    chunk["score"] = float(result.get("relevance_score", 0.0))
                    chunk["llm_reasoning"] = result.get("reasoning", "")
        except Exception as e:
            logger.error(f"Fallback async batch failed completely: {e}")
