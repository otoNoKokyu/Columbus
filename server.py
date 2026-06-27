import asyncio
import json
import logging
from typing import AsyncGenerator
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from Columbus.pipeline.factory import create_research_chain
from Columbus.pipeline.config import ResearchPipelineConfig

app = FastAPI(title="Columbus Agent API")

logger = logging.getLogger(__name__)

# Keep a single instance of the chain for the server lifetime
pipeline_chain = create_research_chain(ResearchPipelineConfig())

class ResearchRequest(BaseModel):
    query: str


async def event_generator(query: str) -> AsyncGenerator[str, None]:
    """
    Executes the research pipeline and yields Server-Sent Events (SSE)
    based on the pipeline's progress.
    """
    logger.info("Starting stream for query: %s", query)
    
    # version="v2" is recommended for LangChain >= 0.2
    events_stream = pipeline_chain.astream_events(
        {"query": query},
        version="v2",
    )
    
    try:
        async for event in events_stream:
            event_type = event["event"]
            run_name = event.get("name", "")
            
            # 1. Overview of stages
            if event_type == "on_chain_start" and run_name.startswith("Stage"):
                yield json.dumps({
                    "type": "stage_start",
                    "stage": run_name,
                })
            elif event_type == "on_chain_end" and run_name.startswith("Stage"):
                yield json.dumps({
                    "type": "stage_end",
                    "stage": run_name,
                })
            
            # 2. Detailed stream for Query Decomposition (Rewrite)
            elif event_type == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content"):
                    yield json.dumps({
                        "type": "token",
                        "content": chunk.content,
                        "node": run_name
                    })

    except Exception as e:
        logger.error("Pipeline streaming error: %s", e)
        yield json.dumps({"type": "error", "message": str(e)})


@app.post("/api/research/stream")
async def research_stream(request: ResearchRequest):
    """
    Endpoint to start a research pipeline and stream the logs back via SSE.
    """
    return EventSourceResponse(event_generator(request.query))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Columbus.server:app", host="127.0.0.1", port=8000, reload=True)
