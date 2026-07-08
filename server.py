import asyncio
import json
from typing import AsyncGenerator
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from Columbus.agents.agents import graph

app = FastAPI(title="Columbus Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from Columbus.utils.logger import logger

class ResearchRequest(BaseModel):
    query: str


async def event_generator(query: str) -> AsyncGenerator[str, None]:
    """
    Executes the research pipeline and yields Server-Sent Events (SSE)
    based on the pipeline's progress.
    """
    logger.info("Starting stream for query: %s", query)
    
    # version="v2" is recommended for LangChain >= 0.2
    events_stream = graph.astream_events(
        {"user_input": query},
        version="v2",
    )
    
    valid_nodes = {"DECOMPOSE", "RESEARCH", "CRITIC", "SYNTHESIZER"}
    active_node = None
    
    try:
        async for event in events_stream:
            event_type = event["event"]
            run_name = event.get("name", "")
            
            # 1. Overview of stages
            if event_type == "on_chain_start" and run_name in valid_nodes:
                active_node = run_name
                yield json.dumps({
                    "type": "stage_start",
                    "stage": run_name,
                })
            elif event_type == "on_chain_end" and run_name in valid_nodes:
                if active_node == run_name:
                    active_node = None
                yield json.dumps({
                    "type": "stage_end",
                    "stage": run_name,
                })
            
            # 2. Detailed stream for Synthesis
            elif event_type == "on_chat_model_stream":
                # Only stream tokens to the frontend if we are in the SYNTHESIZER node
                if active_node == "SYNTHESIZER":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content"):
                        content_val = chunk.content
                        if isinstance(content_val, list):
                            texts = [item.get("text", "") for item in content_val if isinstance(item, dict) and "text" in item]
                            content_val = "".join(texts)
                        elif not isinstance(content_val, str):
                            content_val = str(content_val)
                            
                        yield json.dumps({
                            "type": "token",
                            "content": content_val,
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
