"""Multi-provider LangChain LLM factory.

Copied from refactor/summarizer/__init__.py get_langchain_llm().
Supports: bedrock, ollama, genai, openrouter.
"""

import logging
import os
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

logger = logging.getLogger(__name__)

# Load .env from Columbus root
_dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_dotenv_path)


def get_langchain_llm(temperature: float = 0.0, provider: Optional[str] = None):
    """Factory to get the LangChain LLM chat model.

    Provider priority: explicit arg > LLM_PROVIDER env > default 'bedrock'.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "bedrock")).lower()
    logger.info("Initializing LLM provider: %s", provider)

    if provider == "bedrock":
        import boto3
        from langchain_aws import ChatBedrockConverse
        from ..utils.callbacks import TokenMeasurerCallbackHandler

        model = os.getenv("BEDROCK_MODEL", "deepseek.v3-v1:0")
        region_name = os.getenv("AWS_DEFAULT_REGION", "ap-southeast-2")
        profile_name = os.getenv("AWS_PROFILE")

        session = boto3.Session(profile_name=profile_name) if profile_name else boto3.Session()
        client = session.client(service_name="bedrock-runtime", region_name=region_name)

        logger.info("Bedrock LLM: model=%s, region=%s", model, region_name)
        return ChatBedrockConverse(
            model_id=model,
            client=client,
            temperature=temperature,
            callbacks=[TokenMeasurerCallbackHandler()],
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        logger.info("Ollama LLM: model=%s, base_url=%s", model, base_url)
        return ChatOllama(model=model, base_url=base_url, temperature=temperature)

    elif provider == "genai":
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ..utils.callbacks import TokenMeasurerCallbackHandler

        api_key = os.getenv("GOOGLE_GENAI_API_KEY")
        model = os.getenv("GENAI_MODEL", "gemini-2.5-flash")
        logger.info("GenAI LLM: model=%s", model)
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            callbacks=[TokenMeasurerCallbackHandler()],
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
