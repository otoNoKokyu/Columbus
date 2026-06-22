"""Research pipeline module.

Usage::

    from Columbus.pipeline import create_research_chain, ResearchPipelineConfig

    config = ResearchPipelineConfig()
    chain = create_research_chain(config)
    state = chain.invoke(
        {"query": "your research question"},
        config={"run_name": "ResearchPipeline"},
    )
"""

from .config import ResearchPipelineConfig


def create_research_chain(*args, **kwargs):
    """Lazy import to avoid pulling in langchain at package import time."""
    from .factory import create_research_chain as _create
    return _create(*args, **kwargs)


__all__ = [
    "ResearchPipelineConfig",
    "create_research_chain",
]
