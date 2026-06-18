# ResearchAgent package


def create_research_chain(*args, **kwargs):
    """Lazy import to avoid heavy dependency loading at package import time."""
    from .pipeline.factory import create_research_chain as _create
    return _create(*args, **kwargs)


def __getattr__(name):
    if name == "ResearchPipelineConfig":
        from .pipeline.config import ResearchPipelineConfig
        return ResearchPipelineConfig
    raise AttributeError(f"module 'ResearchAgent' has no attribute {name!r}")


__all__ = [
    "create_research_chain",
    "ResearchPipelineConfig",
]
