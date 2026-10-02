from __future__ import annotations

import os
import logging
from langsmith import traceable

logger = logging.getLogger(__name__)

def setup_tracing() -> None:
    """Initialize LangSmith tracing environment variables if they are not already set."""
    if not os.getenv("LANGCHAIN_TRACING_V2"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        logger.info("Enabled LangSmith tracing (LANGCHAIN_TRACING_V2=true)")
    
    if not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = "rag_github"
        logger.info(f"Set LangSmith project to {os.environ['LANGCHAIN_PROJECT']}")

# Export the traceable decorator so other modules can use it
__all__ = ["setup_tracing", "traceable"]
