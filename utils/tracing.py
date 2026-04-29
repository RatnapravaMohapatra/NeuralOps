# utils/tracing.py

import os
import logging

logger = logging.getLogger(__name__)


def setup_langsmith():
    """
    Initialize LangSmith tracing safely.
    Ensures environment variables are set before any LLM is created.
    """

    # Enable tracing
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

    # Project name (important for dashboard grouping)
    os.environ.setdefault("LANGCHAIN_PROJECT", "NeuralOps")

    # Optional endpoint (safe default)
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    # Debug logs (helps in Render)
    logger.info(
        "LangSmith tracing initialized | enabled=%s | project=%s",
        os.getenv("LANGCHAIN_TRACING_V2"),
        os.getenv("LANGCHAIN_PROJECT"),
    )

    # Optional: warn if API key missing
    if not os.getenv("LANGCHAIN_API_KEY"):
        logger.warning("LANGCHAIN_API_KEY not set → tracing may not work properly")