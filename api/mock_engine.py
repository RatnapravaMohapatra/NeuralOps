"""
mock_engine.py — DEVELOPMENT ONLY

⚠️ This module is NOT part of the production pipeline.
Do NOT import in:
- api/main.py
- graph/incident_graph.py

Purpose:
- Local testing
- UI prototyping
- Offline fallback (optional)

Production uses:
→ graph/incident_graph.py (LangGraph pipeline)
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Hard guard to prevent accidental production usage
ALLOW_MOCK = False


def mock_analyze(log_input: str) -> Dict:
    """
    Mock analysis function.

    Raises error by default to prevent misuse in production.
    """

    if not ALLOW_MOCK:
        raise RuntimeError(
            "❌ mock_engine is disabled.\n"
            "Use the real pipeline: graph/incident_graph.py"
        )

    logger.warning("⚠️ Using mock_engine (DEV MODE ONLY)")

    # ─────────────────────────────
    # Fake structured response
    # ─────────────────────────────
    return {
        "incident_id": "MOCK-123456",
        "root_cause": "Simulated database connection pool exhaustion",
        "confidence": 0.75,
        "severity": "High",
        "service_name": "mock-service",
        "evaluation": "Medium",
        "fix_suggestion": "Increase DB pool size and monitor connection usage",
        "latency": 0.01,
    }
