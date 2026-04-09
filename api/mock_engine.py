"""
mock_engine.py — NOT used in production pipeline.
Reserved as a local testing stub only. Do not import in api/main.py or graph/.
"""


def mock_analyze(log_input: str) -> dict:
    raise NotImplementedError(
        "mock_engine is disabled. Use the real LangGraph pipeline via graph/incident_graph.py."
    )
