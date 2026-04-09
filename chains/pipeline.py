"""
Standalone chain pipeline — used for testing individual chain stages
outside of the full LangGraph workflow.
"""
import logging
from chains.log_parsing_chain import build_log_parsing_chain
from chains.enrichment_chain import build_enrichment_chain
from chains.solution_chain import build_solution_chain

logger = logging.getLogger(__name__)


def run_chain_pipeline(log_input: str) -> dict:
    parser = build_log_parsing_chain()
    parsed = parser.invoke({"log_input": log_input})
    logger.info("Parsed: %s", parsed)

    enricher = build_enrichment_chain()
    enriched = enricher.invoke({
        "summary": parsed.get("summary", ""),
        "service_name": parsed.get("service_name", ""),
        "severity": parsed.get("severity", ""),
    })
    logger.info("Enriched: %s", enriched)

    solver = build_solution_chain()
    solution = solver.invoke({
        "root_cause": parsed.get("summary", ""),
        "service_name": parsed.get("service_name", ""),
        "severity": parsed.get("severity", ""),
    })
    logger.info("Solution: %s", solution)

    return {"parsed": parsed, "enriched": enriched, "solution": solution}
