import logging
from chains.log_parsing_chain import build_log_parsing_chain
from chains.enrichment_chain import build_enrichment_chain
from chains.solution_chain import build_solution_chain

logger = logging.getLogger(__name__)


def run_chain_pipeline(log_input: str) -> dict:

    parser = build_log_parsing_chain()
    parsed = parser(log_input)
    logger.info("Parsed: %s", parsed)

    enricher = build_enrichment_chain()
    enriched = enricher(
        parsed.get("summary", ""),
        parsed.get("service_name", ""),
        parsed.get("severity", "")
    )
    logger.info("Enriched: %s", enriched)

    solver = build_solution_chain()
    solution = solver(
        parsed.get("summary", ""),
        parsed.get("service_name", ""),
        parsed.get("severity", "")
    )
    logger.info("Solution: %s", solution)

    return {
        "parsed": parsed,
        "enriched": enriched,
        "solution": solution
    }
