"""
Standalone chain pipeline — used for testing individual chain stages
outside of the full LangGraph workflow.
"""

import logging
import time
from typing import Dict, Any

from chains.log_parsing_chain import build_log_parsing_chain
from chains.enrichment_chain import build_enrichment_chain
from chains.solution_chain import build_solution_chain

# 🔥 Optional (if you added these)
from memory.self_learning import retrieve_memory, store_memory

logger = logging.getLogger(__name__)

MIN_INPUT_LENGTH = 10


# ── Helpers ────────────────────────────────────────────
def _safe_dict(obj: Any) -> Dict:
    return obj if isinstance(obj, dict) else {}


# ── Main Pipeline ──────────────────────────────────────
def run_chain_pipeline(log_input: str) -> dict:
    start = time.time()

    # ─────────────────────────────
    # 1. Input Validation
    # ─────────────────────────────
    if not log_input or len(log_input.strip()) < MIN_INPUT_LENGTH:
        raise ValueError("Invalid log input")

    log_input = log_input.strip()

    # ─────────────────────────────
    # 2. Parse Logs
    # ─────────────────────────────
    parser = build_log_parsing_chain()

    try:
        parsed = parser({"log_input": log_input})  # ⚠️ not .invoke
    except Exception as e:
        logger.error("Parsing failed: %s", e)
        parsed = {
            "error_type": "UnknownError",
            "service_name": "unknown",
            "severity": "Medium",
            "summary": log_input[:200],
        }

    parsed = _safe_dict(parsed)
    logger.info("Parsed: %s", parsed)

    # ─────────────────────────────
    # 3. Enrichment
    # ─────────────────────────────
    enricher = build_enrichment_chain()

    try:
        enriched = enricher({
            "summary": parsed.get("summary", ""),
            "service_name": parsed.get("service_name", ""),
            "severity": parsed.get("severity", ""),
        })
    except Exception as e:
        logger.error("Enrichment failed: %s", e)
        enriched = {
            "affected_components": [],
            "business_impact": "Unknown",
            "urgency_score": 5,
        }

    enriched = _safe_dict(enriched)
    logger.info("Enriched: %s", enriched)

    # ─────────────────────────────
    # 4. Memory Retrieval (🔥 learning)
    # ─────────────────────────────
    try:
        memory_context = retrieve_memory(log_input)
    except Exception as e:
        logger.warning("Memory retrieval failed: %s", e)
        memory_context = []

    # ─────────────────────────────
    # 5. Solution Generation
    # ─────────────────────────────
    solver = build_solution_chain()

    try:
        solution = solver({
            "root_cause": parsed.get("summary", ""),
            "service_name": parsed.get("service_name", ""),
            "severity": parsed.get("severity", ""),
            "memory_context": memory_context,
        })
    except Exception as e:
        logger.error("Solution failed: %s", e)
        solution = {
            "immediate_fix": "Restart service",
            "short_term_fix": "Check logs and metrics",
            "long_term_fix": "Improve monitoring",
            "fix_summary": "Fallback solution",
        }

    solution = _safe_dict(solution)
    logger.info("Solution: %s", solution)

    # ─────────────────────────────
    # 6. Confidence (simple logic)
    # ─────────────────────────────
    confidence = 0.5
    if memory_context:
        confidence = min(confidence + 0.1, 0.9)

    # ─────────────────────────────
    # 7. Latency
    # ─────────────────────────────
    latency = round(time.time() - start, 3)

    result = {
        "parsed": parsed,
        "enriched": enriched,
        "solution": solution,
        "confidence": confidence,
        "latency": latency,
    }

    # ─────────────────────────────
    # 8. Store Learning Data
    # ─────────────────────────────
    try:
        store_memory({
            "incident_id": f"CHAIN-{int(time.time())}",
            "raw_input": log_input,
            "root_cause": parsed.get("summary", ""),
            "fix_summary": solution.get("fix_summary", ""),
            "confidence": confidence,
        })
    except Exception as e:
        logger.warning("Memory store failed: %s", e)

    return result
