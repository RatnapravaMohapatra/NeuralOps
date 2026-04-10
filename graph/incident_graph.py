import os
import logging
from typing import TypedDict, Any

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from agents.log_analyzer import build_log_analyzer
from agents.root_cause import build_root_cause_agent
from agents.fix_agent import build_fix_agent
from agents.tools import generate_incident_id, evaluate_confidence, sanitize_log
from rag.hybrid_retriever import retrieve_similar
from data.seed_db import save_incident

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.environ["GROQ_API_KEY"]

log_analyzer = build_log_analyzer(GROQ_API_KEY)
root_cause_agent = build_root_cause_agent(GROQ_API_KEY)
fix_agent = build_fix_agent(GROQ_API_KEY)

CONFIDENCE_THRESHOLD = 0.65


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

class IncidentState(TypedDict):
    raw_input: str
    parsed_data: dict
    rag_results: list
    root_cause_data: dict
    fix_data: dict
    confidence: float
    retry_count: int
    incident_id: str
    escalated: bool


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #

def node_parse_logs(state: IncidentState) -> dict:
    log = sanitize_log(state["raw_input"])
    parsed = log_analyzer(log)
    logger.info("node_parse_logs: error_type=%s service=%s",
                parsed.get("error_type"), parsed.get("service_name"))
    return {"parsed_data": parsed}


def node_retrieve(state: IncidentState) -> dict:
    # Use raw_input only — avoid double-biasing with LLM-generated summary
    query = state["raw_input"]
    results = retrieve_similar(query, top_k=3)
    logger.info("node_retrieve: %d similar incidents found", len(results))
    return {"rag_results": results}


def node_analyze(state: IncidentState) -> dict:
    result = root_cause_agent(state["parsed_data"], state["rag_results"])
    conf = result.get("confidence", 0.5)
    logger.info("node_analyze: confidence=%.3f evidence=%s",
                conf, result.get("evidence", "")[:60])
    return {"root_cause_data": result, "confidence": conf}


def node_generate_fix(state: IncidentState) -> dict:
    fix = fix_agent(state["root_cause_data"], state["parsed_data"])
    logger.info("node_generate_fix: done for service=%s",
                state["parsed_data"].get("service_name"))
    return {"fix_data": fix}


def node_retry(state: IncidentState) -> dict:
    count = state.get("retry_count", 0) + 1
    logger.warning("node_retry: attempt %d — re-running root cause analysis", count)
    result = root_cause_agent(state["parsed_data"], state["rag_results"])
    conf = result.get("confidence", 0.5)
    return {"root_cause_data": result, "confidence": conf, "retry_count": count}


def node_escalate(state: IncidentState) -> dict:
    logger.warning("node_escalate: confidence below threshold after retries — escalating to human review")
    return {"escalated": True}


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #

def route_after_analyze(state: IncidentState) -> str:
    conf = state.get("confidence", 0.0)
    retries = state.get("retry_count", 0)
    if conf >= CONFIDENCE_THRESHOLD:
        return "generate_fix"
    elif retries < 2:
        return "retry"
    else:
        return "escalate"


def route_after_retry(state: IncidentState) -> str:
    conf = state.get("confidence", 0.0)
    retries = state.get("retry_count", 0)
    if conf >= CONFIDENCE_THRESHOLD:
        return "generate_fix"
    elif retries < 2:
        return "retry"
    else:
        return "escalate"


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #

def build_graph() -> Any:
    graph = StateGraph(IncidentState)

    graph.add_node("parse_logs", node_parse_logs)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("analyze", node_analyze)
    graph.add_node("generate_fix", node_generate_fix)
    graph.add_node("retry", node_retry)
    graph.add_node("escalate", node_escalate)

    graph.set_entry_point("parse_logs")
    graph.add_edge("parse_logs", "retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_conditional_edges("analyze", route_after_analyze, {
        "generate_fix": "generate_fix",
        "retry": "retry",
        "escalate": "escalate",
    })
    graph.add_conditional_edges("retry", route_after_retry, {
        "generate_fix": "generate_fix",
        "retry": "retry",
        "escalate": "escalate",
    })
    graph.add_edge("generate_fix", END)
    graph.add_edge("escalate", END)

    return graph.compile()


_graph = build_graph()


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #

async def run_incident_pipeline(log_input: str) -> dict:
    incident_id = generate_incident_id(log_input)

    initial_state: IncidentState = {
        "raw_input": log_input,
        "parsed_data": {},
        "rag_results": [],
        "root_cause_data": {},
        "fix_data": {},
        "confidence": 0.0,
        "retry_count": 0,
        "incident_id": incident_id,
        "escalated": False,
    }

    final_state = _graph.invoke(initial_state)

    parsed = final_state.get("parsed_data", {})
    root = final_state.get("root_cause_data", {})
    fix = final_state.get("fix_data", {})
    conf = final_state.get("confidence", 0.0)
    escalated = final_state.get("escalated", False)
    rag_results = final_state.get("rag_results", [])

    # Build structured fix output
    fix_parts = []
    if fix.get("immediate_fix"):
        fix_parts.append(f"Immediate: {fix['immediate_fix']}")
    if fix.get("short_term_fix"):
        fix_parts.append(f"Short-term: {fix['short_term_fix']}")
    if fix.get("long_term_fix"):
        fix_parts.append(f"Long-term: {fix['long_term_fix']}")

    fix_suggestion = "\n".join(fix_parts) if fix_parts else (
        "Confidence too low for automated fix. Escalated for manual review."
        if escalated else "No fix generated."
    )

    # Build similar incidents summary for UI display
    similar_incidents = [
        {
            "service": r.get("service_name", "unknown"),
            "root_cause": r.get("root_cause", ""),
            "score": r.get("boosted_score", r.get("bm25_score", 0)),
        }
        for r in rag_results
    ]

    result = {
        "incident_id": incident_id,
        "root_cause": root.get("root_cause", "Could not determine root cause."),
        "evidence": root.get("evidence", ""),
        "fix_suggestion": fix_suggestion,
        "fix_summary": fix.get("fix_summary", ""),
        "confidence": conf,
        "severity": parsed.get("severity", "Unknown"),
        "service_name": parsed.get("service_name", "unknown"),
        "evaluation": evaluate_confidence(conf),
        "similar_incidents": similar_incidents,
        "latency": 0.0,
        "raw_input": log_input,
    }

    save_incident(result)
    return result
