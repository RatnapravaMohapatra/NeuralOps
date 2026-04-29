import time
import logging
import os
from typing import TypedDict

from langgraph.graph import StateGraph, END

from agents.log_analyzer import build_log_analyzer
from agents.root_cause import build_root_cause_agent
from agents.fix_agent import build_fix_agent
from agents.tools import (
    generate_incident_id,
    evaluate_confidence,
    infer_service_name,
)
from rag.hybrid_retriever import retrieve_similar
from data.seed_db import save_incident

logger = logging.getLogger(__name__)


# ─────────────────────────────
# STATE
# ─────────────────────────────
class IncidentState(TypedDict):
    raw_input: str
    parsed_data: dict
    rag_results: list
    root_cause_data: dict
    fix_data: dict
    confidence: float
    incident_id: str


# ─────────────────────────────
# 🔥 LOAD API KEY (SAFE)
# ─────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.warning("⚠️ GROQ_API_KEY not set → running in fallback mode")


# ─────────────────────────────
# AGENTS (ALWAYS SAFE)
# ─────────────────────────────
log_analyzer = build_log_analyzer(GROQ_API_KEY)
root_cause_agent = build_root_cause_agent(GROQ_API_KEY)
fix_agent = build_fix_agent(GROQ_API_KEY)


# ─────────────────────────────
# NODES
# ─────────────────────────────
def node_parse_logs(state: IncidentState):
    try:
        parsed = log_analyzer({"log_input": state["raw_input"]})
    except Exception as e:
        logger.error("Parse failed: %s", e)
        parsed = {
            "error_type": "UnknownError",
            "service_name": "core-platform-service",
            "severity": "Medium",
            "summary": state["raw_input"][:300],
        }

    # 🔥 FORCE SERVICE (never unknown)
    service = infer_service_name(state["raw_input"], parsed)
    parsed["service_name"] = service

    return {"parsed_data": parsed}


def node_retrieve(state: IncidentState):
    try:
        query = state["raw_input"] + " " + state["parsed_data"].get("summary", "")
        results = retrieve_similar(query, top_k=3)
    except Exception as e:
        logger.error("RAG failed: %s", e)
        results = []

    return {"rag_results": results}


def node_analyze(state: IncidentState):
    try:
        result = root_cause_agent(
            {
                **state["parsed_data"],
                "raw_input": state["raw_input"],
            },
            state["rag_results"],
        )
    except Exception as e:
        logger.error("Root cause failed: %s", e)
        result = {
            "root_cause": "Service degradation due to system constraints",
            "confidence": 0.4,
            "evidence": "Fallback",
        }

    return {
        "root_cause_data": result,
        "confidence": result.get("confidence", 0.5),
    }


def node_generate_fix(state: IncidentState):
    root = state.get("root_cause_data", {})
    parsed = state.get("parsed_data", {})

    try:
        fix = fix_agent({
            "root_cause": root.get("root_cause", ""),
            "service_name": parsed.get("service_name", ""),
            "severity": parsed.get("severity", ""),
        })
    except Exception as e:
        logger.error("Fix generation failed: %s", e)
        fix = {
            "fix_summary": "Restart service and check logs",
        }

    return {"fix_data": fix}


# ─────────────────────────────
# GRAPH
# ─────────────────────────────
def build_graph():
    graph = StateGraph(IncidentState)

    graph.add_node("parse_logs", node_parse_logs)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("analyze", node_analyze)
    graph.add_node("generate_fix", node_generate_fix)

    graph.set_entry_point("parse_logs")

    graph.add_edge("parse_logs", "retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", "generate_fix")
    graph.add_edge("generate_fix", END)

    return graph.compile()


_graph = build_graph()


# ─────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────
def run_incident_pipeline(log_input: str) -> dict:
    start = time.time()

    incident_id = generate_incident_id(log_input)

    state: IncidentState = {
        "raw_input": log_input,
        "parsed_data": {},
        "rag_results": [],
        "root_cause_data": {},
        "fix_data": {},
        "confidence": 0.0,
        "incident_id": incident_id,
    }

    try:
        final_state = _graph.invoke(state)
    except Exception as e:
        logger.exception("Pipeline execution failed")

        return {
            "incident_id": incident_id,
            "root_cause": "Pipeline execution failure",
            "confidence": 0.3,
            "severity": "Medium",
            "service_name": "core-platform-service",
            "fix_suggestion": "Check system logs and deployment",
            "evaluation": "Low",
            "latency": round(time.time() - start, 3),
        }

    root = final_state.get("root_cause_data", {})
    fix = final_state.get("fix_data", {})
    parsed = final_state.get("parsed_data", {})

    latency = round(time.time() - start, 3)

    result = {
        "incident_id": incident_id,
        "root_cause": root.get("root_cause", "No root cause identified"),
        "confidence": round(root.get("confidence", 0.5), 3),
        "severity": parsed.get("severity", "Medium"),
        "service_name": parsed.get("service_name", "core-platform-service"),
        "fix_suggestion": fix.get("fix_summary", "Apply standard remediation"),
        "evaluation": evaluate_confidence(root.get("confidence", 0.5)),
        "latency": latency,
    }

    try:
        save_incident(result)
    except Exception as e:
        logger.warning("DB save failed: %s", e)

    return result
