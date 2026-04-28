import os
import time
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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

CONF_THRESHOLD = 0.65
CONF_MIN = 0.15
MAX_RETRIES = 2
MIN_INPUT_LENGTH = 10

# ── Agents ─────────────────────────────
log_analyzer = build_log_analyzer(GROQ_API_KEY)
root_cause_agent = build_root_cause_agent(GROQ_API_KEY)
fix_agent = build_fix_agent(GROQ_API_KEY)


# ── State ─────────────────────────────
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


# ── Nodes ─────────────────────────────
def node_parse_logs(state: IncidentState):
    try:
        log = sanitize_log(state["raw_input"])
        parsed = log_analyzer(log)
    except Exception as e:
        logger.error("parse_logs failed: %s", e)
        parsed = {
            "error_type": "UnknownError",
            "service_name": "unknown",
            "severity": "Medium",
            "summary": state["raw_input"][:200],
        }
    return {"parsed_data": parsed}


def node_retrieve(state: IncidentState):
    try:
        retry = state.get("retry_count", 0)
        top_k = 5 if retry > 0 else 3

        query = state["raw_input"] + " " + state["parsed_data"].get("summary", "")
        results = retrieve_similar(query, top_k=top_k)
    except Exception as e:
        logger.error("retrieve failed: %s", e)
        results = []

    return {"rag_results": results}


def node_analyze(state: IncidentState):
    try:
        retry = state.get("retry_count", 0)

        result = root_cause_agent(
            state["parsed_data"],
            state["rag_results"],
            retry_count=retry,
            prev_confidence=state.get("confidence", 0.0),
        )

    except Exception as e:
        logger.error("analyze failed: %s", e)
        result = {
            "root_cause": "Unknown cause",
            "confidence": CONF_MIN,
            "reasoning": "fallback",
        }

    conf = result.get("confidence", CONF_MIN)
    conf = max(CONF_MIN, min(conf, 0.95))

    return {"root_cause_data": result, "confidence": conf}


def node_generate_fix(state: IncidentState):
    try:
        fix = fix_agent(state["root_cause_data"], state["parsed_data"])
    except Exception as e:
        logger.error("fix failed: %s", e)
        fix = {
            "immediate_fix": "Restart service",
            "short_term_fix": "Check logs",
            "long_term_fix": "Improve monitoring",
            "fix_summary": "Fallback fix",
        }

    return {"fix_data": fix}


def node_retry(state: IncidentState):
    return {"retry_count": state.get("retry_count", 0) + 1}


def node_escalate(state: IncidentState):
    return {"escalated": True}


# ── Routing ─────────────────────────────
def route(state: IncidentState):
    conf = state.get("confidence", 0.0)
    retries = state.get("retry_count", 0)

    if conf >= CONF_THRESHOLD:
        return "generate_fix"
    elif retries < MAX_RETRIES:
        return "retry"
    else:
        return "escalate"


# ── Graph ─────────────────────────────
def build_graph():
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

    graph.add_conditional_edges("analyze", route, {
        "generate_fix": "generate_fix",
        "retry": "retry",
        "escalate": "escalate",
    })

    graph.add_edge("retry", "analyze")
    graph.add_edge("generate_fix", END)
    graph.add_edge("escalate", END)

    return graph.compile()


_graph = build_graph()


# ── Main Entry ─────────────────────────────
def run_incident_pipeline(log_input: str) -> dict:
    start = time.time()

    if not log_input or len(log_input.strip()) < MIN_INPUT_LENGTH:
        raise ValueError("Invalid log input")

    incident_id = generate_incident_id(log_input)

    state: IncidentState = {
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

    try:
        final_state = _graph.invoke(state)
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        return {
            "incident_id": incident_id,
            "root_cause": "Pipeline failure",
            "confidence": CONF_MIN,
            "latency": round(time.time() - start, 3),
        }

    latency = round(time.time() - start, 3)

    root = final_state.get("root_cause_data", {})
    fix = final_state.get("fix_data", {})
    conf = final_state.get("confidence", CONF_MIN)

    result = {
        "incident_id": incident_id,
        "root_cause": root.get("root_cause", ""),
        "fix_suggestion": fix.get("fix_summary", ""),
        "confidence": round(conf, 3),
        "severity": final_state.get("parsed_data", {}).get("severity", "Unknown"),
        "service_name": final_state.get("parsed_data", {}).get("service_name", "unknown"),
        "evaluation": evaluate_confidence(conf),
        "latency": latency,
        "raw_input": log_input,
    }

    save_incident(result)
    return result
