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
logger = logging.getLogger(**name**)

# ================================

# ENV SAFE LOAD

# ================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
raise ValueError("GROQ_API_KEY not set")

log_analyzer = build_log_analyzer(GROQ_API_KEY)
root_cause_agent = build_root_cause_agent(GROQ_API_KEY)
fix_agent = build_fix_agent(GROQ_API_KEY)

# ================================

# STATE

# ================================

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

# ================================

# NODES

# ================================

def node_parse_logs(state: IncidentState) -> dict:
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

def node_retrieve(state: IncidentState) -> dict:
try:
query = state["parsed_data"].get("summary", "")
results = retrieve_similar(query, top_k=3)
except Exception as e:
logger.error("retrieve failed: %s", e)
results = []
return {"rag_results": results}

def node_analyze(state: IncidentState) -> dict:
try:
result = root_cause_agent(state["parsed_data"], state["rag_results"])
conf = evaluate_confidence(result.get("confidence", 0.5), state["parsed_data"])
except Exception as e:
logger.error("analyze failed: %s", e)
result = {
"root_cause": "Analysis failed.",
"confidence": 0.3,
"reasoning": "LLM failure",
}
conf = 0.3

```
return {"root_cause_data": result, "confidence": conf}
```

def node_generate_fix(state: IncidentState) -> dict:
try:
fix = fix_agent(state["root_cause_data"], state["parsed_data"])
except Exception as e:
logger.error("fix generation failed: %s", e)
fix = {}
return {"fix_data": fix}

def node_retry(state: IncidentState) -> dict:
count = state.get("retry_count", 0) + 1
logger.warning("Retry attempt %d", count)

```
try:
    parsed = state["parsed_data"].copy()
    parsed["summary"] += " (Re-evaluate strictly based on log evidence)"

    result = root_cause_agent(parsed, state["rag_results"])
    conf = evaluate_confidence(result.get("confidence", 0.5), parsed)

except Exception as e:
    logger.error("retry failed: %s", e)
    result = {
        "root_cause": "Retry failed.",
        "confidence": 0.3,
        "reasoning": "Retry failure",
    }
    conf = 0.3

return {
    "root_cause_data": result,
    "confidence": conf,
    "retry_count": count
}
```

def node_escalate(state: IncidentState) -> dict:
logger.warning("Escalating incident due to low confidence.")
return {"escalated": True}

# ================================

# ROUTING

# ================================

def route_after_analyze(state: IncidentState) -> str:
if state["confidence"] >= 0.8:
return "generate_fix"
elif state["retry_count"] < 2:
return "retry"
else:
return "escalate"

def route_after_retry(state: IncidentState) -> str:
if state["confidence"] >= 0.8:
return "generate_fix"
elif state["retry_count"] < 2:
return "retry"
else:
return "escalate"

# ================================

# GRAPH

# ================================

def build_graph() -> Any:
graph = StateGraph(IncidentState)

```
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
```

_graph = build_graph()

# ================================

# PIPELINE ENTRYPOINT

# ================================

async def run_incident_pipeline(log_input: str) -> dict:
incident_id = generate_incident_id(log_input)

```
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

final_state = await _graph.ainvoke(initial_state)

parsed = final_state.get("parsed_data", {})
root = final_state.get("root_cause_data", {})
fix = final_state.get("fix_data", {})
conf = final_state.get("confidence", 0.0)
escalated = final_state.get("escalated", False)

# ================================
# FIX FORMATTING
# ================================
fix_parts = []

if fix.get("immediate_fix"):
    fix_parts.append(f"Immediate: {fix['immediate_fix']}")
if fix.get("short_term_fix"):
    fix_parts.append(f"Short-term: {fix['short_term_fix']}")
if fix.get("long_term_fix"):
    fix_parts.append(f"Long-term: {fix['long_term_fix']}")

if escalated:
    fix_suggestion = (
        "Low confidence after multiple attempts. Manual investigation required.\n"
        f"Evidence: {parsed.get('summary', '')}"
    )
else:
    fix_suggestion = "\n".join(fix_parts) if fix_parts else "No fix generated."

result = {
    "incident_id": incident_id,
    "root_cause": root.get("root_cause", "Could not determine root cause."),
    "fix_suggestion": fix_suggestion,
    "confidence": conf,
    "severity": parsed.get("severity", "Unknown"),
    "service_name": parsed.get("service_name", "unknown"),
    "evaluation": evaluate_confidence(conf),
    "latency": 0.0,
    "raw_input": log_input,
}

try:
    save_incident(result)
except Exception as e:
    logger.error("Failed to save incident: %s", e)

return result
```
