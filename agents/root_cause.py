"""
Root Cause Agent — production-grade version with:
- Evidence-based reasoning
- Keyword intelligence layer (NEW 🔥)
- Retry-aware prompting
- Safe fallback handling
"""

import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONF_MIN = 0.15
CONF_MAX = 0.95
CONF_DEFAULT = 0.50


# ─────────────────────────────
# 🔥 Keyword Intelligence Layer (NEW)
# ─────────────────────────────
KEYWORD_PATTERNS = {
    "timeout": {
        "cause": "downstream service latency or network timeout",
        "confidence": 0.6,
    },
    "timed out": {
        "cause": "service latency or blocking operation",
        "confidence": 0.65,
    },
    "connection refused": {
        "cause": "target service is down or unreachable",
        "confidence": 0.85,
    },
    "outofmemory": {
        "cause": "memory exhaustion due to high usage or leak",
        "confidence": 0.9,
    },
    "heap": {
        "cause": "JVM heap memory exhausted",
        "confidence": 0.9,
    },
    "nullpointerexception": {
        "cause": "null reference bug in application code",
        "confidence": 0.8,
    },
}


def _keyword_override(parsed: dict):
    text = (parsed.get("summary", "") + " " + parsed.get("error_type", "")).lower()

    for key, val in KEYWORD_PATTERNS.items():
        if key in text:
            return {
                "root_cause": (
                    f"The system failed due to {val['cause']}. "
                    f"This is evidenced by '{key}' in the log. "
                    f"This typically occurs when system resources are constrained or services are slow/unavailable."
                ),
                "evidence": key,
                "confidence": val["confidence"],
                "reasoning": f"Keyword '{key}' matched known failure pattern",
            }
    return None


# ─────────────────────────────
# Prompts
# ─────────────────────────────
SYSTEM_PROMPT = """
You are a senior Site Reliability Engineer.

TASK:
Identify the most likely root cause of the issue.

RULES:
- Prioritize signals from log
- If weak signal → infer using common patterns:
  timeout, memory, connection, network, config
- NEVER return "Unknown cause"

OUTPUT JSON:
{
  "root_cause": "...",
  "evidence": "...",
  "confidence": 0.0,
  "reasoning": "..."
}
"""

USER_PROMPT = """
Log Summary:
{summary}

Error Type: {error_type}
Service: {service_name}
Severity: {severity}

Similar Incidents:
{rag_context}
"""


# ─────────────────────────────
# Helpers
# ─────────────────────────────
def _clamp_confidence(val):
    try:
        val = float(val)
    except:
        return CONF_DEFAULT

    if val > 1:
        val = val / 100

    return round(max(CONF_MIN, min(CONF_MAX, val)), 3)


def _safe_result(result):
    if not isinstance(result, dict):
        result = {}

    return {
        "root_cause": result.get("root_cause")
        or "Likely system issue due to insufficient log detail, possibly resource or configuration related.",
        "evidence": result.get("evidence") or "No strong signal found",
        "confidence": _clamp_confidence(result.get("confidence", CONF_DEFAULT)),
        "reasoning": result.get("reasoning") or "Fallback reasoning applied",
    }


# ─────────────────────────────
# Main Builder
# ─────────────────────────────
def build_root_cause_agent(api_key: str):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0,
    )

    chain = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ]) | llm | JsonOutputParser()

    def analyze(parsed, rag_results, retry_count=0, prev_confidence=0.0):
        # 🔥 1. Keyword override (FAST PATH)
        override = _keyword_override(parsed)
        if override:
            logger.info("Keyword override triggered")
            return override

        # ─────────────────────────────
        # 2. Prepare RAG context
        # ─────────────────────────────
        rag_context = "\n".join(
            f"- {r.get('root_cause', '')}" for r in rag_results
        ) or "No similar incidents found."

        try:
            result = chain.invoke({
                "summary": parsed.get("summary", ""),
                "error_type": parsed.get("error_type", ""),
                "service_name": parsed.get("service_name", ""),
                "severity": parsed.get("severity", ""),
                "rag_context": rag_context,
            })

        except Exception as e:
            logger.error("Root cause LLM failed: %s", e)
            return {
                "root_cause": "System failure — unable to analyze log.",
                "evidence": "LLM failure",
                "confidence": CONF_MIN,
                "reasoning": "Pipeline failure",
            }

        return _safe_result(result)

    return analyze
