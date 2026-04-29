import logging
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

CONF_MIN = 0.15
CONF_MAX = 0.95
CONF_DEFAULT = 0.5


# ─────────────────────────────
# 🔥 KEYWORD ENGINE (KEEP)
# ─────────────────────────────
KEYWORD_PATTERNS = {
    "timeout": ("downstream service latency or network timeout", 0.6),
    "timed out": ("service latency or blocking operation", 0.65),
    "connection refused": ("target service is down or unreachable", 0.85),
    "outofmemory": ("memory exhaustion due to high usage", 0.9),
    "heap": ("JVM heap memory exhausted", 0.9),

    "insufficient cpu": ("Kubernetes nodes lack sufficient CPU resources", 0.85),
    "crashloopbackoff": ("container repeatedly crashing on startup", 0.8),
    "oomkilled": ("container killed due to memory limit exceeded", 0.9),
    "failedscheduling": ("pod cannot be scheduled due to resource constraints", 0.85),

    "429": ("API rate limit exceeded", 0.9),
    "rate limit": ("API quota exceeded", 0.9),

    "nullpointerexception": ("null reference bug in application code", 0.8),
}


def _keyword_override(parsed: Dict) -> Dict | None:
    text = (
        (parsed.get("summary", "") + " " +
         parsed.get("error_type", ""))
        .lower()
    )

    for key, (cause, conf) in KEYWORD_PATTERNS.items():
        if key in text:
            return {
                "root_cause": (
                    f"The system failed due to {cause}. "
                    f"Detected keyword '{key}' in logs."
                ),
                "evidence": key,
                "confidence": conf,
                "reasoning": f"Keyword match: {key}",
            }
    return None


# ─────────────────────────────
# PROMPT
# ─────────────────────────────
SYSTEM_PROMPT = """
You are a senior Site Reliability Engineer.

Identify the most likely root cause.

RULES:
- Use log signals first
- Avoid vague answers
- NEVER say "unknown cause"

Return JSON:
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
# HELPERS
# ─────────────────────────────
def _clamp(val):
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
        or "Likely system issue due to insufficient log detail.",
        "evidence": result.get("evidence") or "No strong signal",
        "confidence": _clamp(result.get("confidence", CONF_DEFAULT)),
        "reasoning": result.get("reasoning") or "Fallback reasoning",
    }


# ─────────────────────────────
# MAIN AGENT
# ─────────────────────────────
def build_root_cause_agent(api_key: str | None):

    # 🔥 SAFE FALLBACK MODE
    if not api_key:
        logger.warning("No API key → root cause using keyword only")

        def fallback(parsed: Dict, rag_results: List):
            override = _keyword_override(parsed)
            if override:
                return override

            return {
                "root_cause": "Insufficient data to determine root cause",
                "evidence": "No signal",
                "confidence": 0.3,
                "reasoning": "Fallback mode",
            }

        return fallback

    # 🔥 LLM MODE
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0,
    )

    chain = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ]) | llm | JsonOutputParser()

    def analyze(parsed: Dict, rag_results: List) -> Dict:

        # 1. Keyword override first
        override = _keyword_override(parsed)
        if override:
            return override

        # 2. RAG context
        rag_context = "\n".join(
            f"- {r.get('root_cause', '')}"
            for r in rag_results
        ) or "No similar incidents found."

        # 3. LLM fallback
        try:
            result = chain.invoke({
                "summary": parsed.get("summary", ""),
                "error_type": parsed.get("error_type", ""),
                "service_name": parsed.get("service_name", ""),
                "severity": parsed.get("severity", ""),
                "rag_context": rag_context,
            })
        except Exception as e:
            logger.error("LLM failed: %s", e)
            return {
                "root_cause": "System failure — unable to analyze log.",
                "evidence": "LLM failure",
                "confidence": CONF_MIN,
                "reasoning": "Pipeline failure",
            }

        return _safe_result(result)

    return analyze
