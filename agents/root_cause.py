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
# 🔥 KEYWORD ENGINE
# ─────────────────────────────
KEYWORD_PATTERNS = {
    "timeout": ("downstream service latency or network timeout", 0.6),
    "timed out": ("service latency or blocking operation", 0.65),
    "connection refused": ("target service is down or unreachable", 0.85),
    "outofmemory": ("memory exhaustion due to high usage", 0.9),
    "heap": ("JVM heap memory exhausted", 0.9),

    "insufficient cpu": ("Kubernetes nodes lack sufficient CPU resources", 0.9),
    "crashloopbackoff": ("container repeatedly crashing on startup", 0.85),
    "oomkilled": ("container killed due to memory limit exceeded", 0.95),
    "failedscheduling": ("pod cannot be scheduled due to resource constraints", 0.9),

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
- Be precise and actionable

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


def _boost_confidence(base_conf: float, severity: str, rag_hits: int) -> float:
    boost = 0.0

    if severity == "Critical":
        boost += 0.1
    elif severity == "High":
        boost += 0.05

    if rag_hits > 0:
        boost += 0.05

    return _clamp(base_conf + boost)


def _safe_result(result, severity: str, rag_hits: int):
    if not isinstance(result, dict):
        result = {}

    base_conf = result.get("confidence", CONF_DEFAULT)

    return {
        "root_cause": result.get("root_cause")
        or "System instability detected due to resource or service constraints.",
        "evidence": result.get("evidence") or "Log pattern analysis",
        "confidence": _boost_confidence(base_conf, severity, rag_hits),
        "reasoning": result.get("reasoning") or "Derived from log signals and context",
    }


# ─────────────────────────────
# MAIN AGENT
# ─────────────────────────────
def build_root_cause_agent(api_key: str | None):

    # 🔥 FALLBACK MODE
    if not api_key:
        logger.warning("No API key → keyword-only root cause")

        def fallback(parsed: Dict, rag_results: List):
            override = _keyword_override(parsed)
            if override:
                return override

            return {
                "root_cause": "System instability due to insufficient log evidence",
                "evidence": "No strong signal",
                "confidence": 0.35,
                "reasoning": "Fallback mode (no LLM)",
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

        # 1. Keyword override (strongest signal)
        override = _keyword_override(parsed)
        if override:
            return override

        # 2. Prepare RAG
        rag_hits = len(rag_results)
        rag_context = "\n".join(
            f"- {r.get('root_cause', '')}"
            for r in rag_results
        ) or "No similar incidents found."

        severity = parsed.get("severity", "Medium")

        # 3. LLM inference
        try:
            result = chain.invoke({
                "summary": parsed.get("summary", ""),
                "error_type": parsed.get("error_type", ""),
                "service_name": parsed.get("service_name", ""),
                "severity": severity,
                "rag_context": rag_context,
            })
        except Exception as e:
            logger.error("LLM failed: %s", e)
            return {
                "root_cause": "Service degradation due to infrastructure or dependency issues",
                "evidence": "LLM failure fallback",
                "confidence": 0.4,
                "reasoning": "Fallback due to model error",
            }

        return _safe_result(result, severity, rag_hits)

    return analyze
