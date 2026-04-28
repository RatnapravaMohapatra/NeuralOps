# solution_chain.py
import os
import logging
import time
from typing import Dict, Any, List

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────
MAX_TEXT_LENGTH = 600
RETRY_ATTEMPTS = 2

# ── Helpers ────────────────────────────────────────────
def _safe_str(val: Any, default: str = "") -> str:
    if not val or not isinstance(val, str):
        return default
    return val.strip()

def _format_memory(memory_context: Any) -> str:
    """
    Accepts list[dict] like:
    [{ "root_cause": "...", "fix": "..." }, ...]
    """
    if not memory_context:
        return "None"

    if isinstance(memory_context, list):
        lines = []
        for m in memory_context:
            rc = m.get("root_cause", "")
            fx = m.get("fix", "")
            if rc or fx:
                lines.append(f"- Cause: {rc} | Fix: {fx}")
        return "\n".join(lines)[:800] or "None"

    return "None"

def _safe_output(result: Any) -> Dict[str, str]:
    if not isinstance(result, dict):
        result = {}

    return {
        "immediate_fix": result.get("immediate_fix") or "Restart the affected service and clear transient load.",
        "short_term_fix": result.get("short_term_fix") or "Inspect logs, metrics, and recent deployments for anomalies.",
        "long_term_fix": result.get("long_term_fix") or "Add monitoring, capacity planning, and resilience patterns.",
        "fix_summary": result.get("fix_summary") or "Stabilize service and investigate root cause.",
    }

# ── Builder ────────────────────────────────────────────
def build_solution_chain():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY", ""),
        temperature=0.2,
    )

    SYSTEM_PROMPT = """You are a senior Site Reliability Engineer (SRE).

Generate precise, actionable remediation steps for production incidents.

STRICT RULES:
- Be specific to the error type and service
- NO generic advice like "check logs"
- Include concrete commands, configs, or steps when applicable
- Prioritize immediate stabilization, then recovery, then prevention
- If similar past issues are provided, leverage proven fixes

Return ONLY valid JSON:
{
  "immediate_fix": "...",
  "short_term_fix": "...",
  "long_term_fix": "...",
  "fix_summary": "..."
}
"""

    USER_PROMPT = """
Root Cause:
{root_cause}

Service:
{service_name}

Severity:
{severity}

Past similar issues:
{memory_context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])

    parser = JsonOutputParser()
    chain = prompt | llm | parser

    # ── Runner ─────────────────────────────────────────
    def generate_solution(data: Dict[str, Any]) -> Dict[str, str]:
        start = time.time()

        # 1) Input Safety
        root_cause = _safe_str(data.get("root_cause"), "Unknown issue")[:MAX_TEXT_LENGTH]
        service = _safe_str(data.get("service_name"), "unknown")
        severity = _safe_str(data.get("severity"), "Medium")
        memory_text = _format_memory(data.get("memory_context"))

        payload = {
            "root_cause": root_cause,
            "service_name": service,
            "severity": severity,
            "memory_context": memory_text,
        }

        # 2) Retry Logic
        last_error = None
        for attempt in range(RETRY_ATTEMPTS):
            try:
                result = chain.invoke(payload)
                result = _safe_output(result)
                logger.info("SolutionChain success (attempt=%d, latency=%.2fs)", attempt + 1, time.time() - start)
                return result
            except Exception as e:
                last_error = e
                logger.warning("SolutionChain attempt %d failed: %s", attempt + 1, e)
                time.sleep(0.8)

        # 3) Fallback (never crash)
        logger.error("SolutionChain failed after retries: %s", last_error)
        return {
            "immediate_fix": "Restart the affected service to restore availability and reduce load.",
            "short_term_fix": "Check system metrics (CPU, memory, connections) and recent changes; roll back if needed.",
            "long_term_fix": "Implement monitoring, autoscaling, and resilience patterns (timeouts, retries, circuit breakers).",
            "fix_summary": "Fallback remediation applied due to analysis failure.",
        }

    return generate_solution
