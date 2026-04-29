import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior SRE fix generator.

Return JSON:
{
  "immediate_fix": "...",
  "short_term_fix": "...",
  "long_term_fix": "...",
  "fix_summary": "..."
}

Rules:
- No generic advice
- Provide concrete actions (commands, configs, scaling)
"""

USER_PROMPT = """
Root Cause: {root_cause}
Service: {service_name}
Severity: {severity}
"""


# ─────────────────────────────
# 🔥 RULE-BASED FALLBACK (SMART)
# ─────────────────────────────
def _rule_based_fix(data: dict):
    rc = (data.get("root_cause") or "").lower()

    # Kubernetes CPU issue
    if "insufficient cpu" in rc or "cpu" in rc:
        return {
            "immediate_fix": "Scale up Kubernetes node pool or reduce pod CPU requests",
            "short_term_fix": "Adjust resource requests/limits and reschedule pending pods",
            "long_term_fix": "Enable cluster autoscaler and optimize resource allocation",
            "fix_summary": "Increase cluster capacity and optimize CPU allocation",
        }

    # Memory issue
    if "memory" in rc or "oom" in rc:
        return {
            "immediate_fix": "Restart affected pods and increase memory limits",
            "short_term_fix": "Analyze memory usage and fix leaks",
            "long_term_fix": "Optimize application memory usage and enable autoscaling",
            "fix_summary": "Resolve memory pressure and optimize resource usage",
        }

    # Timeout / network issue
    if "timeout" in rc:
        return {
            "immediate_fix": "Retry failed requests and increase timeout thresholds",
            "short_term_fix": "Check downstream service latency and availability",
            "long_term_fix": "Implement retries, circuit breakers, and monitoring",
            "fix_summary": "Stabilize network calls and downstream service latency",
        }

    # Default intelligent fallback
    return {
        "immediate_fix": "Restart affected service or pod",
        "short_term_fix": "Check logs and system dependencies",
        "long_term_fix": "Improve monitoring, alerting, and autoscaling",
        "fix_summary": "Apply standard remediation steps",
    }


# ─────────────────────────────
# MAIN AGENT
# ─────────────────────────────
def build_fix_agent(api_key: str | None):

    # 🔥 If no API key → use smart fallback
    if not api_key:
        logger.warning("GROQ_API_KEY missing → using rule-based fix agent")
        return _rule_based_fix

    # 🔥 LLM setup
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.1,
    )

    chain = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ]) | llm | JsonOutputParser()

    def generate_fix(data: dict):
        root_cause = (data.get("root_cause") or "")[:500]
        service = data.get("service_name", "unknown")
        severity = data.get("severity", "Medium")

        try:
            result = chain.invoke({
                "root_cause": root_cause,
                "service_name": service,
                "severity": severity,
            })

        except Exception as e:
            logger.error("FixAgent LLM failed: %s", e)
            # 🔥 fallback to smart rule-based (not generic)
            return _rule_based_fix(data)

        if not isinstance(result, dict):
            return _rule_based_fix(data)

        return {
            "immediate_fix": result.get("immediate_fix") or _rule_based_fix(data)["immediate_fix"],
            "short_term_fix": result.get("short_term_fix") or _rule_based_fix(data)["short_term_fix"],
            "long_term_fix": result.get("long_term_fix") or _rule_based_fix(data)["long_term_fix"],
            "fix_summary": result.get("fix_summary") or _rule_based_fix(data)["fix_summary"],
        }

    return generate_fix
