import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from langchain_core.callbacks import CallbackManager

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
# 🔥 RULE-BASED SMART FIX ENGINE
# ─────────────────────────────
def _rule_based_fix(data: dict):
    rc = (data.get("root_cause") or "").lower()
    service = (data.get("service_name") or "").lower()

    # Kubernetes CPU issue
    if "insufficient cpu" in rc or "cpu" in rc:
        return {
            "immediate_fix": "kubectl scale nodepool or reduce pod CPU requests",
            "short_term_fix": "Adjust resource requests/limits and reschedule pending pods",
            "long_term_fix": "Enable cluster autoscaler and optimize resource allocation",
            "fix_summary": "Increase cluster capacity and optimize CPU allocation",
        }

    # Memory issue
    if "memory" in rc or "oom" in rc:
        return {
            "immediate_fix": "Restart pods and increase memory limits",
            "short_term_fix": "Analyze memory usage and identify leaks",
            "long_term_fix": "Optimize memory usage and enable autoscaling",
            "fix_summary": "Resolve memory pressure and stabilize workloads",
        }

    # Timeout / network issue
    if "timeout" in rc or "latency" in rc:
        return {
            "immediate_fix": "Retry failed requests and increase timeout settings",
            "short_term_fix": "Check downstream service health and latency",
            "long_term_fix": "Implement retries, circuit breakers, and observability",
            "fix_summary": "Stabilize network communication and reduce latency",
        }

    # Database issue
    if "database" in rc or "sql" in rc:
        return {
            "immediate_fix": "Restart DB connection pool and check queries",
            "short_term_fix": "Optimize slow queries and connection usage",
            "long_term_fix": "Implement indexing and DB monitoring",
            "fix_summary": "Improve database performance and reliability",
        }

    # Auth issue
    if "auth" in rc or "token" in rc:
        return {
            "immediate_fix": "Validate authentication tokens and restart auth service",
            "short_term_fix": "Check token expiry and auth configuration",
            "long_term_fix": "Implement centralized auth monitoring and retries",
            "fix_summary": "Stabilize authentication flow and token handling",
        }

    # Default intelligent fallback (NEVER weak)
    return {
        "immediate_fix": "Restart affected service or pod",
        "short_term_fix": "Check logs, dependencies, and resource usage",
        "long_term_fix": "Improve monitoring, alerting, and autoscaling",
        "fix_summary": "Apply standard remediation to stabilize the system",
    }


# ─────────────────────────────
# MAIN AGENT BUILDER
# ─────────────────────────────
def build_fix_agent(api_key: str | None):

    # 🔥 No API key → always use smart rule-based engine
    if not api_key:
        logger.warning("GROQ_API_KEY missing → using rule-based fix agent")
        return _rule_based_fix

    # 🔥 LLM with LangSmith tracing
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.1,
        callbacks=CallbackManager(),  # 🔥 REQUIRED for LangSmith
    )

    chain = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ]) | llm | JsonOutputParser()

    def generate_fix(data: dict):
        root_cause = (data.get("root_cause") or "")[:500]
        service = data.get("service_name") or "core-platform-service"
        severity = data.get("severity") or "Medium"

        try:
            result = chain.invoke({
                "root_cause": root_cause,
                "service_name": service,
                "severity": severity,
            })

        except Exception as e:
            logger.error("FixAgent LLM failed: %s", e)
            return _rule_based_fix(data)

        if not isinstance(result, dict):
            return _rule_based_fix(data)

        fallback = _rule_based_fix(data)

        return {
            "immediate_fix": result.get("immediate_fix") or fallback["immediate_fix"],
            "short_term_fix": result.get("short_term_fix") or fallback["short_term_fix"],
            "long_term_fix": result.get("long_term_fix") or fallback["long_term_fix"],
            "fix_summary": result.get("fix_summary") or fallback["fix_summary"],
        }

    return generate_fix
