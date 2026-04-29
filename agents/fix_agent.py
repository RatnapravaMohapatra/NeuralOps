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


def build_fix_agent(api_key: str | None):
    # 🔥 SAFE FALLBACK (no crash if key missing)
    if not api_key:
        logger.warning("GROQ_API_KEY missing → using fallback fix agent")

        def fallback(data: dict):
            return {
                "immediate_fix": "Restart affected service or pod",
                "short_term_fix": "Check logs and resource usage",
                "long_term_fix": "Implement monitoring and autoscaling",
                "fix_summary": "Fallback remediation (no AI)",
            }

        return fallback

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
        # 🔥 Input safety
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
            logger.error("FixAgent failed: %s", e)
            return {
                "immediate_fix": "Restart affected service or pod",
                "short_term_fix": "Check logs and resource usage",
                "long_term_fix": "Implement monitoring and autoscaling",
                "fix_summary": "Fallback remediation (LLM failure)",
            }

        # 🔥 Ensure valid output
        if not isinstance(result, dict):
            result = {}

        return {
            "immediate_fix": result.get("immediate_fix") or "Restart affected service",
            "short_term_fix": result.get("short_term_fix") or "Investigate logs and dependencies",
            "long_term_fix": result.get("long_term_fix") or "Improve monitoring and scaling",
            "fix_summary": result.get("fix_summary") or "Apply standard remediation",
        }

    return generate_fix
