import os
import logging
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
logger = logging.getLogger(__name__)


# ─────────────────────────────
# SAFE OUTPUT HANDLER
# ─────────────────────────────
def _safe_output(result: Dict[str, Any]) -> Dict[str, str]:
    if not isinstance(result, dict):
        result = {}

    return {
        "immediate_fix": result.get("immediate_fix") or "Restart affected service or pod",
        "short_term_fix": result.get("short_term_fix") or "Check logs, resource usage, and system health",
        "long_term_fix": result.get("long_term_fix") or "Implement monitoring, autoscaling, and alerting",
        "fix_summary": result.get("fix_summary") or "Apply standard remediation and monitor system",
    }


# ─────────────────────────────
# MAIN CHAIN BUILDER
# ─────────────────────────────
def build_solution_chain():

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.2,
    )

    SYSTEM_PROMPT = """
You are a senior Site Reliability Engineer (SRE).

Generate actionable fixes based on the root cause.

STRICT RULES:
- NEVER return empty fields
- DO NOT give generic advice like "check logs"
- Provide practical, real-world actions

Return ONLY JSON:
{
  "immediate_fix": "...",
  "short_term_fix": "...",
  "long_term_fix": "...",
  "fix_summary": "..."
}

EXAMPLE:

Root cause: Kubernetes insufficient CPU

Output:
{
  "immediate_fix": "Scale up Kubernetes nodes or reduce pod CPU requests",
  "short_term_fix": "Reschedule pods and adjust CPU limits",
  "long_term_fix": "Enable cluster autoscaling and optimize workload distribution",
  "fix_summary": "Increase cluster capacity and optimize CPU usage"
}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", """
Root Cause: {root_cause}
Service: {service_name}
Severity: {severity}
""")
    ])

    chain = prompt | llm | JsonOutputParser()

    # ─────────────────────────────
    # RUN FUNCTION
    # ─────────────────────────────
    def run(data: Dict[str, Any]) -> Dict[str, str]:
        try:
            result = chain.invoke(data)

            # 🔍 Debug (remove later)
            print("DEBUG SOLUTION RAW:", result)

            return _safe_output(result)

        except Exception as e:
            logger.error("Solution generation failed: %s", e)

            return {
                "immediate_fix": "Restart affected service or pod",
                "short_term_fix": "Check system logs and resource usage",
                "long_term_fix": "Improve monitoring, scaling, and alerting",
                "fix_summary": "Fallback remediation applied",
            }

    return run
