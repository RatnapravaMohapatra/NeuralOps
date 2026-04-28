import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

CONF_THRESHOLD = 0.5

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
- Use tools, commands, configs
"""

USER_PROMPT = """
Root Cause: {root_cause}
Evidence: {evidence}
Service: {service}
Error: {error_type}
Confidence: {confidence}
"""


def build_fix_agent(api_key: str):
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.1)

    chain = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT)
    ]) | llm | JsonOutputParser()

    def generate_fix(root, parsed):
        conf = root.get("confidence", 0.5)

        root_cause = (root.get("root_cause") or "")[:500]
        evidence = (root.get("evidence") or "")[:300]

        try:
            result = chain.invoke({
                "root_cause": root_cause,
                "evidence": evidence,
                "service": parsed.get("service_name"),
                "error_type": parsed.get("error_type"),
                "confidence": round(conf, 2),
            })
        except Exception as e:
            logger.error("FixAgent failed: %s", e)
            return {
                "immediate_fix": "Restart service",
                "short_term_fix": "Check logs",
                "long_term_fix": "Add monitoring",
                "fix_summary": "Fallback fix"
            }

        if not isinstance(result, dict):
            result = {}

        result["immediate_fix"] = result.get("immediate_fix") or "Check service"
        result["short_term_fix"] = result.get("short_term_fix") or "Investigate issue"
        result["long_term_fix"] = result.get("long_term_fix") or "Improve system"
        result["fix_summary"] = result.get("fix_summary") or "See above"

        return result

    return generate_fix
