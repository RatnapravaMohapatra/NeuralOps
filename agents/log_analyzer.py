import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = {"Critical", "High", "Medium", "Low"}

SYSTEM_PROMPT = """You are an expert SRE log analysis agent.
Given a raw log, extract structured information.

Use ONLY what is explicitly in the log. Do not assume.

Service name rules:
1. "Service: X" → X
2. URL path → /predict → predict-service
3. Known pattern → payment, auth, api
4. Else → "unknown"

Severity:
Critical / High / Medium / Low

Return ONLY JSON:
- error_type
- service_name
- severity
- summary
"""

USER_PROMPT = """Analyze this log:

{log_input}"""


def build_log_analyzer(groq_api_key: str) -> callable:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])

    parser = JsonOutputParser()
    chain = prompt | llm | parser

    def analyze(log_input: str) -> dict:
        logger.info("LogAnalyzer: parsing log (len=%d)", len(log_input))

        try:
            result = chain.invoke({"log_input": log_input})

        except Exception as e:
            logger.error("LogAnalyzer failed: %s", str(e))

            # ✅ FALLBACK (VERY IMPORTANT)
            return {
                "error_type": "UnknownError",
                "service_name": "unknown",
                "severity": "Medium",
                "summary": log_input[:200]
            }

        # ✅ SAFE GUARDS
        result = result or {}

        result.setdefault("error_type", "UnknownError")
        result.setdefault("service_name", "unknown")
        result.setdefault("severity", "Medium")
        result.setdefault("summary", log_input[:200])

        if result["severity"] not in SEVERITY_LEVELS:
            result["severity"] = "Medium"

        logger.info(
            "LogAnalyzer: error_type=%s service=%s severity=%s",
            result["error_type"],
            result["service_name"],
            result["severity"],
        )

        return result

    return analyze