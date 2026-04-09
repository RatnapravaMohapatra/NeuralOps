import logging
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = {"Critical", "High", "Medium", "Low"}

SYSTEM_PROMPT = """You are an SRE log parser.

STRICT RULE: Extract ONLY explicit facts. No guessing.

error_type:
- Use exact exception name if present
- Else use short phrase from log (max 3 words)

service_name:
- "Service: X" → X
- URL → /predict → predict-service
- keyword → payment/auth/order → use as service
- else → unknown

severity:
- Critical: full outage
- High: major failure
- Medium: timeout / degraded
- Low: warning

summary:
- Copy EXACT log line containing the error
- DO NOT modify it

Return JSON only:
error_type, service_name, severity, summary
"""

USER_PROMPT = """LOG:
{log_input}"""


def normalize_service_name(name: str) -> str:
    """Ensure consistent service naming"""
    if not name or name == "unknown":
        return "unknown"

    name = name.lower().strip()

    if not name.endswith("-service"):
        name = f"{name}-service"

    return name


def extract_error_type_fallback(log: str) -> str:
    """Fallback if LLM fails"""
    match = re.search(r"([A-Za-z]+Error|Exception)", log)
    return match.group(1) if match else "UnknownError"


def extract_summary_fallback(log: str) -> str:
    """Pick most relevant line"""
    for line in log.split("\n"):
        if any(k in line.lower() for k in ["error", "exception", "timeout", "failed"]):
            return line.strip()
    return log[:300]


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
            logger.error("LLM failed: %s", e)
            return {
                "error_type": extract_error_type_fallback(log_input),
                "service_name": "unknown",
                "severity": "Medium",
                "summary": extract_summary_fallback(log_input),
            }

        result = result or {}

        # ✅ Validate + fix fields
        error_type = result.get("error_type") or extract_error_type_fallback(log_input)
        service_name = normalize_service_name(result.get("service_name"))
        severity = result.get("severity", "Medium")
        summary = result.get("summary") or extract_summary_fallback(log_input)

        if severity not in SEVERITY_LEVELS:
            severity = "Medium"

        final = {
            "error_type": error_type,
            "service_name": service_name,
            "severity": severity,
            "summary": summary,
        }

        logger.info(
            "Parsed → error=%s service=%s severity=%s",
            final["error_type"],
            final["service_name"],
            final["severity"],
        )

        return final

    return analyze
