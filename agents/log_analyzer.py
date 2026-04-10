import logging
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = {"Critical", "High", "Medium", "Low"}


SYSTEM_PROMPT = """You are an expert SRE log analysis agent.

YOUR ONLY JOB: Extract facts directly from the log.

GOLDEN RULE:
"If it is not in the log, it is not in the answer."

STRICT:
- DO NOT paraphrase summary
- MUST copy exact line from log
- DO NOT invent anything

Return ONLY JSON:
{
  "error_type": "...",
  "service_name": "...",
  "severity": "...",
  "summary": "..."
}
"""


USER_PROMPT = """Extract facts from this log.

LOG:
{log_input}"""


def normalize_service(service: str) -> str:
    if not service or service == "unknown":
        return "unknown"

    service = service.lower()

    # convert "order" → "order-service"
    if not service.endswith("-service"):
        return f"{service}-service"

    return service


def extract_verbatim_summary(log: str, predicted: str) -> str:
    """
    Ensure summary is actually present in log.
    If not, fallback to closest matching line.
    """
    if predicted and predicted in log:
        return predicted

    lines = log.splitlines()
    for line in lines:
        if predicted.strip()[:20] in line:
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
            logger.error("LogAnalyzer failed: %s", e)
            return {
                "error_type": "UnknownError",
                "service_name": "unknown",
                "severity": "Medium",
                "summary": log_input[:300],
            }

        result = result or {}

        error_type = result.get("error_type") or "UnknownError"
        service_name = normalize_service(result.get("service_name", "unknown"))
        severity = result.get("severity", "Medium")
        summary = extract_verbatim_summary(log_input, result.get("summary", ""))

        if severity not in SEVERITY_LEVELS:
            severity = "Medium"

        final = {
            "error_type": error_type,
            "service_name": service_name,
            "severity": severity,
            "summary": summary,
        }

        logger.info(
            "LogAnalyzer: error_type=%s service=%s severity=%s",
            final["error_type"],
            final["service_name"],
            final["severity"],
        )

        return final

    return analyze
