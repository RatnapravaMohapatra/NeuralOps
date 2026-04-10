import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = {"Critical", "High", "Medium", "Low"}

SYSTEM_PROMPT = """You are an expert SRE log analysis agent.

YOUR ONLY JOB: Extract facts directly from the log. Nothing else.

GOLDEN RULE: "If it is not in the log, it is not in the answer."

════════════════════════════════════════
EXTRACTION RULES
════════════════════════════════════════

error_type:
- Use the exact exception class name if present
  Example: SQLTimeoutException, OutOfMemoryError, NullPointerException
- If no class name → use 3-word description of the error
  Example: "request timed out", "connection refused", "disk space full"
- NEVER invent an error type not visible in the log

service_name — apply in strict priority order, stop at first match:
1. Log says "Service: X" explicitly → use X exactly as written
2. Log contains a URL like /predict, /order, /payment → use as predict-service, order-service
3. Log contains service keyword: payment, auth, order, gateway, recommendation, session → use it
4. Log contains class name like OrderService → convert to order-service format
5. None of the above → use "unknown"

severity — based ONLY on impact described in log:
- Critical: complete outage, data loss, security breach, service fully down
- High: major feature broken, repeated failures, significant user impact
- Medium: single timeout, degraded performance, partial failure
- Low: warning only, no user impact

summary:
- Copy the SINGLE most important line from the log VERBATIM
- Do not paraphrase, summarize, or add context
- This exact line will be used as primary evidence for root cause analysis
- If multiple lines are equally important, pick the one with the error class or signal

Return ONLY valid JSON — no markdown, no explanation:
{
  "error_type": "string",
  "service_name": "string",
  "severity": "Critical|High|Medium|Low",
  "summary": "verbatim copy of most important log line"
}"""

USER_PROMPT = """Extract facts from this log. Do not infer. Do not assume. Do not add context.

LOG:
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
            logger.error("LogAnalyzer failed: %s", e)
            return {
                "error_type": "UnknownError",
                "service_name": "unknown",
                "severity": "Medium",
                "summary": log_input[:300],
            }

        result = result or {}
        result.setdefault("error_type", "UnknownError")
        result.setdefault("service_name", "unknown")
        result.setdefault("severity", "Medium")
        result.setdefault("summary", log_input[:300])

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
