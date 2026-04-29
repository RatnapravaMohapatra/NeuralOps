import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = {"Critical", "High", "Medium", "Low"}
MIN_INPUT_LENGTH = 5
MAX_INPUT_LENGTH = 2000


SYSTEM_PROMPT = """You are an expert SRE log analysis agent.

Extract facts ONLY from the log. No assumptions.

Return JSON:
{
  "error_type": "...",
  "service_name": "...",
  "severity": "Critical/High/Medium/Low",
  "summary": "..."
}
"""

USER_PROMPT = "Log:\n{log_input}"


# ─────────────────────────────
# 🔥 RULE-BASED SEVERITY BOOST
# ─────────────────────────────
def _infer_severity_from_log(text: str) -> str:
    text = text.lower()

    if any(k in text for k in ["outofmemory", "oom", "oomkilled"]):
        return "Critical"

    if any(k in text for k in ["insufficient cpu", "failedscheduling"]):
        return "High"

    if any(k in text for k in ["timeout", "timed out"]):
        return "Medium"

    if any(k in text for k in ["warning", "deprecated"]):
        return "Low"

    return "Medium"


# ─────────────────────────────
# MAIN BUILDER
# ─────────────────────────────
def build_log_analyzer(api_key: str | None):

    # 🔥 Fallback mode (no API key)
    if not api_key:
        logger.warning("GROQ_API_KEY missing → using fallback analyzer")

        def fallback(input_data):
            log_input = (
                input_data.get("log_input", "")
                if isinstance(input_data, dict)
                else str(input_data)
            )

            return {
                "error_type": "UnknownError",
                "service_name": "unknown",
                "severity": _infer_severity_from_log(log_input),
                "summary": log_input[:300] or "No input",
            }

        return fallback

    # 🔥 LLM mode
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0,
    )

    chain = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT)
    ]) | llm | JsonOutputParser()

    def analyze(input_data):
        # 🔥 Handle both dict and string input
        log_input = (
            input_data.get("log_input", "")
            if isinstance(input_data, dict)
            else str(input_data)
        )

        # 🔥 Validation
        if not log_input or len(log_input.strip()) < MIN_INPUT_LENGTH:
            return {
                "error_type": "InvalidInput",
                "service_name": "unknown",
                "severity": "Low",
                "summary": "Log input too short or invalid",
            }

        log_input = log_input[:MAX_INPUT_LENGTH]

        try:
            result = chain.invoke({"log_input": log_input})

        except Exception as e:
            logger.error("LogAnalyzer failed: %s", e)
            return {
                "error_type": "UnknownError",
                "service_name": "unknown",
                "severity": _infer_severity_from_log(log_input),
                "summary": log_input[:300],
            }

        if not isinstance(result, dict):
            result = {}

        # 🔥 Final normalization + severity correction
        severity = result.get("severity")
        if severity not in SEVERITY_LEVELS:
            severity = _infer_severity_from_log(log_input)

        return {
            "error_type": result.get("error_type") or "UnknownError",
            "service_name": result.get("service_name") or "unknown",
            "severity": severity,
            "summary": result.get("summary") or log_input[:300],
        }

    return analyze
