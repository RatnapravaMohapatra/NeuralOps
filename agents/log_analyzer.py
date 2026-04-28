import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = {"Critical", "High", "Medium", "Low"}
MIN_INPUT_LENGTH = 10
MAX_INPUT_LENGTH = 2000

SYSTEM_PROMPT = """You are an expert SRE log analysis agent.

Extract facts ONLY from the log. No assumptions.

Return JSON:
{
  "error_type": "...",
  "service_name": "...",
  "severity": "...",
  "summary": "..."
}
"""

USER_PROMPT = "Log:\n{log_input}"


def build_log_analyzer(api_key: str):
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0)
    chain = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT)
    ]) | llm | JsonOutputParser()

    def analyze(log_input: str):
        if not log_input or len(log_input.strip()) < MIN_INPUT_LENGTH:
            raise ValueError("Invalid log input")

        log_input = log_input[:MAX_INPUT_LENGTH]

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

        if not isinstance(result, dict):
            result = {}

        result["error_type"] = result.get("error_type") or "UnknownError"
        result["service_name"] = result.get("service_name") or "unknown"
        result["severity"] = result.get("severity") or "Medium"
        result["summary"] = result.get("summary") or log_input[:300]

        if result["severity"] not in SEVERITY_LEVELS:
            result["severity"] = "Medium"

        return result

    return analyze
