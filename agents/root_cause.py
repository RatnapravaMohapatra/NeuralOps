import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

CONF_MIN = 0.15
CONF_MAX = 0.95
CONF_DEFAULT = 0.5

SYSTEM_PROMPT = """You are an AI system that finds root cause from logs.

Return JSON:
{
  "root_cause": "...",
  "evidence": "...",
  "confidence": 0.0,
  "reasoning": "..."
}

Rules:
- confidence must be between 0.15 and 0.95
- never return null
"""

RETRY_PROMPT = """Retry analysis carefully.
Previous confidence was low.
Find even weak signals.
"""

USER_PROMPT = """
Log: {summary}
Error: {error_type}
Service: {service_name}
Severity: {severity}
Context: {rag_context}
"""


def build_root_cause_agent(api_key: str):
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0)

    normal_chain = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT)
    ]) | llm | JsonOutputParser()

    retry_chain = ChatPromptTemplate.from_messages([
        ("system", RETRY_PROMPT),
        ("human", USER_PROMPT)
    ]) | llm | JsonOutputParser()

    def analyze(parsed, rag, retry_count=0, prev_confidence=0.0):
        summary = parsed.get("summary", "")[:500]

        rag_text = "\n".join([
            r.get("root_cause", "") for r in rag
        ]) or "No context"

        chain = retry_chain if retry_count > 0 else normal_chain

        try:
            result = chain.invoke({
                "summary": summary,
                "error_type": parsed.get("error_type"),
                "service_name": parsed.get("service_name"),
                "severity": parsed.get("severity"),
                "rag_context": rag_text
            })
        except Exception as e:
            logger.error("RootCause failed: %s", e)
            return {
                "root_cause": "Unknown cause",
                "evidence": "LLM failure",
                "confidence": CONF_MIN,
                "reasoning": "fallback"
            }

        if not isinstance(result, dict):
            result = {}

        result["root_cause"] = result.get("root_cause") or "Unknown cause"
        result["evidence"] = result.get("evidence") or "No evidence"
        result["reasoning"] = result.get("reasoning") or "No reasoning"

        try:
            conf = float(result.get("confidence", CONF_DEFAULT))
            if conf > 1:
                conf /= 100
        except:
            conf = CONF_DEFAULT

        conf = max(CONF_MIN, min(CONF_MAX, conf))
        result["confidence"] = round(conf, 3)

        return result

    return analyze
